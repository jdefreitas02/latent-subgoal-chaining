"""
eval_cem_over_reps.py  —  Row 12 ablation: CEM-over-reps baseline.

Loads a trained WGSP checkpoint (V, φ, LL, decoder, WM) but discards the
trained HL actor.  At every subgoal_steps the subgoal rep is chosen by CEM
in 10-D rep space: a population of reps is scored via WM rollout + V + geometric
residual (exactly the same J^(i,m) as WGSP training), the elite set is refitted,
and the best rep is passed to the LL actor.

This isolates the contribution of "trained HL via AWR" vs "search at test time
with the same WM + V".  If CEM-over-reps ≈ WGSP, the gain is mostly from the
scoring (WM + V + geometry), not the AWR update on π^H.

Usage (224×224 WGSP checkpoint, with decoder):
    python latent_hindsight_rl/eval_cem_over_reps.py \\
        --ckpt_dir ./checkpoints_hiql_wgsp_... \\
        --decoder_ckpt ./checkpoints_action_decoder/action_decoder.pth \\
        --use_decoder True \\
        --ll_out_dim 5 \\
        --results_dir ./eval_cem_over_reps_results

CEM hyper-parameters (sensible defaults, can be overridden):
    --cem_pop       64   candidates per iteration
    --cem_elites     8   elite set size
    --cem_iters      3   CEM iterations per subgoal decision
    --cem_k          8   WM rollout horizon for scoring (matches k_plan)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torchvision.transforms import v2 as transforms

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import stable_pretraining as spt
import stable_worldmodel as swm
from sklearn import preprocessing as sk_pre

from train_hiql_wgsp import (
    GoalRep, GaussianActor, EnsembleValue,
    wgsp_rollout, _score_endpoints, _load_jepa_from_ckpt,
)
from train_action_decoder import ActionChunkDecoder


# =============================================================================
# Image encoding helpers
# =============================================================================

_IMG_TRANSFORM = None


def _make_img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def _encode(wm_model, obs_hwc, device):
    """Encode a single (H, W, C) uint8 obs → [1, 192] latent."""
    t = _IMG_TRANSFORM(obs_hwc).unsqueeze(0).unsqueeze(0)   # [1, 1, C, H, W]
    with torch.no_grad():
        emb = wm_model.encode({'pixels': t.to(device)})['emb']  # [1, 1, 192]
    return emb[:, -1]                                            # [1, 192]


# =============================================================================
# CEM in rep space
# =============================================================================

@torch.no_grad()
def cem_select_rep(
    z_curr,           # [1, 192]
    g_ult,            # [1, 192]
    wm_model,
    ll_actor,
    decoder,
    goal_rep,
    value_net,
    rep_dim,
    k,
    beta_geom,
    use_decoder,
    use_geometric_term,
    use_v_in_J,
    action_scale,
    pop,
    elites,
    iters,
    device,
):
    """Select the best rep for (z_curr, g_ult) via CEM in rep space.

    Returns:
        best_rep [1, rep_dim]  — best elite rep (mean of final elite set)
        best_score float
    """
    # Initialise from unit Gaussian on the sphere
    mu  = torch.zeros(rep_dim, device=device)
    std = torch.ones(rep_dim, device=device) * (rep_dim ** 0.5)

    best_rep   = None
    best_score = -1e9

    for _ in range(iters):
        # Sample population and length-normalise onto sphere of radius √rep_dim
        eps = torch.randn(pop, rep_dim, device=device)
        reps_raw = mu.unsqueeze(0) + std.unsqueeze(0) * eps          # [pop, rep_dim]
        reps = reps_raw / (reps_raw.norm(dim=-1, keepdim=True) + 1e-8) * (rep_dim ** 0.5)

        # Tile (z_curr, g_ult) to match population size
        z_pop = z_curr.expand(pop, -1)      # [pop, 192]
        g_pop = g_ult.expand(pop, -1)       # [pop, 192]

        # WM rollout for all pop reps (M=1 rollout each — stochastic but cheap)
        traj_z, _, _ = wgsp_rollout(
            wm_model, ll_actor, decoder,
            z_pop, g_pop, reps,
            k=k, action_scale=action_scale, use_decoder=use_decoder)
        z_k = traj_z[:, -1, :]             # [pop, 192]

        scores = _score_endpoints(
            z_k, g_pop, value_net, goal_rep,
            beta_geom, use_geometric_term, use_v_in_J,
            lambda_mopo=0.0, return_diag=False)  # [pop]

        # Elite update: refit Gaussian to top-elites reps (pre-normalisation)
        topk = scores.topk(elites).indices   # [elites]
        elite_raws = reps_raw[topk]          # [elites, rep_dim]
        mu  = elite_raws.mean(dim=0)
        std = elite_raws.std(dim=0) + 1e-6

        best_idx = scores.argmax().item()
        if scores[best_idx].item() > best_score:
            best_score = scores[best_idx].item()
            best_rep   = reps[best_idx].unsqueeze(0)   # [1, rep_dim]

    return best_rep, best_score


# =============================================================================
# Policy class
# =============================================================================

class CEMOverRepsPolicy:
    """Drop-in policy for the OGBench eval runner.

    Replaces the trained HL actor with CEM search in rep space.
    Everything else (LL actor, decoder, WM, V, φ) is identical to
    the trained WGSP checkpoint.
    """

    def __init__(
        self,
        wm_model, ll_actor, decoder,
        goal_rep, value_net,
        action_scaler,
        rep_dim, k, beta_geom,
        use_decoder, use_geometric_term, use_v_in_J,
        action_scale,
        subgoal_steps,
        cem_pop, cem_elites, cem_iters,
        device,
    ):
        self.wm_model  = wm_model
        self.ll_actor  = ll_actor
        self.decoder   = decoder
        self.goal_rep  = goal_rep
        self.value_net = value_net
        self.action_scaler = action_scaler

        self.rep_dim   = rep_dim
        self.k         = k
        self.beta_geom = beta_geom
        self.use_decoder         = use_decoder
        self.use_geometric_term  = use_geometric_term
        self.use_v_in_J          = use_v_in_J
        self.action_scale        = action_scale
        self.subgoal_steps       = subgoal_steps
        self.cem_pop     = cem_pop
        self.cem_elites  = cem_elites
        self.cem_iters   = cem_iters
        self.device      = device

        self._buf             = []
        self._wm_steps        = 0
        self._current_rep     = None
        self._subgoal_switches = 0

    def reset(self):
        self._buf              = []
        self._wm_steps         = 0
        self._current_rep      = None
        self._subgoal_switches = 0

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        diag = None
        if not self._buf:
            z_curr = _encode(self.wm_model, obs_hwc, self.device)  # [1, 192]
            if goal_latent is not None:
                g_ult = goal_latent.to(self.device)
            else:
                g_ult = _encode(self.wm_model, goal_hwc, self.device)  # [1, 192]

            switched = (self._current_rep is None or
                        self._wm_steps >= self.subgoal_steps)
            if switched:
                self._current_rep, _ = cem_select_rep(
                    z_curr, g_ult,
                    self.wm_model, self.ll_actor, self.decoder,
                    self.goal_rep, self.value_net,
                    self.rep_dim, self.k, self.beta_geom,
                    self.use_decoder, self.use_geometric_term, self.use_v_in_J,
                    self.action_scale,
                    self.cem_pop, self.cem_elites, self.cem_iters,
                    self.device,
                )
                self._wm_steps         = 0
                self._subgoal_switches += 1

            with torch.no_grad():
                _, _, ll_out = self.ll_actor.sample(z_curr, self._current_rep)
                if self.decoder is not None:
                    chunk_25d = self.decoder(ll_out, z_curr)
                else:
                    chunk_25d = ll_out

            self._wm_steps += 1
            raw      = chunk_25d.cpu().numpy().reshape(5, 5)
            physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)
            diag = {
                'dist_to_goal':         torch.norm(z_curr - g_ult, p=2, dim=-1).item(),
                'subgoal_switches':     self._subgoal_switches,
                'subgoal_switched':     switched,
                'action_norm':          float(np.mean(np.abs(physical))),
            }
            self._buf = [physical[t] for t in range(5)]
        return self._buf.pop(0), diag


# =============================================================================
# Eval runner
# =============================================================================

def run_task(env, policy, task_id, task_name, num_episodes, max_steps,
             goal_info_key='target', diagnose=False):
    successes = []
    for ep in range(num_episodes):
        obs, info = env.reset(options=dict(task_id=task_id))
        goal = info[goal_info_key]
        policy.reset()
        done = False
        step = 0
        wm_step = 0
        while not done and step < max_steps:
            action, diag = policy.get_action(obs, goal)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if diag is not None:
                wm_step += 1
                if diagnose and ep == 0 and wm_step % 4 == 0:
                    print(
                        f"  [task={task_id} ep={ep} phys={step:3d}]"
                        f"  d2g={diag['dist_to_goal']:.3f}"
                        f"  switches={diag['subgoal_switches']}"
                        f"  action_norm={diag['action_norm']:.3f}",
                        flush=True,
                    )
        successes.append(float(info.get('success', 0.0)))
    mean_sr = np.mean(successes)
    print(f"  Task {task_id:d} ({task_name}): {mean_sr*100:5.1f}%  "
          f"({int(sum(successes))}/{num_episodes})", flush=True)
    return mean_sr, successes


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Row 12: CEM-over-reps baseline — trained HL replaced by CEM.')
    parser.add_argument('--ckpt_dir',      required=True,
                        help='WGSP checkpoint directory (hl_actor.pth is ignored).')
    parser.add_argument('--decoder_ckpt',  default=None,
                        help='Path to action_decoder.pth (required if --use_decoder True).')
    parser.add_argument('--wm_ckpt_path',  default=None,
                        help='LeJEPA checkpoint dir (defaults to $STABLEWM_HOME/cube/lejepa).')
    parser.add_argument('--dataset_path',  default=None,
                        help='HDF5 dataset for action scaler fit.')
    parser.add_argument('--results_dir',   default=None)

    parser.add_argument('--rep_dim',       type=int,   default=10)
    parser.add_argument('--latent_dim',    type=int,   default=192)
    parser.add_argument('--ll_out_dim',    type=int,   default=25,
                        help='5 if --use_decoder True, else 25.')
    parser.add_argument('--action_scale',  type=float, default=3.0)
    parser.add_argument('--n_value_heads', type=int,   default=2)

    def _bool(s): return str(s).lower() in ('1', 'true', 'yes', 'y', 't')
    parser.add_argument('--use_decoder',        type=_bool, default=False)
    parser.add_argument('--use_geometric_term', type=_bool, default=True)
    parser.add_argument('--use_v_in_J',         type=_bool, default=True)

    parser.add_argument('--subgoal_steps', type=int,   default=8)
    parser.add_argument('--k',             type=int,   default=8,
                        help='WM rollout horizon for CEM scoring.')
    parser.add_argument('--beta_geom',     type=float, default=0.1)

    parser.add_argument('--cem_pop',    type=int,   default=64)
    parser.add_argument('--cem_elites', type=int,   default=8)
    parser.add_argument('--cem_iters',  type=int,   default=3)

    parser.add_argument('--img_size',    type=int,   default=224)
    parser.add_argument('--patch_size',  type=int,   default=14)
    parser.add_argument('--num_episodes', type=int,  default=50)
    parser.add_argument('--max_steps',    type=int,  default=200)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--seed',         type=int,  default=0)
    parser.add_argument('--diagnose',     action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    STABLEWM_HOME = os.environ.get(
        'STABLEWM_HOME', os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    wm_ckpt  = args.wm_ckpt_path or os.path.join(STABLEWM_HOME, 'cube', 'lejepa')
    data_path = args.dataset_path or os.path.join(STABLEWM_HOME, 'ogbench', 'cube_single_expert')
    results_dir = args.results_dir or (
        f'./eval_cem_over_reps_{os.path.basename(args.ckpt_dir.rstrip("/"))}')
    os.makedirs(results_dir, exist_ok=True)

    # ── Env ───────────────────────────────────────────────────────────────────
    import ogbench   # noqa: F401 — registers gymnasium envs
    import gymnasium
    if args.img_size == 64:
        env = gymnasium.make('visual-cube-single-v0')
        goal_info_key = 'goal'
    else:
        env = gymnasium.make('swm/OGBCube-v0', ob_type='pixels',
                             env_type='single', visualize_info=False)
        goal_info_key = 'target'
    task_infos = (env.unwrapped.task_infos
                  if hasattr(env.unwrapped, 'task_infos') else env.task_infos)
    num_tasks  = len(task_infos)
    print(f'Env: {num_tasks} tasks')

    # ── Action scaler ─────────────────────────────────────────────────────────
    print('Fitting action scaler ...')
    ds = swm.data.HDF5Dataset(data_path, keys_to_cache=['action'],
                              cache_dir=str(os.path.dirname(data_path)))
    a_raw = ds.get_col_data('action')
    a_raw = a_raw[~np.isnan(a_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(a_raw)
    print(f'  Scaler fit on {len(a_raw):,} steps')

    # ── Image transform ───────────────────────────────────────────────────────
    global _IMG_TRANSFORM
    _IMG_TRANSFORM = _make_img_transform()

    # ── WM ────────────────────────────────────────────────────────────────────
    print(f'Loading WM from {wm_ckpt} ...')
    wm_model = _load_jepa_from_ckpt(wm_ckpt, device, args.img_size, args.patch_size)

    # ── WGSP networks ─────────────────────────────────────────────────────────
    HIDDEN = (512, 512, 512)
    goal_rep = GoalRep(latent_dim=args.latent_dim, rep_dim=args.rep_dim,
                       hidden_dims=HIDDEN, layer_norm=True).to(device).eval()
    ll_actor = GaussianActor(state_dim=args.latent_dim, goal_dim=args.rep_dim,
                             output_dim=args.ll_out_dim, hidden_dims=HIDDEN,
                             tanh_squash=True,
                             action_scale=args.action_scale).to(device).eval()
    value_net = EnsembleValue(latent_dim=args.latent_dim, rep_dim=args.rep_dim,
                              hidden_dims=HIDDEN,
                              n_heads=args.n_value_heads).to(device).eval()

    goal_rep.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, 'goal_rep.pth'), map_location=device))
    ll_actor.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, 'll_actor.pth'), map_location=device))
    # Load value_net — handles legacy TwinValue checkpoints automatically
    value_sd = torch.load(os.path.join(args.ckpt_dir, 'value_net.pth'), map_location=device)
    try:
        value_net.load_state_dict(value_sd)
    except RuntimeError:
        # Legacy TwinValue: remap v1./v2. → heads.0./heads.1.
        from train_hiql_wgsp import _load_legacy_twin_value
        _load_legacy_twin_value(value_net, value_sd)

    decoder = None
    if args.use_decoder:
        dec_path = (args.decoder_ckpt or
                    os.path.join(args.ckpt_dir, 'action_decoder.pth'))
        decoder = ActionChunkDecoder(in_dim=5, out_dim=25,
                                     latent_dim=args.latent_dim,
                                     hidden_dims=(256, 256)).to(device).eval()
        decoder.load_state_dict(torch.load(dec_path, map_location=device))
        for p in decoder.parameters():
            p.requires_grad = False
        print(f'  Loaded ActionChunkDecoder from {dec_path}')

    for net in (goal_rep, ll_actor, value_net):
        for p in net.parameters():
            p.requires_grad = False

    print(f'\nCEM config: pop={args.cem_pop} elites={args.cem_elites} '
          f'iters={args.cem_iters} k={args.k}')

    policy = CEMOverRepsPolicy(
        wm_model=wm_model, ll_actor=ll_actor, decoder=decoder,
        goal_rep=goal_rep, value_net=value_net,
        action_scaler=action_scaler,
        rep_dim=args.rep_dim, k=args.k, beta_geom=args.beta_geom,
        use_decoder=args.use_decoder,
        use_geometric_term=args.use_geometric_term,
        use_v_in_J=args.use_v_in_J,
        action_scale=args.action_scale,
        subgoal_steps=args.subgoal_steps,
        cem_pop=args.cem_pop, cem_elites=args.cem_elites,
        cem_iters=args.cem_iters,
        device=device,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results = {}
    all_srs     = []
    t0 = time.time()
    for i, ti in enumerate(task_infos):
        task_name = ti.get('task_name', f'task_{i}')
        sr, succs = run_task(
            env, policy, task_id=i,
            task_name=task_name,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            goal_info_key=goal_info_key,
            diagnose=args.diagnose,
        )
        all_results[task_name] = {'success_rate': sr, 'successes': succs}
        all_srs.append(sr)

    mean_sr = np.mean(all_srs)
    elapsed = time.time() - t0
    print(f'\nMean success rate: {mean_sr*100:.1f}%  ({elapsed/60:.1f} min)')

    out = {
        'ckpt_dir':   args.ckpt_dir,
        'mean_sr':    mean_sr,
        'tasks':      all_results,
        'cem_pop':    args.cem_pop,
        'cem_elites': args.cem_elites,
        'cem_iters':  args.cem_iters,
        'k':          args.k,
        'beta_geom':  args.beta_geom,
        'use_decoder': args.use_decoder,
        'use_geometric_term': args.use_geometric_term,
        'use_v_in_J': args.use_v_in_J,
    }
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
