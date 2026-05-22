"""
Diagnostic: use the HIQL baseline LL actor (good, 62%) as the first-action
input to the WGSP decoder, with the HIQL baseline HL.

If this scores near 62%: the decoder + adapter pipeline is fine and the WGSP
LL actor itself was degraded.
If this also scores 0%: the decoder/adapter pipeline is fundamentally broken.

Usage:
    python latent_hindsight_rl/eval_baseline_through_decoder.py
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.transforms import v2 as transforms

_parent = os.path.abspath(os.path.dirname(__file__))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import stable_worldmodel as swm
from sklearn import preprocessing

sys.path.insert(0, str(Path(__file__).parent))
from eval_ogbench import (
    _LewmGaussianActor, _ActionChunkDecoder, _LatentAdapter, _load_adapter,
    _WGSPHierarchicalPolicy, _HIQLBaselineHighLevelWrapper,
    _BaselineGaussianActor, encode_and_project, load_jepa,
    _make_img_transform, run_task, nn_clamp_subgoal,
)
import eval_ogbench as _eval_mod


class _BaselineLLThroughDecoder:
    """HIQL baseline HL + HIQL baseline LL → feed LL output through WGSP decoder.

    The baseline LL outputs 5D scaled actions (the "first action anchor").
    We pass those through the pre-trained WGSP decoder to get a 25D chunk.
    """

    def __init__(self, jepa_model, baseline_ll, baseline_hl, decoder,
                 action_scaler, gap, device, adapter, rep_dim=10):
        self.jepa_model     = jepa_model
        self.actor          = baseline_ll
        self.high_level     = baseline_hl
        self.decoder        = decoder
        self.action_scaler  = action_scaler
        self.gap            = gap
        self.device         = device
        self.adapter        = adapter
        self.rep_dim        = rep_dim
        self._buf           = []
        self._wm_steps      = 0
        self._z_subgoal     = None
        self._subgoal_switches = 0

    def reset(self):
        self._buf, self._wm_steps = [], 0
        self._z_subgoal, self._subgoal_switches = None, 0

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        diag = None
        if not self._buf:
            z_raw = encode_and_project(self.jepa_model, obs_hwc, self.device)
            if goal_latent is not None:
                z_raw_goal = goal_latent.to(self.device)
            else:
                z_raw_goal = encode_and_project(self.jepa_model, goal_hwc, self.device)

            with torch.no_grad():
                # Adapt for HL and LL
                z_curr = self.adapter(z_raw)
                z_goal = self.adapter(z_raw_goal)

                switched = (self._z_subgoal is None or self._wm_steps >= self.gap)
                if switched:
                    self._z_subgoal = self.high_level.predict(z_curr, z_goal)
                    self._wm_steps = 0
                    self._subgoal_switches += 1

                # Use BASELINE LL for the first-action anchor
                _, _, ll_out_5d = self.actor.sample(z_curr, self._z_subgoal)  # (1,5)

                # Pass baseline LL output through WGSP decoder
                chunk_25d = self.decoder(ll_out_5d, z_raw)  # (1,25)

            self._wm_steps += 1
            raw      = chunk_25d.cpu().numpy().reshape(5, 5)
            physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)
            diag = {
                'dist_to_goal':         torch.norm(z_raw - z_raw_goal, p=2, dim=-1).item(),
                'dist_to_subgoal':      float('nan'),
                'dist_subgoal_to_goal': float('nan'),
                'subgoal_switches':     self._subgoal_switches,
                'subgoal_switched':     switched,
                'action_norm':          float(np.mean(np.abs(physical))),
            }
            self._buf = [physical[t] for t in range(5)]
        return self._buf.pop(0), diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wgsp_ckpt',     default='checkpoints_hiql_wgsp_row1_headline_s0')
    parser.add_argument('--baseline_ckpt', default='checkpoints_hiql_baseline_k10_rep10_adapter256')
    parser.add_argument('--decoder_ckpt',  default='checkpoints_action_decoder/action_decoder.pth')
    parser.add_argument('--dataset_path',  default=None)
    parser.add_argument('--ckpt_path',     default=None)
    parser.add_argument('--num_episodes',  type=int, default=10)
    parser.add_argument('--rep_dim',       type=int, default=10)
    parser.add_argument('--img_size',      type=int, default=224)
    parser.add_argument('--patch_size',    type=int, default=14)
    parser.add_argument('--diagnose',      action='store_true')
    parser.add_argument('--ogbench_dir',   default=None)
    args = parser.parse_args()

    stablewm_home = os.environ.get('STABLEWM_HOME',
                                   os.path.join(os.path.expanduser('~'), 'stable_wm_data'))
    if args.dataset_path is None:
        args.dataset_path = os.path.join(stablewm_home, 'ogbench', 'visual-cube-single-play-v0_224')
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(stablewm_home, 'cube', 'lejepa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.ogbench_dir:
        sys.path.insert(0, os.path.abspath(args.ogbench_dir))
    import ogbench
    import gymnasium

    env = gymnasium.make('swm/OGBCube-v0', ob_type='pixels', env_type='single', visualize_info=False)
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

    print("Fitting action scaler...")
    dataset = swm.data.HDF5Dataset(args.dataset_path, keys_to_cache=['action'],
                                   cache_dir=str(Path(args.dataset_path).parent))
    action_data = dataset.get_col_data('action')
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Fit on {len(action_data):,} steps.")

    _eval_mod._IMG_TRANSFORM = _make_img_transform(args.img_size)
    print(f"Loading JEPA...")
    jepa_model = load_jepa(args.ckpt_path, device, img_size=args.img_size, patch_size=args.patch_size)

    # --- Shared adapter (from WGSP checkpoint, same architecture as baseline) ---
    wgsp_dir = Path(args.wgsp_ckpt)
    wgsp_adapter, _, policy_dim = _load_adapter(wgsp_dir / 'adapter.pth', device)
    print(f"Adapter: 192 → {policy_dim}")

    # --- WGSP decoder ---
    dec_ckpt = wgsp_dir / 'action_decoder.pth'
    if not dec_ckpt.exists():
        dec_ckpt = Path(args.decoder_ckpt)
    decoder = _ActionChunkDecoder(in_dim=5, out_dim=25, latent_dim=192, hidden_dims=(256, 256)).to(device)
    decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
    decoder.eval()
    for p in decoder.parameters(): p.requires_grad_(False)
    print(f"Decoder from {dec_ckpt}")

    # --- HIQL baseline HL ---
    bl_dir = Path(args.baseline_ckpt)
    hl_actor_net = _BaselineGaussianActor(
        input_dim=policy_dim * 2, output_dim=args.rep_dim,
        hidden_dims=(512, 512, 512), tanh_squash=False,
    ).to(device)
    hl_actor_net.load_state_dict(torch.load(bl_dir / 'hl_actor.pth', map_location=device))
    hl_actor_net.eval()
    for p in hl_actor_net.parameters(): p.requires_grad_(False)
    hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)
    print(f"Baseline HL loaded (input={policy_dim*2}, out={args.rep_dim})")

    # --- HIQL baseline LL (direct 5D, same input_dim as WGSP LL) ---
    ll_actor = _BaselineGaussianActor(
        input_dim=policy_dim + args.rep_dim,  # 266
        output_dim=5,
        hidden_dims=(512, 512, 512), action_scale=1.0, tanh_squash=False,
    ).to(device)
    ll_actor.load_state_dict(torch.load(bl_dir / 'll_actor.pth', map_location=device))
    ll_actor.eval()
    for p in ll_actor.parameters(): p.requires_grad_(False)
    print(f"Baseline LL loaded (input={policy_dim+args.rep_dim}, out=5)")

    # --- Build policy ---
    policy = _BaselineLLThroughDecoder(
        jepa_model, ll_actor, hl_model, decoder,
        action_scaler, gap=8, device=device,
        adapter=wgsp_adapter, rep_dim=args.rep_dim,
    )

    print("\n" + "="*60)
    print("Diagnostic: HIQL baseline HL + baseline LL → WGSP decoder")
    print(f"  {len(task_infos)} tasks × {args.num_episodes} eps each")
    print("="*60)

    t0 = time.time()
    per_task = {}
    all_successes = []
    for task_id in range(1, len(task_infos) + 1):
        task_name = task_infos[task_id - 1].get('task_name', f'task{task_id}')
        mean_sr, eps = run_task(
            env, policy, task_id, task_name, args.num_episodes, 200,
            diagnose=args.diagnose, goal_info_key='target')
        per_task[task_name] = mean_sr
        all_successes.extend(eps)

    overall = np.mean(all_successes)
    print(f"\n{'='*60}")
    print("RESULTS: baseline HL + baseline LL → WGSP decoder")
    print(f"{'='*60}")
    for name, sr in per_task.items():
        print(f"  {name:<30s} {sr*100:5.1f}%")
    print(f"  {'─'*36}")
    print(f"  {'overall':<30s} {overall*100:5.1f}%")
    print(f"  elapsed: {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
