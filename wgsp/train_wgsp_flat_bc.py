"""
train_wgsp_flat_bc.py
WGSP-weighted flat behavioural cloning.

Trains a single goal-conditioned policy π(a | z_t, g) on the offline dataset,
where each transition is weighted by the WGSP scoring function

    J(z_{t+k}, g) = V(z_{t+k}, φ(z_{t+k}, g)) − β‖z_{t+k} − g‖₂

with g a HER-sampled future goal from the same episode.  The BC loss is

    L = −Σ_i exp(α(J_i − mean J)) · log π(a_i | z_i, g_i)

where V + φ come from a value_offline.pt checkpoint trained by
train_value_offline.py on the SAME dataset.

This tests whether WGSP scoring improves BC outside the HIQL framework —
no hierarchical structure, no world-model rollouts.

Usage:
    python latent_hindsight_rl/train_wgsp_flat_bc.py \\
        --cache_path /path/to/play_latents_cache.pt \\
        --dataset_path /path/to/visual-cube-single-play-v0_224 \\
        --value_ckpt  /path/to/checkpoints_value_offline_play/value_offline.pt \\
        --save_dir ./checkpoints_wgsp_flat_bc_s0

    # Geometry-only (no V) ablation:
        --use_v_in_J False

    # Value-only (no geometry) ablation:
        --use_geometric_term False

    # Vanilla BC (no weighting):
        --alpha_weight 0.0

Output:
    {save_dir}/flat_bc.pt — dict with keys:
        actor     : GaussianActor state_dict (input: 192+192 → output: 25)
        step      : int
        args      : training args dict

Evaluation:
    Use eval_wgsp_flat_bc.py (which uses swm.World + evaluate_from_dataset
    for a fair comparison with the LeWorldModel paper results) or run a
    simple sanity check with the quick_eval() function at the bottom of
    this file.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from sklearn import preprocessing as sk_pre

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import stable_worldmodel as swm

from train_hiql_wgsp import (
    GoalRep,
    EnsembleValue,
    GaussianActor,
    RealOfflineCache,
    _score_endpoints,
)


def _bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('1', 'true', 'yes')


def train_flat_bc(
    real_cache,
    actor,            actor_optimizer,
    value_net,        goal_rep,
    total_steps, batch_size, subgoal_steps,
    beta_geom, alpha_weight,
    use_v_in_J, use_geometric_term,
    save_dir, device,
    log_interval=500,
    save_interval=10_000,
    actor_args=None,
):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'bc_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'bc_loss', 'j_mean', 'j_std', 'elapsed_s'])

    vanilla_bc = (alpha_weight == 0.0)

    print(f"\n{'='*60}")
    print(f"  WGSP-weighted flat BC training")
    print(f"  vanilla_bc       : {vanilla_bc}  (alpha_weight={alpha_weight})")
    print(f"  use_v_in_J       : {use_v_in_J}")
    print(f"  use_geometric    : {use_geometric_term}  (β={beta_geom})")
    print(f"  total_steps      : {total_steps:,}")
    print(f"  batch_size       : {batch_size}")
    print(f"  subgoal_steps    : {subgoal_steps}")
    print(f"{'='*60}\n")

    t0 = time.time()
    sum_loss = 0.0
    sum_j    = 0.0
    sum_j2   = 0.0
    log_n    = 0

    for step in range(total_steps):
        # Sample (z_t, a_t, z_{t+k}, g_HER) from the same episode
        # z_{t+k} = subgoal used for WGSP scoring
        # g_HER   = HER-sampled future goal (farther than z_{t+k})
        idx = torch.randint(0, real_cache.total, (batch_size,), device=device)
        z_t   = real_cache.z_t_flat[idx]             # (B, 192)
        a_t   = real_cache.a_flat[idx]               # (B, 25)
        z_tk  = real_cache._kstep_latents(idx, subgoal_steps)   # (B, 192)
        g_idx, ep_id, _, _ = real_cache._her_future_idxs(idx)
        g_flat = real_cache.ep_wm_offsets[ep_id] + g_idx
        g_her  = real_cache.wm_latents_flat[g_flat]  # (B, 192)

        # Compute WGSP score for each dataset transition
        with torch.no_grad():
            J = _score_endpoints(
                z_k=z_tk,
                g_ult=g_her,
                value_net=value_net,
                goal_rep=goal_rep,
                beta_geom=beta_geom,
                use_geometric_term=use_geometric_term,
                use_v_in_J=use_v_in_J,
                lambda_mopo=0.0,
            )   # (B,)

        # AWR-style weights: softmax(α·(J − mean(J)))
        if vanilla_bc:
            w = torch.ones_like(J)
        else:
            J_adv = J - J.mean()
            w = torch.softmax(alpha_weight * J_adv, dim=0) * batch_size
            w = w.detach()

        # BC loss: −Σ_i w_i · log π(a_t^(i) | z_t^(i), g^(i))
        # Actor takes (state, goal) where goal is the full 192-D latent
        log_p = actor.log_prob(z_t, g_her, a_t)     # (B,)
        loss  = -(w * log_p).mean()

        actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()

        sum_loss += loss.item()
        sum_j    += J.mean().item()
        sum_j2   += J.pow(2).mean().item()
        log_n    += 1

        if step % log_interval == 0:
            d     = max(log_n, 1)
            avg_l = sum_loss / d
            avg_j = sum_j / d
            std_j = (max(sum_j2 / d - avg_j ** 2, 0)) ** 0.5
            elapsed = time.time() - t0
            print(f"  step {step:>7,}/{total_steps:,}  bc_loss={avg_l:.4f}  "
                  f"J_mean={avg_j:.3f}  J_std={std_j:.3f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, avg_l, avg_j, std_j, elapsed])
            sum_loss = sum_j = sum_j2 = 0.0
            log_n = 0

        if (step + 1) % save_interval == 0 or step == total_steps - 1:
            ckpt = {
                'actor': actor.state_dict(),
                'step':  step + 1,
                'args':  actor_args or {},
            }
            out = os.path.join(save_dir, 'flat_bc.pt')
            torch.save(ckpt, out)
            print(f"  Saved → {out}  (step {step+1:,})", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGSP-weighted flat BC')
    parser.add_argument('--cache_path',   type=str, required=True,
                        help='Precomputed latent cache .pt (play dataset).')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='HDF5 dataset path (without .h5) for action scaler.')
    parser.add_argument('--value_ckpt',   type=str, required=True,
                        help='value_offline.pt from train_value_offline.py '
                             '(trained on the SAME play dataset cache).')
    parser.add_argument('--save_dir',     type=str, default=None)

    parser.add_argument('--total_steps',   type=int,   default=200_000)
    parser.add_argument('--batch_size',    type=int,   default=256)
    parser.add_argument('--subgoal_steps', type=int,   default=8,
                        help='k — the future step z_{t+k} used for WGSP scoring.')
    parser.add_argument('--alpha_weight',  type=float, default=3.0,
                        help='AWR temperature. Set to 0 for vanilla BC.')
    parser.add_argument('--beta_geom',     type=float, default=0.1)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--action_scale',  type=float, default=3.0)

    parser.add_argument('--use_v_in_J',         type=_bool, default=True,
                        help='Include V(z_k, φ) in the WGSP score.')
    parser.add_argument('--use_geometric_term',  type=_bool, default=True,
                        help='Include β‖z_k − g‖₂ in the WGSP score.')
    parser.add_argument('--seed',  type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vanilla = (args.alpha_weight == 0.0)
    tag = (
        f"_vanilla" if vanilla else
        f"_v{int(args.use_v_in_J)}_g{int(args.use_geometric_term)}"
        f"_a{args.alpha_weight}_b{args.beta_geom}"
    )
    save_dir = args.save_dir or f'./checkpoints_wgsp_flat_bc{tag}_s{args.seed}'

    # ── Load latent cache ────────────────────────────────────────────────────
    print(f'Loading cache from {args.cache_path} ...')
    cache_data  = torch.load(args.cache_path, map_location='cpu', weights_only=False)
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'.")

    # ── Fit action scaler ────────────────────────────────────────────────────
    print(f'Fitting action scaler on {args.dataset_path} ...')
    _ds = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=['action'],
        cache_dir=os.path.dirname(args.dataset_path))
    action_raw = _ds.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Fit on {len(action_raw):,} frames')

    # ── Dataset ──────────────────────────────────────────────────────────────
    real_cache = RealOfflineCache(
        all_latents=all_latents,
        all_actions=all_actions,
        action_scaler=action_scaler,
        device=device,
        frameskip=5,
    )

    # ── Load V + φ ────────────────────────────────────────────────────────────
    print(f'Loading V + φ from {args.value_ckpt} ...')
    vc = torch.load(args.value_ckpt, map_location='cpu', weights_only=False)
    rep_dim = vc['rep_dim']
    n_heads = vc['n_heads']

    goal_rep = GoalRep(
        latent_dim=192, rep_dim=rep_dim,
        hidden_dims=(512, 512, 512), layer_norm=True).to(device).eval()
    goal_rep.load_state_dict(vc['goal_rep'])
    goal_rep.requires_grad_(False)

    value_net = EnsembleValue(
        latent_dim=192, rep_dim=rep_dim,
        hidden_dims=(512, 512, 512), n_heads=n_heads).to(device).eval()
    value_net.load_state_dict(vc['value'])
    value_net.requires_grad_(False)
    print(f'  rep_dim={rep_dim}, n_heads={n_heads}, '
          f'trained for {vc.get("step","?")} steps')

    # ── Flat BC actor: π(a | z_t, g_her) ─────────────────────────────────────
    # goal_dim=192 so the policy takes the full goal latent (no rep compression)
    actor = GaussianActor(
        state_dim=192, goal_dim=192, output_dim=25,
        hidden_dims=(512, 512, 512),
        tanh_squash=True, action_scale=args.action_scale,
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    actor_args = dict(
        state_dim=192, goal_dim=192, output_dim=25,
        action_scale=args.action_scale,
        use_v_in_J=args.use_v_in_J,
        use_geometric_term=args.use_geometric_term,
        alpha_weight=args.alpha_weight,
        beta_geom=args.beta_geom,
        rep_dim=rep_dim,
        n_heads=n_heads,
        seed=args.seed,
    )

    train_flat_bc(
        real_cache=real_cache,
        actor=actor,                  actor_optimizer=actor_optimizer,
        value_net=value_net,          goal_rep=goal_rep,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        subgoal_steps=args.subgoal_steps,
        beta_geom=args.beta_geom,
        alpha_weight=args.alpha_weight,
        use_v_in_J=args.use_v_in_J,
        use_geometric_term=args.use_geometric_term,
        save_dir=save_dir,
        device=device,
        actor_args=actor_args,
    )
    print('Done.')
