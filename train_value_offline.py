"""
train_value_offline.py
Standalone IQL training of V (EnsembleValue) + φ (GoalRep) on a precomputed
latent cache — no actors, no WGSP, no world-model queries.

This produces the value-function checkpoint required by:
  - eval_wgsp_cem_ogbench.py  (train on cube_single_expert cache)
  - train_wgsp_flat_bc.py     (train on visual-cube-single-play-v0_224 cache)

The training is identical to Phase 1 of train_hiql_wgsp.py: two IQL expectile
steps per iteration (subgoal batch + HER goal batch) plus soft target updates.

Usage:
    python latent_hindsight_rl/train_value_offline.py \\
        --cache_path /path/to/lewm_224_latents_cache.pt \\
        --dataset_path /path/to/cube_single_expert \\
        --save_dir ./checkpoints_value_offline_expert

Output:
    {save_dir}/value_offline.pt — dict with keys:
        value      : EnsembleValue state_dict
        goal_rep   : GoalRep state_dict
        rep_dim    : int
        n_heads    : int
        step       : int
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from sklearn import preprocessing as sk_pre

_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import stable_worldmodel as swm

from train_hiql_wgsp import (
    GoalRep,
    EnsembleValue,
    RealOfflineCache,
    _value_step,
)


def _bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('1', 'true', 'yes')


def train_value(
    real_cache,
    goal_rep,       goal_rep_target,
    value_net,      value_target,
    value_optimizer,
    total_steps, batch_size, subgoal_steps,
    gamma, tau, expectile,
    save_dir, device,
    log_interval=500,
    save_interval=10_000,
):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'value_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'v_loss', 'elapsed_s'])

    t0 = time.time()
    sum_v = 0.0
    log_n = 0

    print(f"\n{'='*55}")
    print(f"  Offline V + φ training")
    print(f"  total_steps    : {total_steps:,}")
    print(f"  batch_size     : {batch_size}")
    print(f"  subgoal_steps  : {subgoal_steps}")
    print(f"  gamma / tau    : {gamma} / {tau}")
    print(f"  expectile      : {expectile}")
    print(f"{'='*55}\n")

    for step in range(total_steps):
        # Subgoal-batch V update (r = -||z' - z_sub||)
        z, _, r, z_next, z_sub = real_cache.sample_ll_batch(
            batch_size, subgoal_steps, use_5d=False)
        v_loss = _value_step(
            value_net, value_target, goal_rep, goal_rep_target,
            z, z_next, z_sub, r, gamma, expectile, value_optimizer)
        sum_v += v_loss

        # HER-goal auxiliary V update
        z_t, z_next_t, g_her = real_cache.sample_value_her_batch(batch_size)
        r_her = -torch.norm(z_next_t - g_her, p=2, dim=-1)
        sum_v += _value_step(
            value_net, value_target, goal_rep, goal_rep_target,
            z_t, z_next_t, g_her, r_her, gamma, expectile, value_optimizer)

        # Soft-update target networks
        with torch.no_grad():
            for tp, p in zip(value_target.parameters(), value_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)
            for tp, p in zip(goal_rep_target.parameters(), goal_rep.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data * tau)

        log_n += 1

        if step % log_interval == 0:
            d = max(log_n, 1)
            elapsed = time.time() - t0
            avg_v = sum_v / (2 * d)
            print(f"  step {step:>7,}/{total_steps:,}  v_loss={avg_v:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, avg_v, elapsed])
            sum_v = 0.0
            log_n = 0

        if (step + 1) % save_interval == 0 or step == total_steps - 1:
            ckpt = {
                'value':    value_net.state_dict(),
                'goal_rep': goal_rep.state_dict(),
                'rep_dim':  goal_rep.rep_dim,
                'n_heads':  value_net.n_heads,
                'step':     step + 1,
            }
            out = os.path.join(save_dir, 'value_offline.pt')
            torch.save(ckpt, out)
            print(f"  Saved checkpoint → {out}  (step {step+1:,})", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline IQL: train V + φ only')
    parser.add_argument('--cache_path',   type=str, required=True,
                        help='Path to precomputed latent cache .pt file '
                             '(all_latents + all_actions, produced by analyse_lewm_224.py).')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the HDF5 dataset (without .h5) used only '
                             'to fit the action StandardScaler.')
    parser.add_argument('--save_dir',     type=str, default=None)

    parser.add_argument('--total_steps',    type=int,   default=100_000)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--subgoal_steps',  type=int,   default=8)
    parser.add_argument('--rep_dim',        type=int,   default=10)
    parser.add_argument('--n_heads',        type=int,   default=2)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--gamma',          type=float, default=0.99)
    parser.add_argument('--tau',            type=float, default=0.005)
    parser.add_argument('--expectile',      type=float, default=0.7)
    parser.add_argument('--seed',           type=int,   default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tag = os.path.basename(args.cache_path.rstrip('/').replace('.pt', ''))
    save_dir = args.save_dir or (
        f'./checkpoints_value_offline_{tag}_rep{args.rep_dim}_s{args.seed}')

    # ── Load latent cache ────────────────────────────────────────────────────
    print(f'Loading cache from {args.cache_path} ...')
    cache_data  = torch.load(args.cache_path, map_location='cpu', weights_only=False)
    all_latents = cache_data['all_latents']
    all_actions = cache_data.get('all_actions', [])
    if not all_actions:
        raise RuntimeError("Cache has no 'all_actions'.")

    # ── Fit action scaler ────────────────────────────────────────────────────
    print(f'Fitting StandardScaler on {args.dataset_path} ...')
    _ds = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=['action'],
        cache_dir=os.path.dirname(args.dataset_path))
    action_raw = _ds.get_col_data('action')
    action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
    action_scaler = sk_pre.StandardScaler()
    action_scaler.fit(action_raw)
    print(f'  Scaler fit on {len(action_raw):,} frames')

    # ── Dataset ──────────────────────────────────────────────────────────────
    real_cache = RealOfflineCache(
        all_latents=all_latents,
        all_actions=all_actions,
        action_scaler=action_scaler,
        device=device,
        frameskip=5,
    )

    # ── Networks ─────────────────────────────────────────────────────────────
    LATENT_DIM  = 192
    HIDDEN_DIMS = (512, 512, 512)

    goal_rep = GoalRep(
        latent_dim=LATENT_DIM, rep_dim=args.rep_dim,
        hidden_dims=HIDDEN_DIMS, layer_norm=True).to(device)
    goal_rep_target = GoalRep(
        latent_dim=LATENT_DIM, rep_dim=args.rep_dim,
        hidden_dims=HIDDEN_DIMS, layer_norm=True).to(device)
    goal_rep_target.load_state_dict(goal_rep.state_dict())
    for p in goal_rep_target.parameters():
        p.requires_grad = False

    value_net = EnsembleValue(
        latent_dim=LATENT_DIM, rep_dim=args.rep_dim,
        hidden_dims=HIDDEN_DIMS, n_heads=args.n_heads).to(device)
    value_target = EnsembleValue(
        latent_dim=LATENT_DIM, rep_dim=args.rep_dim,
        hidden_dims=HIDDEN_DIMS, n_heads=args.n_heads).to(device)
    value_target.load_state_dict(value_net.state_dict())
    for p in value_target.parameters():
        p.requires_grad = False

    value_optimizer = torch.optim.Adam(
        list(value_net.parameters()) + list(goal_rep.parameters()),
        lr=args.lr,
    )

    train_value(
        real_cache=real_cache,
        goal_rep=goal_rep,             goal_rep_target=goal_rep_target,
        value_net=value_net,           value_target=value_target,
        value_optimizer=value_optimizer,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        subgoal_steps=args.subgoal_steps,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        save_dir=save_dir,
        device=device,
    )
    print('Done.')
