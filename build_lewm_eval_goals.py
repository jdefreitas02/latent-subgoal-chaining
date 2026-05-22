#!/usr/bin/env python3
"""
Select 5 diverse in-distribution goal latents from the leworldmodel cache for evaluation.

Uses k-means (k=5) over episode-final latents to find 5 representative,
maximally-spread goal states. These replace the 5 hardcoded OGBench task goals
when the training data doesn't cover those specific configurations.

Output: $STABLEWM_HOME/lewm_eval_goals.pt
  {
    'goal_latents':  Tensor [5, 192]   — the 5 goal latents
    'task_names':    list[str]          — e.g. ['goal_0', ..., 'goal_4']
    'source_ep_ids': list[int]          — which episode each goal came from
    'source_frames': list[int]          — frame index within that episode
  }

Usage:
    python latent_hindsight_rl/build_lewm_eval_goals.py
    python latent_hindsight_rl/build_lewm_eval_goals.py --n_goals 5 --use_all_frames
"""

import os
import sys
import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path',   default=None)
    parser.add_argument('--out_dir',      default=None)
    parser.add_argument('--n_goals',      type=int, default=5,
                        help='Number of goal latents to select (default: 5)')
    parser.add_argument('--use_all_frames', action='store_true',
                        help='Run k-means over ALL latents, not just episode-final ones. '
                             'Better coverage but goals may be mid-episode states.')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    stablewm = os.environ.get('STABLEWM_HOME', os.path.expanduser('~/stable_wm_data'))
    if args.cache_path is None:
        args.cache_path = os.path.join(stablewm, 'lewm_224_latents_cache.pt')
    if args.out_dir is None:
        args.out_dir = stablewm

    print(f'Cache:   {args.cache_path}')
    print(f'Out dir: {args.out_dir}')

    # ── Load cache ───────────────────────────────────────────────────────────────
    print('Loading cache ...')
    cache = torch.load(args.cache_path, map_location='cpu', weights_only=False)
    all_latents = cache['all_latents']   # list of [T, 192] tensors
    n_eps = len(all_latents)
    print(f'  {n_eps} episodes')

    # ── Build candidate pool ─────────────────────────────────────────────────────
    if args.use_all_frames:
        # All latents — richer coverage but goals may be transient mid-episode states
        print('Building candidate pool from ALL frames ...')
        all_z   = torch.cat(all_latents, dim=0).numpy()   # [N_total, 192]
        # Track which (ep, frame) each index came from
        ep_ids_list = []
        frame_ids_list = []
        for ep_i, ep in enumerate(all_latents):
            for t in range(ep.shape[0]):
                ep_ids_list.append(ep_i)
                frame_ids_list.append(t)
        ep_ids    = np.array(ep_ids_list)
        frame_ids = np.array(frame_ids_list)
    else:
        # Episode-final latents only — these are states the expert actually reached
        print('Building candidate pool from episode-final frames ...')
        all_z     = torch.stack([ep[-1] for ep in all_latents]).numpy()   # [N_eps, 192]
        ep_ids    = np.arange(n_eps)
        frame_ids = np.array([ep.shape[0] - 1 for ep in all_latents])

    print(f'  Candidate pool size: {len(all_z):,}  |  latent_dim: {all_z.shape[1]}')

    # ── K-means ─────────────────────────────────────────────────────────────────
    print(f'Running k-means (k={args.n_goals}) ...')
    km = KMeans(n_clusters=args.n_goals, n_init=20, random_state=args.seed)
    km.fit(all_z)
    centers = km.cluster_centers_   # [k, 192]

    # For each cluster, find the nearest actual data point (so goals are real states)
    chosen_latents = []
    chosen_ep_ids  = []
    chosen_frames  = []
    for k in range(args.n_goals):
        dists = np.linalg.norm(all_z - centers[k], axis=-1)
        nearest = int(np.argmin(dists))
        chosen_latents.append(all_z[nearest])
        chosen_ep_ids.append(int(ep_ids[nearest]))
        chosen_frames.append(int(frame_ids[nearest]))

    goal_latents = torch.tensor(np.stack(chosen_latents), dtype=torch.float32)  # [5, 192]
    task_names   = [f'goal_{i}' for i in range(args.n_goals)]

    # ── Diversity check ──────────────────────────────────────────────────────────
    print('\nGoal diversity check (pairwise L2 distances):')
    print(f"  {'':>10}", end='')
    for i in range(args.n_goals):
        print(f"  {task_names[i]:>8}", end='')
    print()
    for i in range(args.n_goals):
        print(f"  {task_names[i]:>10}", end='')
        for j in range(args.n_goals):
            d = float(torch.norm(goal_latents[i] - goal_latents[j]))
            print(f"  {d:>8.2f}", end='')
        print(f"   (ep={chosen_ep_ids[i]}, frame={chosen_frames[i]})")

    min_dist = float('inf')
    for i in range(args.n_goals):
        for j in range(i + 1, args.n_goals):
            d = float(torch.norm(goal_latents[i] - goal_latents[j]))
            if d < min_dist:
                min_dist = d
    print(f'\n  Min pairwise distance: {min_dist:.3f}')
    if min_dist < 5.0:
        print('  WARNING: some goals are close — consider --use_all_frames for more diversity')
    else:
        print('  Goals are well-separated (>5.0 apart)')

    # ── Save ─────────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'lewm_eval_goals.pt')
    torch.save({
        'goal_latents':  goal_latents,           # [5, 192]
        'task_names':    task_names,
        'source_ep_ids': chosen_ep_ids,
        'source_frames': chosen_frames,
        'n_goals':       args.n_goals,
        'use_all_frames': args.use_all_frames,
    }, out_path)
    print(f'\nSaved {args.n_goals} eval goals → {out_path}')
    print('\nPass to eval_ogbench.py:')
    print(f'  --lewm_goals_path {out_path}')


if __name__ == '__main__':
    main()
