#!/usr/bin/env python3
"""
Convert the LeWM latent cache to an OGBench-compatible numpy dataset for
use with HIQL / GCIVL / GCBC baselines in ogbench/impls/.

Dataset format (WM-step granularity: 1 step = 5 physical frames):
  observations  [N, 192]  float32  — encoder latents
  actions       [N, 25]   float32  — StandardScaler-normalised (5 frames × 5D)
  terminals     [N]       float32  — 1.0 at the LAST step of each episode
  rewards       [N]       float32  — all zeros (goal-conditioned, computed at sample time)
  valids        [N]       float32  — 0.0 at last step (for next-obs shifting)

Also saves action scaler params so the eval wrapper can denormalise at test time.

Outputs (all under $STABLEWM_HOME):
  lewm_latent_dataset.npz   — train + val split
  lewm_action_scaler.npz    — StandardScaler mean_ and scale_ (5D)

Usage:
    export STABLEWM_HOME=~/stable_wm_data
    python latent_hindsight_rl/build_ogbench_latent_dataset.py
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


FRAMESKIP = 5   # WM-step = 5 physical frames, matching train_joint.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path",    default=None,
                        help="Path to lewm_224_latents_cache.pt")
    parser.add_argument("--dataset_path",  default=None,
                        help="HDF5 dataset (without .h5) for fitting action scaler")
    parser.add_argument("--out_dir",       default=None,
                        help="Output directory (default: $STABLEWM_HOME)")
    parser.add_argument("--val_fraction",  type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    stablewm = os.environ.get("STABLEWM_HOME", os.path.expanduser("~/stable_wm_data"))
    if args.cache_path   is None: args.cache_path   = os.path.join(stablewm, "lewm_224_latents_cache.pt")
    if args.dataset_path is None: args.dataset_path = os.path.join(stablewm, "ogbench", "cube_single_expert")
    if args.out_dir      is None: args.out_dir      = stablewm

    print(f"Cache:   {args.cache_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Out:     {args.out_dir}")

    # ── Load cache ──────────────────────────────────────────────────────────────
    print("\nLoading latent cache ...")
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"]   # list of [T_raw, 192]  (T_raw = raw frame count ≈ 201)
    all_actions = cache["all_actions"]   # list of [T_raw, 5]    (5D physical action per frame)
    n_eps = len(all_latents)
    print(f"  {n_eps} episodes, "
          f"{all_latents[0].shape[0]} raw frames each, "
          f"action_dim={all_actions[0].shape[-1]}")

    raw_action_dim = all_actions[0].shape[-1]   # 5
    wm_action_dim  = FRAMESKIP * raw_action_dim  # 25

    # ── Fit action scaler on dataset raw 5D actions ─────────────────────────────
    print("Fitting action scaler ...")
    import stable_worldmodel as swm
    dataset = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=["action"],
        cache_dir=str(Path(args.dataset_path).parent),
    )
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler()
    scaler.fit(action_data)
    print(f"  Scaler fit on {len(action_data):,} frames  "
          f"(mean abs scale: {np.mean(np.abs(scaler.scale_)):.4f})")

    # ── Build flat WM-step arrays ────────────────────────────────────────────────
    print("Building flat arrays ...")
    obs_list   = []
    acts_list  = []
    terms_list = []
    ep_lengths = []   # WM-step counts per episode

    for ep_z_raw, ep_a_raw in zip(all_latents, all_actions):
        T_raw = ep_z_raw.shape[0]          # raw frame count
        n_wm  = (T_raw - 1) // FRAMESKIP  # complete WM-step transitions available
        if n_wm <= 0:
            continue

        # WM-step latents at frames 0, 5, 10, …  (inclusive of final step)
        step_idx = torch.arange(0, (n_wm + 1) * FRAMESKIP, FRAMESKIP)[:n_wm + 1]
        ep_z_wm  = ep_z_raw[step_idx].numpy().astype(np.float32)  # [n_wm+1, 192]

        # Scale raw 5D actions then stack into 25D per WM-step
        n_raw    = n_wm * FRAMESKIP
        ep_a_np  = ep_a_raw[:n_raw].numpy().astype(np.float32)    # [n_wm*5, 5]
        ep_a_sc  = scaler.transform(ep_a_np)                      # [n_wm*5, 5]
        ep_a_wm  = ep_a_sc.reshape(n_wm, wm_action_dim)           # [n_wm, 25]

        # Use z_t (not z_t+1) as observations — same convention as CacheTransitionMixer
        obs_list.append(ep_z_wm[:-1])   # [n_wm, 192] — state at start of each WM-step
        acts_list.append(ep_a_wm)        # [n_wm, 25]

        t_arr        = np.zeros(n_wm, dtype=np.float32)
        t_arr[-1]    = 1.0               # terminal at last WM-step of episode
        terms_list.append(t_arr)
        ep_lengths.append(n_wm)

    observations = np.concatenate(obs_list,   axis=0)   # [N, 192]
    actions      = np.concatenate(acts_list,  axis=0)   # [N, 25]
    terminals    = np.concatenate(terms_list, axis=0)   # [N]
    rewards      = np.zeros(len(observations), dtype=np.float32)
    valids       = (1.0 - terminals).astype(np.float32)

    n_valid_eps = len(ep_lengths)
    N           = len(observations)
    print(f"  Valid episodes: {n_valid_eps}  |  Total WM-step transitions: {N:,}  "
          f"(avg {np.mean(ep_lengths):.1f} steps/ep)")
    print(f"  Observations: {observations.shape}  "
          f"Actions: {actions.shape}  "
          f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    # ── Train / val split at episode boundaries ──────────────────────────────────
    rng = np.random.RandomState(args.seed)
    ep_order   = rng.permutation(n_valid_eps)
    n_val      = max(1, int(n_valid_eps * args.val_fraction))
    val_set    = set(ep_order[:n_val].tolist())
    train_set  = set(ep_order[n_val:].tolist())

    # Build flat index arrays for each split (preserving episode order within split)
    ep_starts_flat = np.cumsum([0] + ep_lengths[:-1])

    def gather(ep_set):
        idxs = []
        for ep_i in sorted(ep_set):
            s = ep_starts_flat[ep_i]
            idxs.extend(range(s, s + ep_lengths[ep_i]))
        return np.array(idxs, dtype=np.int64)

    tr_idx = gather(train_set)
    va_idx = gather(val_set)
    print(f"  Train: {len(tr_idx):,} steps ({len(train_set)} eps)  "
          f"Val: {len(va_idx):,} steps ({len(val_set)} eps)")

    # Sanity checks expected by GCDataset
    assert terminals[tr_idx[-1]] == 1.0, "Train split must end on a terminal"
    assert terminals[va_idx[-1]] == 1.0, "Val split must end on a terminal"

    # ── Save dataset ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "lewm_latent_dataset.npz")
    np.savez(
        out_path,
        train_observations = observations[tr_idx],
        train_actions      = actions[tr_idx],
        train_terminals    = terminals[tr_idx],
        train_rewards      = rewards[tr_idx],
        train_valids       = valids[tr_idx],
        val_observations   = observations[va_idx],
        val_actions        = actions[va_idx],
        val_terminals      = terminals[va_idx],
        val_rewards        = rewards[va_idx],
        val_valids         = valids[va_idx],
    )
    print(f"\nSaved dataset → {out_path}")

    scaler_path = os.path.join(args.out_dir, "lewm_action_scaler.npz")
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Saved action scaler → {scaler_path}")
    print("\nDone. Next step:")
    print("  python ogbench/impls/run_latent.py --agent agents/hiql.py \\")
    print(f"      --latent_dataset_path {out_path} \\")
    print(f"      --action_scaler_path {scaler_path}")


if __name__ == "__main__":
    main()
