"""Diagnostic B: how good is the WM at planning a short action sequence that
takes z_A -> z_B for arbitrary close latent pairs from the offline dataset?

Method: for each sampled pair (z_A, z_B) with initial L2 distance in a chosen
bin, run a simple CEM over action sequences:
  - population = 64
  - iterations = 5
  - top-k = 8
  - sequence length = h (5 real-env steps per WM step)
  - cost = ||WM(z_A, actions) - z_B||_2
Report per (initial_distance_bin, horizon):
  - median achieved L2 (after CEM) vs initial L2
  - fraction of pairs whose achieved L2 < initial L2 / 2
  - fraction of pairs whose achieved L2 < 1.0 (median intra-D NN distance from diag 2)
This isolates whether the WM is a usable short-range navigator. The Q-learning
direction we tried treats the WM as a transition generator with bootstrap; the
hierarchical alternative needs the WM to behave as a planner over a short
horizon. This script tests the latter directly.
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn import preprocessing

from envs.jepa_loader import load_jepa


LATENT_DIM = 192
ACTION_CHUNK_DIM = 25  # 5 sub-actions x 5 dims per WM step


def _load_latents_flat(cache_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"] if isinstance(cache, dict) and "all_latents" in cache else cache
    out = []
    for ep in all_latents:
        z = ep.cpu().numpy() if torch.is_tensor(ep) else np.asarray(ep)
        out.append(z.astype(np.float32))
    flat = np.concatenate(out, axis=0)
    return flat


def _load_action_scaler(hdf5_path):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][...].astype(np.float32)
    a = actions[~np.isnan(actions).any(axis=1)]
    sc = preprocessing.StandardScaler()
    sc.fit(a)
    return sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)


def sample_pairs(Z, target_l2, tol, n, rng):
    """Sample n pairs (i, j) of indices into Z whose L2 distance is in
    [target_l2 - tol, target_l2 + tol]. Brute-force rejection sampling.
    Returns indices arrays (i, j) length <=n."""
    N = Z.shape[0]
    out_i, out_j = [], []
    attempts = 0
    max_attempts = n * 200
    while len(out_i) < n and attempts < max_attempts:
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            attempts += 1
            continue
        d = float(np.linalg.norm(Z[i] - Z[j]))
        if abs(d - target_l2) <= tol:
            out_i.append(i)
            out_j.append(j)
        attempts += 1
    return np.array(out_i, dtype=np.int64), np.array(out_j, dtype=np.int64)


@torch.no_grad()
def wm_rollout_batch(jepa, z0, action_chunks_25, sm_t, ss_t):
    """Batched WM rollout.
    Args:
      z0:                (B, 192)  -- start latent
      action_chunks_25:  (B, h, 25) -- h chunks of 25-D
      sm_t, ss_t:        (1, 1, 5) torch -- scaler mean/std
    Returns:
      z_final: (B, 192)
    """
    device = z0.device
    B, h, _ = action_chunks_25.shape
    z = z0.unsqueeze(1).contiguous()  # (B, 1, 192)
    for k in range(h):
        a = action_chunks_25[:, k].view(B, 1, 5, 5)
        a_scaled = (a - sm_t) / ss_t
        a_flat = a_scaled.reshape(B, 1, 25)
        act_emb = jepa.action_encoder(a_flat)
        z_next = jepa.predict(z, act_emb)[:, -1:]   # (B, 1, 192)
        z = z_next
    return z.squeeze(1)


def cem_plan(jepa, z_A, z_B, h, sm_t, ss_t, device, n_iters=5, pop=64, topk=8,
             init_std=0.5, min_std=0.05):
    """CEM in 25-D-per-chunk action space, h chunks, isotropic Gaussian per element.
    Returns (best_actions_25 (h, 25), best_l2_to_z_B).
    """
    mean = torch.zeros(h, 25, device=device)
    std = torch.full((h, 25), init_std, device=device)
    z_A_t = torch.from_numpy(z_A).to(device).unsqueeze(0).repeat(pop, 1)  # (pop, 192)
    z_B_t = torch.from_numpy(z_B).to(device)                              # (192,)
    best_l2 = float("inf")
    best_a = None
    for it in range(n_iters):
        # Sample population (pop, h, 25), clip per 5-D sub-action to [-1, 1]
        eps = torch.randn(pop, h, 25, device=device)
        a = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        a = a.clamp(-1.0, 1.0)
        # Rollout
        z_f = wm_rollout_batch(jepa, z_A_t, a, sm_t, ss_t)               # (pop, 192)
        dists = torch.norm(z_f - z_B_t.unsqueeze(0), p=2, dim=1)         # (pop,)
        # Track best
        idx_min = int(torch.argmin(dists).item())
        if float(dists[idx_min].item()) < best_l2:
            best_l2 = float(dists[idx_min].item())
            best_a = a[idx_min].detach().cpu().numpy()
        # Refit Gaussian on top-k
        topk_idx = torch.topk(-dists, topk).indices
        elite = a[topk_idx]                                              # (topk, h, 25)
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(min_std)
    return best_a, best_l2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--latent_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--hdf5_dataset", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_pairs_per_bin", type=int, default=100)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--bins", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 8.0, 16.0],
                   help="Target L2 distances between z_A and z_B for sampling bins.")
    p.add_argument("--bin_tol", type=float, default=0.5,
                   help="Acceptance tolerance around target L2 when sampling pairs.")
    p.add_argument("--cem_pop", type=int, default=64)
    p.add_argument("--cem_iters", type=int, default=5)
    p.add_argument("--cem_topk", type=int, default=8)
    p.add_argument("--cem_init_std", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="diagnostics_wm_planner")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] JEPA from {args.wm_ckpt}", flush=True)
    jepa = load_jepa(args.wm_ckpt, device=args.device, img_size=224, patch_size=14)
    print(f"[load] flat latents from {args.latent_cache}", flush=True)
    Z = _load_latents_flat(args.latent_cache)
    print(f"  Z: {Z.shape}")
    sm, ss = _load_action_scaler(args.hdf5_dataset)
    sm_t = torch.from_numpy(sm).to(args.device).view(1, 1, 5)
    ss_t = torch.from_numpy(ss).to(args.device).view(1, 1, 5)

    # Random baseline: just take a single sample of uniform random actions per pair, no CEM.
    @torch.no_grad()
    def random_baseline(z_A, z_B, h, n=64):
        z_A_t = torch.from_numpy(z_A).to(args.device).unsqueeze(0).repeat(n, 1)
        z_B_t = torch.from_numpy(z_B).to(args.device)
        a = (torch.rand(n, h, 25, device=args.device) * 2.0 - 1.0)
        z_f = wm_rollout_batch(jepa, z_A_t, a, sm_t, ss_t)
        d = torch.norm(z_f - z_B_t.unsqueeze(0), p=2, dim=1)
        return float(torch.min(d).item())

    # Sweep
    rows = []
    abs_thresh = 1.0  # "reached" if final dist < this (median intra-D NN dist ~0.78 from diag 2)
    for target_l2 in args.bins:
        i_idx, j_idx = sample_pairs(Z, target_l2, args.bin_tol, args.n_pairs_per_bin, rng)
        actual = len(i_idx)
        if actual == 0:
            print(f"  bin={target_l2}: no pairs found within tol={args.bin_tol}", flush=True)
            continue
        print(f"\n=== bin: initial L2 ~ {target_l2:.2f} (±{args.bin_tol})  n_pairs={actual} ===",
              flush=True)
        for h in args.horizons:
            t0 = time.time()
            init_l2 = []
            cem_l2 = []
            rand_l2 = []
            for k in range(actual):
                z_A = Z[i_idx[k]]
                z_B = Z[j_idx[k]]
                init_l2.append(float(np.linalg.norm(z_A - z_B)))
                best_a, best_l2 = cem_plan(
                    jepa, z_A, z_B, h, sm_t, ss_t, args.device,
                    n_iters=args.cem_iters, pop=args.cem_pop, topk=args.cem_topk,
                    init_std=args.cem_init_std,
                )
                cem_l2.append(best_l2)
                rand_l2.append(random_baseline(z_A, z_B, h, n=args.cem_pop))
            init_l2 = np.array(init_l2)
            cem_l2 = np.array(cem_l2)
            rand_l2 = np.array(rand_l2)
            frac_half = float((cem_l2 < init_l2 / 2.0).mean())
            frac_abs  = float((cem_l2 < abs_thresh).mean())
            print(f"  h={h:2d}  median: init={np.median(init_l2):.3f}  "
                  f"CEM={np.median(cem_l2):.3f}  rand={np.median(rand_l2):.3f}  "
                  f"frac(<init/2)={frac_half:.3f}  frac(<{abs_thresh:.1f})={frac_abs:.3f}  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
            rows.append(dict(
                bin=float(target_l2), h=int(h), n=int(actual),
                init_l2_median=float(np.median(init_l2)),
                cem_l2_median=float(np.median(cem_l2)),
                rand_l2_median=float(np.median(rand_l2)),
                cem_l2_p25=float(np.percentile(cem_l2, 25)),
                cem_l2_p75=float(np.percentile(cem_l2, 75)),
                frac_below_init_half=frac_half,
                frac_below_abs_thresh=frac_abs,
                abs_thresh=float(abs_thresh),
            ))

    # Save
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(dict(rows=rows, args=vars(args)), f, indent=2)
    print(f"\nWritten to {args.out_dir}/report.json")

    # Pretty-print compact summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"  {'bin_init_l2':<13} {'h':<3} {'CEM_med':<9} {'rand_med':<9} "
          f"{'frac<init/2':<12} {'frac<abs':<10}")
    for r in rows:
        print(f"  {r['bin']:<13.2f} {r['h']:<3d} {r['cem_l2_median']:<9.3f} "
              f"{r['rand_l2_median']:<9.3f} {r['frac_below_init_half']:<12.3f} "
              f"{r['frac_below_abs_thresh']:<10.3f}")


if __name__ == "__main__":
    main()
