"""Diagnostics for the proposed "anchored restart rollouts" online phase.

Two questions:
  (1) How fast does the JEPA world model's prediction error compound along
      a real trajectory? -> determines the safe branch length h.
  (2) Is the JEPA latent space well-conditioned for nearest-neighbor anchoring
      to the offline dataset? -> determines whether anchoring has signal at all.

If both come back positive, variant B (anchored restart rollouts) is worth
implementing.

Output: ./diagnostics_anchor/diag_report.json + a couple of .npz/.png files.
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import h5py
from sklearn import preprocessing

from envs.jepa_loader import load_jepa


LATENT_DIM = 192


def _load_latents(cache_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"] if isinstance(cache, dict) and "all_latents" in cache else cache
    out = []
    for ep in all_latents:
        if torch.is_tensor(ep):
            ep = ep.cpu().numpy()
        out.append(ep.astype(np.float32))
    return out


def _load_actions(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][...].astype(np.float32)
        ep_len = f["ep_len"][...].astype(np.int64)
        ep_offset = f["ep_offset"][...].astype(np.int64)
    per_ep = [actions[o:o + L] for o, L in zip(ep_offset, ep_len)]
    return per_ep, actions


def _fit_scaler(action_data):
    a = action_data[~np.isnan(action_data).any(axis=1)]
    sc = preprocessing.StandardScaler()
    sc.fit(a)
    return sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)


@torch.no_grad()
def diagnostic_compounding(jepa, latents_per_ep, actions_per_ep,
                           scaler_mean, scaler_std, device, n_traj=128,
                           K=20, stride=5, seed=0):
    """For each sampled trajectory, predict z_k iteratively from z_0 using the
    real action chunks, and compare to the real cached z_5k.

    Returns arrays of cosine sim and L2 distance, shape (n_traj, K+1) where
    column 0 is the (trivially perfect) start state.
    """
    rng = np.random.default_rng(seed)
    n_eps = len(latents_per_ep)
    cosines = np.full((n_traj, K + 1), np.nan, dtype=np.float32)
    l2s = np.full((n_traj, K + 1), np.nan, dtype=np.float32)
    z_real_norms = np.full((n_traj, K + 1), np.nan, dtype=np.float32)
    z_pred_drift_to_NN = np.full((n_traj, K + 1), np.nan, dtype=np.float32)  # filled below

    sm = torch.from_numpy(scaler_mean).to(device).view(1, 1, 5)
    ss = torch.from_numpy(scaler_std).to(device).view(1, 1, 5)

    ep_indices = rng.choice(n_eps, size=min(n_traj, n_eps), replace=False)

    pred_traces = []  # list of (K+1, 192) predicted latents for diag 2 use
    real_traces = []  # list of (K+1, 192) real latents (z_ep[stride*k])

    for i, ep_idx in enumerate(ep_indices):
        z_ep = latents_per_ep[ep_idx]      # (T_z, 192)
        a_ep = actions_per_ep[ep_idx]      # (T_a, 5)
        T_pair = min(z_ep.shape[0], a_ep.shape[0] + 1)
        num_chunks = (T_pair - 1) // stride
        if num_chunks < K:
            continue
        z_state = torch.from_numpy(z_ep[0]).to(device).view(1, 1, LATENT_DIM).contiguous()
        cos0 = 1.0
        l20 = 0.0
        cosines[i, 0] = cos0
        l2s[i, 0] = l20
        z_real_norms[i, 0] = float(np.linalg.norm(z_ep[0]))

        pred_traj = [z_ep[0].copy()]
        real_traj = [z_ep[0].copy()]

        for k in range(K):
            chunk = a_ep[stride * k:stride * (k + 1)].reshape(1, 1, 5, 5)
            a = torch.from_numpy(chunk.astype(np.float32)).to(device).view(1, 1, 5, 5)
            a_scaled = (a - sm) / ss
            a_flat = a_scaled.reshape(1, 1, 25)
            act_emb = jepa.action_encoder(a_flat)
            z_next = jepa.predict(z_state, act_emb)[:, -1:]
            z_state = z_next

            z_pred = z_next.squeeze().detach().cpu().numpy()
            z_real = z_ep[stride * (k + 1)]
            denom = (np.linalg.norm(z_pred) * np.linalg.norm(z_real) + 1e-8)
            cosines[i, k + 1] = float(np.dot(z_pred, z_real) / denom)
            l2s[i, k + 1] = float(np.linalg.norm(z_pred - z_real))
            z_real_norms[i, k + 1] = float(np.linalg.norm(z_real))
            pred_traj.append(z_pred.copy())
            real_traj.append(z_real.copy())

        pred_traces.append(np.stack(pred_traj))
        real_traces.append(np.stack(real_traj))

    return cosines, l2s, z_real_norms, pred_traces, real_traces


def diagnostic_nn_quality(latents_per_ep, n_query=2000, exclude_window=10, seed=0):
    """For each query latent, find its nearest neighbor in the offline dataset
    excluding same-episode within `exclude_window` steps. Compare to several
    references:
      - within-trajectory step distance ||z_{t+1} - z_t||
      - random-pair distance between any two latents
      - distance from z_t to its NN (the anchor candidate)
    """
    rng = np.random.default_rng(seed)
    # Flatten all latents into one big array, with bookkeeping arrays
    ep_idx_flat = np.concatenate([
        np.full(z.shape[0], i, dtype=np.int32) for i, z in enumerate(latents_per_ep)
    ])
    step_idx_flat = np.concatenate([
        np.arange(z.shape[0], dtype=np.int32) for z in latents_per_ep
    ])
    Z = np.concatenate(latents_per_ep, axis=0).astype(np.float32)
    N = Z.shape[0]

    # Normalize for cosine NN
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    # Try FAISS; fall back to torch on GPU otherwise.
    try:
        import faiss
        index = faiss.IndexFlatIP(LATENT_DIM)
        index.add(Zn)
        use_faiss = True
    except Exception:
        index = None
        use_faiss = False

    # Sample queries from random positions across the dataset
    query_idxs = rng.choice(N, size=min(n_query, N), replace=False)

    K_FETCH = max(2 * exclude_window + 5, 32)  # over-fetch to allow filtering

    nn_l2 = np.full(query_idxs.shape, np.nan, dtype=np.float32)
    nn_cos = np.full(query_idxs.shape, np.nan, dtype=np.float32)
    nn_ep = np.full(query_idxs.shape, -1, dtype=np.int32)
    nn_step = np.full(query_idxs.shape, -1, dtype=np.int32)
    q_ep = ep_idx_flat[query_idxs]
    q_step = step_idx_flat[query_idxs]

    # We don't need to worry about FAISS chunking — 2000x192 is tiny
    if use_faiss:
        sims, idxs = index.search(Zn[query_idxs], K_FETCH)  # (n_query, K_FETCH)
    else:
        Zt = torch.from_numpy(Zn).cuda()
        Qt = torch.from_numpy(Zn[query_idxs]).cuda()
        s = Qt @ Zt.T
        sims_t, idxs_t = torch.topk(s, K_FETCH, dim=1)
        sims = sims_t.detach().cpu().numpy()
        idxs = idxs_t.detach().cpu().numpy()

    for i in range(len(query_idxs)):
        for r in range(K_FETCH):
            cand = idxs[i, r]
            if cand == query_idxs[i]:
                continue
            if ep_idx_flat[cand] == q_ep[i] and abs(step_idx_flat[cand] - q_step[i]) <= exclude_window:
                continue
            # Accept this neighbor
            nn_cos[i] = sims[i, r]
            nn_l2[i] = float(np.linalg.norm(Z[query_idxs[i]] - Z[cand]))
            nn_ep[i] = ep_idx_flat[cand]
            nn_step[i] = step_idx_flat[cand]
            break

    # Reference: within-trajectory consecutive step distance
    step_dists = []
    for z in latents_per_ep:
        if z.shape[0] < 2:
            continue
        step_dists.append(np.linalg.norm(z[1:] - z[:-1], axis=1))
    step_dists = np.concatenate(step_dists) if step_dists else np.array([np.nan])

    # Reference: random-pair distance
    n_rand = min(50000, N * (N - 1) // 2)
    a = rng.integers(0, N, size=n_rand)
    b = rng.integers(0, N, size=n_rand)
    keep = a != b
    a, b = a[keep], b[keep]
    rand_l2 = np.linalg.norm(Z[a] - Z[b], axis=1)
    rand_cos = (Zn[a] * Zn[b]).sum(axis=1)

    return dict(
        nn_l2=nn_l2,
        nn_cos=nn_cos,
        nn_ep=nn_ep,
        nn_step=nn_step,
        q_ep=q_ep,
        q_step=q_step,
        step_l2=step_dists,
        rand_l2=rand_l2,
        rand_cos=rand_cos,
    )


def diagnostic_pred_vs_anchor_distance(pred_traces, real_traces, latents_per_ep, K, seed=0):
    """After each predicted step, what's the distance from z_pred to its NN
    in the offline dataset? If anchoring works, this should be small enough
    (< some ε) for the prediction to be 'caught' before drifting too far.
    """
    # Build flat normalized index once
    Z = np.concatenate(latents_per_ep, axis=0).astype(np.float32)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    try:
        import faiss
        index = faiss.IndexFlatIP(LATENT_DIM)
        index.add(Zn)
        use_faiss = True
    except Exception:
        index = None
        use_faiss = False

    # For each (predicted) latent along each rollout, find NN distance.
    pred_l2_to_nn = np.full((len(pred_traces), K + 1), np.nan, dtype=np.float32)
    pred_cos_to_nn = np.full((len(pred_traces), K + 1), np.nan, dtype=np.float32)
    real_l2_to_nn  = np.full((len(real_traces), K + 1), np.nan, dtype=np.float32)
    real_cos_to_nn = np.full((len(real_traces), K + 1), np.nan, dtype=np.float32)

    def _nn_dists(traces, l2_out, cos_out):
        flat = np.concatenate(traces, axis=0).astype(np.float32)
        flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        if use_faiss:
            sims, idxs = index.search(flat_n, 1)
        else:
            Zt = torch.from_numpy(Zn).cuda()
            Qt = torch.from_numpy(flat_n).cuda()
            s = Qt @ Zt.T
            sims_t, idxs_t = torch.max(s, dim=1, keepdim=True)
            sims, idxs = sims_t.detach().cpu().numpy(), idxs_t.detach().cpu().numpy()
        # Compute L2 against retrieved unnormalized vectors
        nn_vecs = Z[idxs[:, 0]]
        l2 = np.linalg.norm(flat - nn_vecs, axis=1)
        cos = sims[:, 0]
        # Stamp back into (n_traj, K+1)
        offsets = np.cumsum([0] + [t.shape[0] for t in traces])
        for i, t in enumerate(traces):
            s_, e_ = offsets[i], offsets[i + 1]
            l2_out[i, :t.shape[0]] = l2[s_:e_]
            cos_out[i, :t.shape[0]] = cos[s_:e_]

    _nn_dists(pred_traces, pred_l2_to_nn, pred_cos_to_nn)
    _nn_dists(real_traces, real_l2_to_nn, real_cos_to_nn)
    return pred_l2_to_nn, pred_cos_to_nn, real_l2_to_nn, real_cos_to_nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--latent_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--hdf5_dataset", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_traj", type=int, default=128)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--n_query", type=int, default=2000)
    p.add_argument("--exclude_window", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="diagnostics_anchor")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[load] JEPA from {args.wm_ckpt}", flush=True)
    jepa = load_jepa(args.wm_ckpt, device=args.device, img_size=224, patch_size=14)
    print(f"[load] latents from {args.latent_cache}", flush=True)
    latents_per_ep = _load_latents(args.latent_cache)
    print(f"  n_eps={len(latents_per_ep)}, "
          f"total_steps={sum(z.shape[0] for z in latents_per_ep)}", flush=True)
    print(f"[load] actions from {args.hdf5_dataset}", flush=True)
    actions_per_ep, action_flat = _load_actions(args.hdf5_dataset)
    sm, ss = _fit_scaler(action_flat)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== Diagnostic 1: compounding error over {args.K} chunks "
          f"({args.K*5} env steps) on {args.n_traj} trajs ===", flush=True)
    t0 = time.time()
    cos, l2, real_norms, pred_traces, real_traces = diagnostic_compounding(
        jepa, latents_per_ep, actions_per_ep, sm, ss, args.device,
        n_traj=args.n_traj, K=args.K, stride=5, seed=args.seed,
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    cos_med = np.nanmedian(cos, axis=0)
    cos_p25 = np.nanpercentile(cos, 25, axis=0)
    cos_p75 = np.nanpercentile(cos, 75, axis=0)
    l2_med = np.nanmedian(l2, axis=0)
    real_norm_med = np.nanmedian(real_norms, axis=0)

    print("\n  k  cos_p25  cos_med  cos_p75   l2_med  ||z_real||_med  rel_l2")
    for k in range(args.K + 1):
        rel = l2_med[k] / (real_norm_med[k] + 1e-8)
        print(f"  {k:2d}   {cos_p25[k]:.3f}    {cos_med[k]:.3f}    {cos_p75[k]:.3f}    "
              f"{l2_med[k]:.3f}    {real_norm_med[k]:.3f}     {rel:.3f}",
              flush=True)

    print(f"\n=== Diagnostic 2: NN quality on {args.n_query} random queries ===", flush=True)
    t0 = time.time()
    nn = diagnostic_nn_quality(
        latents_per_ep, n_query=args.n_query, exclude_window=args.exclude_window,
        seed=args.seed,
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    valid = ~np.isnan(nn["nn_l2"])
    print(f"  valid_NNs: {valid.sum()}/{valid.size}")
    print(f"  step_l2  median={np.median(nn['step_l2']):.3f}  "
          f"p25={np.percentile(nn['step_l2'], 25):.3f}  "
          f"p75={np.percentile(nn['step_l2'], 75):.3f}")
    print(f"  nn_l2    median={np.median(nn['nn_l2'][valid]):.3f}  "
          f"p25={np.percentile(nn['nn_l2'][valid], 25):.3f}  "
          f"p75={np.percentile(nn['nn_l2'][valid], 75):.3f}")
    print(f"  rand_l2  median={np.median(nn['rand_l2']):.3f}  "
          f"p25={np.percentile(nn['rand_l2'], 25):.3f}  "
          f"p75={np.percentile(nn['rand_l2'], 75):.3f}")
    print(f"  nn_cos   median={np.median(nn['nn_cos'][valid]):.4f}  "
          f"p25={np.percentile(nn['nn_cos'][valid], 25):.4f}")
    print(f"  rand_cos median={np.median(nn['rand_cos']):.4f}")

    # How often is NN from a DIFFERENT trajectory? (true stitching evidence)
    diff_traj = (nn["nn_ep"][valid] != nn["q_ep"][valid]).mean()
    print(f"  fraction NN from different trajectory: {diff_traj:.3f}")

    print(f"\n=== Diagnostic 3 (derived): pred-to-NN distance vs k ===", flush=True)
    t0 = time.time()
    pl2, pcos, rl2, rcos = diagnostic_pred_vs_anchor_distance(
        pred_traces, real_traces, latents_per_ep, args.K, seed=args.seed,
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    print("  k  pred_NN_l2_med  real_NN_l2_med  pred_NN_cos_med  real_NN_cos_med")
    for k in range(args.K + 1):
        print(f"  {k:2d}   {np.nanmedian(pl2[:, k]):.3f}        "
              f"{np.nanmedian(rl2[:, k]):.3f}        "
              f"{np.nanmedian(pcos[:, k]):.4f}         "
              f"{np.nanmedian(rcos[:, k]):.4f}", flush=True)

    # Save raw arrays for later plotting / analysis
    np.savez(
        os.path.join(args.out_dir, "raw.npz"),
        cos=cos, l2=l2, real_norms=real_norms,
        nn_l2=nn["nn_l2"], nn_cos=nn["nn_cos"],
        nn_ep=nn["nn_ep"], nn_step=nn["nn_step"],
        q_ep=nn["q_ep"], q_step=nn["q_step"],
        step_l2=nn["step_l2"], rand_l2=nn["rand_l2"], rand_cos=nn["rand_cos"],
        pl2=pl2, pcos=pcos, rl2=rl2, rcos=rcos,
    )
    report = dict(
        compounding=dict(
            cos_med=cos_med.tolist(),
            cos_p25=cos_p25.tolist(),
            cos_p75=cos_p75.tolist(),
            l2_med=l2_med.tolist(),
            real_norm_med=real_norm_med.tolist(),
            rel_l2_med=(l2_med / (real_norm_med + 1e-8)).tolist(),
        ),
        nn_quality=dict(
            valid=int(valid.sum()),
            nn_l2_median=float(np.median(nn["nn_l2"][valid])),
            nn_l2_p25=float(np.percentile(nn["nn_l2"][valid], 25)),
            nn_l2_p75=float(np.percentile(nn["nn_l2"][valid], 75)),
            step_l2_median=float(np.median(nn["step_l2"])),
            step_l2_p25=float(np.percentile(nn["step_l2"], 25)),
            step_l2_p75=float(np.percentile(nn["step_l2"], 75)),
            rand_l2_median=float(np.median(nn["rand_l2"])),
            nn_cos_median=float(np.median(nn["nn_cos"][valid])),
            rand_cos_median=float(np.median(nn["rand_cos"])),
            frac_NN_different_trajectory=float(diff_traj),
        ),
        pred_to_NN_l2_med=[float(np.nanmedian(pl2[:, k])) for k in range(args.K + 1)],
        real_to_NN_l2_med=[float(np.nanmedian(rl2[:, k])) for k in range(args.K + 1)],
        pred_to_NN_cos_med=[float(np.nanmedian(pcos[:, k])) for k in range(args.K + 1)],
        real_to_NN_cos_med=[float(np.nanmedian(rcos[:, k])) for k in range(args.K + 1)],
        args=vars(args),
    )
    with open(os.path.join(args.out_dir, "diag_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten to {args.out_dir}/diag_report.json and raw.npz")


if __name__ == "__main__":
    main()
