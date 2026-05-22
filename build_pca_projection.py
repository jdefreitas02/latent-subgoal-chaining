"""
Build a PCA projection from the 192D LeWM latent cache to a lower-dimensional
space (target: 10D) suitable for goal-conditioned RL.

Why:
  - AWR advantage signal in 192D: ~0.5/20 ≈ 2.5% of scale → nearly uniform weights
  - In 10D: same absolute advantage, ~5× sharper relative signal
  - SAC goal conditioning: 10D goal vectors are tractable; 192D goals are high-variance noise
  - FAISS NN lookup: 2M × 10D fits in ~80MB vs 2M × 192D = ~1.5GB

Outputs (saved to STABLEWM_HOME by default):
  lewm_pca_10d.pt          — PCA parameters: pca_matrix [192, D], pca_mean [192],
                              pca_dim, explained_variance [192], cumulative_variance [192]
  lewm_10d_faiss.index     — FAISS FlatL2 index over all projected cache latents

Usage:
    python latent_hindsight_rl/build_pca_projection.py
    python latent_hindsight_rl/build_pca_projection.py --target_dim 16 --variance_threshold 0.90
"""

import os
import sys
import argparse
import numpy as np
import torch

_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def main():
    parser = argparse.ArgumentParser(description="Build PCA projection for LeWM latent cache")
    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    parser.add_argument("--cache_path",
        default=os.path.join(stablewm_home, "lewm_224_latents_cache.pt"),
        help="Path to lewm_224_latents_cache.pt")
    parser.add_argument("--output_dir", default=stablewm_home,
        help="Directory to save PCA parameters and FAISS index")
    parser.add_argument("--target_dim", type=int, default=10,
        help="Target PCA dimension (default: 10, like HIQL)")
    parser.add_argument("--variance_threshold", type=float, default=0.80,
        help="Minimum cumulative variance fraction to accept (default: 0.80 = 80%%)")
    parser.add_argument("--max_frames", type=int, default=None,
        help="Subsample frames for PCA fit (default: use all). Recommended: None (use all).")
    args = parser.parse_args()

    print(f"Loading latent cache from {args.cache_path} ...")
    cache = torch.load(args.cache_path, map_location="cpu")
    all_latents = cache["all_latents"]  # List[Tensor[T_ep, 192]]

    # Flatten all latents to [N_total, 192]
    print("Stacking all latents ...")
    flat = torch.cat(all_latents, dim=0).numpy()  # [N, 192]
    N, D = flat.shape
    print(f"  Total frames: {N:,}  Latent dim: {D}")

    if args.max_frames is not None and N > args.max_frames:
        idx = np.random.choice(N, args.max_frames, replace=False)
        flat_fit = flat[idx]
        print(f"  Subsampled {args.max_frames:,} frames for PCA fit")
    else:
        flat_fit = flat

    # Fit PCA (sklearn for reliability, torch for storage)
    from sklearn.decomposition import PCA as SklearnPCA

    print(f"Fitting PCA on {len(flat_fit):,} frames, keeping {D} components ...")
    pca = SklearnPCA(n_components=D, svd_solver="full")
    pca.fit(flat_fit)

    expl_var = pca.explained_variance_ratio_           # [D]
    cum_var  = np.cumsum(expl_var)                     # [D]

    # Report
    print("\n  Cumulative variance explained:")
    for k in [5, 8, 10, 12, 16, 20, 32, 64, 128, D]:
        if k <= D:
            print(f"    Top {k:3d} PCs: {cum_var[k-1]*100:.1f}%")

    target_dim = args.target_dim
    top_k_var  = cum_var[target_dim - 1]
    print(f"\n  Target dim = {target_dim}: explains {top_k_var*100:.1f}% of variance")

    if top_k_var < args.variance_threshold:
        # Find minimum k that meets the threshold
        k_needed = int(np.searchsorted(cum_var, args.variance_threshold)) + 1
        print(f"  WARNING: {target_dim}D only explains {top_k_var*100:.1f}% < {args.variance_threshold*100:.0f}%.")
        print(f"           Minimum dimension for {args.variance_threshold*100:.0f}% variance: {k_needed}D")
        print(f"           Proceeding with {target_dim}D as requested (override with --target_dim {k_needed})")
    else:
        print(f"  OK: {target_dim}D explains ≥ {args.variance_threshold*100:.0f}% of variance.")

    # Extract projection matrix for top-k PCs
    pca_matrix = torch.tensor(pca.components_[:target_dim].T, dtype=torch.float32)  # [192, D]
    pca_mean   = torch.tensor(pca.mean_,                      dtype=torch.float32)  # [192]

    pca_params = {
        "pca_matrix":            pca_matrix,         # [192, target_dim]
        "pca_mean":              pca_mean,            # [192]
        "pca_dim":               target_dim,
        "explained_variance":    torch.tensor(expl_var, dtype=torch.float32),
        "cumulative_variance":   torch.tensor(cum_var,  dtype=torch.float32),
        "top_k_variance":        float(top_k_var),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    pca_path = os.path.join(args.output_dir, f"lewm_pca_{target_dim}d.pt")
    torch.save(pca_params, pca_path)
    print(f"\nSaved PCA params → {pca_path}")
    print(f"  pca_matrix shape: {pca_matrix.shape}  (192 → {target_dim}D)")

    # Build FAISS index over all projected latents
    print(f"\nProjecting {N:,} frames to {target_dim}D for FAISS index ...")
    pca_mean_np   = pca_mean.numpy()
    pca_matrix_np = pca_matrix.numpy()  # [192, D]
    flat_proj     = (flat - pca_mean_np) @ pca_matrix_np  # [N, D]
    flat_proj     = flat_proj.astype(np.float32)

    import faiss
    print(f"Building FAISS FlatL2 index over {N:,} × {target_dim}D vectors ...")
    index = faiss.IndexFlatL2(target_dim)
    index.add(flat_proj)

    faiss_path = os.path.join(args.output_dir, f"lewm_{target_dim}d_faiss.index")
    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index → {faiss_path}  ({index.ntotal:,} vectors)")

    # Quick sanity check: project a random latent and find NN
    test_idx = np.random.randint(0, N)
    test_vec = flat_proj[test_idx:test_idx+1]
    D_sq, I  = index.search(test_vec, k=1)
    assert I[0][0] == test_idx, "NN of a vector should be itself!"
    print(f"\nFAISS sanity check: NN(frame_{test_idx}) = frame_{I[0][0]} ✓")

    print(f"\nDone. To use in training:")
    print(f"  python latent_hindsight_rl/train_joint.py \\")
    print(f"      --pca_path {pca_path} \\")
    print(f"      --done_threshold <recalibrate in {target_dim}D via check_predictor_consistency.py>")


if __name__ == "__main__":
    main()
