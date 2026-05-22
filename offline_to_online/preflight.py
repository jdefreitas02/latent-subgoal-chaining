"""Preflight checks for the qc + JEPA WM experiments.

Run before any training to verify the dataset / WM contract:
  1. Episode alignment between the latent cache and the swm HDF5 dataset.
  2. Action dim of the HDF5 actions (must be 5 for cube-single).
  3. Action ordering inside the 25-D chunk (verify by running WMEnv.step on a
     known offline transition and comparing to the cached z[t+5]).
  4. Latent-distance threshold calibration vs each of the 5 task goals
     (prints P85/P90/P95 to pick a value that yields ~5-15% positive rate).

Example:
  python preflight.py \
      --wm_ckpt_path ~/.stable_worldmodel/cube/lejepa \
      --wm_latent_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache.pt \
      --hdf5_dataset_path ~/stable_wm_data/ogbench/visual-cube-single-play-v0_224
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch


def _add_paths():
    # Put offline-online/ FIRST so our `utils/` subpackage takes priority
    # over ~/leworldmodel/utils.py.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    # Append leworldmodel/ at the END so stable_pretraining/jepa.py resolve
    # without shadowing this package.
    leworldmodel = os.path.abspath(os.path.join(here, "..", ".."))
    if leworldmodel not in sys.path:
        sys.path.append(leworldmodel)


def check_alignment(latent_cache_path, hdf5_dataset_path):
    print("\n=== (1) Episode alignment: latent cache vs HDF5 dataset ===")
    import stable_worldmodel as swm

    cache = torch.load(latent_cache_path, map_location="cpu", weights_only=False)
    if isinstance(cache, dict) and "all_latents" in cache:
        all_latents = cache["all_latents"]
    else:
        all_latents = cache
    n_cache = len(all_latents)

    ds = swm.data.HDF5Dataset(
        hdf5_dataset_path,
        keys_to_cache=["action"],
        cache_dir=str(Path(hdf5_dataset_path).parent),
    )
    n_hdf5 = len(ds.lengths)

    print(f"  cache episodes: {n_cache}")
    print(f"  hdf5  episodes: {n_hdf5}")
    if n_cache != n_hdf5:
        print(f"  MISMATCH in episode count")
        return False, ds, all_latents

    # Check first few episode lengths match
    mismatches = 0
    for i in range(min(20, n_cache)):
        t_cache = (all_latents[i].shape[0] if torch.is_tensor(all_latents[i])
                   else np.asarray(all_latents[i]).shape[0])
        t_hdf5 = int(ds.lengths[i])
        if abs(t_cache - t_hdf5) > 1:
            mismatches += 1
            print(f"    ep {i}: cache T={t_cache} vs hdf5 T={t_hdf5}")
    if mismatches:
        print(f"  {mismatches}/20 sampled episodes mismatched (>1 frame diff)")
        return False, ds, all_latents
    print(f"  OK on first 20 episodes")
    return True, ds, all_latents


def check_action_dim(ds):
    print("\n=== (2) Action dim from HDF5 ===")
    ch = ds.load_chunk(np.array([0]), np.array([0]), np.array([1]))[0]
    act = ch["action"]
    if hasattr(act, "numpy"):
        act = act.numpy()
    a_dim = act.shape[-1]
    print(f"  HDF5 action dim per step: {a_dim}")
    if a_dim != 5:
        print(f"  EXPECTED 5; got {a_dim}. Verify which env this dataset is for.")
        return False, a_dim

    # Per-element range
    action_data = ds.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    print(f"  action per-element stats (after NaN filter):")
    print(f"    mean: {np.mean(action_data, axis=0).round(3).tolist()}")
    print(f"    std : {np.std(action_data, axis=0).round(3).tolist()}")
    print(f"    min : {np.min(action_data, axis=0).round(3).tolist()}")
    print(f"    max : {np.max(action_data, axis=0).round(3).tolist()}")
    return True, a_dim


def _fit_action_scaler(ds):
    """Fit a StandardScaler on HDF5 actions, matching train.py get_column_normalizer."""
    from sklearn import preprocessing
    action_data = ds.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = preprocessing.StandardScaler()
    scaler.fit(action_data)
    return scaler


def check_action_ordering(all_latents, ds, wm_ckpt_path, wm_device, img_size,
                          stride=5, n_episodes=200):
    """Compare row-major vs col-major chunk encoding with StandardScaler-normalized actions.

    train.py applies get_column_normalizer (StandardScaler) to actions before
    feeding them to the WM. eval.py does the same via WorldModelPolicy(process=...).
    WMEnv must apply the same normalization at inference time.
    """
    print(f"\n=== (3) 25-D action chunk ordering vs WM (avg over {n_episodes} chunks) ===")
    from envs.jepa_loader import load_jepa

    jepa = load_jepa(wm_ckpt_path, device=wm_device, img_size=img_size)
    scaler = _fit_action_scaler(ds)
    print(f"  action scaler  mean={scaler.mean_.round(3).tolist()}  "
          f"std={scaler.scale_.round(3).tolist()}")

    eligible = []
    for k in range(len(all_latents)):
        T = (all_latents[k].shape[0] if torch.is_tensor(all_latents[k])
             else np.asarray(all_latents[k]).shape[0])
        T_a = int(ds.lengths[k])
        if T >= stride + 1 and T_a >= stride:
            eligible.append(k)
    if not eligible:
        print("  no episode with >= stride+1 frames; skipping")
        return None

    rng = np.random.default_rng(0)
    samples = rng.choice(eligible, size=min(n_episodes, len(eligible)), replace=False)

    err_row_list, err_col_list, drift_list = [], [], []

    for ep_idx in samples:
        z_ep = all_latents[ep_idx]
        if torch.is_tensor(z_ep):
            z_ep = z_ep.cpu().numpy()
        T = z_ep.shape[0]
        T_a = int(ds.lengths[ep_idx])
        # pick a random starting index t such that t + stride < T and t + stride <= T_a
        max_t = min(T, T_a + 1) - stride - 1
        if max_t <= 0:
            continue
        t = int(rng.integers(0, max_t + 1))
        ch = ds.load_chunk(np.array([ep_idx]), np.array([t]), np.array([t + stride]))[0]
        a_seg = ch["action"]
        if hasattr(a_seg, "numpy"):
            a_seg = a_seg.numpy()
        a_seg = a_seg.astype(np.float32)  # (stride, 5)
        a_seg_scaled = scaler.transform(a_seg).astype(np.float32)

        z_t = torch.from_numpy(z_ep[t].astype(np.float32)).to(wm_device).view(1, 1, 192)
        z_target = z_ep[t + stride].astype(np.float32)

        # Row-major after scaling -- matches train.py / WMEnv.step convention
        chunk_row = a_seg_scaled.reshape(-1).astype(np.float32)
        chunk_col = a_seg_scaled.T.reshape(-1).astype(np.float32)

        with torch.no_grad():
            for chunk_25, errs in [(chunk_row, err_row_list), (chunk_col, err_col_list)]:
                a_t = torch.from_numpy(chunk_25).to(wm_device).view(1, 1, 25)
                a_emb = jepa.action_encoder(a_t)
                z_next = jepa.predict(z_t, a_emb)[:, -1:]
                z_pred = z_next.squeeze().cpu().numpy().astype(np.float32)
                errs.append(float(np.linalg.norm(z_pred - z_target)))
        drift_list.append(float(np.linalg.norm(z_ep[t] - z_target)))

    err_row = float(np.mean(err_row_list))
    err_col = float(np.mean(err_col_list))
    drift = float(np.mean(drift_list))
    print(f"  (StandardScaler-normalized actions — matching train.py/eval.py convention)")
    print(f"  averaged over {len(err_row_list)} chunks")
    print(f"  mean ||predict(row-major) - z[t+{stride}]||  = {err_row:.4f}  "
          f"(median {np.median(err_row_list):.4f})")
    print(f"  mean ||predict(col-major) - z[t+{stride}]||  = {err_col:.4f}  "
          f"(median {np.median(err_col_list):.4f})")
    print(f"  mean ||z[t] - z[t+{stride}]|| (no-movement)  = {drift:.4f}  "
          f"(median {np.median(drift_list):.4f})")
    chosen = "row-major" if err_row <= err_col else "col-major"
    delta = abs(err_row - err_col)
    print(f"  best: {chosen}   (row-vs-col delta: {delta:.4f})")
    if min(err_row, err_col) > drift:
        print(f"  WARNING: best WM prediction is WORSE than no-movement drift; "
              f"either ordering is still wrong or WM has high prediction error on this data.")
    return {"row": err_row, "col": err_col, "drift": drift,
            "chosen": chosen, "jepa": jepa}


def check_thresholds(all_latents, jepa, wm_ckpt_path, wm_device, img_size,
                     task_ids=(1, 2, 3, 4, 5), stride=5):
    print("\n=== (4) Latent-distance threshold calibration vs task goals ===")
    from envs.jepa_loader import encode_pixels_to_latent
    from envs.swm_env_register import make_swm_cube_env

    # Encode each task's target image
    real_env = make_swm_cube_env(seed=0)
    z_goals = {}
    for t in task_ids:
        _, info = real_env.reset(options=dict(task_id=t))
        if "target" not in info:
            print(f"  task {t}: info has no 'target' key; got {list(info.keys())}")
            continue
        z_goals[t] = encode_pixels_to_latent(jepa, info["target"], wm_device)

    # For each task, compute distance distribution to z[stride*(k+1)]
    for t, z_goal in z_goals.items():
        dists = []
        for ep in all_latents:
            if torch.is_tensor(ep):
                ep = ep.cpu().numpy()
            T = ep.shape[0]
            num_chunks = (T - 1) // stride
            for k in range(num_chunks):
                z_next = ep[stride * (k + 1)]
                dists.append(float(np.linalg.norm(z_next - z_goal)))
        if not dists:
            continue
        dists = np.array(dists)
        print(f"  task {t}: n={len(dists)} chunk-end latents")
        print(f"    distance percentiles  "
              f"P05={np.percentile(dists, 5):.2f}  "
              f"P10={np.percentile(dists, 10):.2f}  "
              f"P15={np.percentile(dists, 15):.2f}  "
              f"P25={np.percentile(dists, 25):.2f}  "
              f"P50={np.percentile(dists, 50):.2f}  "
              f"min={dists.min():.2f}")
        # Threshold candidates that yield ~5-15% positive offline rate
        thr_p10 = np.percentile(dists, 10)
        pos_rate = float((dists < thr_p10).mean())
        print(f"    candidate threshold (P10): {thr_p10:.3f} -> positive rate {pos_rate*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_ckpt_path",
                        default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    parser.add_argument("--wm_latent_cache",
                        default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    parser.add_argument("--hdf5_dataset_path",
                        default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    parser.add_argument("--wm_device", default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    _add_paths()

    ok_align, ds, all_latents = check_alignment(args.wm_latent_cache, args.hdf5_dataset_path)
    if not ok_align:
        print("\n[FAIL] alignment check failed; cannot continue.")
        sys.exit(1)

    ok_dim, a_dim = check_action_dim(ds)
    if not ok_dim:
        print(f"\n[WARN] action dim {a_dim} != 5; continuing but expect issues.")

    ordering = check_action_ordering(all_latents, ds, args.wm_ckpt_path,
                                     args.wm_device, args.img_size)
    if ordering is None:
        print("\n[WARN] could not verify action ordering")
        sys.exit(0)

    check_thresholds(all_latents, ordering["jepa"], args.wm_ckpt_path,
                     args.wm_device, args.img_size)

    print("\n[OK] preflight complete.")


if __name__ == "__main__":
    main()
