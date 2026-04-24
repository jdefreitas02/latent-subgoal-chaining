"""
Diagnostic tests for the LeWM latent space: is it a well-posed space for
dense L2-distance rewards to goal images?

Three tests:
  1. L2 vs frame gap:   ||z_t - z_{t+k}|| for k ∈ [1, max_gap] over random
                        trajectories. Cheap sanity check. Flat line → encoder
                        is useless for L2 rewards. Monotone increasing up to
                        some plateau is expected.
  2. OOD check:         Compare ||z|| and nearest-cache-distance for each of
                        the 5 OGBench goal images against the distribution of
                        play-dataset latents. Large z-scores / outlier norms
                        mean the goal lives off the encoder's manifold, so L2
                        distances to it are geometrically untrustworthy.
  3. Goal-distance
     histograms:        For each goal, histogram of ||z_goal - z_t|| across a
                        dataset sample, alongside a random-pair baseline. If
                        the two histograms overlap completely, the goal is
                        "just another point" and L2 distance carries no signal
                        about how close you are to it.

Outputs plots + summary.txt to --results_dir (default: diagnostics_lewm/).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import stable_pretraining as spt
import stable_worldmodel as swm


def make_img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def _to_hwc(imgs):
    """Accepts numpy or torch, (N, H, W, C) or (N, C, H, W), returns (N, H, W, C) np.uint8."""
    if torch.is_tensor(imgs):
        imgs = imgs.cpu().numpy()
    imgs = np.asarray(imgs)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3) and imgs.shape[-1] not in (1, 3):
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    return imgs


@torch.inference_mode()
def encode_batch(model, imgs_hwc: np.ndarray, device, transform, chunk: int = 32) -> torch.Tensor:
    """Encode (N, H, W, C) uint8/float images → (N, D) latents on CPU."""
    out = []
    for i in range(0, len(imgs_hwc), chunk):
        batch = imgs_hwc[i : i + chunk]
        tensors = torch.stack([transform(img) for img in batch])  # (B, C, H, W)
        tensors = tensors.unsqueeze(1).to(device)                 # (B, 1, C, H, W)
        emb = model.encode({"pixels": tensors})["emb"]            # (B, 1, D)
        out.append(emb[:, -1].cpu())
    return torch.cat(out, dim=0)


def gather_trajectory_windows(dataset, num_traj: int, max_gap: int, rng: np.random.Generator):
    """Return a list of (ep_idx, global_row_indices) pairs for random windows.

    Uses dataset.offsets / dataset.lengths (always present on HDF5Dataset) instead
    of episode-index columns, which may not exist in all datasets.
    """
    ep_offsets = dataset.offsets   # (num_episodes,) global row of each episode start
    ep_lengths = dataset.lengths   # (num_episodes,) frames per episode
    num_eps = len(ep_offsets)

    if num_eps == 0:
        raise RuntimeError("Dataset has no episodes.")

    # Only consider episodes long enough to contain a window of max_gap+1 frames.
    valid_eps = np.where(ep_lengths >= max_gap + 1)[0]
    if len(valid_eps) == 0:
        raise RuntimeError(f"No episodes with >= {max_gap + 1} frames.")

    chosen = rng.choice(valid_eps, size=min(num_traj, len(valid_eps)), replace=False)
    windows = []
    for ep_idx in chosen:
        ep_len = int(ep_lengths[ep_idx])
        ep_start = int(ep_offsets[ep_idx])
        window_start = int(rng.integers(0, ep_len - max_gap))
        row_indices = np.arange(ep_start + window_start,
                                ep_start + window_start + max_gap + 1)
        windows.append((int(ep_idx), row_indices))
    return windows


def run_test_l2_vs_gap(model, dataset, transform, device, out_dir, args, rng):
    print("\n== Test 1: L2 distance vs frame gap ==")
    windows = gather_trajectory_windows(dataset, args.num_trajectories, args.max_gap, rng)
    if len(windows) == 0:
        print("  (no episodes long enough — skipping)")
        return None

    all_dists = []
    for _, row_idx in tqdm(windows, desc="  encoding"):
        rows_data = dataset.get_row_data(row_idx.tolist())
        imgs = _to_hwc(rows_data["pixels"])
        lat = encode_batch(model, imgs, device, transform)  # (max_gap+1, D)
        dists = torch.norm(lat[1:] - lat[:1], dim=-1).numpy()  # (max_gap,)
        all_dists.append(dists)

    all_dists = np.stack(all_dists)                       # (T, max_gap)
    mean = all_dists.mean(axis=0)
    std = all_dists.std(axis=0)
    gaps = np.arange(1, args.max_gap + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gaps, mean, label="mean", color="steelblue")
    ax.fill_between(gaps, mean - std, mean + std, alpha=0.25, color="steelblue", label="±1σ")
    ax.set_xlabel("frame gap k")
    ax.set_ylabel(r"$\| z_t - z_{t+k} \|_2$")
    ax.set_title(f"Latent L2 distance vs temporal gap  (n={len(windows)} trajectories)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "l2_vs_gap.png", dpi=110)
    plt.close(fig)

    ratio = mean[-1] / max(mean[0], 1e-8)
    print(f"  mean dist at k=1       : {mean[0]:.3f} ± {std[0]:.3f}")
    print(f"  mean dist at k={args.max_gap:<5d}: {mean[-1]:.3f} ± {std[-1]:.3f}")
    print(f"  ratio (k={args.max_gap})/(k=1) = {ratio:.2f}  "
          "(≈1 → encoder ignores time; ≫1 → encoder captures it)")
    print(f"  saved {out_dir/'l2_vs_gap.png'}")

    return {
        "gaps":     gaps,
        "mean":     mean,
        "std":      std,
        "ratio":    ratio,
        "num_traj": len(windows),
    }


def run_test_ood_and_goal_distances(model, dataset, transform, device, out_dir, args, rng):
    import gymnasium
    import ogbench  # noqa: F401  (registers envs)

    print("\n== Test 2+3: OGBench goal OOD + dataset distance histograms ==")

    # ─ Get the 5 goal images from OGBCube-v0 ─────────────────────────────
    env_name = "swm/OGBCube-v0" if args.img_size == 224 else "visual-cube-single-v0"
    print(f"  Creating {env_name} to extract task goal images...")
    if args.img_size == 224:
        env = gymnasium.make(env_name, ob_type="pixels", env_type="single", visualize_info=False)
        goal_key = "target"
    else:
        env = gymnasium.make(env_name)
        goal_key = "goal"

    task_infos = env.unwrapped.task_infos
    goal_imgs, task_names = [], []
    for task_id in range(1, len(task_infos) + 1):
        _, info = env.reset(options=dict(task_id=task_id))
        goal_imgs.append(np.asarray(info[goal_key]))
        task_names.append(task_infos[task_id - 1].get("task_name", f"task{task_id}"))
    env.close()
    goal_imgs = np.stack(goal_imgs)  # (5, H, W, C)
    print(f"  Got {len(task_names)} goals: {task_names}")

    goal_lat = encode_batch(model, goal_imgs, device, transform)  # (5, D)
    goal_norms = torch.norm(goal_lat, dim=-1).numpy()

    # ─ Random dataset sample ─────────────────────────────────────────────
    n_sample = min(args.num_dataset_samples, len(dataset))
    rand_rows = rng.choice(len(dataset), size=n_sample, replace=False)
    rand_rows = np.sort(rand_rows)
    print(f"  Encoding {n_sample} random dataset frames...")
    rand_data = dataset.get_row_data(rand_rows.tolist())
    rand_imgs = _to_hwc(rand_data["pixels"])
    rand_lat = []
    for i in tqdm(range(0, len(rand_imgs), 128), desc="  encoding dataset"):
        rand_lat.append(encode_batch(model, rand_imgs[i : i + 128], device, transform))
    rand_lat = torch.cat(rand_lat, dim=0)  # (n_sample, D)

    rand_norms = torch.norm(rand_lat, dim=-1).numpy()
    rn_mean, rn_std = rand_norms.mean(), rand_norms.std()
    print(f"  Dataset latent norm: {rn_mean:.3f} ± {rn_std:.3f}")

    # Random-pair baseline (how far apart are two unrelated dataset frames?)
    half = len(rand_lat) // 2
    rand_pair_dists = torch.norm(rand_lat[:half] - rand_lat[half : 2 * half], dim=-1).numpy()
    print(f"  Random-pair dist (baseline):   mean={rand_pair_dists.mean():.3f}  "
          f"median={np.median(rand_pair_dists):.3f}")

    goal_to_dataset = []
    ood_rows = []
    for i, name in enumerate(task_names):
        dists = torch.norm(rand_lat - goal_lat[i : i + 1], dim=-1).numpy()  # (n_sample,)
        goal_to_dataset.append(dists)
        z = (goal_norms[i] - rn_mean) / max(rn_std, 1e-8)
        ood_rows.append({
            "task": name,
            "goal_norm":     float(goal_norms[i]),
            "z_score":       float(z),
            "nn_dist":       float(dists.min()),
            "mean_dist":     float(dists.mean()),
            "baseline_mean": float(rand_pair_dists.mean()),
        })
        print(f"  {name:>22s}: goal‖z‖={goal_norms[i]:.3f} (z={z:+.2f}σ)  "
              f"nn={dists.min():.3f}  mean={dists.mean():.3f}  "
              f"vs rand-pair mean={rand_pair_dists.mean():.3f}")

    # Histogram plot
    n_plots = len(task_names) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3.6), sharey=True)
    axes[0].hist(rand_pair_dists, bins=40, color="0.55")
    axes[0].set_title("random pairs (baseline)")
    axes[0].set_xlabel("L2")
    axes[0].set_ylabel("count")
    xmax = max(rand_pair_dists.max(), max(d.max() for d in goal_to_dataset))
    axes[0].set_xlim(0, xmax * 1.05)
    for i, name in enumerate(task_names):
        axes[i + 1].hist(goal_to_dataset[i], bins=40, color="steelblue")
        axes[i + 1].axvline(rand_pair_dists.mean(), color="k", ls="--", lw=1,
                            label="baseline mean")
        axes[i + 1].set_title(name)
        axes[i + 1].set_xlabel("L2")
        axes[i + 1].set_xlim(0, xmax * 1.05)
        axes[i + 1].legend(fontsize=7)
    fig.suptitle("‖z_dataset − z_goal‖ vs random-pair baseline\n"
                 "(signal = goal histogram shifted LEFT vs baseline)")
    fig.tight_layout()
    fig.savefig(out_dir / "goal_distance_histograms.png", dpi=110)
    plt.close(fig)
    print(f"  saved {out_dir/'goal_distance_histograms.png'}")

    # Latent-norm histogram (OOD visual)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rand_norms, bins=60, color="0.6", label="dataset")
    for i, name in enumerate(task_names):
        ax.axvline(goal_norms[i], color=f"C{i}", lw=2, label=f"{name}")
    ax.set_xlabel("‖z‖")
    ax.set_ylabel("count")
    ax.set_title("Goal latents vs dataset latent-norm distribution")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "goal_norms_ood.png", dpi=110)
    plt.close(fig)
    print(f"  saved {out_dir/'goal_norms_ood.png'}")

    return {
        "task_names":         task_names,
        "goal_norms":         goal_norms,
        "dataset_norm_mean":  rn_mean,
        "dataset_norm_std":   rn_std,
        "rand_pair_mean":     float(rand_pair_dists.mean()),
        "rand_pair_median":   float(np.median(rand_pair_dists)),
        "rows":               ood_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostics for the LeWM latent space / dense-L2-reward assumption."
    )
    parser.add_argument("--ckpt_path", default=None,
                        help="Base path for AutoCostModel. Default: $STABLEWM_HOME/cube/lejepa")
    parser.add_argument("--dataset_path", default=None,
                        help="HDF5 dataset (no .h5 extension). For 224 tests, must be a "
                             "224-rendered dataset. Default: $STABLEWM_HOME/cube/cube_single_expert")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_gap", type=int, default=50,
                        help="Maximum frame gap k for Test 1.")
    parser.add_argument("--num_trajectories", type=int, default=80,
                        help="Number of random trajectory windows for Test 1.")
    parser.add_argument("--num_dataset_samples", type=int, default=4000,
                        help="Number of random dataset frames to encode for Tests 2+3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="diagnostics_lewm")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(stablewm_home, "cube", "lejepa")
    if args.dataset_path is None:
        args.dataset_path = os.path.join(stablewm_home, "cube", "cube_single_expert")
    print(f"Weights:  {args.ckpt_path}")
    print(f"Dataset:  {args.dataset_path}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("Loading AutoCostModel...")
    model = swm.policy.AutoCostModel(args.ckpt_path).to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    transform = make_img_transform(args.img_size)

    print("Opening dataset...")
    dataset = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=[],
        cache_dir=str(Path(args.dataset_path).parent),
    )
    print(f"  {len(dataset):,} frames, columns: {dataset.column_names}")

    test1 = run_test_l2_vs_gap(model, dataset, transform, device, results_dir, args, rng)
    test23 = run_test_ood_and_goal_distances(model, dataset, transform, device, results_dir, args, rng)

    # Summary
    summary = results_dir / "summary.txt"
    with summary.open("w") as f:
        f.write(f"ckpt:    {args.ckpt_path}\n")
        f.write(f"dataset: {args.dataset_path}\n")
        f.write(f"img_size: {args.img_size}\n\n")

        if test1 is not None:
            f.write("== Test 1: L2 distance vs frame gap ==\n")
            f.write(f"  num_trajectories: {test1['num_traj']}\n")
            for k, m, s in zip(test1["gaps"], test1["mean"], test1["std"]):
                f.write(f"  k={k:3d}  mean={m:.4f}  std={s:.4f}\n")
            f.write(f"  ratio mean(k={args.max_gap}) / mean(k=1) = {test1['ratio']:.2f}\n")
            verdict1 = (
                "FAIL — encoder does not capture temporal structure"
                if test1["ratio"] < 1.2 else
                "OK — L2 distance grows meaningfully with temporal gap"
            )
            f.write(f"  verdict: {verdict1}\n\n")

        if test23 is not None:
            f.write("== Test 2+3: Goal OOD + distance-to-dataset ==\n")
            f.write(f"  dataset ‖z‖: {test23['dataset_norm_mean']:.4f}"
                    f" ± {test23['dataset_norm_std']:.4f}\n")
            f.write(f"  random-pair dist (baseline): mean={test23['rand_pair_mean']:.4f}"
                    f" median={test23['rand_pair_median']:.4f}\n\n")

            f.write(f"  {'task':<24s}  {'‖z‖':>7s}  {'z':>6s}  {'nn':>8s}  "
                    f"{'mean':>8s}  {'base':>8s}  {'signal':>8s}\n")
            for row in test23["rows"]:
                signal = row["baseline_mean"] - row["mean_dist"]   # positive → goal is CLOSER to dataset than typical pair
                f.write(
                    f"  {row['task']:<24s}  "
                    f"{row['goal_norm']:7.3f}  "
                    f"{row['z_score']:+6.2f}  "
                    f"{row['nn_dist']:8.3f}  "
                    f"{row['mean_dist']:8.3f}  "
                    f"{row['baseline_mean']:8.3f}  "
                    f"{signal:+8.3f}\n"
                )
            f.write("\n  signal > 0 → goal is closer to typical dataset frames than "
                    "a random pair (good for L2 reward)\n")
            f.write("  |z_score| < 2 → goal norm sits within dataset distribution "
                    "(SIGReg constraint held for this image)\n")

    print(f"\nSummary saved to {summary}")


if __name__ == "__main__":
    main()
