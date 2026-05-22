"""
224x224 LeWM World Model Analysis
 - Loads the pretrained 224x224 LeWM from HuggingFace weights
 - Builds latent cache from the lewm-cube dataset
 - Analyses latent space geometry, prediction accuracy, autoregressive drift
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stable_pretraining as spt
import stable_worldmodel as swm

# ─────────────────────────────────────────────────────────────────────────────
#  Constants matching the pretrained 224x224 LeWM config.json
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
PATCH_SIZE  = 14
EMBED_DIM   = 192
FRAMESKIP   = 5
ACTION_DIM  = 5
EFF_ACT_DIM = FRAMESKIP * ACTION_DIM   # = 25
HISTORY_SZ  = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(weights_path, device):
    """Load the 224×224 LeWM. Pass base path (without _object.ckpt suffix)."""
    model = swm.policy.AutoCostModel(weights_path)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded 224×224 JEPA via AutoCostModel from {weights_path}")
    return model


def get_transform():
    return tv_transforms.Compose([
        tv_transforms.ToDtype(torch.float32, scale=True),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


ENCODE_MODES = ('cls_projected', 'cls_raw', 'patch_mean', 'cls_patch_cat')
"""
Encoding modes for the latent cache:
  cls_projected  — projector(CLS token). Current default. SIGReg-shaped space.
  cls_raw        — raw CLS token before the projector MLP. Tests if projector is lossy.
  patch_mean     — mean of all 256 spatial patch tokens. Tests if spatial detail
                   is lost during CLS attention pooling.
  cls_patch_cat  — concat(cls_projected, patch_mean) → 384D. Best-of-both baseline.
"""


def _encode_frames(model, pix_TCHW, encode_mode, transform, device):
    """Encode [T, C, H, W] uint8 frames according to encode_mode.

    Returns [T, D] float32 tensor where D depends on mode:
      cls_projected / cls_raw / patch_mean → 192D
      cls_patch_cat                        → 384D
    """
    pix = transform(pix_TCHW.to(device))   # [T, C, H, W] float, normalised

    if encode_mode == 'cls_projected':
        z = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)  # [T, 192]
        return z

    # For raw variants, bypass model.encode() and call the HuggingFace ViT directly.
    # model.encode() flattens [B, T, ...] → [B*T, ...]; we do the same (B=1 here).
    vit_out = model.encoder(pixel_values=pix, interpolate_pos_encoding=True)
    tokens = vit_out.last_hidden_state   # [T, 1+256, 192]

    if encode_mode == 'cls_raw':
        return tokens[:, 0, :]           # [T, 192]  CLS before projector

    if encode_mode == 'patch_mean':
        return tokens[:, 1:, :].mean(dim=1)  # [T, 192]  spatial average

    if encode_mode == 'cls_patch_cat':
        cls_proj = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)  # [T, 192]
        patch_mean = tokens[:, 1:, :].mean(dim=1)                                 # [T, 192]
        return torch.cat([cls_proj, patch_mean], dim=-1)                          # [T, 384]

    raise ValueError(f"Unknown encode_mode '{encode_mode}'. "
                     f"Choose from: {ENCODE_MODES}")


def build_cache(model, dataset, device, save_path, batch_size=8,
                encode_mode='cls_projected'):
    transform = get_transform()
    num_eps = len(dataset.lengths)
    all_latents = []
    all_actions = []
    total_frames = 0
    t0 = time.time()

    print(f"\n  Encoding {num_eps} episodes  [mode={encode_mode}] …")
    with torch.no_grad():
        for i in range(0, num_eps, batch_size):
            end_idx = min(i + batch_size, num_eps)
            ep_idx = np.arange(i, end_idx)
            ep_lens = dataset.lengths[ep_idx]
            starts = np.zeros(len(ep_idx), dtype=int)
            chunks = dataset.load_chunk(ep_idx, starts, ep_lens)

            for chunk in chunks:
                raw = chunk["pixels"]
                z = _encode_frames(model, raw, encode_mode, transform, device)
                all_latents.append(z.cpu())
                if "action" in chunk:
                    all_actions.append(chunk["action"].cpu().float())
                total_frames += len(z)

            elapsed = time.time() - t0
            print(f"  {end_idx:4d}/{num_eps}  frames={total_frames:,}  "
                  f"elapsed={elapsed:.1f}s", end="  ")

    print()
    cache = {
        "all_latents": all_latents,
        "all_actions": all_actions,
        "total_frames": total_frames,
        "encode_mode": encode_mode,
        "latent_dim": all_latents[0].shape[-1] if all_latents else None,
    }
    torch.save(cache, save_path)
    print(f"  Saved cache → {save_path}  "
          f"({total_frames:,} frames, mode={encode_mode}, "
          f"dim={cache['latent_dim']})")
    return cache


def analyse_latent_space(all_latents):
    print("\n" + "=" * 60)
    print("  LATENT SPACE GEOMETRY")
    print("=" * 60)

    gap1_dists, gap5_dists, gap25_dists, max_dists, norms = [], [], [], [], []
    for ep in all_latents[:1000]:
        T = len(ep)
        norms.extend(torch.norm(ep, dim=-1).tolist())
        if T > 1:
            gap1_dists.append(torch.norm(ep[1:] - ep[:-1], dim=-1).mean().item())
        if T > 5:
            gap5_dists.append(torch.norm(ep[5:] - ep[:-5], dim=-1).mean().item())
        if T > 25:
            gap25_dists.append(torch.norm(ep[25:] - ep[:-25], dim=-1).mean().item())
        max_dists.append(torch.norm(ep[-1] - ep[0], dim=-1).item())

    def _stats(arr, label):
        arr = np.array(arr)
        print(f"  {label:<35s}  mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

    _stats(norms, "Latent L2 norm")
    _stats(gap1_dists, "1-raw-frame distance  (1 phys action, 0.2s)")
    _stats(gap5_dists, "5-raw-frame distance  (1 WM step = 1s)")
    if gap25_dists:
        _stats(gap25_dists, "25-step distance (goal_offset)")
    _stats(max_dists, "Start→End distance (full ep)")

    mean1 = np.mean(gap1_dists)
    mean5 = np.mean(gap5_dists)
    print(f"\n  NOTE: 1 WM step = {FRAMESKIP} raw frames. Natural WM-step distance = {mean5:.3f}")
    print(f"  Recommended done_threshold ≈ {mean5 * 1.5:.3f}  (1.5× 1-WM-step mean)")
    if gap25_dists:
        print(f"  goal_offset=25 steps spans {np.mean(gap25_dists):.3f} latent units")

    all_z = torch.cat(all_latents[:200])
    mean_z = all_z.mean(0)
    std_z = all_z.std(0)
    print(f"\n  Latent distribution (SIGReg check):")
    print(f"    mean  →  L2={mean_z.norm():.4f}  "
          f"(want ≈0,  mean per-dim={mean_z.abs().mean():.4f})")
    print(f"    std   →  mean={std_z.mean():.4f}  "
          f"std={std_z.std():.4f}  (want ≈1.0 per dim)")


def test_wm_prediction(model, dataset, device, n_test=200):
    print("\n" + "=" * 60)
    print("  WORLD MODEL PREDICTION ACCURACY")
    print("=" * 60)
    transform = get_transform()

    errors_1step, baselines_1step = [], []
    errors_5step, baselines_5step = [], []

    rng = np.random.default_rng(0)
    ep_indices = rng.choice(len(dataset.lengths),
                            size=min(n_test, len(dataset.lengths)), replace=False)

    with torch.no_grad():
        for ep_i in ep_indices:
            ep_len = dataset.lengths[ep_i]
            if ep_len < 10:
                continue

            chunk = dataset.load_chunk(
                np.array([ep_i]), np.array([0]), np.array([ep_len])
            )[0]

            raw_pix = chunk["pixels"].to(device)
            actions = chunk["action"].to(device).float()

            pix = transform(raw_pix)
            z_all = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)

            T = z_all.shape[0]
            max_t = T - FRAMESKIP - 1
            if max_t < 1:
                continue

            for t in rng.choice(min(max_t, 50), size=min(10, max_t), replace=False):
                t = int(t)
                act_block = actions[t:t + FRAMESKIP].reshape(1, 1, EFF_ACT_DIM)
                z_ctx = z_all[t].unsqueeze(0).unsqueeze(0)
                act_emb = model.action_encoder(act_block)
                z_pred = model.predict(z_ctx, act_emb)[:, -1, :]
                z_true = z_all[t + FRAMESKIP].unsqueeze(0)

                err = torch.norm(z_pred - z_true, dim=-1).item()
                base = torch.norm(z_ctx.squeeze() - z_true.squeeze(), dim=-1).item()
                errors_1step.append(err)
                baselines_1step.append(base)

                if t + 5 * FRAMESKIP < T:
                    z_ctx_r = z_all[t].unsqueeze(0).unsqueeze(0)
                    for step in range(5):
                        a_blk = actions[t + step * FRAMESKIP:t + (step + 1) * FRAMESKIP]
                        a_blk = a_blk.reshape(1, 1, EFF_ACT_DIM)
                        ae = model.action_encoder(a_blk)
                        z_ctx_r = model.predict(z_ctx_r, ae)[:, -1:, :]
                    z_true5 = z_all[t + 5 * FRAMESKIP].unsqueeze(0)
                    err5 = torch.norm(z_ctx_r.squeeze() - z_true5.squeeze(), dim=-1).item()
                    base5 = torch.norm(z_all[t] - z_true5.squeeze(), dim=-1).item()
                    errors_5step.append(err5)
                    baselines_5step.append(base5)

    def _report(errs, bases, label):
        errs = np.array(errs)
        bases = np.array(bases)
        ratio = errs / (bases + 1e-8)
        print(f"\n  {label}")
        print(f"    WM prediction error : mean={errs.mean():.4f}  "
              f"median={np.median(errs):.4f}  p90={np.percentile(errs, 90):.4f}")
        print(f"    No-change baseline  : mean={bases.mean():.4f}  "
              f"median={np.median(bases):.4f}")
        print(f"    WM/baseline ratio   : mean={ratio.mean():.3f}  "
              f"(< 1 = WM beats trivial predictor)")

    _report(errors_1step, baselines_1step,
            f"1-step prediction  (1 WM step = {FRAMESKIP} dataset frames)")
    _report(errors_5step, baselines_5step,
            "5-step rollout     (5 WM steps = 25 dataset frames)")

    ratio_1 = np.mean(np.array(errors_1step) / (np.array(baselines_1step) + 1e-8))
    print(f"\n  VERDICT: 1-step WM/baseline = {ratio_1:.3f}")
    if ratio_1 < 0.5:
        print("    → Excellent: WM predicts much better than no-change.")
    elif ratio_1 < 1.0:
        print("    → Good: WM is better than the trivial predictor.")
    elif ratio_1 < 1.5:
        print("    → Weak: WM is barely useful; consider more training.")
    else:
        print("    → Bad: WM is worse than doing nothing.")


def test_wm_long_rollout_drift(model, dataset, device, n_test=100,
                               gaps=(1, 3, 5, 10, 20, 30)):
    print("\n" + "=" * 60)
    print("  LONG-HORIZON AUTOREGRESSIVE DRIFT")
    print("=" * 60)
    print(f"  (mirrors LatentEnv.step: 1-frame history, 25-dim action blocks)")

    transform = get_transform()
    max_gap = max(gaps)
    drift_by_gap = {k: [] for k in gaps}
    baseline_by_gap = {k: [] for k in gaps}
    natural_step = []

    rng = np.random.default_rng(0)
    ep_pool = rng.choice(len(dataset.lengths),
                         size=min(n_test, len(dataset.lengths)), replace=False)

    needed_frames = max_gap * FRAMESKIP + 1

    with torch.no_grad():
        for ep_i in ep_pool:
            ep_len = dataset.lengths[ep_i]
            if ep_len < needed_frames:
                continue

            chunk = dataset.load_chunk(
                np.array([ep_i]), np.array([0]), np.array([ep_len])
            )[0]
            raw_pix = chunk["pixels"].to(device)
            actions = chunk["action"].to(device).float()
            pix = transform(raw_pix)
            z_all = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)

            if ep_len > FRAMESKIP:
                deltas = torch.norm(
                    z_all[FRAMESKIP::FRAMESKIP] - z_all[:-FRAMESKIP:FRAMESKIP], dim=-1
                )
                natural_step.extend(deltas.tolist())

            valid_starts = ep_len - needed_frames
            n_windows = min(3, max(1, valid_starts // 10))
            starts = rng.choice(max(1, valid_starts), size=n_windows, replace=False)

            for t0 in starts:
                t0 = int(t0)
                z_state = z_all[t0].unsqueeze(0).unsqueeze(0)
                z0 = z_all[t0]

                for k in range(1, max_gap + 1):
                    a_lo = t0 + (k - 1) * FRAMESKIP
                    a_hi = t0 + k * FRAMESKIP
                    if a_hi > len(actions):
                        break
                    act_block = actions[a_lo:a_hi].reshape(1, 1, EFF_ACT_DIM)
                    act_emb = model.action_encoder(act_block)
                    z_state = model.predict(z_state, act_emb)[:, -1:, :]

                    if k in drift_by_gap:
                        true_idx = t0 + k * FRAMESKIP
                        if true_idx < len(z_all):
                            z_true = z_all[true_idx]
                            drift_by_gap[k].append(
                                torch.norm(z_state.squeeze() - z_true, dim=-1).item()
                            )
                            baseline_by_gap[k].append(
                                torch.norm(z0 - z_true, dim=-1).item()
                            )

    nat = np.array(natural_step)
    print(f"\n  Natural 1-WM-step ‖z_enc[t+1] − z_enc[t]‖:")
    print(f"    mean={nat.mean():.4f}  median={np.median(nat):.4f}  "
          f"p90={np.percentile(nat, 90):.4f}")

    mean1 = nat.mean()
    recommended_thr = mean1 * 1.5
    print(f"\n  Recommended done_threshold = {recommended_thr:.3f}  (1.5× natural 1-step)")

    print(f"\n  {'gap (WM steps)':<16s}{'pred drift mean':<20s}"
          f"{'median':<12s}{'p90':<12s}{'baseline Δ₀ mean':<20s}"
          f"{'drift / rec_thr':<18s}")
    print(f"  {'─' * 98}")

    breach_gap = None
    for k in gaps:
        if not drift_by_gap[k]:
            continue
        d = np.array(drift_by_gap[k])
        b = np.array(baseline_by_gap[k])
        ratio = d.mean() / recommended_thr
        print(f"  {k:<16d}{d.mean():<20.4f}{np.median(d):<12.4f}"
              f"{np.percentile(d, 90):<12.4f}{b.mean():<20.4f}{ratio:<18.2f}")
        if breach_gap is None and d.mean() > recommended_thr:
            breach_gap = k

    print()
    if breach_gap is None:
        print(f"  ✓ Drift stays under recommended threshold={recommended_thr:.3f} at all gaps.")
        print(f"    The 224x224 model can support RL-in-WM at all tested gap lengths!")
    else:
        max_safe_gap = max((k for k in gaps if k < breach_gap), default=0)
        print(f"  ✗ Drift first exceeds recommended threshold at gap={breach_gap}.")
        print(f"    Maximum safe low-level gap: {max_safe_gap} WM steps.")
        largest_gap = max(k for k in gaps if drift_by_gap[k])
        mean_drift = np.mean(drift_by_gap[largest_gap])
        print(f"    At gap={largest_gap}: mean drift = {mean_drift:.2f} L2")


def main():
    parser = argparse.ArgumentParser(description="Analyze 224x224 LeWM world model")
    parser.add_argument("--weights", default=os.path.join(
        os.path.expanduser("~"), ".stable_worldmodel", "cube", "lejepa", "weights.pt"))
    parser.add_argument("--dataset", default=None,
                        help="HDF5 dataset path (no .h5 extension)")
    parser.add_argument("--cache_path", default=None,
                        help="Path to save/load latent cache. "
                             "Defaults to lewm_224_latents_{encode_mode}.pt.")
    parser.add_argument("--encode_mode", default="cls_projected",
                        choices=ENCODE_MODES,
                        help="How to aggregate ViT tokens into a latent vector:\n"
                             "  cls_projected  — projector(CLS) [default, current cache]\n"
                             "  cls_raw        — CLS before projector (lossy-projector test)\n"
                             "  patch_mean     — mean of 256 spatial patches (spatial-loss test)\n"
                             "  cls_patch_cat  — concat(cls_projected, patch_mean) → 384D")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stablewm_home = os.environ.get(
        "STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))

    data_path = args.dataset or os.path.join(
        stablewm_home, "ogbench", "visual-cube-single-play-v0_224")
    # Default cache name encodes the mode so different modes don't overwrite each other.
    default_cache = os.path.join(
        stablewm_home, "ogbench",
        f"lewm_224_latents_{args.encode_mode}.pt")
    cache_path = args.cache_path or default_cache

    print("=" * 60)
    print("  224x224 LeWM Analysis")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Weights     : {args.weights}")
    print(f"  Dataset     : {data_path}.h5")
    print(f"  Cache       : {cache_path}")
    print(f"  Encode mode : {args.encode_mode}")

    print("\n[1/4] Loading model …")
    model = load_model(args.weights, device)

    print("\n[2/4] Loading dataset …")
    dataset = swm.data.HDF5Dataset(data_path)
    n_eps = len(dataset.lengths)
    print(f"  Episodes : {n_eps}")
    print(f"  Lengths  : min={dataset.lengths.min()}  "
          f"max={dataset.lengths.max()}  "
          f"mean={dataset.lengths.mean():.1f}")
    print(f"  Total frames: {dataset.lengths.sum():,}")

    print("\n[3/4] Building latent cache …")
    if os.path.exists(cache_path):
        print(f"  Cache exists at {cache_path}, loading …")
        cache = torch.load(cache_path, map_location="cpu")
        all_latents = cache["all_latents"]
        cached_mode = cache.get("encode_mode", "cls_projected")
        print(f"  Loaded {len(all_latents)} episodes, "
              f"{cache['total_frames']:,} frames  [mode={cached_mode}]")
        if cached_mode != args.encode_mode:
            print(f"  WARNING: cached mode '{cached_mode}' != requested "
                  f"'{args.encode_mode}'. Delete the cache to rebuild.")
    else:
        cache = build_cache(model, dataset, device, cache_path,
                            encode_mode=args.encode_mode)
        all_latents = cache["all_latents"]

    print("\n[4/4] Running analysis …")
    analyse_latent_space(all_latents)
    test_wm_prediction(model, dataset, device)
    test_wm_long_rollout_drift(model, dataset, device)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
