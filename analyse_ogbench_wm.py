"""
OGBench World Model Analysis
 - Builds the latent cache from the OGBench play dataset using lewm_ogbench_weights.ckpt
 - Analyses latent space geometry (distances, separability)
 - Tests WM prediction accuracy vs. actual next states
"""

import os
import sys
import time
import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stable_pretraining as spt
import stable_worldmodel as swm
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


# ─────────────────────────────────────────────────────────────────────────────
#  Constants matching lewm_ogbench.yaml / ogbench_play.yaml / train.py
# ─────────────────────────────────────────────────────────────────────────────
CKPT_PATH   = os.path.join(ROOT, "lewm_ogbench_weights.ckpt")
IMG_SIZE    = 64
PATCH_SIZE  = 8
EMBED_DIM   = 192
FRAMESKIP   = 5        # ogbench_play.yaml
ACTION_DIM  = 5        # OGBench cube has 5-DoF actions
EFF_ACT_DIM = FRAMESKIP * ACTION_DIM   # = 25
HISTORY_SZ  = 3        # history_size in lewm_ogbench.yaml

# ImageNet normalisation used by get_img_preprocessor
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
#  Build model from scratch and load checkpoint weights
# ─────────────────────────────────────────────────────────────────────────────
def load_model(device):
    encoder = spt.backbone.utils.vit_hf(
        "tiny",
        patch_size=PATCH_SIZE,
        image_size=IMG_SIZE,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size   # 192

    predictor = ARPredictor(
        num_frames=HISTORY_SZ,
        input_dim=EMBED_DIM,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=EFF_ACT_DIM, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=hidden_dim, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=hidden_dim, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    # Strip the "model." prefix added by spt.Module wrapper
    raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(raw_sd, strict=True)
    assert not missing and not unexpected, f"Load mismatch: {missing} / {unexpected}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Image transform (mirrors training)
# ─────────────────────────────────────────────────────────────────────────────
def get_transform():
    return tv_transforms.Compose([
        tv_transforms.ToDtype(torch.float32, scale=True),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Build latent cache
# ─────────────────────────────────────────────────────────────────────────────
def build_cache(model, dataset, device, save_path):
    transform = get_transform()
    num_eps    = len(dataset.lengths)
    batch_size = 16
    all_latents = []
    total_frames = 0
    t0 = time.time()

    print(f"\n  Encoding {num_eps} episodes …")
    with torch.no_grad():
        for i in range(0, num_eps, batch_size):
            end_idx  = min(i + batch_size, num_eps)
            ep_idx   = np.arange(i, end_idx)
            ep_lens  = dataset.lengths[ep_idx]
            starts   = np.zeros(len(ep_idx), dtype=int)
            chunks   = dataset.load_chunk(ep_idx, starts, ep_lens)

            for chunk in chunks:
                raw = chunk["pixels"].to(device)          # [T, C, H, W] uint8
                pix = transform(raw)                      # [T, C, H, W] float
                z   = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)
                all_latents.append(z.cpu())
                total_frames += len(z)

            elapsed = time.time() - t0
            print(f"  {end_idx:4d}/{num_eps}  frames={total_frames:,}  "
                  f"elapsed={elapsed:.1f}s", end="\r")

    print()
    cache = {"all_latents": all_latents, "total_frames": total_frames}
    torch.save(cache, save_path)
    print(f"  Saved cache → {save_path}  ({total_frames:,} frames)")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
#  Latent space analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_latent_space(all_latents):
    print("\n" + "="*60)
    print("  LATENT SPACE GEOMETRY")
    print("="*60)

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
              f"std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    _stats(norms,       "Latent L2 norm")
    _stats(gap1_dists,  "1-step distance  (1 WM step)")
    _stats(gap5_dists,  "5-step distance  (1 sec @ 5fps)")
    _stats(gap25_dists, "25-step distance (goal_offset)")
    _stats(max_dists,   "Start→End distance (full ep)")

    mean1  = np.mean(gap1_dists)
    mean25 = np.mean(gap25_dists)
    print(f"\n  Recommended success_threshold ≈ {mean1 * 1.5:.3f}  "
          f"(1.5× 1-step mean)")
    print(f"  goal_offset=25 steps spans {mean25:.3f} latent units on average")

    # Check SIGReg target: latents should be near-unit Gaussian
    all_z = torch.cat(all_latents[:200])      # first 200 episodes
    mean_z = all_z.mean(0)
    std_z  = all_z.std(0)
    print(f"\n  Latent distribution (SIGReg check):")
    print(f"    mean  →  L2={mean_z.norm():.4f}  "
          f"(want ≈0,  mean per-dim={mean_z.abs().mean():.4f})")
    print(f"    std   →  mean={std_z.mean():.4f}  "
          f"std={std_z.std():.4f}  (want ≈1.0 per dim)")


# ─────────────────────────────────────────────────────────────────────────────
#  WM prediction accuracy test
# ─────────────────────────────────────────────────────────────────────────────
def test_wm_prediction(model, dataset, device, n_test=200):
    """Test how well the WM predicts the next latent given (z_t, a_t)→z_{t+1}."""
    print("\n" + "="*60)
    print("  WORLD MODEL PREDICTION ACCURACY")
    print("="*60)
    transform = get_transform()

    errors_1step = []       # WM prediction error
    baselines_1step = []    # "no-change" baseline error (predict z_t = z_{t+1})
    errors_5step = []
    baselines_5step = []

    rng = np.random.default_rng(0)
    ep_indices = rng.choice(len(dataset.lengths), size=min(n_test, len(dataset.lengths)),
                            replace=False)

    with torch.no_grad():
        for ep_i in ep_indices:
            ep_len = dataset.lengths[ep_i]
            if ep_len < 10:
                continue

            chunk = dataset.load_chunk(
                np.array([ep_i]), np.array([0]), np.array([ep_len])
            )[0]

            raw_pix = chunk["pixels"].to(device)                  # [T, C, H, W]
            actions = chunk["action"].to(device).float()          # [T, 5]

            # Encode all frames
            pix = transform(raw_pix)
            z_all = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)  # [T, 192]

            # Build 25-dim action chunks (frameskip=5, but dataset actions are already
            # per control step; the WM was trained on groups of 5 consecutive actions)
            # For the prediction test we replicate what latent_env does:
            # a single 25-dim block = 5 consecutive 5-dim actions concatenated
            T = z_all.shape[0]
            max_t = T - FRAMESKIP - 1
            if max_t < 1:
                continue

            for t in rng.choice(min(max_t, 50), size=min(10, max_t), replace=False):
                t = int(t)
                # Build 25-dim action block from 5 consecutive raw actions
                act_block = actions[t : t + FRAMESKIP].reshape(1, 1, EFF_ACT_DIM)  # [1,1,25]

                # 1-step prediction: z_t → z_{t+frameskip}
                z_ctx = z_all[t].unsqueeze(0).unsqueeze(0)   # [1, 1, 192]
                act_emb = model.action_encoder(act_block)
                z_pred = model.predict(z_ctx, act_emb)[:, -1, :]  # [1, 192]
                z_true = z_all[t + FRAMESKIP].unsqueeze(0)         # [1, 192]

                err   = torch.norm(z_pred - z_true, dim=-1).item()
                base  = torch.norm(z_ctx.squeeze() - z_true.squeeze(), dim=-1).item()
                errors_1step.append(err)
                baselines_1step.append(base)

                # 5-step rollout: predict 5 WM steps (= 25 dataset frames) ahead
                if t + 5 * FRAMESKIP < T:
                    z_ctx_r = z_all[t].unsqueeze(0).unsqueeze(0)
                    for step in range(5):
                        a_blk = actions[t + step*FRAMESKIP : t + (step+1)*FRAMESKIP]
                        a_blk = a_blk.reshape(1, 1, EFF_ACT_DIM)
                        ae    = model.action_encoder(a_blk)
                        z_ctx_r = model.predict(z_ctx_r, ae)[:, -1:, :]
                    z_true5 = z_all[t + 5*FRAMESKIP].unsqueeze(0)
                    err5  = torch.norm(z_ctx_r.squeeze() - z_true5.squeeze(), dim=-1).item()
                    base5 = torch.norm(z_all[t] - z_true5.squeeze(), dim=-1).item()
                    errors_5step.append(err5)
                    baselines_5step.append(base5)

    def _report(errs, bases, label):
        errs  = np.array(errs)
        bases = np.array(bases)
        ratio = errs / (bases + 1e-8)
        print(f"\n  {label}")
        print(f"    WM prediction error : mean={errs.mean():.4f}  "
              f"median={np.median(errs):.4f}  p90={np.percentile(errs,90):.4f}")
        print(f"    No-change baseline  : mean={bases.mean():.4f}  "
              f"median={np.median(bases):.4f}")
        print(f"    WM/baseline ratio   : mean={ratio.mean():.3f}  "
              f"(< 1 = WM beats trivial predictor)")

    _report(errors_1step,  baselines_1step,
            f"1-step prediction  (1 WM step = {FRAMESKIP} dataset frames)")
    _report(errors_5step,  baselines_5step,
            f"5-step rollout     (5 WM steps = 25 dataset frames = goal_offset)")

    # Sanity: a perfect WM would get ratio ≈ 0; ratio > 1 means WM is useless
    ratio_1 = np.mean(np.array(errors_1step) / (np.array(baselines_1step) + 1e-8))
    print(f"\n  ✓ VERDICT: 1-step WM/baseline = {ratio_1:.3f}")
    if ratio_1 < 0.5:
        print("    → Excellent: WM predicts much better than no-change.")
    elif ratio_1 < 1.0:
        print("    → Good: WM is better than the trivial predictor.")
    elif ratio_1 < 1.5:
        print("    → Weak: WM is barely useful; consider more training.")
    else:
        print("    → Bad: WM is worse than doing nothing. Check config/transform.")


# ─────────────────────────────────────────────────────────────────────────────
#  Long-horizon autoregressive drift (mirrors LatentEnv.step exactly)
# ─────────────────────────────────────────────────────────────────────────────
def test_wm_long_rollout_drift(model, dataset, device, n_test=100,
                                gaps=(1, 3, 5, 10, 20, 30)):
    """
    Autoregressively roll the predictor from z_enc[0] using the ground-truth
    action sequence, exactly as latent_env.LatentEnv.step does:
        z_state = z_enc[0]                           (single-frame history)
        for k in range(max_gap):
            z_state = model.predict(z_state, action_encoder(a_k))[:, -1:]
    Compare against encode(real_frame[k]) at each of the requested gaps.

    This is the definitive Tier 1.1 measurement: if drift at gap=30 is
    comfortably below the training done_threshold (=2.0), the "exposure
    bias" hypothesis is wrong. If drift at gap=30 greatly exceeds 2.0, it
    confirms the diagnosis that the SAC critic was trained in a fictional
    predictor-drift space.

    Each dataset action row is a single 5-DoF control step; the world model
    was trained on 25-dim blocks of 5 consecutive raw actions (frameskip=5).
    One "WM step" = 5 dataset frames.
    """
    print("\n" + "="*60)
    print("  LONG-HORIZON AUTOREGRESSIVE DRIFT")
    print("="*60)
    print(f"  (mirrors LatentEnv.step: 1-frame history, 25-dim action blocks)")

    transform = get_transform()
    max_gap = max(gaps)
    # drift_by_gap[k] = list of L2 distances between predict^k(z_enc[0], a...)
    # and z_enc[k*FRAMESKIP] across all test windows
    drift_by_gap      = {k: [] for k in gaps}
    baseline_by_gap   = {k: [] for k in gaps}  # ‖z_enc[0] - z_enc[k*FRAMESKIP]‖
    natural_step      = []  # ‖z_enc[t+1 WM-step] − z_enc[t]‖

    rng = np.random.default_rng(0)
    ep_pool = rng.choice(len(dataset.lengths),
                         size=min(n_test, len(dataset.lengths)), replace=False)

    needed_frames = max_gap * FRAMESKIP + 1   # frames 0 … max_gap·FRAMESKIP

    with torch.no_grad():
        for ep_i in ep_pool:
            ep_len = dataset.lengths[ep_i]
            if ep_len < needed_frames:
                continue

            # Load & encode the entire (trimmed) episode once
            chunk = dataset.load_chunk(
                np.array([ep_i]), np.array([0]), np.array([ep_len])
            )[0]
            raw_pix = chunk["pixels"].to(device)
            actions = chunk["action"].to(device).float()   # [ep_len, 5]
            pix     = transform(raw_pix)
            z_all   = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)  # [ep_len, 192]

            # Natural 1-WM-step scale (one per episode)
            if ep_len > FRAMESKIP:
                deltas = torch.norm(
                    z_all[FRAMESKIP::FRAMESKIP] - z_all[:-FRAMESKIP:FRAMESKIP],
                    dim=-1
                )
                natural_step.extend(deltas.tolist())

            # Only a few windows per episode to keep the cost bounded
            valid_starts = ep_len - needed_frames
            n_windows = min(3, max(1, valid_starts // 10))
            starts = rng.choice(max(1, valid_starts), size=n_windows, replace=False)

            for t0 in starts:
                t0 = int(t0)
                # 1-frame history seeded with the real encoder latent
                z_state = z_all[t0].unsqueeze(0).unsqueeze(0)    # [1, 1, 192]
                z0      = z_all[t0]

                # Walk the predictor forward for max_gap WM steps,
                # recording divergence at each requested checkpoint.
                for k in range(1, max_gap + 1):
                    a_lo = t0 + (k - 1) * FRAMESKIP
                    a_hi = t0 + k       * FRAMESKIP
                    act_block = actions[a_lo:a_hi].reshape(1, 1, EFF_ACT_DIM)
                    act_emb   = model.action_encoder(act_block)
                    z_state   = model.predict(z_state, act_emb)[:, -1:, :]

                    if k in drift_by_gap:
                        z_true = z_all[t0 + k * FRAMESKIP]
                        drift_by_gap[k].append(
                            torch.norm(z_state.squeeze() - z_true, dim=-1).item()
                        )
                        baseline_by_gap[k].append(
                            torch.norm(z0 - z_true, dim=-1).item()
                        )

    # Report
    nat = np.array(natural_step)
    print(f"\n  Natural 1-WM-step ‖z_enc[t+1] − z_enc[t]‖  "
          f"(the scale pred_loss is measured against):")
    print(f"    mean={nat.mean():.4f}  median={np.median(nat):.4f}  "
          f"p90={np.percentile(nat, 90):.4f}")

    done_thr = 2.0
    print(f"\n  Training done_threshold = {done_thr}  (sparse reward boundary)")
    print(f"\n  {'gap (WM steps)':<16s}{'pred drift mean':<20s}"
          f"{'median':<12s}{'p90':<12s}{'baseline Δ₀ mean':<20s}{'drift / done_thr':<18s}")
    print(f"  {'─'*98}")

    breach_gap = None
    for k in gaps:
        if not drift_by_gap[k]:
            continue
        d = np.array(drift_by_gap[k])
        b = np.array(baseline_by_gap[k])
        ratio = d.mean() / done_thr
        print(f"  {k:<16d}{d.mean():<20.4f}{np.median(d):<12.4f}"
              f"{np.percentile(d, 90):<12.4f}{b.mean():<20.4f}{ratio:<18.2f}")
        if breach_gap is None and d.mean() > done_thr:
            breach_gap = k

    print()
    if breach_gap is None:
        print(f"  ✗ Drift stays under done_threshold={done_thr} at all tested gaps.")
        print(f"    Exposure-bias hypothesis is NOT supported — look elsewhere.")
    else:
        largest_gap = max(k for k in gaps if drift_by_gap[k])
        mean_drift  = np.mean(drift_by_gap[largest_gap])
        print(f"  ✓ Mean drift first exceeds done_threshold={done_thr} "
              f"at gap={breach_gap} WM steps.")
        print(f"    At gap={largest_gap} the SAC critic was training inside a ")
        print(f"    predictor-drift cloud ~{mean_drift:.2f} L2 away on average from")
        print(f"    any real encoder latent — confirming exposure-bias diagnosis.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))

    data_path  = os.path.join(stablewm_home, "ogbench", "cube_single_play_v0")
    cache_path = os.path.join(stablewm_home, "ogbench_latents_cache.pt")

    print("="*60)
    print("  OGBench LeWM Analysis")
    print("="*60)
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Dataset   : {data_path}.h5")

    # 1. Load model
    print("\n[1/4] Loading model …")
    model = load_model(device)

    # 2. Load dataset
    print("\n[2/4] Loading dataset …")
    dataset = swm.data.HDF5Dataset(data_path)
    n_eps   = len(dataset.lengths)
    print(f"  Episodes : {n_eps}")
    print(f"  Lengths  : min={dataset.lengths.min()}  "
          f"max={dataset.lengths.max()}  "
          f"mean={dataset.lengths.mean():.1f}")
    print(f"  Total frames: {dataset.lengths.sum():,}")

    # 3. Build / load cache
    print("\n[3/4] Building latent cache …")
    if os.path.exists(cache_path):
        print(f"  Cache exists at {cache_path}, loading …")
        cache = torch.load(cache_path, map_location="cpu")
        all_latents = cache["all_latents"]
        print(f"  Loaded {len(all_latents)} episodes, "
              f"{cache['total_frames']:,} frames")
    else:
        cache = build_cache(model, dataset, device, cache_path)
        all_latents = cache["all_latents"]

    # 4. Analysis
    print("\n[4/4] Running analysis …")
    analyse_latent_space(all_latents)
    test_wm_prediction(model, dataset, device)
    test_wm_long_rollout_drift(model, dataset, device)

    print("\n" + "="*60)
    print("  Done.")
    print("="*60)


if __name__ == "__main__":
    main()
