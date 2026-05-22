"""
Diagnostic: do predictor-space latents match encoder-space latents after 1 real step?

This tests Hypothesis 1 (H1): the SAC trains on predictor-drifted transitions
  (z_pred_t, a, z_pred_{t+1}) but at eval time sees encoder latents (z_enc_t).
If the predictor drifts away from the encoder even after 1 step, the SAC policy
is effectively out-of-distribution at every real step.

Outputs (per episode):
  ||z_pred - z_enc_next||   — how far the predictor thinks we ended up vs reality
  ||z_enc  - z_enc_next||   — how far we actually moved in encoder space
  ratio = pred_err / real_move — if >> 1, predictor is useless for navigation

Usage:
    python latent_hindsight_rl/check_predictor_consistency.py \
        --ckpt_path $STABLEWM_HOME/cube/lejepa_weights.ckpt \
        --num_episodes 20 --num_steps 10
"""

import os
import sys
import argparse
import numpy as np
import torch

_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import stable_pretraining as spt
import stable_worldmodel as swm
import torch.nn as nn
from torchvision.transforms import v2 as transforms


# ── Reuse load_jepa and _make_img_transform from eval_ogbench ─────────────────
def _make_img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


def load_jepa(ckpt_path, device, img_size=224, patch_size=14):
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False)
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("model.")} if "state_dict" in ckpt else dict(ckpt)
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded JEPA from {ckpt_path}")
    return model.to(device).eval()


def encode_obs(model, obs_hwc, transform, device):
    """uint8 HWC → [1, 192] encoder latent."""
    t = transform(obs_hwc).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,C,H,W]
    with torch.no_grad():
        emb = model.encode({"pixels": t})["emb"]  # [1,1,192]
    return emb[:, -1]  # [1, 192]


def predict_next(model, z_curr, action_scaled, device):
    """Run the predictor for 1 WM step.

    action_scaled: np.ndarray [5, 5] — 5 physical actions in scaled (StandardScaler) space.
    Returns z_pred [1, 192].

    Mirrors latent_env.py:88-93:
        act_emb = model.action_encoder(actions)   # [B, T, 192]
        z_next  = model.predict(z_state, act_emb)[:, -1:]
    """
    # Flatten 5-step block → [25] then embed via action_encoder
    a_block = torch.tensor(action_scaled.flatten(), dtype=torch.float32, device=device)
    a_t = a_block.unsqueeze(0).unsqueeze(0)  # [1, 1, 25]
    with torch.no_grad():
        act_emb = model.action_encoder(a_t)           # [1, 1, 192]
        z_pred  = model.predict(z_curr.unsqueeze(1), act_emb)  # [1, 1, 192]
    return z_pred[:, -1]  # [1, 192]


def main():
    parser = argparse.ArgumentParser(description="Check predictor vs encoder consistency in real env.")
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--ogbench_dir', default=None)
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=20,
                        help="WM steps per episode (each WM step = 5 physical actions)")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pca_path', type=str, default=None,
                        help="Path to PCA projection params (.pt) from build_pca_projection.py. "
                             "If provided, also reports predictor consistency and calibrated "
                             "done_threshold in the projected (pca_dim) space.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(stablewm_home, "cube", "lejepa_weights.ckpt")

    if args.ogbench_dir:
        sys.path.insert(0, os.path.abspath(args.ogbench_dir))
    import ogbench
    import gymnasium

    # ── PCA projection (optional) ─────────────────────────────────────────────
    pca_mean = pca_matrix = None
    pca_dim = None
    if args.pca_path is not None:
        print(f"Loading PCA projection from {args.pca_path} ...")
        pca_data = torch.load(args.pca_path, map_location='cpu')
        pca_mean   = pca_data['pca_mean'].to(device)    # [192]
        pca_matrix = pca_data['pca_matrix'].to(device)  # [192, D]
        pca_dim    = int(pca_data['pca_dim'])
        top_k_var  = pca_data.get('top_k_variance', float('nan'))
        print(f"  PCA: 192D → {pca_dim}D  (top-k variance: {top_k_var*100:.1f}%)")

    print(f"Device: {device}")
    print("Creating swm/OGBCube-v0 environment...")
    env = gymnasium.make('swm/OGBCube-v0', ob_type='pixels', env_type='single', visualize_info=False)

    print("Loading JEPA model...")
    model = load_jepa(args.ckpt_path, device, img_size=args.img_size, patch_size=args.patch_size)
    transform = _make_img_transform()

    # Need an action scaler to get scaled actions back for the predictor.
    # Use StandardScaler fitted on dataset — same as in eval.
    from sklearn import preprocessing
    import stable_worldmodel as swm
    dataset_path = os.path.join(stablewm_home, "ogbench", "cube_single_expert")
    dataset = swm.data.HDF5Dataset(dataset_path, keys_to_cache=['action'],
                                    cache_dir=os.path.dirname(dataset_path))
    action_data = dataset.get_col_data('action')
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = preprocessing.StandardScaler()
    scaler.fit(action_data)
    print(f"  Action scaler fit on {len(action_data):,} steps.")

    all_pred_errs   = []  # ||z_pred - z_enc_next|| after 1 WM step  (192D)
    all_real_moves  = []  # ||z_enc - z_enc_next|| after 1 WM step   (192D)
    all_enc_starts  = []  # ||z_enc_start - z_enc_next|| — cumulative drift from episode start
    all_pred_errs_proj  = []  # same in pca_dim space (if --pca_path provided)
    all_real_moves_proj = []  # same in pca_dim space

    print(f"\nRunning {args.num_episodes} episodes × {args.num_steps} WM steps each...")
    print(f"  Each WM step = 5 physical actions (frameskip=5)\n")

    for ep in range(args.num_episodes):
        obs, info = env.reset(options=dict(task_id=((ep % 5) + 1)))
        z_enc = encode_obs(model, obs, transform, device)
        z_enc_start = z_enc.clone()

        ep_pred_errs  = []
        ep_real_moves = []

        for wm_step in range(args.num_steps):
            # Sample 5 random actions (1 WM step)
            actions_physical = np.array([env.action_space.sample() for _ in range(5)])
            actions_scaled   = scaler.transform(actions_physical)  # [5, 5]

            # Predict next latent using the predictor
            z_pred_next = predict_next(model, z_enc, actions_scaled, device)

            # Execute the 5 actions in the real env
            for a in actions_physical:
                obs, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    break

            # Encode the resulting real observation
            z_enc_next = encode_obs(model, obs, transform, device)

            pred_err  = torch.norm(z_pred_next - z_enc_next, p=2, dim=-1).item()
            real_move = torch.norm(z_enc       - z_enc_next, p=2, dim=-1).item()

            ep_pred_errs.append(pred_err)
            ep_real_moves.append(real_move)
            all_pred_errs.append(pred_err)
            all_real_moves.append(real_move)

            # Project to pca_dim and record distances there too
            if pca_matrix is not None:
                def proj(z):
                    return (z - pca_mean) @ pca_matrix  # [1, D]
                pred_err_proj  = torch.norm(proj(z_pred_next) - proj(z_enc_next), p=2, dim=-1).item()
                real_move_proj = torch.norm(proj(z_enc)       - proj(z_enc_next), p=2, dim=-1).item()
                all_pred_errs_proj.append(pred_err_proj)
                all_real_moves_proj.append(real_move_proj)

            z_enc = z_enc_next  # advance

            if terminated or truncated:
                break

        enc_start_drift = torch.norm(z_enc - z_enc_start, p=2, dim=-1).item()
        all_enc_starts.append(enc_start_drift)

    env.close()

    pred_errs  = np.array(all_pred_errs)
    real_moves = np.array(all_real_moves)
    ratios     = pred_errs / (real_moves + 1e-6)

    print("=" * 60)
    print("PREDICTOR CONSISTENCY RESULTS")
    print("=" * 60)
    print(f"  WM steps measured:              {len(pred_errs)}")
    print()
    print(f"  ||z_pred - z_enc_next||  (predictor error after 1 WM step):")
    print(f"    mean={pred_errs.mean():.3f}  median={np.median(pred_errs):.3f}  "
          f"p90={np.percentile(pred_errs, 90):.3f}")
    print()
    print(f"  ||z_enc - z_enc_next||   (real encoder displacement in 1 WM step):")
    print(f"    mean={real_moves.mean():.3f}  median={np.median(real_moves):.3f}  "
          f"p90={np.percentile(real_moves, 90):.3f}")
    print()
    print(f"  pred_err / real_move ratio  (1.0 = predictor as accurate as doing nothing):")
    print(f"    mean={ratios.mean():.2f}  median={np.median(ratios):.2f}  "
          f"p90={np.percentile(ratios, 90):.2f}")
    print()
    print("  Reference distances (from prior drift analysis):")
    print("    done_threshold = 2.41   (training success boundary, 192D)")
    print("    1 WM-step distance (random policy, mean predictor drift) = 4.83")
    print("    start-to-goal distance (gap=1) ≈ 6.29")
    print()

    # ── PCA-projected statistics ───────────────────────────────────────────────
    if pca_matrix is not None and all_pred_errs_proj:
        pred_errs_proj  = np.array(all_pred_errs_proj)
        real_moves_proj = np.array(all_real_moves_proj)
        ratios_proj     = pred_errs_proj / (real_moves_proj + 1e-6)

        print("=" * 60)
        print(f"PCA-PROJECTED STATISTICS ({pca_dim}D)")
        print("=" * 60)
        print(f"  ||proj(z_pred) - proj(z_enc_next)||  (predictor error in {pca_dim}D):")
        print(f"    mean={pred_errs_proj.mean():.3f}  median={np.median(pred_errs_proj):.3f}  "
              f"p90={np.percentile(pred_errs_proj, 90):.3f}")
        print()
        print(f"  ||proj(z_enc) - proj(z_enc_next)||   (real displacement in {pca_dim}D):")
        print(f"    mean={real_moves_proj.mean():.3f}  median={np.median(real_moves_proj):.3f}  "
              f"p90={np.percentile(real_moves_proj, 90):.3f}")
        print()
        print(f"  pred_err / real_move ratio in {pca_dim}D:")
        print(f"    mean={ratios_proj.mean():.2f}  median={np.median(ratios_proj):.2f}  "
              f"p90={np.percentile(ratios_proj, 90):.2f}")
        print()

        # Calibrated done_threshold: median real 1-step displacement in pca_dim space
        # (= "1 WM step away" in projected space → suitable success radius)
        done_threshold_proj = float(np.median(real_moves_proj))
        print(f"  Recommended done_threshold in {pca_dim}D:  {done_threshold_proj:.4f}")
        print(f"    (= median 1-step real encoder displacement in {pca_dim}D)")
        print()
        print(f"  Use in training and eval:")
        print(f"    python latent_hindsight_rl/train_joint.py \\")
        print(f"        --pca_path {args.pca_path} \\")
        print(f"        --done_threshold {done_threshold_proj:.4f}")
        print()

    if pred_errs.mean() > 4.0:
        print("  ⚠ DIAGNOSIS: Predictor error >> done_threshold.")
        print("    H1 CONFIRMED: predictor drifts far from real encoder after 1 step.")
        print("    The SAC trained on predictor rollouts is out-of-distribution at eval.")
        print("    Fix: add encoder-cache transitions to the SAC replay buffer.")
    elif pred_errs.mean() < 1.5:
        print("  ✓ Predictor is consistent with encoder. H1 likely NOT the problem.")
        print("    Check H2 (threshold artifact) or H4 (sparse reward instability).")
    else:
        print("  ⚠ Predictor error moderate. H1 may be a partial contributor.")


if __name__ == "__main__":
    main()
