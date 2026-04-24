"""
Check whether the lewm-cube latent cache covers OGBench's 5 task goal states.

Encodes each OGBench task's goal image (64x64, resized to 224x224) with the
224x224 LeWM encoder and finds its nearest neighbour in the lewm-cube cache.
If NN distance is < 2× the natural 1-WM-step distance (6.29), the cache
covers that goal state well enough for training and evaluation.

Usage:
    python latent_hindsight_rl/check_ogbench_coverage.py \
        --weights $STABLEWM_HOME/cube/lejepa_weights.ckpt \
        --cache   $STABLEWM_HOME/lewm_224_latents_cache.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

IMG_SIZE   = 224
PATCH_SIZE = 14
EMBED_DIM  = 192
NATURAL_WM_STEP_DIST = 6.29   # from analyse_lewm_224 output
PASS_THRESHOLD = NATURAL_WM_STEP_DIST * 2  # 12.58 — within 2 WM steps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(weights_path, device):
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=PATCH_SIZE, image_size=IMG_SIZE,
        pretrained=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size

    predictor    = ARPredictor(num_frames=3, input_dim=EMBED_DIM, hidden_dim=hidden_dim,
                               output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048,
                               dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_enc   = Embedder(input_dim=25, emb_dim=EMBED_DIM)
    projector    = MLP(input_dim=hidden_dim, output_dim=EMBED_DIM,
                       hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj    = MLP(input_dim=hidden_dim, output_dim=EMBED_DIM,
                       hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_enc, projector=projector, pred_proj=pred_proj)
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle raw state dict or dict-wrapped
    sd = weights if not isinstance(weights, dict) or "state_dict" not in weights else \
         {k[len("model."):]: v for k, v in weights["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def encode_obs(model, obs_hwc, transform, device):
    """obs_hwc: (H, W, 3) uint8 numpy array → [192] latent tensor"""
    t = transform(torch.from_numpy(obs_hwc).permute(2, 0, 1))  # [3, H, W]
    t = t.unsqueeze(0).unsqueeze(0).to(device)                  # [1, 1, 3, H, W]
    with torch.no_grad():
        emb = model.encode({"pixels": t})["emb"]               # [1, 1, 192]
    return emb[0, 0].cpu()                                       # [192]


def main():
    parser = argparse.ArgumentParser(description="OGBench coverage check against lewm-cube cache")
    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    parser.add_argument("--weights",
        default=os.path.join(stablewm_home, "cube", "lejepa_weights.ckpt"))
    parser.add_argument("--cache",
        default=os.path.join(stablewm_home, "lewm_224_latents_cache.pt"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.weights} ...")
    model = load_model(args.weights, device)

    # swm/OGBCube-v0 renders natively at IMG_SIZE — no Resize needed (matches eval_ogbench.py)
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"Loading latent cache from {args.cache} ...")
    cache = torch.load(args.cache, map_location="cpu")
    all_latents = cache["all_latents"]
    # Build a flat [N, 192] tensor for fast NN search
    flat_cache = torch.cat(all_latents, dim=0)  # [N, 192]
    print(f"  Cache: {len(all_latents)} episodes, {flat_cache.shape[0]:,} total frames")

    # ── Cache geometry (reference) ────────────────────────────────────────────
    within_dists = []
    for ep in all_latents[:500]:
        if ep.shape[0] > 5:
            d = torch.norm(ep[5:] - ep[:-5], dim=-1)
            within_dists.extend(d.tolist())
    within_mean = float(np.mean(within_dists))
    print(f"\n  Within-cache 1-WM-step distance: mean={within_mean:.3f}")
    print(f"  Pass threshold (2× that):        {within_mean * 2:.3f}")

    # ── OGBench goal encoding ─────────────────────────────────────────────────
    try:
        import gymnasium
        import ogbench  # registers environments
    except ImportError:
        print("\nERROR: ogbench / gymnasium not available. Activate the correct venv.")
        return

    print("\nEncoding OGBench task goals ...")
    # Use swm/OGBCube-v0 (same env as eval_ogbench.py) — renders at 224×224 natively.
    # Previous version incorrectly used "visual-cube-single-v0" (64×64 OGBench env)
    # and info["goal"] instead of info["target"], giving untrustworthy coverage results.
    env = gymnasium.make('swm/OGBCube-v0', ob_type='pixels', env_type='single', visualize_info=False)
    task_infos = env.unwrapped.task_infos
    num_tasks = len(task_infos)

    results = []
    all_pass = True

    for task_id in range(1, num_tasks + 1):
        task_name = task_infos[task_id - 1].get("task_name", f"task{task_id}")
        _, info = env.reset(options=dict(task_id=task_id))
        goal_img = info["target"]  # (H, W, 3) uint8 — swm/OGBCube-v0 stores goal under 'target'

        z_goal = encode_obs(model, goal_img, transform, device)  # [192]

        # Nearest-neighbour search in cache
        dists = torch.norm(flat_cache - z_goal.unsqueeze(0), dim=-1)  # [N]
        nn_dist = dists.min().item()
        nn_idx  = dists.argmin().item()

        passed = nn_dist <= within_mean * 2
        if not passed:
            all_pass = False
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Task {task_id} ({task_name:30s}): NN dist = {nn_dist:.3f}  {status}")
        results.append((task_name, nn_dist, passed))

    env.close()

    print(f"\n{'='*60}")
    if all_pass:
        print("RESULT: ALL tasks PASS — lewm-cube cache covers OGBench goals.")
        print("        Proceed with training using the existing cache.")
    else:
        failed = [r[0] for r in results if not r[2]]
        print(f"RESULT: {len(failed)} task(s) FAIL — goals not well covered:")
        for name in failed:
            print(f"  - {name}")
        print("\nRecommendation: rebuild cache from OGBench data re-rendered at 224x224.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
