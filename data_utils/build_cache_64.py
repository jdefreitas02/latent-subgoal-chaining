"""
build_cache_64.py
Build the encoder-latent cache for the 64×64 OGBench-trained LeWM.

Encodes every frame of every episode in the HDF5 dataset and saves:
  {
      'all_latents': List[Tensor[T_ep, 192]],  # per-episode encoder latents
      'all_actions': List[Tensor[T_ep, 5]],    # per-episode raw 5-D actions
      'total_frames': int,
  }

The resulting .pt file is required by train_hiql_lewm.py's RealOfflineCache.

Usage:
    python latent_hindsight_rl/build_cache_64.py \\
        --weights  ~/leworldmodel/lewm_ogbench_weights.ckpt \\
        --dataset  ~/stable_wm_data/ogbench/visual-cube-single-play-v0 \\
        --cache    ~/stable_wm_data/ogbench/lewm_64_latents_cache.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import stable_pretraining as spt
import stable_worldmodel as swm
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

# ── Constants matching the 64×64 OGBench-trained LeWM ────────────────────────
IMG_SIZE    = 64
PATCH_SIZE  = 8
EMBED_DIM   = 192
FRAMESKIP   = 5
ACTION_DIM  = 5
EFF_ACT_DIM = FRAMESKIP * ACTION_DIM   # 25

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(weights_path, device):
    """Load the 64×64 LeWM from a Lightning-style .ckpt file."""
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=PATCH_SIZE, image_size=IMG_SIZE,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=EFF_ACT_DIM, emb_dim=EMBED_DIM)
    projector = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=EMBED_DIM, output_dim=EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        # Lightning-style checkpoint
        raw_sd = {k[len("model."):]: v
                  for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        epoch = ckpt.get('epoch', '?')
    else:
        raw_sd = dict(ckpt)
        epoch = '?'

    missing, unexpected = model.load_state_dict(raw_sd, strict=True)
    assert not missing and not unexpected, \
        f"Checkpoint mismatch — missing: {missing}  unexpected: {unexpected}"

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded 64×64 LeWM from {weights_path}  (epoch={epoch})")
    return model


def get_transform():
    return tv_transforms.Compose([
        tv_transforms.ToDtype(torch.float32, scale=True),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_cache(model, dataset, device, save_path, batch_size=16):
    """Encode all episodes and save the latent cache.

    Args:
        model:      Loaded LeWM model (frozen).
        dataset:    swm.data.HDF5Dataset instance.
        device:     Torch device.
        save_path:  Output .pt path.
        batch_size: Episodes processed per GPU batch.
    """
    transform   = get_transform()
    num_eps     = len(dataset.lengths)
    all_latents = []
    all_actions = []
    total_frames = 0
    t0 = time.time()

    print(f"\n  Encoding {num_eps} episodes  (batch_size={batch_size}) ...")
    with torch.no_grad():
        for i in range(0, num_eps, batch_size):
            end_idx  = min(i + batch_size, num_eps)
            ep_idx   = np.arange(i, end_idx)
            ep_lens  = dataset.lengths[ep_idx]
            starts   = np.zeros(len(ep_idx), dtype=int)
            chunks   = dataset.load_chunk(ep_idx, starts, ep_lens)

            for chunk in chunks:
                raw = chunk["pixels"].to(device)   # [T, C, H, W]  uint8, 64×64
                pix = transform(raw)               # [T, C, H, W]  float32, ImageNet-normalised

                # JEPA encode expects [B, T, C, H, W]; B=1 here
                z_ep = model.encode({"pixels": pix.unsqueeze(0)})["emb"].squeeze(0)  # [T, 192]

                all_latents.append(z_ep.cpu())

                if "action" in chunk:
                    all_actions.append(chunk["action"].cpu().float())   # [T, 5]

                total_frames += z_ep.shape[0]

            elapsed = time.time() - t0
            print(f"  {end_idx:5d}/{num_eps}  frames={total_frames:,}  elapsed={elapsed:.1f}s",
                  end="\r", flush=True)

    print()  # newline after \r
    cache = {
        "all_latents":  all_latents,
        "all_actions":  all_actions,
        "total_frames": total_frames,
        "img_size":     IMG_SIZE,
        "patch_size":   PATCH_SIZE,
    }
    torch.save(cache, save_path)
    print(f"\n  Cache saved → {save_path}  ({total_frames:,} frames, {num_eps} episodes)")
    return cache


def analyse_latent_space(all_latents):
    """Print basic geometry stats to validate the SIGReg prior."""
    print("\n" + "=" * 60)
    print("  LATENT SPACE GEOMETRY (64×64 LeWM)")
    print("=" * 60)

    gap1_dists, gap5_dists, max_dists, norms = [], [], [], []
    for ep in all_latents[:1000]:
        T = len(ep)
        norms.extend(torch.norm(ep, dim=-1).tolist())
        if T > 1:
            gap1_dists.append(torch.norm(ep[1:] - ep[:-1], dim=-1).mean().item())
        if T > FRAMESKIP:
            gap5_dists.append(torch.norm(ep[FRAMESKIP:] - ep[:-FRAMESKIP], dim=-1).mean().item())
        max_dists.append(torch.norm(ep[-1] - ep[0], dim=-1).item())

    def _stats(arr, label):
        arr = np.array(arr)
        print(f"  {label:<40s}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    _stats(norms,      "Latent L2 norm")
    _stats(gap1_dists, "1-frame distance  (0.2 s)")
    _stats(gap5_dists, f"{FRAMESKIP}-frame distance  (1 WM step = 1 s)")
    _stats(max_dists,  "Start→End distance (full ep)")

    mean5 = np.mean(gap5_dists) if gap5_dists else float('nan')
    print(f"\n  Recommended done_threshold ≈ {mean5 * 1.5:.3f}  (1.5× 1-WM-step mean)")
    print(f"  (Use this value for --done_threshold in train_hiql_lewm.py and eval_ogbench.py)")

    all_z = torch.cat(all_latents[:200])
    mean_z = all_z.mean(0)
    std_z  = all_z.std(0)
    print(f"\n  SIGReg check:")
    print(f"    mean  L2={mean_z.norm():.4f}  (want ≈0, per-dim mean={mean_z.abs().mean():.4f})")
    print(f"    std   mean={std_z.mean():.4f}  std={std_z.std():.4f}  (want ≈1.0 per dim)")


def main():
    parser = argparse.ArgumentParser(
        description="Build latent cache for the 64×64 OGBench-trained LeWM.")
    parser.add_argument('--weights',    required=True,
                        help="Path to lewm_ogbench_weights.ckpt")
    parser.add_argument('--dataset',    required=True,
                        help="HDF5 dataset path WITHOUT .h5 extension, e.g. "
                             "~/stable_wm_data/ogbench/visual-cube-single-play-v0")
    parser.add_argument('--cache',      required=True,
                        help="Output cache path, e.g. "
                             "~/stable_wm_data/ogbench/lewm_64_latents_cache.pt")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Episodes per GPU encoding batch (default 16).")
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Weights:  {args.weights}")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {args.cache}")

    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)

    model   = load_model(args.weights, device)
    dataset = swm.data.HDF5Dataset(args.dataset)
    print(f"  Dataset: {len(dataset.lengths)} episodes, "
          f"{dataset.lengths.sum():,} total frames")

    cache = build_cache(model, dataset, device, args.cache, args.batch_size)
    analyse_latent_space(cache['all_latents'])


if __name__ == '__main__':
    main()
