"""Re-encode the OGBench play HDF5 with a (possibly fine-tuned) JEPA encoder,
saving a new latent cache that mirrors the structure of
``lewm_224_latents_cache.pt`` (a dict ``{"all_latents": [tensor(T_i, 192), ...]}``).

Why we need this: when we fine-tune the WM's encoder, the latent space changes.
The existing cache was built with the original encoder; downstream code
(anchor pool, offline dataset, QC critic) assumes one consistent latent space.
To use a full-FT WM cleanly, we re-encode the dataset with the new encoder
into a SEPARATE cache file. The original cache is never touched.

Inputs:
    --wm_ckpt        Base path of the WM checkpoint (load_jepa convention)
    --hdf5           Path to the visual-cube-single-play-v0_224.h5 file
    --out_cache      Output .pt path for the new cache (do NOT overwrite)

The script:
    1. Loads the WM via load_jepa(...).
    2. Streams batches of pixels from the HDF5.
    3. Encodes each batch via jepa.encode({"pixels": ...}) -> info["emb"].
    4. Splits the resulting flat (N, 192) array back into per-episode lists
       using the dataset's ep_offset/ep_len arrays.
    5. Saves as torch.save({"all_latents": [tensors]}, out_cache).
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "offline_to_online"))

import h5py
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

from envs.jepa_loader import load_jepa


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def encode_chunk(jepa, pixels_uint8_NHWC, device, tx):
    """Encode (N, H, W, 3) uint8 -> (N, 192) float32."""
    N = pixels_uint8_NHWC.shape[0]
    # Apply per-image tx then stack (CPU side), then batch transfer
    imgs = torch.empty((N, 3, pixels_uint8_NHWC.shape[1], pixels_uint8_NHWC.shape[2]),
                       dtype=torch.float32)
    for i in range(N):
        imgs[i] = tx(torch.from_numpy(pixels_uint8_NHWC[i]).permute(2, 0, 1).contiguous())
    imgs = imgs.to(device, non_blocking=True)
    # encode expects (B, T, 3, H, W); we use T=1 per call
    info = {"pixels": imgs.unsqueeze(1)}
    info = jepa.encode(info)
    emb = info["emb"].squeeze(1)   # (N, 192)
    return emb.cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", required=True,
                   help="Base path of the WM checkpoint, e.g. "
                        "~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full")
    p.add_argument("--hdf5", default=os.path.expanduser(
                   "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"))
    p.add_argument("--out_cache", required=True,
                   help="Output .pt path. Must NOT exist (we don't overwrite).")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if os.path.exists(args.out_cache):
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {args.out_cache}. "
            f"Delete it manually if you really want to redo this.")
    Path(args.out_cache).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] WM from {args.wm_ckpt}", flush=True)
    jepa = load_jepa(args.wm_ckpt, device=args.device, img_size=224, patch_size=14)
    jepa.eval()

    print(f"[open] HDF5 {args.hdf5}", flush=True)
    with h5py.File(args.hdf5, "r") as f:
        N = int(f["pixels"].shape[0])
        H, W = int(f["pixels"].shape[1]), int(f["pixels"].shape[2])
        ep_len = f["ep_len"][...].astype(np.int64)
        ep_offset = f["ep_offset"][...].astype(np.int64)

        print(f"  N={N}, H={H}, W={W}, episodes={len(ep_len)}")
        latents_flat = np.empty((N, 192), dtype=np.float32)

        tx = img_transform()
        t0 = time.time()
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            chunk = f["pixels"][start:end]
            latents_flat[start:end] = encode_chunk(jepa, chunk, args.device, tx)
            if (start // args.batch_size) % 100 == 0:
                done = end / N
                elapsed = time.time() - t0
                eta = elapsed * (1.0 - done) / max(done, 1e-6)
                print(f"  {end}/{N} ({100*done:.1f}%) elapsed={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)

    print(f"\n[done] encoded {N} frames in {time.time()-t0:.1f}s "
          f"({N / (time.time()-t0):.1f} fps)", flush=True)

    # Reshape into per-episode list-of-tensors
    print(f"[reshape] splitting flat latents back into per-episode arrays", flush=True)
    all_latents = []
    for off, L in zip(ep_offset, ep_len):
        all_latents.append(torch.from_numpy(latents_flat[off:off + int(L)]).clone())
    print(f"  produced {len(all_latents)} episode latents "
          f"(first 5 lengths: {[t.shape[0] for t in all_latents[:5]]})", flush=True)

    print(f"[save] writing {args.out_cache}", flush=True)
    torch.save({"all_latents": all_latents}, args.out_cache)
    sz = os.path.getsize(args.out_cache) / 1e6
    print(f"  wrote {sz:.1f} MB")


if __name__ == "__main__":
    main()
