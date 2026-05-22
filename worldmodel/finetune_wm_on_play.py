"""Fine-tune the JEPA world model on the OGBench play dataset.

Motivation: the existing WM (`~/stable_wm_data/cube/lejepa`) was trained on
cube_single_expert (narrow, goal-completing trajectories). The QC offline
dataset, the anchor pool, and the policy's online state distribution all come
from cube_single_play (random goal pursuit -- much broader state and action
coverage). Diagnostic B showed the WM is OOD-ish under these conditions: it
can steer roughly but cannot land precisely on arbitrary in-distribution
latents.

This script:
  1. Loads the pretrained WM (same architecture as eval / WMEnv).
  2. Builds a torch Dataset over visual-cube-single-play-v0_224.h5 that yields
     (4-frame pixel sequences, 3 stride-5 action chunks) -- matching the
     original LeWM training recipe.
  3. Fine-tunes the WM with the same pred_loss as train.py:lejepa_forward
     (MSE between predicted and real next-frame latents).
  4. Saves a new ckpt directory in the AutoCostModel format so it can be
     loaded by load_jepa(...) drop-in for WMEnv / our diagnostics.

We DROP the SIGReg term -- it requires the full spt.Module setup. Fine-tune
should be short (a few epochs) and use a small LR (1e-5), so encoder collapse
risk is minimal.
"""

import argparse
import os
import sys
import time
from pathlib import Path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Make jepa.py / module.py importable from the leworldmodel root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "latent_hindsight_rl", "offline-online"))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms

from envs.jepa_loader import load_jepa


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CostShim(torch.nn.Module):
    """AutoCostModel-compatible wrapper for a fine-tuned JEPA.

    Module-level (not nested inside main) so it pickles cleanly. After fine-tune
    we wrap the JEPA in this shim so swm.policy.AutoCostModel(...) can scan
    the loaded object, find a `get_cost` method, and return us a module that
    exposes encode / predict / action_encoder for WMEnv to use.
    """

    def __init__(self, jepa):
        super().__init__()
        self.jepa = jepa
        # Mirror submodules as attributes so any caller code that does
        # `model.encoder`, `model.predictor`, etc. still works.
        self.encoder = jepa.encoder
        self.predictor = jepa.predictor
        self.action_encoder = jepa.action_encoder
        self.projector = jepa.projector
        self.pred_proj = jepa.pred_proj

    def encode(self, info):
        return self.jepa.encode(info)

    def predict(self, emb, act_emb):
        return self.jepa.predict(emb, act_emb)

    def rollout(self, *a, **k):
        return self.jepa.rollout(*a, **k)

    def get_cost(self, info, goal_emb=None):
        """Latent L2 distance from current emb to goal_emb."""
        return torch.linalg.vector_norm(info["emb"] - goal_emb, dim=-1)


def img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PlayDataset(Dataset):
    """Yields (pixels (T, 3, 224, 224), action_chunks (T-1, 25)).

    pixels are sampled every `frameskip` env steps starting from a random valid
    offset in a random episode. The action chunks are stride-`frameskip`
    flattened (frameskip * action_dim) per WM-step.
    """

    def __init__(self, hdf5_path, num_steps=4, frameskip=5, max_samples=None, seed=0):
        self.hdf5_path = str(hdf5_path)
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.seq_env_len = (num_steps - 1) * frameskip + 1  # inclusive

        with h5py.File(self.hdf5_path, "r") as f:
            self.ep_len = f["ep_len"][...].astype(np.int64)
            self.ep_offset = f["ep_offset"][...].astype(np.int64)
            self.action_dim = int(f["action"].shape[1])
            self.img_hw = (int(f["pixels"].shape[1]), int(f["pixels"].shape[2]))

        # Pre-compute (episode_idx, start_step) pairs for every valid sample
        valid = []
        last_valid_start = self.ep_len - self.seq_env_len   # inclusive max start
        for ep_i, (L, off, vmax) in enumerate(zip(self.ep_len, self.ep_offset, last_valid_start)):
            if vmax < 0:
                continue
            for s in range(int(vmax) + 1):
                valid.append((ep_i, s))
        rng = np.random.default_rng(seed)
        rng.shuffle(valid)
        if max_samples is not None:
            valid = valid[:int(max_samples)]
        self.samples = np.array(valid, dtype=np.int64)
        self._tx = img_transform()
        self._hdf5_handle = None

    def _f(self):
        # h5py file objects must be opened per-worker to avoid races
        if self._hdf5_handle is None:
            self._hdf5_handle = h5py.File(self.hdf5_path, "r")
        return self._hdf5_handle

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_i, start = self.samples[idx]
        off = int(self.ep_offset[ep_i])
        # Frame indices in the flat array
        frame_idx = [off + start + k * self.frameskip for k in range(self.num_steps)]
        # Action chunks: (num_steps - 1) chunks of `frameskip` raw actions, flattened
        chunks = []
        for k in range(self.num_steps - 1):
            a_lo = off + start + k * self.frameskip
            a_hi = a_lo + self.frameskip
            chunks.append(self._f()["action"][a_lo:a_hi].astype(np.float32).reshape(-1))
        chunks = np.stack(chunks, axis=0)   # (num_steps-1, frameskip*action_dim)
        # Pixels: read each frame, transform
        pix = []
        for fi in frame_idx:
            p = self._f()["pixels"][fi]      # (H, W, 3) uint8
            t = self._tx(torch.from_numpy(p).permute(2, 0, 1).contiguous())  # (3, H, W) float
            pix.append(t)
        pix = torch.stack(pix, dim=0)        # (num_steps, 3, H, W)
        # NaN-safe (matches train.py)
        chunks = np.nan_to_num(chunks, nan=0.0)
        return pix, torch.from_numpy(chunks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--hdf5", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"))
    p.add_argument("--out_dir", default=os.path.expanduser("~/stable_wm_data/cube/lejepa_play_ft"))
    p.add_argument("--out_name", default="lejepa_play_ft")
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=200000,
                   help="Subsample the play dataset to keep an epoch reasonable.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--predictor_only", action="store_true",
                   help="Freeze encoder + action_encoder + projectors, train predictor only.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] WM from {args.wm_ckpt}", flush=True)
    jepa = load_jepa(args.wm_ckpt, device=args.device, img_size=224, patch_size=14)
    # load_jepa returns the JEPA module with requires_grad=False; flip back on.
    for p_ in jepa.parameters():
        p_.requires_grad_(True)
    if args.predictor_only:
        for p_ in jepa.encoder.parameters():       p_.requires_grad_(False)
        for p_ in jepa.action_encoder.parameters(): p_.requires_grad_(False)
        for p_ in jepa.projector.parameters():     p_.requires_grad_(False)
        for p_ in jepa.pred_proj.parameters():     p_.requires_grad_(False)
        # eval mode for frozen modules (no BN updates)
        jepa.encoder.eval()
        jepa.action_encoder.eval()
        jepa.projector.eval()
        jepa.pred_proj.eval()
    else:
        jepa.train()
    n_train_params = sum(p.numel() for p in jepa.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train_params/1e6:.2f}M", flush=True)

    print(f"[load] play dataset from {args.hdf5}", flush=True)
    ds = PlayDataset(args.hdf5, num_steps=args.num_steps, frameskip=args.frameskip,
                     max_samples=args.max_samples, seed=args.seed)
    print(f"  total samples: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optim = torch.optim.AdamW(
        [p_ for p_ in jepa.parameters() if p_.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(enabled=False)  # we use bf16 manually below
    ctx_len = args.history_size
    n_preds = args.num_preds

    print(f"\n=== fine-tune: epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  pred_only={args.predictor_only}\n", flush=True)

    sd_path = out_dir / f"{args.out_name}_state_dict.pt"

    total_steps = 0
    t0 = time.time()
    losses = []
    for ep in range(args.epochs):
        ep_losses = []
        ep_t0 = time.time()
        for it, (pixels, actions) in enumerate(loader):
            pixels = pixels.to(args.device, non_blocking=True)    # (B, T, 3, H, W)
            actions = actions.to(args.device, non_blocking=True)  # (B, T-1, 25)
            batch = {"pixels": pixels, "action": actions}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = jepa.encode(batch)
                emb = out["emb"]         # (B, T, D)
                act_emb = out["act_emb"] # (B, T-1, D_a)
                ctx_emb = emb[:, :ctx_len]            # (B, 3, D)
                ctx_act = act_emb[:, :ctx_len]        # (B, 3, D)
                tgt_emb = emb[:, n_preds:]            # (B, T-n_preds, D)  = (B, 3, D)
                pred_emb = jepa.predict(ctx_emb, ctx_act)
                pred_loss = F.mse_loss(pred_emb, tgt_emb)
            optim.zero_grad(set_to_none=True)
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in jepa.parameters() if p_.requires_grad], 1.0,
            )
            optim.step()
            ep_losses.append(float(pred_loss.detach().item()))
            total_steps += 1
            if it % 50 == 0:
                print(f"  ep {ep+1}/{args.epochs}  it {it}/{len(loader)}  "
                      f"loss={ep_losses[-1]:.6f}  "
                      f"running_mean(last 50)={np.mean(ep_losses[-50:]):.6f}",
                      flush=True)
        ep_mean = float(np.mean(ep_losses))
        losses.extend(ep_losses)
        # Save state_dict at end of every epoch so we always have a recoverable
        # checkpoint even if the final wrapper-pickle step fails.
        torch.save({k: v.cpu() for k, v in jepa.state_dict().items()}, sd_path)
        print(f"\n  epoch {ep+1} done: mean_loss={ep_mean:.6f}  "
              f"min={np.min(ep_losses):.6f}  max={np.max(ep_losses):.6f}  "
              f"[{time.time()-ep_t0:.1f}s]  saved state_dict to {sd_path}\n",
              flush=True)

    print(f"\n[done] total {total_steps} iters in {time.time()-t0:.1f}s")

    # Save in AutoCostModel-compatible format (pickle of a module with get_cost)
    obj_path = out_dir / f"{args.out_name}_object.ckpt"
    shim = CostShim(jepa).cpu().eval()
    try:
        torch.save(shim, obj_path)
        print(f"[save] wrote {obj_path}")
    except Exception as e:
        # Don't fail the run — state_dict was already saved per-epoch above.
        print(f"[save] WARNING: shim pickle failed ({e!r}); "
              f"state_dict at {sd_path} is the recovery path", flush=True)

    # Summary file
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"finetuned WM from {args.wm_ckpt}\n")
        f.write(f"on hdf5={args.hdf5}\n")
        f.write(f"samples={len(ds)} epochs={args.epochs} lr={args.lr} "
                f"batch={args.batch_size} predictor_only={args.predictor_only}\n")
        f.write(f"final_mean_loss(last_epoch)={np.mean(losses[-len(loader):]):.6f}\n")
        f.write(f"first_epoch_mean_loss={np.mean(losses[:len(loader)]):.6f}\n")


if __name__ == "__main__":
    main()
