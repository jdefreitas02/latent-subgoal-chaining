"""Fine-tune JEPA from a starting checkpoint with original objectives plus a
task-aware auxiliary loss.

The aux loss makes the encoder predict cube-to-goal distance from its 192-D
latent. This shapes JEPA's features so that "near goal" vs "far from goal"
states are linearly distinguishable -- which is the discriminative signal the
frozen JEPA lacks for B2.

Saves an updated JEPA object checkpoint that load_jepa() can read directly.

Usage:
    python finetune_jepa.py --jepa_in PATH \
                            --jepa_out PATH \
                            --hdf5 PATH \
                            --task_id 1 \
                            --steps 2000 \
                            --batch_size 16
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  -- registers Blosc plugin for compressed HDF5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

sys.path.append('/home/jrd21/leworldmodel')

from module import SIGReg


# ============================================================
# Dataset: returns (pixels_seq, action_seq, distance_to_task1_goal_at_t_last)
# ============================================================

CUBE_QPOS_START = 14  # ogbench manipspace cube xyz position in qpos
HISTORY = 3            # JEPA was trained with history_size=3
NUM_PREDS = 1
FRAMESKIP = 5          # JEPA was trained with frameskip=5
N_FRAMES_PER_SAMPLE = HISTORY + NUM_PREDS   # 4

# The sequence we need from the dataset is: 4 frames at frameskip=5 apart
# I.e., for a sample starting at t, we need frames at t, t+5, t+10, t+15
# Each "frame" has 5 actions associated, concatenated to 25-D


def get_task1_goal_xyz():
    """Get task-1 goal xyz from OGBench's singletask env."""
    import gymnasium as gym
    import ogbench
    env = gym.make("cube-single-singletask-task1-v0")
    env.reset()
    goal = env.unwrapped._data.mocap_pos[0].copy()
    env.close()
    return goal  # (3,) xyz


class JEPATrainDataset(Dataset):
    """Sample sequences of (history+pred) frames at frameskip=5 with their
    corresponding actions (concatenated 25-D per frame), plus a per-sample
    auxiliary target:
      - if `q_targets` / `actor_targets` provided -> policy-distillation targets
        for the last frame of the sample
      - otherwise -> the task-1 distance for the LAST frame
    """

    def __init__(self, hdf5_root, task1_goal, img_size=224,
                 q_targets=None, actor_targets=None,
                 frames_per_episode_dataset=1000):
        self.hdf5_root = hdf5_root
        self.task1_goal = torch.from_numpy(task1_goal.astype(np.float32))  # (3,)
        # Optional policy-distillation targets, indexed by dataset transition idx
        # build_for_B2 produces 1000 transitions per 1001-frame episode (drops last)
        self.q_targets = q_targets        # (N_total_transitions,) or None
        self.actor_targets = actor_targets  # (N_total_transitions, 25) or None
        self.frames_per_episode_dataset = frames_per_episode_dataset

        # Open HDF5 to read metadata
        h5_path = hdf5_root if hdf5_root.endswith('.h5') else hdf5_root + '.h5'
        with h5py.File(h5_path, 'r') as f:
            self.actions = f['action'][...].astype(np.float32)  # (N, 5)
            self.qpos = f['qpos'][...].astype(np.float32)        # (N, 21)
            self.ep_len = f['ep_len'][...].astype(np.int64)
            self.ep_offset = f['ep_offset'][...].astype(np.int64)
        self.h5_path = h5_path

        # Pre-compute valid start indices per episode.
        # For a sample starting at t in episode i (ep_offset[i] <= t):
        # we need frames at t, t+5, t+10, t+15 (frameskip=5, N_FRAMES_PER_SAMPLE=4)
        # so we need t + 15 < ep_offset[i] + ep_len[i]
        self.valid_starts = []
        max_offset = (N_FRAMES_PER_SAMPLE - 1) * FRAMESKIP  # 15
        for i in range(len(self.ep_len)):
            ep_start = int(self.ep_offset[i])
            ep_end = int(self.ep_offset[i] + self.ep_len[i])
            # frame indices into this episode: 0 .. ep_len[i]-1
            # so absolute starts: ep_start .. ep_start + ep_len[i] - 1 - max_offset
            valid_end = ep_end - max_offset - FRAMESKIP  # need last frame's actions too
            for t in range(ep_start, valid_end):
                self.valid_starts.append(t)
        self.valid_starts = np.array(self.valid_starts, dtype=np.int64)
        print(f"JEPATrainDataset: {len(self.valid_starts)} valid sample starts "
              f"({len(self.ep_len)} episodes)")

        # Set up image transform
        import stable_pretraining as spt
        self.img_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        ])

        # Lazily open HDF5 per worker (avoid concurrent access issues)
        self._h5 = None

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        f = self._ensure_h5()
        t = int(self.valid_starts[idx])
        # Read N_FRAMES_PER_SAMPLE frames at frameskip=5 apart
        frames_pix = []
        frames_act_chunks = []
        for j in range(N_FRAMES_PER_SAMPLE):
            pos = t + j * FRAMESKIP
            pix = f['pixels'][pos]  # (224, 224, 3) uint8
            # Apply transform: HWC uint8 -> CHW float normalized
            pix_t = self.img_transform(torch.from_numpy(pix.transpose(2, 0, 1)))
            frames_pix.append(pix_t)
            # The 5 actions starting at pos
            act_chunk = self.actions[pos:pos + FRAMESKIP]  # (5, 5)
            frames_act_chunks.append(torch.from_numpy(act_chunk.reshape(-1)))  # (25,)

        pixels = torch.stack(frames_pix, dim=0)        # (4, 3, 224, 224)
        actions = torch.stack(frames_act_chunks, dim=0)  # (4, 25)

        # Distance to task-1 goal at the LAST frame
        last_pos = t + (N_FRAMES_PER_SAMPLE - 1) * FRAMESKIP
        cube_xyz = torch.from_numpy(self.qpos[last_pos, CUBE_QPOS_START:CUBE_QPOS_START + 3])
        distance = torch.norm(cube_xyz - self.task1_goal, p=2)

        sample = {
            'pixels': pixels,           # (T=4, 3, 224, 224)
            'action': actions,          # (T=4, 25)
            'distance': distance,       # scalar
        }

        # Look up policy distillation targets if provided.
        # last_pos is the HDF5 frame index of the last frame. Map to dataset
        # transition idx: each episode contributes 1001 HDF5 frames but only
        # `frames_per_episode_dataset`=1000 dataset transitions (drop last frame).
        if self.q_targets is not None or self.actor_targets is not None:
            ep_i = last_pos // 1001
            frame_in_ep = last_pos % 1001
            if frame_in_ep < self.frames_per_episode_dataset:
                dataset_idx = ep_i * self.frames_per_episode_dataset + frame_in_ep
                sample['target_valid'] = torch.tensor(1.0, dtype=torch.float32)
                if self.q_targets is not None:
                    sample['q_target'] = self.q_targets[dataset_idx].clone()
                if self.actor_targets is not None:
                    sample['actor_target'] = self.actor_targets[dataset_idx].clone()
            else:
                sample['target_valid'] = torch.tensor(0.0, dtype=torch.float32)
                if self.q_targets is not None:
                    sample['q_target'] = torch.tensor(0.0, dtype=torch.float32)
                if self.actor_targets is not None:
                    sample['actor_target'] = torch.zeros(25, dtype=torch.float32)

        return sample


# ============================================================
# Fine-tune loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jepa_in', required=True, help='Path to input lejepa_object.ckpt')
    parser.add_argument('--jepa_out', required=True, help='Path to output lejepa_object.ckpt')
    parser.add_argument('--hdf5', required=True, help='Path to 224x224 HDF5 dataset (without .h5)')
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda_sigreg', type=float, default=0.09)
    parser.add_argument('--lambda_aux', type=float, default=0.0,
                       help='Weight on the distance-prediction aux loss (0 = disabled)')
    parser.add_argument('--distill_targets', default=None,
                       help='Optional .pt with q_targets/actor_targets from a JAX policy. '
                            'When set, enables policy distillation as the task signal.')
    parser.add_argument('--lambda_q_distill', type=float, default=0.1,
                       help='Weight on Q distillation loss (only if --distill_targets is set)')
    parser.add_argument('--lambda_a_distill', type=float, default=1.0,
                       help='Weight on action distillation loss (only if --distill_targets is set)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading JEPA from {args.jepa_in}")
    jepa = torch.load(args.jepa_in, weights_only=False, map_location='cpu')
    jepa = jepa.to(args.device).train()
    # Unfreeze everything
    for p in jepa.parameters():
        p.requires_grad_(True)
    print(f"JEPA total params: {sum(p.numel() for p in jepa.parameters()):,}")

    # Add distance head (192-D -> 64 -> 1) -- used if --lambda_aux > 0
    distance_head = nn.Sequential(
        nn.Linear(192, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    ).to(args.device).train()

    # Policy distillation heads -- used if --distill_targets is set
    q_head = nn.Sequential(
        nn.Linear(192, 256),
        nn.GELU(),
        nn.Linear(256, 1),
    ).to(args.device).train()
    action_head = nn.Sequential(
        nn.Linear(192, 256),
        nn.GELU(),
        nn.Linear(256, 25),
    ).to(args.device).train()

    sigreg = SIGReg(knots=17, num_proj=1024).to(args.device)

    if args.task_id != 1:
        raise NotImplementedError("Only task 1 supported for now")
    task1_goal = get_task1_goal_xyz()
    print(f"Task-1 goal xyz: {task1_goal}")

    # Load policy distillation targets if provided
    q_targets, actor_targets = None, None
    using_distill = args.distill_targets is not None
    if using_distill:
        print(f"Loading distillation targets from {args.distill_targets}")
        d = torch.load(args.distill_targets, weights_only=False, map_location='cpu')
        q_targets = d['q_targets'].float()
        actor_targets = d['actor_targets'].float()
        # Z-normalize Q targets so MSE has unit-scale magnitude (raw Q values
        # can have range ~[-32, 0] which makes MSE values dominate other losses)
        q_mean, q_std = q_targets.mean().item(), q_targets.std().item() + 1e-8
        q_targets = (q_targets - q_mean) / q_std
        print(f"  q_targets: shape={q_targets.shape}, "
              f"raw_mean={q_mean:.3f}, raw_std={q_std:.3f} "
              f"(z-normalized to mean=0, std=1)")
        print(f"  actor_targets: shape={actor_targets.shape}, "
              f"per-dim std mean={actor_targets.std(0).mean():.4f}")
    print(f"Loss weights: pred=1.0  sigreg={args.lambda_sigreg}  "
          f"distance_aux={args.lambda_aux}  "
          f"q_distill={args.lambda_q_distill if using_distill else 0.0}  "
          f"action_distill={args.lambda_a_distill if using_distill else 0.0}")

    dataset = JEPATrainDataset(args.hdf5, task1_goal,
                               q_targets=q_targets, actor_targets=actor_targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset)} samples, "
          f"batch_size={args.batch_size}, "
          f"batches per epoch={len(loader)}")

    # Optimizer for JEPA + heads together
    head_params = list(distance_head.parameters())
    if using_distill:
        head_params = head_params + list(q_head.parameters()) + list(action_head.parameters())
    opt = torch.optim.AdamW(
        list(jepa.parameters()) + head_params,
        lr=args.lr, weight_decay=1e-3,
    )

    step = 0
    pbar = tqdm(total=args.steps)
    loader_iter = iter(loader)
    losses_pred, losses_sigreg, losses_aux = [], [], []
    losses_q, losses_a = [], []

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)

        # Forward through JEPA (encoder, action_encoder)
        # encode() expects "pixels" key; sets info["emb"] and info["act_emb"]
        output = jepa.encode(batch)
        emb = output['emb']        # (B, T=4, D=192)
        act_emb = output['act_emb']  # (B, T=4, D=192)

        # Predict: input first 3 frames -> predict frames 1..3 (autoregressive within T)
        # Following train.py:
        #   ctx_emb = emb[:, :ctx_len]  (B, 3, D)
        #   pred_emb = predict(ctx_emb, ctx_act)
        #   tgt_emb = emb[:, n_preds:]   (B, T - n_preds, D)  = emb[:, 1:]
        ctx_len = HISTORY  # 3
        ctx_emb = emb[:, :ctx_len]      # (B, 3, D)
        ctx_act = act_emb[:, :ctx_len]  # (B, 3, D)
        tgt_emb = emb[:, NUM_PREDS:]    # (B, T - 1=3, D)
        pred_emb = jepa.predict(ctx_emb, ctx_act)  # (B, 3, D)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = sigreg(emb.transpose(0, 1))

        z_last = emb[:, -1]  # (B, 192)

        # Distance aux loss (only if lambda_aux > 0)
        if args.lambda_aux > 0:
            dist_pred = distance_head(z_last).squeeze(-1)  # (B,)
            dist_target = batch['distance']
            aux_loss = F.mse_loss(dist_pred, dist_target)
        else:
            aux_loss = torch.zeros((), device=args.device)

        # Policy distillation losses (only if --distill_targets is set)
        if using_distill:
            valid = batch['target_valid']  # (B,)
            # Q distillation
            q_pred = q_head(z_last).squeeze(-1)  # (B,)
            q_target = batch['q_target']
            q_loss = ((q_pred - q_target) ** 2 * valid).sum() / (valid.sum().clamp_min(1.0))
            # Action distillation
            a_pred = action_head(z_last)  # (B, 25)
            a_target = batch['actor_target']
            a_loss = ((a_pred - a_target) ** 2).sum(dim=-1) * valid
            a_loss = a_loss.sum() / (valid.sum().clamp_min(1.0))
        else:
            q_loss = torch.zeros((), device=args.device)
            a_loss = torch.zeros((), device=args.device)

        loss = (pred_loss
                + args.lambda_sigreg * sigreg_loss
                + args.lambda_aux * aux_loss
                + (args.lambda_q_distill * q_loss if using_distill else 0.0)
                + (args.lambda_a_distill * a_loss if using_distill else 0.0))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
        opt.step()

        losses_pred.append(pred_loss.item())
        losses_sigreg.append(sigreg_loss.item())
        losses_aux.append(aux_loss.item())
        losses_q.append(q_loss.item())
        losses_a.append(a_loss.item())

        step += 1
        if step % 50 == 0:
            desc = (f"pred={np.mean(losses_pred[-50:]):.4f} "
                    f"sigreg={np.mean(losses_sigreg[-50:]):.4f}")
            if args.lambda_aux > 0:
                desc += f" dist={np.mean(losses_aux[-50:]):.4f}"
            if using_distill:
                desc += (f" q={np.mean(losses_q[-50:]):.4f} "
                         f"a={np.mean(losses_a[-50:]):.4f}")
            pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    print(f"\nFinal losses (last 50):")
    print(f"  pred={np.mean(losses_pred[-50:]):.4f}")
    print(f"  sigreg={np.mean(losses_sigreg[-50:]):.4f}")
    if args.lambda_aux > 0:
        print(f"  distance_aux={np.mean(losses_aux[-50:]):.4f}")
    if using_distill:
        print(f"  q_distill={np.mean(losses_q[-50:]):.4f}")
        print(f"  action_distill={np.mean(losses_a[-50:]):.4f}")

    # Save updated JEPA (object pickle)
    out_dir = os.path.dirname(args.jepa_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    jepa = jepa.eval().cpu()
    torch.save(jepa, args.jepa_out)
    print(f"Saved updated JEPA to {args.jepa_out}")

    # Save heads separately for later inspection
    heads_path = args.jepa_out.replace('object.ckpt', 'heads.pt')
    torch.save({
        'distance_head': distance_head.eval().cpu().state_dict(),
        'q_head': q_head.eval().cpu().state_dict() if using_distill else None,
        'action_head': action_head.eval().cpu().state_dict() if using_distill else None,
    }, heads_path)
    print(f"Saved heads to {heads_path}")

if __name__ == '__main__':
    main()
