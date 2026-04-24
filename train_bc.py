"""Train a deterministic goal-conditioned behavioural cloning policy.

The BC policy is trained on the offline expert dataset.  For each expert
WM-step transition (z_t -> z_{t+5}), we sample several random future
latents from the same episode as goals (HER-style), giving the model a
diverse set of (state, goal) -> action targets from ground-truth data.

Expert 5-D per-frame actions are normalised with StandardScaler and then
passed through tanh * action_scale to land in (-3, 3), the same range as
GoalConditionedActor.  The mean/std are saved with the checkpoint so they
are available for inspection, but train.py does not need them (the BC
model handles the transform internally).

Usage:
    python train_bc.py [--save_dir ./checkpoints_bc] [--epochs 50]
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import stable_worldmodel as swm

from bc_policy import BCPolicy

LATENT_DIM  = 192
ACTION_DIM  = 25    # 5 joints × 5 video frames per WM step
FRAME_SKIP  = 5     # 1 WM step = 5 video frames
ACTION_SCALE = 3.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """In-memory dataset of (z_start, z_goal, action_25d) triples.

    For each expert episode and each valid WM-step start frame t, we sample
    `goals_per_step` random future latents as goals.  The 25-D action is
    StandardScaler-normalised then tanh-squashed to (-3, 3).
    """

    def __init__(self, all_latents, all_ep_actions, action_mean, action_std,
                 action_scale=ACTION_SCALE, goals_per_step=4):

        states_list = []
        goals_list  = []
        acts_list   = []

        for ep_idx, (latents, ep_actions) in enumerate(zip(all_latents, all_ep_actions)):
            ep_len = latents.shape[0]
            # Need at least FRAME_SKIP frames for the action chunk and one future frame for the goal
            if ep_len < FRAME_SKIP + 1:
                continue

            # ep_actions is [ep_len, 5] numpy; we need t+FRAME_SKIP <= ep_len-1
            max_start = ep_len - FRAME_SKIP - 1   # inclusive upper bound
            for t in range(0, max_start + 1):
                chunk = ep_actions[t : t + FRAME_SKIP]   # [5, 5]

                if np.any(np.isnan(chunk)):
                    continue

                # Normalise: per-dim StandardScaler on the 5-D per-frame space, tiled across frames
                norm_chunk = (chunk - action_mean) / (action_std + 1e-8)   # [5, 5]
                # Squash to (-action_scale, action_scale) matching the RL actor's output range
                action_25d = np.tanh(norm_chunk.flatten()).astype(np.float32) * action_scale  # [25]

                z_t = latents[t].numpy() if isinstance(latents[t], torch.Tensor) else latents[t]

                # Earliest valid goal: first frame after the action chunk
                goal_start = t + FRAME_SKIP
                for _ in range(goals_per_step):
                    goal_t  = np.random.randint(goal_start, ep_len)
                    z_goal  = latents[goal_t]
                    z_goal  = z_goal.numpy() if isinstance(z_goal, torch.Tensor) else z_goal

                    states_list.append(z_t)
                    goals_list.append(z_goal)
                    acts_list.append(action_25d)

        self.states  = torch.tensor(np.stack(states_list), dtype=torch.float32)
        self.goals   = torch.tensor(np.stack(goals_list),  dtype=torch.float32)
        self.actions = torch.tensor(np.stack(acts_list),   dtype=torch.float32)

        print(f"BCDataset: {len(self.states):,} samples from {len(all_latents)} episodes")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.goals[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_actions_by_episode(dataset):
    """Return a list of per-episode numpy arrays with shape [ep_len, 5]."""
    raw = dataset.get_col_data("action")   # [total_frames, 5]
    ep_actions = []
    offset = 0
    for length in dataset.lengths:
        ep_actions.append(raw[offset : offset + int(length)])
        offset += int(length)
    return ep_actions


def compute_action_stats(all_ep_actions):
    """Per-dim mean and std across all valid (non-NaN) 5-D per-frame actions."""
    all_frames = np.concatenate(all_ep_actions, axis=0)   # [total_frames, 5]
    valid      = all_frames[~np.isnan(all_frames).any(axis=1)]
    return valid.mean(axis=0).astype(np.float32), valid.std(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))

    dataset_path = args.dataset_path or os.path.join(stablewm_home, "ogbench", "cube_single_expert")
    cache_path   = args.cache_path   or os.path.join(stablewm_home, "cube_all_latents_cache.pt")

    print(f"Loading dataset from:     {dataset_path}")
    print(f"Loading latents cache from {cache_path}...")

    dataset = swm.data.HDF5Dataset(dataset_path)

    # --- Latent cache ---
    cache = torch.load(cache_path, map_location="cpu")
    all_latents = cache['all_latents']   # list of [ep_len, 192] CPU tensors

    # --- Expert actions ---
    print("Loading expert actions from dataset...")
    all_ep_actions = split_actions_by_episode(dataset)

    action_mean, action_std = compute_action_stats(all_ep_actions)
    print(f"Action mean (5D): {np.round(action_mean, 4)}")
    print(f"Action std  (5D): {np.round(action_std, 4)}")

    # --- Build training dataset ---
    bc_dataset = BCDataset(
        all_latents, all_ep_actions, action_mean, action_std,
        action_scale=ACTION_SCALE, goals_per_step=args.goals_per_step,
    )

    loader = DataLoader(
        bc_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = BCPolicy(latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                     action_scale=ACTION_SCALE).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for z_curr, z_goal, expert_action in loader:
            z_curr        = z_curr.to(device)
            z_goal        = z_goal.to(device)
            expert_action = expert_action.to(device)

            pred_action = model(z_curr, z_goal)
            loss = F.mse_loss(pred_action, expert_action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(z_curr)

        scheduler.step()
        avg_loss = total_loss / len(bc_dataset)

        print(f"Epoch {epoch:03d}/{args.epochs} | Loss: {avg_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time()-t0:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'action_mean': torch.from_numpy(action_mean),
                    'action_std':  torch.from_numpy(action_std),
                    'action_scale': ACTION_SCALE,
                    'latent_dim':   LATENT_DIM,
                    'action_dim':   ACTION_DIM,
                },
                os.path.join(args.save_dir, "bc_policy.pth"),
            )
            print(f"  -> Checkpoint saved (loss={best_loss:.6f})")

    print(f"\nBC training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioural Cloning Policy Training")
    parser.add_argument('--save_dir',       type=str,   default='./checkpoints_bc')
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=1024)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--goals_per_step', type=int,   default=4,
                        help="Random HER goals sampled per WM-step transition")
    parser.add_argument('--cache_path',   type=str, default=None,
                        help="Path to latents cache .pt file. "
                             "Default: $HOME/stable_wm_data/cube_all_latents_cache.pt")
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to HDF5 dataset (without .h5 extension). "
                             "Default: $HOME/stable_wm_data/ogbench/cube_single_expert")
    args = parser.parse_args()
    train_bc(args)
