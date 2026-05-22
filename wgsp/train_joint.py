import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import numpy as np
import random
import os
import time
import csv
import argparse
from collections import deque
import stable_worldmodel as swm
from hydra import initialize, compose

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path: sys.path.insert(0, _THIS_DIR)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

# Import your vectorized environment
from latent_env import LatentEnv
from bc_policy import BCPolicy


# =============================================================================
# PCA projection helpers
# =============================================================================

def project_latent(z, pca_mean, pca_matrix):
    """Project latents from 192D to pca_dim.

    Args:
        z: [..., 192] tensor
        pca_mean: [192] tensor (or None)
        pca_matrix: [192, D] tensor (or None)
    Returns:
        [..., D] tensor if PCA provided, else z unchanged.
    """
    if pca_matrix is None:
        return z
    return (z - pca_mean.to(z.device)) @ pca_matrix.to(z.device)  # [..., D]


def deproject_latent(z_proj, pca_mean, pca_matrix):
    """Approximate inverse PCA projection from pca_dim back to 192D.

    Used to convert pca_dim subgoals back to 192D for the WM env's done computation.
    Not a perfect reconstruction — PCA truncation loses information.
    """
    if pca_matrix is None:
        return z_proj
    return z_proj @ pca_matrix.to(z_proj.device).T + pca_mean.to(z_proj.device)  # [..., 192]


# =============================================================================
# Cache transition mixer (pure-offline RL data source)
# =============================================================================

class CacheTransitionMixer:
    """GPU-native sampler of (z_t, a_expert, z_next, goal, r, done) from the
    encoder-space latent cache with HER goal relabeling.

    All latents are encoder outputs (no predictor drift). Actions are expert
    actions from the dataset, scaled to match the WM action_encoder's expected
    input format (5 consecutive StandardScaler-normalized actions, stacked to 25D).

    Operates entirely in projected pca_dim space (or 192D if pca_matrix is None).

    Args:
        all_latents:    List[Tensor[T_ep, 192]] — raw encoder latents from cache
        all_actions:    List[Tensor[T_ep, A_raw]] — raw per-frame actions from cache
        action_scaler:  sklearn StandardScaler fit on raw actions
        done_threshold: L2 threshold for sparse success reward
        future_p:       HER future goal sampling probability (default 0.8)
        reward_mode:    'sparse' or 'dense'
        device:         Target device for sampling
        frameskip:      Physical frames per WM step (default 5)
        pca_mean:       [192] PCA mean (None = no projection)
        pca_matrix:     [192, D] PCA matrix (None = no projection)
    """

    def __init__(self, all_latents, all_actions, action_scaler, done_threshold,
                 future_p=0.8, reward_mode='sparse', device='cuda',
                 frameskip=5, pca_mean=None, pca_matrix=None,
                 dense_reward_scale=6.29):

        self.done_threshold    = done_threshold
        self.future_p          = future_p
        self.reward_mode       = reward_mode
        self.device            = device
        self.dense_reward_scale = dense_reward_scale

        raw_action_dim = all_actions[0].shape[-1] if all_actions else 5
        wm_action_dim  = frameskip * raw_action_dim  # 5 × 5 = 25

        print(f"  Building CacheTransitionMixer: frameskip={frameskip}, "
              f"raw_action_dim={raw_action_dim}, wm_action_dim={wm_action_dim}")

        z_t_list, a_list, z_next_list = [], [], []
        ep_start_list, ep_end_list, ep_goal_list = [], [], []
        ep_ids_list, t_within_ep_list = [], []

        offset = 0
        skipped = 0
        for ep_z_raw, ep_a_raw in zip(all_latents, all_actions):
            # Project latents to pca_dim if requested
            if pca_mean is not None:
                ep_z = project_latent(ep_z_raw.float(), pca_mean, pca_matrix)
            else:
                ep_z = ep_z_raw.float()

            T = ep_z.shape[0]
            # Number of valid WM-step transitions (need t and t+frameskip to exist)
            n_wm = (T - 1) // frameskip
            if n_wm <= 0:
                skipped += 1
                continue

            # Latents at WM-step boundaries: 0, frameskip, 2*frameskip, ...
            step_indices = torch.arange(0, (n_wm + 1) * frameskip, frameskip)[:n_wm + 1]
            ep_z_wm = ep_z[step_indices]  # [n_wm+1, D]

            # Scale and stack raw actions per WM step
            n_raw = n_wm * frameskip
            ep_a_raw_np = ep_a_raw[:n_raw].numpy().astype(np.float32)      # [n_wm*5, A_raw]
            scaled      = action_scaler.transform(ep_a_raw_np)             # [n_wm*5, A_raw]
            ep_a_wm     = torch.tensor(
                scaled.reshape(n_wm, wm_action_dim), dtype=torch.float32)  # [n_wm, 25]

            # Store transitions: z_t[i] → a[i] → z_next[i] = z_t[i+1]
            z_t_list.append(ep_z_wm[:-1])   # [n_wm, D]
            a_list.append(ep_a_wm)           # [n_wm, 25]
            z_next_list.append(ep_z_wm[1:]) # [n_wm, D]

            ep_start_list.append(offset)
            ep_end_list.append(offset + n_wm)
            ep_goal_list.append(ep_z_wm[-1])  # episode ultimate goal = last WM-step latent

            # Track which episode and timestep each flat index belongs to
            ep_id = len(ep_start_list) - 1
            ep_ids_list.extend([ep_id] * n_wm)
            t_within_ep_list.extend(range(n_wm))

            offset += n_wm

        if skipped > 0:
            print(f"  Skipped {skipped} episodes too short for WM-step transitions")

        self.total = offset
        self.n_eps = len(ep_start_list)
        print(f"  Cache mixer: {self.n_eps} episodes, {self.total:,} WM-step transitions")

        # Flat GPU tensors for O(1) sampling
        self.z_t_flat    = torch.cat(z_t_list,    dim=0).to(device)   # [N, D]
        self.a_flat      = torch.cat(a_list,      dim=0).to(device)   # [N, 25]
        self.z_next_flat = torch.cat(z_next_list, dim=0).to(device)   # [N, D]
        self.ep_starts   = torch.tensor(ep_start_list, device=device) # [n_eps]
        self.ep_ends     = torch.tensor(ep_end_list,   device=device) # [n_eps]
        self.ep_goals    = torch.stack(ep_goal_list).to(device)        # [n_eps, D]
        self.ep_ids      = torch.tensor(ep_ids_list,      device=device, dtype=torch.long)  # [N]
        self.t_within    = torch.tensor(t_within_ep_list, device=device, dtype=torch.long)  # [N]

    def sample(self, batch_size):
        """Vectorized GPU-native sample. Returns (z_t, a_expert, z_next, goal, r, done)."""
        # 1. Sample random flat transition indices
        idx = torch.randint(0, self.total, (batch_size,), device=self.device)

        z_t    = self.z_t_flat[idx]
        a      = self.a_flat[idx]
        z_next = self.z_next_flat[idx]

        # 2. HER goal relabeling
        ep_id    = self.ep_ids[idx]          # [B]
        t_ep     = self.t_within[idx]        # [B]
        ep_start = self.ep_starts[ep_id]     # [B]
        ep_end   = self.ep_ends[ep_id]       # [B]

        # Number of strictly future timesteps available
        n_future = ep_end - ep_start - 1 - t_ep  # [B], ≥0

        her_mask = (torch.rand(batch_size, device=self.device) < self.future_p) & (n_future > 0)

        # Sample a random future offset ∈ [1, n_future]
        safe_n = n_future.clamp(min=1).float()
        future_offset = (torch.rand(batch_size, device=self.device) * safe_n).long() + 1  # [B]

        future_flat_idx = (ep_start + t_ep + future_offset).clamp(
            max=(ep_end - 1).clamp(min=0)
        )  # [B]

        her_goal = self.z_t_flat[future_flat_idx]   # [B, D] — future state as goal
        ep_goal  = self.ep_goals[ep_id]              # [B, D] — episode final state

        g = torch.where(her_mask.unsqueeze(-1), her_goal, ep_goal)  # [B, D]

        # 3. Reward
        dist_next = torch.norm(z_next - g, p=2, dim=-1)  # [B]
        success   = dist_next < self.done_threshold
        done      = success.float()

        if self.reward_mode == 'sparse':
            r = torch.where(success,
                            torch.zeros_like(dist_next),
                            -torch.ones_like(dist_next))
        else:
            dist_curr   = torch.norm(z_t - g, p=2, dim=-1)
            improvement = (dist_curr - dist_next) / self.dense_reward_scale
            r           = improvement.clamp(-2.0, 2.0)

        return z_t, a, z_next, g, r, done

# Standard SAC Hyperparameters for the Actor
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    """Standard weight initialization from the SAC paper."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GoalConditionedActor(nn.Module):
    """The Policy Network: pi(action | z_curr, z_goal)"""
    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256, action_scale=3.0):
        super(GoalConditionedActor, self).__init__()

        self.action_scale = action_scale

        input_dim = latent_dim * 2
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

        # Force the Actor to start with very small actions to prevent blowing up the World Model
        torch.nn.init.uniform_(self.mean_linear.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.mean_linear.bias, 0)
        torch.nn.init.uniform_(self.log_std_linear.weight, -1e-3, 1e-3)
        # Start with a smaller std (e^-1.0 ~ 0.36) so it explores safely within the expert distribution
        torch.nn.init.constant_(self.log_std_linear.bias, -1.0)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Multiply by the scale so the agent can reach the expert's top speed
        action = y_t * self.action_scale

        log_prob = normal.log_prob(x_t)
        # Correct the log probability density for the wider scale
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean) * self.action_scale

class TwinCritic(nn.Module):
    """The Value Network: Q(z_curr, z_goal, action)"""
    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256):
        super(TwinCritic, self).__init__()

        input_dim = (latent_dim * 2) + action_dim

        self.q1_l1 = nn.Linear(input_dim, hidden_dim)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(input_dim, hidden_dim)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

        # Xavier init on the output layer would produce wildly overestimated initial Q-values,
        # poisoning the critic target. Use small-weight init like the actor output layers.
        torch.nn.init.uniform_(self.q1_l3.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.q1_l3.bias, 0)
        torch.nn.init.uniform_(self.q2_l3.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.q2_l3.bias, 0)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)

        q1 = F.relu(self.q1_ln1(self.q1_l1(x)))
        q1 = F.relu(self.q1_ln2(self.q1_l2(q1)))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_ln1(self.q2_l1(x)))
        q2 = F.relu(self.q2_ln2(self.q2_l2(q2)))
        q2 = self.q2_l3(q2)
        return q1, q2

class HighLevelActor(nn.Module):
    """Predicts z_subgoal (gap-step-ahead waypoint) from (z_curr, z_ultimate_goal).

    Architecture matches MLPHighLevel in train_high_level.py so that a pretrained
    MLPHighLevel checkpoint can be loaded directly as a warm-start via load_state_dict.
    """
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),      nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_curr, z_goal):
        return self.net(torch.cat([z_curr, z_goal], dim=-1))


class VectorizedEpisodicHERBuffer:
    """100% GPU-native, highly optimized 3D Tensor Buffer with dynamic HER sampling.

    Stores two separate goals per transition:
    - low_goal:   the subgoal the low-level actor was actually pursuing (z_subgoal
                  in joint mode, z_target otherwise).  Used for SAC training so that
                  the critic and actor see the same goal distribution at train and
                  rollout time.  HER relabels THIS goal.
    - ultimate_g: the long-horizon episode goal (ep[-1]).  Read only by the high-level
                  AWR update via sample_high_level_triplets().
    """
    def __init__(self, latent_dim=192, action_dim=25, capacity_episodes=20000, max_t=50, future_p=0.8, device="cuda", done_threshold=2.0, dense_reward_scale=1.6):
        self.capacity = capacity_episodes
        self.max_t = max_t
        self.future_p = future_p
        self.device = device
        self.done_threshold = done_threshold
        self.dense_reward_scale = dense_reward_scale

        # Pre-allocate all memory instantly
        self.z_curr = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity_episodes, max_t, action_dim), dtype=torch.float32, device=device)
        self.z_next = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.low_goal = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.ultimate_g = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.ep_lens = torch.zeros((capacity_episodes,), dtype=torch.long, device=device)

        self.position = 0
        self.size = 0

    @property
    def num_transitions(self):
        """Total number of individual (s,a,s') transitions stored across all episodes."""
        return int(self.ep_lens[:self.size].sum().item())

    def store_episodes(self, z_curr_seq, actions_seq, z_next_seq, low_goal_seq, ultimate_g_seq, lengths):
        num_new = z_curr_seq.shape[0]
        seq_len = z_curr_seq.shape[1]
        end_idx = self.position + num_new

        # Vectorized wrap-around injection
        if end_idx <= self.capacity:
            sl = slice(self.position, end_idx)
            self.z_curr[sl, :seq_len] = z_curr_seq
            self.actions[sl, :seq_len] = actions_seq
            self.z_next[sl, :seq_len] = z_next_seq
            self.low_goal[sl, :seq_len] = low_goal_seq
            self.ultimate_g[sl, :seq_len] = ultimate_g_seq
            self.ep_lens[sl] = lengths
        else:
            overflow = end_idx - self.capacity
            valid = num_new - overflow

            sl1 = slice(self.position, self.capacity)
            self.z_curr[sl1, :seq_len] = z_curr_seq[:valid]
            self.actions[sl1, :seq_len] = actions_seq[:valid]
            self.z_next[sl1, :seq_len] = z_next_seq[:valid]
            self.low_goal[sl1, :seq_len] = low_goal_seq[:valid]
            self.ultimate_g[sl1, :seq_len] = ultimate_g_seq[:valid]
            self.ep_lens[sl1] = lengths[:valid]

            sl2 = slice(0, overflow)
            self.z_curr[sl2, :seq_len] = z_curr_seq[valid:]
            self.actions[sl2, :seq_len] = actions_seq[valid:]
            self.z_next[sl2, :seq_len] = z_next_seq[valid:]
            self.low_goal[sl2, :seq_len] = low_goal_seq[valid:]
            self.ultimate_g[sl2, :seq_len] = ultimate_g_seq[valid:]
            self.ep_lens[sl2] = lengths[valid:]

        self.position = end_idx % self.capacity
        self.size = min(self.size + num_new, self.capacity)

    def sample_batch(self, batch_size=256, reward_mode='sparse'):
        # 1. Instantly sample random episodes and a random valid timestep per episode
        ep_idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        sampled_lens = self.ep_lens[ep_idxs]

        safe_lens = torch.clamp(sampled_lens, min=1)
        t_idxs = (torch.rand(batch_size, device=self.device) * safe_lens).long()
        t_idxs = torch.clamp(t_idxs, max=safe_lens - 1)

        # 2. Gather base transitions — low_goal is the subgoal the actor was pursuing
        z_curr_b = self.z_curr[ep_idxs, t_idxs]
        a_b = self.actions[ep_idxs, t_idxs]
        z_next_b = self.z_next[ep_idxs, t_idxs]
        orig_g_b = self.low_goal[ep_idxs, t_idxs]

        # 3. HER goal relabelling: only valid when there is a strictly future step
        valid_future = (t_idxs < sampled_lens - 1)
        her_mask = (torch.rand(batch_size, device=self.device) < self.future_p) & valid_future

        # Calculate how many steps are strictly AFTER t_idxs
        range_len = sampled_lens - 1 - t_idxs
        safe_range = torch.clamp(range_len, min=1)

        # Force offsets to be at least +1 so we never sample the current state
        offsets = (torch.rand(batch_size, device=self.device) * safe_range).long() + 1
        future_t = torch.clamp(t_idxs + offsets, max=sampled_lens - 1)

        future_g_b = self.z_curr[ep_idxs, future_t]

        # Swap goals based on mask.
        # HER is DISABLED for dense reward: when reward_mode='dense', the HER
        # offset=1 trivially gives dist_next=0 (the HER goal IS z_next), making
        # r=+2.0 for all actions regardless of quality and collapsing the Q gradient.
        # Dense reward already provides continuous signal without goal relabelling.
        if reward_mode == 'dense':
            g_target_b = orig_g_b
        else:
            g_target_b = torch.where(her_mask.unsqueeze(-1), future_g_b, orig_g_b)

        # 4. Compute reward based on mode
        action_penalty = torch.norm(a_b, p=2, dim=-1) * 0.01
        dist_next = torch.norm(z_next_b - g_target_b, p=2, dim=-1)
        success = dist_next < self.done_threshold
        done_b = success.to(torch.float32)

        if reward_mode == 'dense':
            dist_curr = torch.norm(z_curr_b - g_target_b, p=2, dim=-1)
            improvement = (dist_curr - dist_next) / self.dense_reward_scale
            improvement = torch.clamp(improvement, -2.0, 2.0)
            r_b = improvement - action_penalty
        else:  # sparse
            r_b = torch.where(success, torch.zeros_like(dist_next), -torch.ones_like(dist_next)) - action_penalty

        return z_curr_b, a_b, z_next_b, g_target_b, r_b, done_b

    def sample_high_level_triplets(self, batch_size):
        """Return (z_start, z_end, z_ultimate) triplets for high-level AWR training.

        In 'joint' mode every stored episode is exactly one gap-step window.
        z_start   = first state of the episode (where the high-level made its decision).
        z_end     = last z_next in the episode (what the low-level actually reached).
        z_ultimate= ultimate_g — the long-horizon task goal.
        """
        ep_idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        ep_lens = self.ep_lens[ep_idxs]
        last_t  = (ep_lens - 1).clamp(min=0)

        z_start   = self.z_curr[ep_idxs, 0]         # [B, latent_dim]
        z_end     = self.z_next[ep_idxs, last_t]     # [B, latent_dim]
        z_ultimate = self.ultimate_g[ep_idxs, 0]     # [B, latent_dim], same for all steps
        return z_start, z_end, z_ultimate


def sample_tasks(all_latents, batch_size, device, mode,
                 fixed_gap=None, gap_schedule=None, current_stage=None):
    """
    Samples (start, goal) pairs according to the training mode.

    Modes:
      'pure_distance' — start=ep[0], goal=ep[-1]; the original goal spans the full
                        episode. HER relabels goals with visited future states, giving
                        the agent dense implicit sub-goals for free.
      'fixed'         — constant gap (in WM steps) between start and goal throughout
                        all of training; no advancement logic.
      'curriculum'    — adaptive gap that increases once the agent masters the current
                        level (original behaviour).

    Returns: z_starts [B, D], z_targets [B, D], gaps_used [B] (int tensor)
    """
    num_eps = len(all_latents)
    z_starts = []
    z_targets = []
    gaps_used = []

    if mode == 'pure_distance':
        ep_idxs = torch.randint(0, num_eps, (batch_size,))
        for ep_idx in ep_idxs:
            ep = all_latents[ep_idx.item()]
            ep_len = ep.shape[0]
            # Random start within the episode for diversity; goal is always the final frame
            start_t = torch.randint(0, ep_len, (1,)).item()
            z_starts.append(ep[start_t])
            z_targets.append(ep[-1])
            gaps_used.append(0)

    elif mode == 'fixed':
        assert fixed_gap is not None, "fixed_gap must be provided for 'fixed' mode"
        ep_idxs = torch.randint(0, num_eps, (batch_size,))
        for ep_idx in ep_idxs:
            ep = all_latents[ep_idx.item()]
            ep_len = ep.shape[0]
            actual_frame_gap = min(fixed_gap * 5, ep_len - 1)
            start_t = torch.randint(0, ep_len - actual_frame_gap, (1,)).item()
            target_t = start_t + actual_frame_gap
            z_starts.append(ep[start_t])
            z_targets.append(ep[target_t])
            gaps_used.append(fixed_gap)

    elif mode == 'curriculum':
        assert gap_schedule is not None and current_stage is not None
        ep_idxs = torch.randint(0, num_eps, (batch_size,))
        for ep_idx in ep_idxs:
            ep = all_latents[ep_idx.item()]
            ep_len = ep.shape[0]

            # 80% current hardest gap, 20% random earlier gap to prevent forgetting
            if current_stage > 0 and random.random() < 0.2:
                gap = random.choice(gap_schedule[:current_stage])
            else:
                gap = gap_schedule[current_stage]

            actual_frame_gap = min(gap * 5, ep_len - 1)
            start_t = torch.randint(0, ep_len - actual_frame_gap, (1,)).item()
            target_t = start_t + actual_frame_gap
            z_starts.append(ep[start_t])
            z_targets.append(ep[target_t])
            gaps_used.append(gap)

    elif mode == 'joint':
        # Joint hierarchical mode: z_start = random frame, z_ultimate = ep[-1].
        # The high-level actor generates the gap-step intermediate subgoal at rollout time.
        # Each episode is T_max=fixed_gap steps (one subgoal window).
        assert fixed_gap is not None, "fixed_gap must be provided for 'joint' mode"
        ep_idxs = torch.randint(0, num_eps, (batch_size,))
        for ep_idx in ep_idxs:
            ep = all_latents[ep_idx.item()]
            ep_len = ep.shape[0]
            # Sample a random start that is at least 1 frame from the end
            start_t = torch.randint(0, max(ep_len - 1, 1), (1,)).item()
            z_starts.append(ep[start_t])
            z_targets.append(ep[-1])   # ultimate goal — high-level decomposes this
            gaps_used.append(fixed_gap)

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose from: pure_distance, fixed, curriculum, joint")

    return (
        torch.stack(z_starts).to(device),
        torch.stack(z_targets).to(device),
        torch.tensor(gaps_used, device=device),
    )


def sample_high_level_from_cache(all_latents, batch_size, gap, device):
    """Sample (z_t, z_{t+gap}, z_end) triplets from the encoder cache.

    All returned latents are encoder outputs (no predictor drift). Used for
    training the high-level actor with distance-based AWR advantage so that
    the high-level operates entirely in encoder space — matching eval.

    Args:
        all_latents: List of [T_ep, D] tensors from the encoder cache.
        batch_size: Number of triplets to sample.
        gap: Gap in dataset frames (NOT WM steps; caller should pass gap * FRAMESKIP).
        device: Target device.

    Returns:
        z_t: [B, D] encoder latent at time t
        z_t_gap: [B, D] encoder latent at time t + gap
        z_ultimate: [B, D] encoder latent at end of episode
    """
    num_eps = len(all_latents)
    z_t_list, z_tg_list, z_ult_list = [], [], []

    ep_idxs = torch.randint(0, num_eps, (batch_size,))
    for ep_idx in ep_idxs:
        ep = all_latents[ep_idx.item()]
        ep_len = ep.shape[0]
        max_start = ep_len - gap - 1
        if max_start < 1:
            max_start = 1
            actual_gap = ep_len - 2
        else:
            actual_gap = gap
        start_t = torch.randint(0, max(max_start, 1), (1,)).item()
        z_t_list.append(ep[start_t])
        z_tg_list.append(ep[min(start_t + actual_gap, ep_len - 1)])
        z_ult_list.append(ep[-1])

    return (
        torch.stack(z_t_list).to(device),
        torch.stack(z_tg_list).to(device),
        torch.stack(z_ult_list).to(device),
    )


def train_loop(env, actor, critic, critic_target,
               actor_optimizer, critic_optimizer, alpha_optimizer,
               log_alpha, target_entropy, replay_buffer, all_latents,
               num_iterations=1000, gamma=0.99, tau=0.005,
               save_dir="./checkpoints",
               mode='curriculum', fixed_gap=None, tmax_pure_distance=40,
               reward_mode='sparse', bc_model=None, bc_alpha=0.0,
               high_actor=None, high_actor_optimizer=None,
               high_level_beta=3.0, high_level_gap=8,
               # --- New: cache-transition offline RL ---
               cache_mixer=None,       # CacheTransitionMixer instance (or None)
               cache_ratio=0.0,        # fraction of each SAC batch from cache (1.0 = pure offline)
               bc_alpha_cache=0.1,     # BC regularization strength on cache transitions
               fixed_alpha=None,       # if not None, fix SAC temperature to this value (disable auto-tuning)
               pca_mean=None,          # [192] PCA mean (for projecting WM rollout latents)
               pca_matrix=None):       # [192, D] PCA matrix (None = no projection)

    os.makedirs(save_dir, exist_ok=True)
    mode_str = mode if mode != 'fixed' else f"fixed (gap={fixed_gap})"
    bc_str = f"BC alpha={bc_alpha}" if bc_model is not None else "no BC"
    if cache_mixer is not None:
        bc_str += f" | Cache BC alpha={bc_alpha_cache} | cache_ratio={cache_ratio:.2f}"
    if fixed_alpha is not None:
        bc_str += f" | fixed_alpha={fixed_alpha}"
    print(f"Starting HER Training | Mode: {mode_str} | Reward: {reward_mode} | {bc_str} | Saving to: {save_dir}")

    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Stage", "Gap", "Success_Rate", "Episodes_in_Buf",
                         "Transitions_in_Buf", "Actor_Loss", "Critic_Loss", "HL_Loss",
                         "BC_Loss", "Alpha",
                         "EnvStep_Time", "BufStore_Time", "Train_Time", "Total_Time"])

    # Effective latent dim for RL components (pca_dim if PCA provided, else 192)
    rl_latent_dim = pca_matrix.shape[1] if pca_matrix is not None else 192
    pure_offline = (cache_ratio >= 1.0) and (cache_mixer is not None)

    # gap_schedule / tmax_schedule are only active in 'curriculum' mode.
    # Each gap is measured in world-model steps (1 WM step = 5 video frames).
    # T_max matches the gap exactly so the budget scales with task difficulty.
    gap_schedule  = [1, 2, 4, 8, 16, 24, 32, 40]
    tmax_schedule = [1, 2, 4, 8, 16, 24, 32, 40]
    stage = 0
    recent_successes = deque(maxlen=2000)

    start_time = time.time()
    env_step_accum = 0.0
    buf_store_accum = 0.0
    train_time_accum = 0.0

    for iteration in range(num_iterations):

        # --- DETERMINE T_max AND CURRENT GAP FOR THIS ITERATION ---
        if mode == 'curriculum':
            current_target_gap = gap_schedule[stage]
            T_max = tmax_schedule[stage]
        elif mode == 'fixed':
            current_target_gap = fixed_gap
            T_max = fixed_gap
        elif mode == 'joint':
            current_target_gap = high_level_gap
            T_max = high_level_gap
        else:  # pure_distance
            current_target_gap = 0
            T_max = tmax_pure_distance

        if not pure_offline:
            assert T_max <= replay_buffer.max_t, (
                f"T_max={T_max} exceeds the buffer's max_t={replay_buffer.max_t}. "
                "Increase max_t when constructing VectorizedEpisodicHERBuffer."
            )

        # --- TASK GENERATION (skipped in pure offline — cache_mixer provides all transitions) ---
        if not pure_offline:
            z_curr, z_target, gaps_used = sample_tasks(
                all_latents, env.num_envs, env.device,
                mode=mode, fixed_gap=fixed_gap,
                gap_schedule=gap_schedule, current_stage=stage,
            )
            env.set_states(z_curr)  # WM env uses 192D latents

            # --- HIGH-LEVEL SUBGOAL GENERATION (joint mode only, WM rollout path) ---
            # In joint mode z_target = ep[-1] (the ultimate goal). The high-level actor
            # predicts an intermediate subgoal that is gap steps ahead. The env pursues
            # z_subgoal (reward/done relative to it). The buffer stores BOTH goals:
            #   low_goal  = z_subgoal  (for SAC: critic + actor train on this)
            #   ultimate_g = z_target  (for high-level AWR advantage computation)
            if mode == 'joint' and high_actor is not None:
                # Project 192D WM latents to rl_latent_dim for the high-level actor
                z_curr_hl   = project_latent(z_curr,   pca_mean, pca_matrix)
                z_target_hl = project_latent(z_target, pca_mean, pca_matrix)
                with torch.no_grad():
                    z_subgoal_rl = high_actor(z_curr_hl, z_target_hl)  # [B, rl_latent_dim]
                # De-project subgoal to 192D for WM env's done computation
                z_subgoal_wm = deproject_latent(z_subgoal_rl, pca_mean, pca_matrix)
                env.z_ultimate_goal = z_subgoal_wm   # env done/reward in 192D
                low_level_goal      = z_subgoal_wm   # rollout actor uses 192D (will project inside)
            else:
                env.z_ultimate_goal = z_target
                low_level_goal      = z_target
        else:
            # Pure offline: set dummy values for logging
            z_curr     = torch.zeros(1, dtype=torch.float32, device=device)
            z_target   = torch.zeros(1, dtype=torch.float32, device=device)
            gaps_used  = torch.zeros(1, dtype=torch.long,    device=device)
            low_level_goal = z_target

        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        rollout_act_mags = []
        avg_start_dist = 0.0
        avg_end_dist = 0.0

        # Pre-allocate temporary trajectory memory for blazing fast episodic collection
        # In pure_offline mode (cache_ratio >= 1.0), we skip WM rollouts entirely.
        # In mixed mode, latents are projected to rl_latent_dim before storage.
        if not pure_offline:
            rollout_z_curr    = torch.zeros((env.num_envs, T_max, rl_latent_dim), dtype=torch.float32, device=env.device)
            rollout_actions   = torch.zeros((env.num_envs, T_max, 25),            dtype=torch.float32, device=env.device)
            rollout_z_next    = torch.zeros((env.num_envs, T_max, rl_latent_dim), dtype=torch.float32, device=env.device)
            rollout_low_goal  = torch.zeros((env.num_envs, T_max, rl_latent_dim), dtype=torch.float32, device=env.device)
            rollout_ultimate  = torch.zeros((env.num_envs, T_max, rl_latent_dim), dtype=torch.float32, device=env.device)
            rollout_lengths   = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        if not pure_offline:
          with torch.no_grad():
            for step in range(T_max):
                if not active_mask.any():
                    break

                # Project WM-space latents to rl_latent_dim before actor sees them
                z_curr_rl       = project_latent(z_curr,          pca_mean, pca_matrix)
                low_goal_rl     = project_latent(low_level_goal,   pca_mean, pca_matrix)

                if step == 0:
                    avg_start_dist = torch.norm(z_curr_rl - low_goal_rl, p=2, dim=-1).mean().item()

                actions, _, _ = actor.sample(z_curr_rl, low_goal_rl)
                rollout_act_mags.append(torch.norm(actions, p=2, dim=-1).mean().item())

                t0 = time.time()
                z_next_wm, _, dones, _, _ = env.step(actions)
                env_step_accum += (time.time() - t0)

                z_next_rl = project_latent(z_next_wm, pca_mean, pca_matrix)

                t1 = time.time()

                # Record data ONLY for environments that are still active (in rl_latent_dim)
                rollout_z_curr[active_mask, step]    = z_curr_rl[active_mask]
                rollout_actions[active_mask, step]   = actions[active_mask]
                rollout_z_next[active_mask, step]    = z_next_rl[active_mask]
                # low_goal = the subgoal the low-level actor is actually pursuing
                rollout_low_goal[active_mask, step]  = low_goal_rl[active_mask]
                # ultimate_g = ep[-1], the long-horizon goal. Only used by HL AWR.
                rollout_ultimate[active_mask, step]  = project_latent(z_target[active_mask], pca_mean, pca_matrix)
                rollout_lengths[active_mask] += 1

                buf_store_accum += (time.time() - t1)

                just_succeeded = dones & active_mask
                success_mask  |= just_succeeded
                active_mask   &= ~just_succeeded
                # Update z_curr in WM space (env operates in 192D)
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next_wm, z_curr)

            avg_end_dist = torch.norm(
                project_latent(z_curr, pca_mean, pca_matrix) - low_goal_rl,
                p=2, dim=-1).mean().item()

          # Inject the entire rollout batch into the episodic HER buffer at once
          t2 = time.time()
          replay_buffer.store_episodes(
              rollout_z_curr, rollout_actions, rollout_z_next,
              rollout_low_goal, rollout_ultimate, rollout_lengths
          )
          buf_store_accum += (time.time() - t2)

        last_batch_z_b      = None
        last_batch_z_next_b = None
        last_batch_g_b      = None
        last_batch_d_b      = None

        # --- CURRICULUM ADVANCEMENT (curriculum mode only) ---
        if pure_offline:
            # No env rollouts — compute proxy metrics from the last training batch.
            # "SR" here is the HER-relabeled done fraction (how often sampled transitions
            # land within done_threshold of their relabeled goal). Not a policy SR, but
            # non-zero and confirms reward/threshold are working correctly.
            if last_batch_d_b is not None:
                current_sr   = last_batch_d_b.float().mean().item()
                avg_start_dist = torch.norm(last_batch_z_b    - last_batch_g_b, p=2, dim=-1).mean().item()
                avg_end_dist   = torch.norm(last_batch_z_next_b - last_batch_g_b, p=2, dim=-1).mean().item()
            else:
                current_sr = 0.0
        elif mode == 'curriculum':
            hard_task_mask = (gaps_used == current_target_gap)
            if hard_task_mask.any():
                recent_successes.extend(success_mask[hard_task_mask].cpu().tolist())

            current_sr = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0

            if len(recent_successes) == recent_successes.maxlen and current_sr > 0.85:
                if stage < len(gap_schedule) - 1:
                    stage += 1
                    recent_successes.clear()
                    print(f"\n*** CURRICULUM ADVANCE: Stage {stage} | "
                          f"New Gap: {gap_schedule[stage]} | "
                          f"New T_max: {tmax_schedule[stage]} ***\n")
        else:
            # For non-curriculum modes, track overall success rate for logging
            recent_successes.extend(success_mask.cpu().tolist())
            current_sr = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0

        # --- SAC Training Phase ---
        iter_train_start = time.time()

        avg_actor_loss  = 0.0
        avg_critic_loss = 0.0
        avg_bc_loss     = 0.0

        # Dense reward provides a richer signal per sample so fewer updates per
        # iteration are needed; more importantly, fewer updates prevent the critic
        # from diverging through aggressive bootstrapping of shaped rewards.
        grad_updates = 20 if reward_mode == 'dense' else 40

        # Determine if we have enough data to train
        # Pure offline: always train (cache has ample data). Mixed: wait for buffer.
        has_data = pure_offline or (replay_buffer.num_transitions >= 256)

        if has_data:
            for _ in range(grad_updates):
                # --- Sample batch: mix cache and WM replay buffer ---
                if pure_offline:
                    # 100% from cache — no WM rollouts, no exploitation
                    assert cache_mixer is not None, "cache_mixer required when cache_ratio >= 1.0"
                    z_b, a_cache_b, z_next_b, g_b, r_b, d_b = cache_mixer.sample(256)
                    a_b = a_cache_b  # expert actions for critic; also BC targets
                elif cache_mixer is not None and cache_ratio > 0.0:
                    # Mixed: cache_ratio fraction from cache, rest from WM replay
                    n_cache = int(256 * cache_ratio)
                    n_wm    = 256 - n_cache
                    zc, ac, znc, gc, rc, dc = cache_mixer.sample(n_cache)
                    zw, aw, znw, gw, rw, dw = replay_buffer.sample_batch(n_wm, reward_mode=reward_mode)
                    z_b     = torch.cat([zc, zw], dim=0)
                    a_b     = torch.cat([ac, aw], dim=0)
                    z_next_b = torch.cat([znc, znw], dim=0)
                    g_b     = torch.cat([gc, gw], dim=0)
                    r_b     = torch.cat([rc, rw], dim=0)
                    d_b     = torch.cat([dc, dw], dim=0)
                    a_cache_b = ac  # only cache actions used for BC
                else:
                    z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(256, reward_mode=reward_mode)
                    a_cache_b = None

                last_batch_z_b      = z_b
                last_batch_z_next_b = z_next_b
                last_batch_g_b      = g_b
                last_batch_d_b      = d_b

                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)

                # SAC temperature: fixed or learned
                if fixed_alpha is not None:
                    alpha = fixed_alpha
                else:
                    alpha = log_alpha.exp().item()

                # Update Critic (with TD3-style target policy smoothing)
                with torch.no_grad():
                    next_actions_raw, next_log_pi, _ = actor.sample(z_next_b, g_b)
                    # TD3 target noise: adds robustness to Q-value overestimation
                    td3_noise = (torch.randn_like(next_actions_raw) * 0.2).clamp(-0.5, 0.5)
                    next_actions = (next_actions_raw + td3_noise).clamp(
                        -actor.action_scale, actor.action_scale)
                    target_q1, target_q2 = critic_target(z_next_b, g_b, next_actions)
                    target_q_min   = torch.min(target_q1, target_q2) - alpha * next_log_pi
                    target_q_value = r_b + (1.0 - d_b) * gamma * target_q_min

                current_q1, current_q2 = critic(z_b, g_b, a_b)
                critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()

                avg_critic_loss += critic_loss.item()

                # Update Actor
                new_actions, log_pi, _ = actor.sample(z_b, g_b)
                for p in critic.parameters():
                    p.requires_grad = False

                q1_new, q2_new = critic(z_b, g_b, new_actions)
                q_min_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_pi - q_min_new).mean()

                # BC regularisation from cache transitions:
                # Pull actor toward expert actions (ReBRAC-style, no pretrained BC model needed).
                # a_cache_b: expert actions from the cache batch.
                bc_loss_val = 0.0
                if a_cache_b is not None and bc_alpha_cache > 0.0:
                    if pure_offline:
                        # All transitions are cache — BC on full batch
                        bc_actions_input = z_b
                        bc_goals_input   = g_b
                        bc_targets       = a_cache_b
                    else:
                        # Mixed — BC only on the cache portion
                        n_cache = a_cache_b.shape[0]
                        bc_actions_input = z_b[:n_cache]
                        bc_goals_input   = g_b[:n_cache]
                        bc_targets       = a_cache_b
                    bc_new_actions, _, _ = actor.sample(bc_actions_input, bc_goals_input)
                    bc_loss_val = F.mse_loss(bc_new_actions, bc_targets)
                    actor_loss  = actor_loss + bc_alpha_cache * bc_loss_val
                elif bc_model is not None and bc_alpha > 0.0:
                    # Legacy: pretrained BC model (only active when cache_mixer is None)
                    with torch.no_grad():
                        bc_actions = bc_model(z_b, g_b)
                    bc_loss_val = F.mse_loss(new_actions, bc_actions)
                    actor_loss  = actor_loss + bc_alpha * bc_loss_val

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_bc_loss    += float(bc_loss_val) if isinstance(bc_loss_val, float) else bc_loss_val.item()

                for p in critic.parameters():
                    p.requires_grad = True

                # Update Alpha (skip if fixed)
                if fixed_alpha is None:
                    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                # Soft Update Target Networks
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # --- HIGH-LEVEL AWR UPDATE (joint mode only) ---
        # Trains the high-level actor to predict achievable gap-step waypoints.
        # Samples from the ENCODER CACHE (not the replay buffer) so that all
        # latents are in encoder space — matching what the high-level sees at eval.
        # Uses distance-based advantage instead of critic-based: under SIGReg's
        # isotropic regularization, L2 distance is a valid proxy for value.
        # Latents are projected to rl_latent_dim (pca_dim) to sharpen the advantage
        # signal: in 192D the advantage per step is ~2.5% of scale; in 10D it's much sharper.
        avg_hl_loss = 0.0
        hl_updates = 10
        hl_device = replay_buffer.device if not pure_offline else (
            cache_mixer.device if cache_mixer is not None else 'cuda')
        hl_ready = pure_offline or (replay_buffer.num_transitions >= 256)
        if mode == 'joint' and high_actor is not None and high_actor_optimizer is not None \
                and hl_ready and iteration >= 10:
            # gap in dataset frames = high_level_gap * FRAMESKIP(=5)
            hl_frame_gap = high_level_gap * 5
            for _ in range(hl_updates):
                z_t_raw, z_t_gap_raw, g_ult_raw = sample_high_level_from_cache(
                    all_latents, 256, hl_frame_gap, device=hl_device
                )

                # Project to rl_latent_dim for sharper advantage signal
                z_t     = project_latent(z_t_raw,     pca_mean, pca_matrix)
                z_t_gap = project_latent(z_t_gap_raw, pca_mean, pca_matrix)
                g_ult   = project_latent(g_ult_raw,   pca_mean, pca_matrix)

                with torch.no_grad():
                    # Distance-based advantage: closer to goal = higher advantage
                    dist_t    = torch.norm(z_t     - g_ult, p=2, dim=-1)   # [B]
                    dist_tgap = torch.norm(z_t_gap - g_ult, p=2, dim=-1)   # [B]
                    adv = dist_t - dist_tgap                                # positive if z_t_gap closer
                    w   = torch.exp(high_level_beta * adv).clamp(max=100.0)

                pred    = high_actor(z_t, g_ult)                                        # [B, rl_latent_dim]
                per_dim = F.mse_loss(pred, z_t_gap, reduction='none').mean(dim=-1)      # [B]
                hl_loss = (w * per_dim).mean()

                high_actor_optimizer.zero_grad()
                hl_loss.backward()
                torch.nn.utils.clip_grad_norm_(high_actor.parameters(), 1.0)
                high_actor_optimizer.step()
                avg_hl_loss += hl_loss.item()

            avg_hl_loss /= hl_updates

        train_time_accum += (time.time() - iter_train_start)

        if iteration % 10 == 0:
            alpha_now = fixed_alpha if fixed_alpha is not None else log_alpha.exp().item()
            actor_val  = avg_actor_loss  / grad_updates if has_data else 0.0
            critic_val = avg_critic_loss / grad_updates if has_data else 0.0
            bc_val     = avg_bc_loss     / grad_updates if has_data else 0.0
            elapsed_time = time.time() - start_time
            avg_act_mag = np.mean(rollout_act_mags) if rollout_act_mags else 0.0
            hl_str = f" | HL Loss: {avg_hl_loss:.4f}" if mode == 'joint' else ""
            buf_info = f"CacheMixer:{cache_mixer.total:,}" if pure_offline else \
                       f"Buf Eps:{replay_buffer.size} Tr:{replay_buffer.num_transitions}"

            print(f"Iter {iteration:04d} | SR: {current_sr*100:.1f}% | "
                  f"{buf_info} | "
                  f"Act: {actor_val:.3f} | Crit: {critic_val:.3f} | BC: {bc_val:.4f} | α: {alpha_now:.4f} | "
                  f"StartD: {avg_start_dist:.2f} | EndD: {avg_end_dist:.2f}"
                  f"{hl_str}")

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, stage, current_target_gap, current_sr,
                                 replay_buffer.size, replay_buffer.num_transitions,
                                 actor_val, critic_val, avg_hl_loss, bc_val, alpha_now,
                                 env_step_accum, buf_store_accum, train_time_accum, elapsed_time])

            start_time = time.time()
            env_step_accum   = 0.0
            buf_store_accum  = 0.0
            train_time_accum = 0.0

        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(),  os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            if mode == 'joint' and high_actor is not None:
                torch.save(high_actor.state_dict(), os.path.join(save_dir, "high_actor.pth"))
            print(f"--> Checkpoint saved at Iteration {iteration}")

def _load_jepa_from_ckpt(ckpt_path, device, img_size=64, patch_size=8):
    """Load a JEPA checkpoint into a bare JEPA model.

    Supports both:
    - Lightning-style checkpoints (state_dict with 'model.' prefix) from train.py
    - Raw state dicts (bare module paths) like the HuggingFace weights.pt

    Args:
        ckpt_path: Path to checkpoint file
        device: Target device
        img_size: Image resolution (64 for OGBench, 224 for pretrained LeWM)
        patch_size: ViT patch size (8 for 64x64, 14 for 224x224)
    """
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        # Lightning-style checkpoint: weights live under 'state_dict' with a 'model.' prefix
        raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        epoch = ckpt.get('epoch', '?')
        step  = ckpt.get('global_step', '?')
    else:
        # Raw state dict saved directly (keys are bare module paths, no prefix)
        raw_sd = dict(ckpt)
        epoch, step = '?', '?'
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded JEPA from {ckpt_path} (epoch {epoch}, step {step}, "
          f"img_size={img_size}, patch_size={patch_size})")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent World Model SAC + HER Training")
    parser.add_argument(
        '--mode', type=str, default='curriculum',
        choices=['pure_distance', 'fixed', 'curriculum', 'joint'],
        help=(
            "Training mode:\n"
            "  pure_distance — no subgoals; start=ep[0], goal=ep[-1]. HER relabels with\n"
            "                  visited future states, providing implicit dense sub-goals.\n"
            "  fixed         — constant WM-step gap between start and goal throughout training.\n"
            "  curriculum    — adaptive gap that increases once the agent masters the current level.\n"
            "  joint         — HIQL-style: high-level actor generates gap-step subgoals toward ep[-1];\n"
            "                  low-level SAC+HER pursues those subgoals; both trained jointly."
        ),
    )
    parser.add_argument(
        '--gap', type=int, default=8,
        help="Gap size (in WM steps) for 'fixed' mode. Each WM step = 5 video frames. Ignored for other modes.",
    )
    parser.add_argument(
        '--tmax', type=int, default=40,
        help="Max rollout steps for 'pure_distance' mode (default: 40). Ignored for other modes.",
    )
    parser.add_argument(
        '--num_iters', type=int, default=None,
        help="Override the number of training iterations.",
    )
    parser.add_argument(
        '--reward_mode', type=str, default='sparse', choices=['sparse', 'dense'],
        help=(
            "Reward mode:\n"
            "  sparse — -1 every step until success (original behaviour).\n"
            "  dense  — potential-based shaping: reward = clamp((dist_curr - dist_next)/scale, -2, 2).\n"
            "            No terminal bonus; 20 gradient updates/iter instead of 40.\n"
            "            HER relabelling still applies in both modes."
        ),
    )
    parser.add_argument(
        '--bc_alpha', type=float, default=0.0,
        help="BC regularisation coefficient for legacy pretrained-BC-model path (0 = disabled).",
    )
    parser.add_argument(
        '--bc_model_path', type=str, default=None,
        help="Path to a BCPolicy checkpoint (.pth) produced by train_bc.py. "
             "Required when --bc_alpha > 0 AND --cache_ratio == 0.",
    )
    parser.add_argument(
        '--cache_ratio', type=float, default=1.0,
        help=(
            "Fraction of each SAC batch drawn from the encoder-cache (expert transitions).\n"
            "  1.0 = pure offline (no WM rollouts — completely eliminates H1 exploitation)\n"
            "  0.5 = mixed (50%% cache + 50%% WM rollouts)\n"
            "  0.0 = WM rollouts only (original behavior)\n"
            "Default: 1.0 (pure offline)."
        ),
    )
    parser.add_argument(
        '--bc_alpha_cache', type=float, default=0.1,
        help=(
            "BC regularisation strength on cache transitions: "
            "actor_loss += bc_alpha_cache * MSE(π(z,g), a_expert). "
            "a_expert comes directly from the cache — no pretrained BC model needed. "
            "Default: 0.1. Set to 0 to disable. Only active when --cache_ratio > 0."
        ),
    )
    parser.add_argument(
        '--fixed_alpha', type=float, default=0.01,
        help=(
            "Fix SAC temperature α to this value (disable auto-tuning). "
            "With BC regularization from cache, entropy pressure is handled by BC. "
            "Set to None to re-enable auto-tuning (not recommended with cache mixing). "
            "Default: 0.01."
        ),
    )
    parser.add_argument(
        '--pca_path', type=str, default=None,
        help=(
            "Path to PCA projection params (.pt) built by build_pca_projection.py. "
            "If provided, all RL latents (actor, critic, AWR) are projected to pca_dim "
            "for sharper goal-conditioning and advantage signals. "
            "Example: $STABLEWM_HOME/lewm_pca_10d.pt"
        ),
    )
    parser.add_argument(
        '--high_level_beta', type=float, default=3.0,
        help="AWR temperature for the high-level actor (default 3.0, same as HIQL high_alpha). "
             "Only used in 'joint' mode.",
    )
    parser.add_argument(
        '--high_level_gap', type=int, default=8,
        help="Subgoal horizon in WM steps for 'joint' mode (default 8). "
             "Also sets T_max for each rollout episode.",
    )
    parser.add_argument(
        '--high_actor_path', type=str, default=None,
        help="Optional path to a pretrained MLPHighLevel checkpoint (.pth) to warm-start "
             "the high-level actor. Only used in 'joint' mode.",
    )
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help="Path to JEPA checkpoint (.ckpt). "
             "Default: $HOME/leworldmodel/lewm_ogbench_weights.ckpt",
    )
    parser.add_argument(
        '--cache_path', type=str, default=None,
        help="Path to precomputed latents cache (.pt). "
             "Default: $HOME/stable_wm_data/cube_all_latents_cache.pt",
    )
    parser.add_argument(
        '--dataset_path', type=str, default=None,
        help="Path to HDF5 dataset (without .h5 extension). "
             "Default: $HOME/stable_wm_data/ogbench/cube_single_play_v0",
    )
    parser.add_argument(
        '--done_threshold', type=float, default=None,
        help="L2 distance in latent space below which a step is considered successful.\n"
             "Must be less than the start-to-goal distance to avoid trivial success.\n"
             "For 224x224 model with gap=1: use 2.41 (median 1-step predictor drift).\n"
             "Rule of thumb: threshold ≈ median predictor error at the chosen gap.\n"
             "Overrides auto-detected default.",
    )
    parser.add_argument(
        '--img_size', type=int, default=224,
        help="Image resolution for the JEPA encoder (224 for pretrained LeWM, 64 for OGBench-trained).",
    )
    parser.add_argument(
        '--patch_size', type=int, default=14,
        help="ViT patch size (14 for 224x224, 8 for 64x64).",
    )
    args = parser.parse_args()

    if args.bc_alpha > 0 and args.bc_model_path is None and args.cache_ratio == 0.0:
        parser.error("--bc_model_path is required when --bc_alpha > 0 and --cache_ratio == 0")
    if args.mode == 'joint' and args.gap != 8:
        # --gap is for 'fixed' mode; in joint mode --high_level_gap controls the horizon
        pass  # --gap is ignored for joint mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DERIVE CHECKPOINT DIRECTORY FROM MODE, REWARD, AND BC ---
    reward_tag = f"_{args.reward_mode}"
    bc_tag     = f"_bc{args.bc_alpha}" if args.bc_alpha > 0 else ""
    # Cache ratio tag: cr1.0 = pure offline, cr0.5 = mixed, omitted if 0.0 (original)
    cr_tag = f"_cr{args.cache_ratio:.1f}" if args.cache_ratio > 0.0 else ""
    pca_tag = f"_pca" if args.pca_path is not None else ""
    if args.mode == 'pure_distance':
        save_dir = f"./checkpoints_her_pure_distance{reward_tag}{bc_tag}{cr_tag}{pca_tag}"
    elif args.mode == 'fixed':
        save_dir = f"./checkpoints_her_fixed_gap_{args.gap}{reward_tag}{bc_tag}{cr_tag}{pca_tag}"
    elif args.mode == 'joint':
        save_dir = f"./checkpoints_joint_gap_{args.high_level_gap}_beta{args.high_level_beta}{reward_tag}{bc_tag}{cr_tag}{pca_tag}"
    else:
        save_dir = f"./checkpoints_her_curriculum{reward_tag}{bc_tag}{cr_tag}{pca_tag}"

    DEBUG_MODE = False
    # Default: 2.41 = median 1-step predictor drift for the 224x224 model.
    # This is less than the natural 1-WM-step distance (6.29), so the agent
    # must actually predict accurately — trivial success is not possible.
    done_threshold = 2.41

    if DEBUG_MODE:
        print("--- RUNNING IN LOCAL DEBUG MODE ---")
        num_envs_to_use = 2
        num_iters_to_run = 25

        class DummyJEPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_encoder = nn.Linear(25, 192)
            def encode(self, info):
                return {'emb': torch.randn(info['pixels'].shape[0], 1, 192, device=info['pixels'].device)}
            def predict(self, emb, act_emb):
                return torch.randn(emb.shape[0], emb.shape[1], 192, device=emb.device)

        class DummyDataset:
            def __init__(self):
                self.lengths = np.array([20] * 10)
            def load_chunk(self, ep_indices, starts, ends):
                return [{'pixels': torch.randn(1, 3, 224, 224)} for _ in range(len(ep_indices))]

        jepa_model = DummyJEPA().to(device)
        dataset = DummyDataset()
        all_latents = [torch.randn(201, 192) for _ in range(10)]
    else:
        print("--- RUNNING IN PRODUCTION MODE ---")
        num_envs_to_use = 256
        num_iters_to_run = 50000

        stablewm_home = os.environ.get("STABLEWM_HOME", os.path.join(os.path.expanduser("~"), "stable_wm_data"))

        data_path  = args.dataset_path or os.path.join(stablewm_home, "ogbench", "cube_single_expert")
        ckpt_path  = args.ckpt_path    or os.path.join(stablewm_home, "cube", "lejepa_weights.ckpt")
        cache_path = args.cache_path   or os.path.join(stablewm_home, "lewm_224_latents_cache.pt")

        print(f"Loading Dataset from:    {data_path}")
        print(f"Loading Checkpoint from: {ckpt_path}")
        print(f"Loading Cache from:      {cache_path}")

        parent_dir = os.path.abspath(os.path.dirname(__file__))

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        dataset = swm.data.HDF5Dataset(data_path)

        if args.ckpt_path is not None:
            # Explicit path provided: load directly
            jepa_model = _load_jepa_from_ckpt(ckpt_path, device,
                                               img_size=args.img_size,
                                               patch_size=args.patch_size)
            done_threshold = 2.887
        else:
            # Default: use the swm AutoCostModel (224x224 model resolved via STABLEWM_HOME)
            with initialize(version_base=None, config_path="../config"):
                cfg = compose(config_name="eval/cube", overrides=["+policy=cube/lejepa"])
            jepa_model = swm.policy.AutoCostModel(cfg.policy)
            jepa_model = jepa_model.to(device)
            jepa_model.eval()
            for param in jepa_model.parameters():
                param.requires_grad = False
            done_threshold = 2.0

        # CLI override takes precedence over auto-detected value
        if args.done_threshold is not None:
            done_threshold = args.done_threshold

        print("Dataset and JEPA Model successfully loaded")

        # Load the latents cache for task sampling
        print(f"Loading latents cache from {cache_path}...")
        cache = torch.load(cache_path, map_location="cpu")
        all_latents = cache['all_latents']
        all_actions = cache.get('all_actions', [])  # List[Tensor[T, 5]]

    # Allow --num_iters to override the default
    if args.num_iters is not None:
        num_iters_to_run = args.num_iters

    # --- Load PCA projection (optional) ---
    pca_mean   = None
    pca_matrix = None
    if not DEBUG_MODE and args.pca_path is not None:
        print(f"Loading PCA projection from {args.pca_path} ...")
        pca_data = torch.load(args.pca_path, map_location="cpu")
        pca_mean   = pca_data['pca_mean'].to(device)       # [192]
        pca_matrix = pca_data['pca_matrix'].to(device)     # [192, D]
        pca_dim    = pca_data['pca_dim']
        top_k_var  = pca_data.get('top_k_variance', float('nan'))
        print(f"  PCA: 192D → {pca_dim}D  (explains {top_k_var*100:.1f}% of cache variance)")
    else:
        pca_dim = 192

    # Effective RL latent dimension (pca_dim or 192 if no PCA)
    rl_latent_dim = pca_dim

    # --- Load BC model (frozen) if requested (legacy path, cache_ratio=0) ---
    bc_model = None
    if args.bc_alpha > 0 and args.bc_model_path is not None:
        ckpt = torch.load(args.bc_model_path, map_location=device, weights_only=False)
        bc_model = BCPolicy(
            latent_dim=ckpt.get('latent_dim', 192),
            action_dim=ckpt.get('action_dim', 25),
            action_scale=ckpt.get('action_scale', 3.0),
        ).to(device)
        bc_model.load_state_dict(ckpt['model_state_dict'])
        bc_model.eval()
        for p in bc_model.parameters():
            p.requires_grad = False
        print(f"BC model loaded from {args.bc_model_path} (bc_alpha={args.bc_alpha})")

    # --- Build CacheTransitionMixer if cache_ratio > 0 ---
    cache_mixer = None
    if not DEBUG_MODE and args.cache_ratio > 0.0 and all_actions:
        from sklearn import preprocessing as sklearn_preprocessing
        print("Fitting StandardScaler on dataset actions for CacheTransitionMixer ...")
        import stable_worldmodel as swm_data
        _dataset_for_scaler = swm_data.data.HDF5Dataset(data_path, keys_to_cache=['action'],
                                                         cache_dir=os.path.dirname(data_path))
        action_raw = _dataset_for_scaler.get_col_data('action')
        action_raw = action_raw[~np.isnan(action_raw).any(axis=1)]
        action_scaler = sklearn_preprocessing.StandardScaler()
        action_scaler.fit(action_raw)
        print(f"  Scaler fit on {len(action_raw):,} action frames")

        cache_mixer = CacheTransitionMixer(
            all_latents=all_latents,
            all_actions=all_actions,
            action_scaler=action_scaler,
            done_threshold=done_threshold,
            future_p=0.8,
            reward_mode=args.reward_mode,
            device=device,
            frameskip=5,
            pca_mean=pca_mean,
            pca_matrix=pca_matrix,
        )
    elif args.cache_ratio > 0.0 and not all_actions:
        print("WARNING: --cache_ratio > 0 but cache has no 'all_actions'. "
              "Rebuild cache with analyse_lewm_224.py. Falling back to WM rollouts only.")

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use,
                    device=device, cache_path=cache_path if not DEBUG_MODE else None,
                    done_threshold=done_threshold)

    # When PCA is active, the buffer stores pca_dim latents. The env still operates
    # in 192D (WM predictor space), but latents are projected before buffer storage.
    # For pure offline (cache_ratio=1.0), the buffer is never used for SAC updates.
    replay_buffer = VectorizedEpisodicHERBuffer(
        latent_dim=rl_latent_dim, action_dim=25, capacity_episodes=20000, max_t=50,
        future_p=0.8, device=device,
        done_threshold=done_threshold,
        dense_reward_scale=6.29,  # 1 WM-step distance; avoids clipping most improvements to ±2
    )

    # Actor, critic, and high-level actor operate in rl_latent_dim (pca_dim or 192)
    actor         = GoalConditionedActor(latent_dim=rl_latent_dim, action_dim=25).to(device)
    critic        = TwinCritic(latent_dim=rl_latent_dim, action_dim=25).to(device)
    critic_target = TwinCritic(latent_dim=rl_latent_dim, action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())
    print(f"Networks: latent_dim={rl_latent_dim} (GoalConditionedActor, TwinCritic)")

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features)

    # log_alpha: start at ~0.13 (exp(-2.0)) but may be overridden by fixed_alpha
    log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    fixed_alpha = args.fixed_alpha  # None = auto-tune, float = fixed value

    # --- High-level actor (joint mode only) ---
    high_actor = None
    high_actor_optimizer = None
    if args.mode == 'joint':
        high_actor = HighLevelActor(latent_dim=rl_latent_dim, hidden_dim=512).to(device)
        if args.high_actor_path is not None:
            ckpt_hl = torch.load(args.high_actor_path, map_location=device, weights_only=False)
            # Only load if the latent_dim matches (warm-start only if same architecture)
            try:
                high_actor.load_state_dict(ckpt_hl, strict=True)
                print(f"High-level actor warm-started from {args.high_actor_path}")
            except RuntimeError as e:
                print(f"WARNING: Could not warm-start high actor (dimension mismatch?): {e}")
                print("  Starting high-level actor from scratch.")
        else:
            print(f"High-level actor initialised from scratch (latent_dim={rl_latent_dim}).")
        high_actor_optimizer = torch.optim.Adam(high_actor.parameters(), lr=3e-4)

    train_loop(
        env=env,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
        replay_buffer=replay_buffer,
        all_latents=all_latents,
        num_iterations=num_iters_to_run,
        gamma=0.99,
        tau=0.005,
        save_dir=save_dir,
        mode=args.mode,
        fixed_gap=args.gap if args.mode == 'fixed' else (args.high_level_gap if args.mode == 'joint' else None),
        tmax_pure_distance=args.tmax,
        reward_mode=args.reward_mode,
        bc_model=bc_model,
        bc_alpha=args.bc_alpha,
        high_actor=high_actor,
        high_actor_optimizer=high_actor_optimizer,
        high_level_beta=args.high_level_beta,
        high_level_gap=args.high_level_gap,
        # New cache-offline params
        cache_mixer=cache_mixer,
        cache_ratio=args.cache_ratio,
        bc_alpha_cache=args.bc_alpha_cache,
        fixed_alpha=fixed_alpha,
        pca_mean=pca_mean,
        pca_matrix=pca_matrix,
    )
