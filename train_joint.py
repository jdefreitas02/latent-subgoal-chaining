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

# Import your vectorized environment
from latent_env import LatentEnv
from bc_policy import BCPolicy

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
               high_level_beta=3.0, high_level_gap=8):

    os.makedirs(save_dir, exist_ok=True)
    mode_str = mode if mode != 'fixed' else f"fixed (gap={fixed_gap})"
    bc_str = f"BC alpha={bc_alpha}" if bc_model is not None else "no BC"
    print(f"Starting HER Training | Mode: {mode_str} | Reward: {reward_mode} | {bc_str} | Saving to: {save_dir}")

    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Stage", "Gap", "Success_Rate", "Episodes_in_Buf",
                         "Transitions_in_Buf", "Actor_Loss", "Critic_Loss", "HL_Loss",
                         "EnvStep_Time", "BufStore_Time", "Train_Time", "Total_Time"])

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

        assert T_max <= replay_buffer.max_t, (
            f"T_max={T_max} exceeds the buffer's max_t={replay_buffer.max_t}. "
            "Increase max_t when constructing VectorizedEpisodicHERBuffer."
        )

        # --- TASK GENERATION ---
        z_curr, z_target, gaps_used = sample_tasks(
            all_latents, env.num_envs, env.device,
            mode=mode, fixed_gap=fixed_gap,
            gap_schedule=gap_schedule, current_stage=stage,
        )

        env.set_states(z_curr)

        # --- HIGH-LEVEL SUBGOAL GENERATION (joint mode only) ---
        # In joint mode z_target = ep[-1] (the ultimate goal). The high-level actor
        # predicts an intermediate subgoal that is gap steps ahead. The env pursues
        # z_subgoal (reward/done relative to it). The buffer stores BOTH goals:
        #   low_goal  = z_subgoal  (for SAC: critic + actor train on this)
        #   ultimate_g = z_target  (for high-level AWR advantage computation)
        if mode == 'joint' and high_actor is not None:
            with torch.no_grad():
                z_subgoal = high_actor(z_curr, z_target)  # [B, latent_dim]
            env.z_ultimate_goal = z_subgoal   # env done/reward toward subgoal
            low_level_goal = z_subgoal        # low-level actor target
        else:
            env.z_ultimate_goal = z_target
            low_level_goal = z_target

        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        rollout_act_mags = []
        avg_start_dist = 0.0
        avg_end_dist = 0.0

        # Pre-allocate temporary trajectory memory for blazing fast episodic collection
        rollout_z_curr    = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_actions   = torch.zeros((env.num_envs, T_max, 25),  dtype=torch.float32, device=env.device)
        rollout_z_next    = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_low_goal  = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_ultimate  = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_lengths   = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        with torch.no_grad():
            for step in range(T_max):
                if not active_mask.any():
                    break

                if step == 0:
                    avg_start_dist = torch.norm(z_curr - low_level_goal, p=2, dim=-1).mean().item()

                actions, _, _ = actor.sample(z_curr, low_level_goal)
                rollout_act_mags.append(torch.norm(actions, p=2, dim=-1).mean().item())

                t0 = time.time()
                z_next, _, dones, _, _ = env.step(actions)
                env_step_accum += (time.time() - t0)

                t1 = time.time()

                # Record data ONLY for environments that are still active
                rollout_z_curr[active_mask, step]    = z_curr[active_mask]
                rollout_actions[active_mask, step]    = actions[active_mask]
                rollout_z_next[active_mask, step]     = z_next[active_mask]
                # low_goal = the subgoal the low-level actor is actually pursuing
                # (z_subgoal in joint mode, z_target otherwise).  SAC trains on this.
                rollout_low_goal[active_mask, step]   = low_level_goal[active_mask]
                # ultimate_g = ep[-1], the long-horizon goal. Only used by HL AWR.
                rollout_ultimate[active_mask, step]    = z_target[active_mask]
                rollout_lengths[active_mask] += 1

                buf_store_accum += (time.time() - t1)

                just_succeeded = dones & active_mask
                success_mask  |= just_succeeded
                active_mask   &= ~just_succeeded
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)

            avg_end_dist = torch.norm(z_curr - low_level_goal, p=2, dim=-1).mean().item()

        # Inject the entire rollout batch into the episodic HER buffer at once
        t2 = time.time()
        replay_buffer.store_episodes(
            rollout_z_curr, rollout_actions, rollout_z_next,
            rollout_low_goal, rollout_ultimate, rollout_lengths
        )
        buf_store_accum += (time.time() - t2)

        # --- CURRICULUM ADVANCEMENT (curriculum mode only) ---
        if mode == 'curriculum':
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

        avg_actor_loss = 0.0
        avg_critic_loss = 0.0

        # Dense reward provides a richer signal per sample so fewer updates per
        # iteration are needed; more importantly, fewer updates prevent the critic
        # from diverging through aggressive bootstrapping of shaped rewards.
        grad_updates = 20 if reward_mode == 'dense' else 40

        if replay_buffer.num_transitions >= 256:
            for _ in range(grad_updates):
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(batch_size=256, reward_mode=reward_mode)

                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                # Update Critic
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.sample(z_next_b, g_b)
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

                # Behavioural cloning regularisation: pull actor toward the
                # BC policy's deterministic action for the same (state, goal).
                if bc_model is not None:
                    with torch.no_grad():
                        bc_actions = bc_model(z_b, g_b)
                    bc_loss = F.mse_loss(new_actions, bc_actions)
                    actor_loss = actor_loss + bc_alpha * bc_loss

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                avg_actor_loss += actor_loss.item()

                for p in critic.parameters():
                    p.requires_grad = True

                # Update Alpha
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
        avg_hl_loss = 0.0
        hl_updates = 10
        if mode == 'joint' and high_actor is not None and high_actor_optimizer is not None \
                and replay_buffer.num_transitions >= 256:
            # gap in dataset frames = high_level_gap * FRAMESKIP(=5)
            hl_frame_gap = high_level_gap * 5
            for _ in range(hl_updates):
                z_t, z_t_gap, g_ult = sample_high_level_from_cache(
                    all_latents, 256, hl_frame_gap, device=replay_buffer.device
                )

                with torch.no_grad():
                    # Distance-based advantage: closer to goal = higher advantage
                    dist_t    = torch.norm(z_t     - g_ult, p=2, dim=-1)   # [B]
                    dist_tgap = torch.norm(z_t_gap - g_ult, p=2, dim=-1)   # [B]
                    adv = dist_t - dist_tgap                                # positive if z_t_gap closer
                    w   = torch.exp(high_level_beta * adv).clamp(max=100.0)

                pred    = high_actor(z_t, g_ult)                                        # [B, 192]
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
            actor_val  = avg_actor_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            critic_val = avg_critic_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            elapsed_time = time.time() - start_time
            avg_act_mag = np.mean(rollout_act_mags) if rollout_act_mags else 0.0
            hl_str = f" | HL Loss: {avg_hl_loss:.4f}" if mode == 'joint' else ""

            print(f"Iter {iteration:04d} | SR: {current_sr*100:.1f}% | "
                  f"Buf Eps: {replay_buffer.size} Tr: {replay_buffer.num_transitions} | "
                  f"Act Loss: {actor_val:.1f} | Crit Loss: {critic_val:.1f} | "
                  f"ActMag: {avg_act_mag:.2f} | StartD: {avg_start_dist:.2f} | EndD: {avg_end_dist:.2f}"
                  f"{hl_str}")

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, stage, current_target_gap, current_sr,
                                 replay_buffer.size, replay_buffer.num_transitions,
                                 actor_val, critic_val, avg_hl_loss,
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
        help="BC regularisation coefficient added to the actor loss (0 = disabled).",
    )
    parser.add_argument(
        '--bc_model_path', type=str, default=None,
        help="Path to a BCPolicy checkpoint (.pth) produced by train_bc.py. "
             "Required when --bc_alpha > 0.",
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

    if args.bc_alpha > 0 and args.bc_model_path is None:
        parser.error("--bc_model_path is required when --bc_alpha > 0")
    if args.mode == 'joint' and args.gap != 8:
        # --gap is for 'fixed' mode; in joint mode --high_level_gap controls the horizon
        pass  # --gap is ignored for joint mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DERIVE CHECKPOINT DIRECTORY FROM MODE, REWARD, AND BC ---
    reward_tag = f"_{args.reward_mode}"
    bc_tag     = f"_bc{args.bc_alpha}" if args.bc_alpha > 0 else ""
    if args.mode == 'pure_distance':
        save_dir = f"./checkpoints_her_pure_distance{reward_tag}{bc_tag}"
    elif args.mode == 'fixed':
        save_dir = f"./checkpoints_her_fixed_gap_{args.gap}{reward_tag}{bc_tag}"
    elif args.mode == 'joint':
        save_dir = f"./checkpoints_joint_gap_{args.high_level_gap}_beta{args.high_level_beta}{reward_tag}{bc_tag}"
    else:
        save_dir = f"./checkpoints_her_curriculum{reward_tag}{bc_tag}"

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

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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

    # Allow --num_iters to override the default
    if args.num_iters is not None:
        num_iters_to_run = args.num_iters

    # --- Load BC model (frozen) if requested ---
    bc_model = None
    if args.bc_alpha > 0:
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

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use,
                    device=device, cache_path=cache_path if not DEBUG_MODE else None,
                    done_threshold=done_threshold)

    # max_t=50 comfortably fits the largest T_max of 40 across all modes
    replay_buffer = VectorizedEpisodicHERBuffer(
        latent_dim=192, action_dim=25, capacity_episodes=20000, max_t=50, future_p=0.8, device=device,
        done_threshold=done_threshold,
        dense_reward_scale=6.29,  # 1 WM-step distance; avoids clipping most improvements to ±2
    )

    actor         = GoalConditionedActor(action_dim=25).to(device)
    critic        = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features)

    # Start alpha at ~0.13 instead of 1.0 so Q-values matter immediately
    log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

    # --- High-level actor (joint mode only) ---
    high_actor = None
    high_actor_optimizer = None
    if args.mode == 'joint':
        high_actor = HighLevelActor(latent_dim=192, hidden_dim=512).to(device)
        if args.high_actor_path is not None:
            ckpt_hl = torch.load(args.high_actor_path, map_location=device, weights_only=False)
            # MLPHighLevel stores weights under 'net.*'; HighLevelActor uses the same keys
            high_actor.load_state_dict(ckpt_hl, strict=True)
            print(f"High-level actor warm-started from {args.high_actor_path}")
        else:
            print("High-level actor initialised from scratch (no warm-start checkpoint provided).")
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
    )
