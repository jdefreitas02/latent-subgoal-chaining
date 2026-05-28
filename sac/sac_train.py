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

class VectorizedEpisodicHERBuffer:
    """100% GPU-native, highly optimized 3D Tensor Buffer with dynamic HER sampling.

    All three training modes store full episodes here. During sample_batch, HER
    relabels a fraction (future_p) of goals with a future state from the same
    episode, then computes sparse rewards on the fly. This applies equally to
    pure_distance, fixed-gap, and curriculum modes — the mode only affects how
    (start, goal) pairs are drawn at rollout time.
    """
    def __init__(self, latent_dim=192, action_dim=25, capacity_episodes=20000, max_t=50, future_p=0.8, device="cuda"):
        self.capacity = capacity_episodes
        self.max_t = max_t
        self.future_p = future_p
        self.device = device

        # Pre-allocate all memory instantly
        self.z_curr = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity_episodes, max_t, action_dim), dtype=torch.float32, device=device)
        self.z_next = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.original_g = torch.zeros((capacity_episodes, max_t, latent_dim), dtype=torch.float32, device=device)
        self.ep_lens = torch.zeros((capacity_episodes,), dtype=torch.long, device=device)

        self.position = 0
        self.size = 0

    @property
    def num_transitions(self):
        """Total number of individual (s,a,s') transitions stored across all episodes."""
        return int(self.ep_lens[:self.size].sum().item())

    def store_episodes(self, z_curr_seq, actions_seq, z_next_seq, target_seq, lengths):
        num_new = z_curr_seq.shape[0]
        seq_len = z_curr_seq.shape[1]
        end_idx = self.position + num_new

        # Vectorized wrap-around injection
        if end_idx <= self.capacity:
            self.z_curr[self.position:end_idx, :seq_len] = z_curr_seq
            self.actions[self.position:end_idx, :seq_len] = actions_seq
            self.z_next[self.position:end_idx, :seq_len] = z_next_seq
            self.original_g[self.position:end_idx, :seq_len] = target_seq
            self.ep_lens[self.position:end_idx] = lengths
        else:
            overflow = end_idx - self.capacity
            valid = num_new - overflow

            self.z_curr[self.position:self.capacity, :seq_len] = z_curr_seq[:valid]
            self.actions[self.position:self.capacity, :seq_len] = actions_seq[:valid]
            self.z_next[self.position:self.capacity, :seq_len] = z_next_seq[:valid]
            self.original_g[self.position:self.capacity, :seq_len] = target_seq[:valid]
            self.ep_lens[self.position:self.capacity] = lengths[:valid]

            self.z_curr[0:overflow, :seq_len] = z_curr_seq[valid:]
            self.actions[0:overflow, :seq_len] = actions_seq[valid:]
            self.z_next[0:overflow, :seq_len] = z_next_seq[valid:]
            self.original_g[0:overflow, :seq_len] = target_seq[valid:]
            self.ep_lens[0:overflow] = lengths[valid:]

        self.position = end_idx % self.capacity
        self.size = min(self.size + num_new, self.capacity)

    def sample_batch(self, batch_size=256, reward_mode='sparse'):
        # 1. Instantly sample random episodes and a random valid timestep per episode
        ep_idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        sampled_lens = self.ep_lens[ep_idxs]

        safe_lens = torch.clamp(sampled_lens, min=1)
        t_idxs = (torch.rand(batch_size, device=self.device) * safe_lens).long()
        t_idxs = torch.clamp(t_idxs, max=safe_lens - 1)

        # 2. Gather base transitions
        z_curr_b = self.z_curr[ep_idxs, t_idxs]
        a_b = self.actions[ep_idxs, t_idxs]
        z_next_b = self.z_next[ep_idxs, t_idxs]
        orig_g_b = self.original_g[ep_idxs, t_idxs]

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

        # Swap goals based on mask (HER relabelling applies regardless of reward mode)
        g_target_b = torch.where(her_mask.unsqueeze(-1), future_g_b, orig_g_b)

        # 4. Compute reward based on mode
        action_penalty = torch.norm(a_b, p=2, dim=-1) * 0.01
        dist_next = torch.norm(z_next_b - g_target_b, p=2, dim=-1)
        success = dist_next < 2.0
        done_b = success.to(torch.float32)

        if reward_mode == 'dense':
            # Potential-based shaping: reward = improvement in distance toward goal,
            # normalised by average 1-step latent distance (1.6) so scale ~= 1.0 per step.
            # Clipped to [-2, 2] to prevent noisy transitions from producing large rewards
            # that cause critic divergence through bootstrapping.
            dist_curr = torch.norm(z_curr_b - g_target_b, p=2, dim=-1)
            improvement = (dist_curr - dist_next) / 1.6
            improvement = torch.clamp(improvement, -2.0, 2.0)
            r_b = improvement - action_penalty
        else:  # sparse
            r_b = torch.where(success, torch.zeros_like(dist_next), -torch.ones_like(dist_next)) - action_penalty

        return z_curr_b, a_b, z_next_b, g_target_b, r_b, done_b


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

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose from: pure_distance, fixed, curriculum")

    return (
        torch.stack(z_starts).to(device),
        torch.stack(z_targets).to(device),
        torch.tensor(gaps_used, device=device),
    )


def train_loop(env, actor, critic, critic_target,
               actor_optimizer, critic_optimizer, alpha_optimizer,
               log_alpha, target_entropy, replay_buffer, all_latents,
               num_iterations=1000, gamma=0.99, tau=0.005,
               save_dir="./checkpoints",
               mode='curriculum', fixed_gap=None, tmax_pure_distance=40,
               reward_mode='sparse', bc_model=None, bc_alpha=0.0):

    os.makedirs(save_dir, exist_ok=True)
    mode_str = mode if mode != 'fixed' else f"fixed (gap={fixed_gap})"
    bc_str = f"BC alpha={bc_alpha}" if bc_model is not None else "no BC"
    print(f"Starting HER Training | Mode: {mode_str} | Reward: {reward_mode} | {bc_str} | Saving to: {save_dir}")

    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Stage", "Gap", "Success_Rate", "Episodes_in_Buf",
                         "Transitions_in_Buf", "Actor_Loss", "Critic_Loss",
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
        env.z_ultimate_goal = z_target

        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        rollout_act_mags = []
        avg_start_dist = 0.0
        avg_end_dist = 0.0

        # Pre-allocate temporary trajectory memory for blazing fast episodic collection
        rollout_z_curr   = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_actions  = torch.zeros((env.num_envs, T_max, 25),  dtype=torch.float32, device=env.device)
        rollout_z_next   = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_z_target = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_lengths  = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        with torch.no_grad():
            for step in range(T_max):
                if not active_mask.any():
                    break

                if step == 0:
                    avg_start_dist = torch.norm(z_curr - z_target, p=2, dim=-1).mean().item()

                actions, _, _ = actor.sample(z_curr, z_target)
                rollout_act_mags.append(torch.norm(actions, p=2, dim=-1).mean().item())

                t0 = time.time()
                z_next, _, dones, _, _ = env.step(actions)
                env_step_accum += (time.time() - t0)

                t1 = time.time()

                # Record data ONLY for environments that are still active
                rollout_z_curr[active_mask, step]   = z_curr[active_mask]
                rollout_actions[active_mask, step]   = actions[active_mask]
                rollout_z_next[active_mask, step]    = z_next[active_mask]
                rollout_z_target[active_mask, step]  = z_target[active_mask]
                rollout_lengths[active_mask] += 1

                buf_store_accum += (time.time() - t1)

                just_succeeded = dones & active_mask
                success_mask  |= just_succeeded
                active_mask   &= ~just_succeeded
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)

            avg_end_dist = torch.norm(z_curr - z_target, p=2, dim=-1).mean().item()

        # Inject the entire rollout batch into the episodic HER buffer at once
        t2 = time.time()
        replay_buffer.store_episodes(
            rollout_z_curr, rollout_actions, rollout_z_next, rollout_z_target, rollout_lengths
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

                # Update Alpha — clamp so α never drops below 0.01
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=-4.6)   # exp(-4.6) ≈ 0.01

                # Soft Update Target Networks
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        train_time_accum += (time.time() - iter_train_start)

        if iteration % 10 == 0:
            actor_val  = avg_actor_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            critic_val = avg_critic_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            elapsed_time = time.time() - start_time
            avg_act_mag = np.mean(rollout_act_mags) if rollout_act_mags else 0.0

            print(f"Iter {iteration:04d} | SR: {current_sr*100:.1f}% | "
                  f"Buf Eps: {replay_buffer.size} Tr: {replay_buffer.num_transitions} | "
                  f"Act Loss: {actor_val:.1f} | Crit Loss: {critic_val:.1f} | "
                  f"ActMag: {avg_act_mag:.2f} | StartD: {avg_start_dist:.2f} | EndD: {avg_end_dist:.2f}")

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, stage, current_target_gap, current_sr,
                                 replay_buffer.size, replay_buffer.num_transitions,
                                 actor_val, critic_val,
                                 env_step_accum, buf_store_accum, train_time_accum, elapsed_time])

            start_time = time.time()
            env_step_accum   = 0.0
            buf_store_accum  = 0.0
            train_time_accum = 0.0

        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(),  os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            print(f"--> Checkpoint saved at Iteration {iteration}")

def _load_jepa_from_ckpt(ckpt_path, device):
    """Load an OGBench-style pytorch-lightning state_dict checkpoint into a bare JEPA model.

    The checkpoint stores weights under 'state_dict' with a 'model.' prefix added by
    the spt.Module wrapper. This mirrors the loading logic in analyse_ogbench_wm.py.
    Architecture is fixed to the OGBench config: ViT-Tiny, patch_size=8, img_size=64.
    """
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=8, image_size=64, pretrained=False, use_mask_token=False
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
    print(f"  Loaded OGBench JEPA from {ckpt_path} (epoch {epoch}, step {step})")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent World Model SAC + HER Training")
    parser.add_argument(
        '--mode', type=str, default='curriculum',
        choices=['pure_distance', 'fixed', 'curriculum'],
        help=(
            "Training mode:\n"
            "  pure_distance — no subgoals; start=ep[0], goal=ep[-1]. HER relabels with\n"
            "                  visited future states, providing implicit dense sub-goals.\n"
            "  fixed         — constant WM-step gap between start and goal throughout training.\n"
            "  curriculum    — adaptive gap that increases once the agent masters the current level."
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
            "  dense  — potential-based shaping: reward = clamp((dist_curr - dist_next)/1.6, -2, 2).\n"
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
        '--ckpt_path', type=str, default=None,
        help="Path to JEPA checkpoint (.ckpt). "
             "Default: $HOME/stable_wm_data/cube/lejepa_weights.ckpt",
    )
    parser.add_argument(
        '--cache_path', type=str, default=None,
        help="Path to precomputed latents cache (.pt). "
             "Default: $HOME/stable_wm_data/cube_all_latents_cache.pt",
    )
    parser.add_argument(
        '--dataset_path', type=str, default=None,
        help="Path to HDF5 dataset (without .h5 extension). "
             "Default: $HOME/stable_wm_data/ogbench/cube_single_expert",
    )
    args = parser.parse_args()

    if args.bc_alpha > 0 and args.bc_model_path is None:
        parser.error("--bc_model_path is required when --bc_alpha > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DERIVE CHECKPOINT DIRECTORY FROM MODE, REWARD, AND BC ---
    reward_tag = f"_{args.reward_mode}"
    bc_tag     = f"_bc{args.bc_alpha}" if args.bc_alpha > 0 else ""
    if args.mode == 'pure_distance':
        save_dir = f"./checkpoints_her_pure_distance{reward_tag}{bc_tag}"
    elif args.mode == 'fixed':
        save_dir = f"./checkpoints_her_fixed_gap_{args.gap}{reward_tag}{bc_tag}"
    else:
        save_dir = f"./checkpoints_her_curriculum{reward_tag}{bc_tag}"

    DEBUG_MODE = False
    done_threshold = 2.0  # overridden in production based on model type

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

        data_path  = args.dataset_path or f"{stablewm_home}/ogbench/cube_single_expert"
        ckpt_path  = args.ckpt_path    or f"{stablewm_home}/cube/lejepa_weights.ckpt"
        cache_path = args.cache_path   or os.path.join(stablewm_home, "cube_all_latents_cache.pt")

        print(f"Loading Dataset from:    {data_path}")
        print(f"Loading Checkpoint from: {ckpt_path}")
        print(f"Loading Cache from:      {cache_path}")

        parent_dir = os.path.abspath(os.path.dirname(__file__))

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        dataset = swm.data.HDF5Dataset(data_path)

        if args.ckpt_path is not None:
            # Explicit path provided: load directly (used for the custom 64x64 JEPA model)
            jepa_model = _load_jepa_from_ckpt(ckpt_path, device)
            done_threshold = 2.887
        else:
            # Default: use the swm AutoCostModel (224x224 model resolved via STABLEWM_HOME).
            # AutoCostModel expects the base name (e.g. "lejepa"), appends "_object.ckpt"
            # internally — pass the directory-level path, not lejepa_weights.ckpt.
            auto_path = os.path.join(stablewm_home, "cube", "lejepa")
            jepa_model = swm.policy.AutoCostModel(auto_path)
            jepa_model = jepa_model.to(device)
            jepa_model.eval()
            for param in jepa_model.parameters():
                param.requires_grad = False
            done_threshold = 2.0

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
        latent_dim=192, action_dim=25, capacity_episodes=20000, max_t=50, future_p=0.8, device=device
    )

    actor         = GoalConditionedActor(action_dim=25).to(device)
    critic        = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features)

    # Start alpha at 1.0 (log_alpha=0) matching SB3 default — prevents
    # premature collapse under HER's inflated early Q-values (proven to
    # cause α → 0.003 within 330 iters when starting at -2.0).
    log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

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
        fixed_gap=args.gap if args.mode == 'fixed' else None,
        tmax_pure_distance=args.tmax,
        reward_mode=args.reward_mode,
        bc_model=bc_model,
        bc_alpha=args.bc_alpha,
    )
