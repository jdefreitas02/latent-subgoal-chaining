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

        # Multiply by the new scale so the agent can reach the expert's top speed
        action = y_t * self.action_scale

        log_prob = normal.log_prob(x_t)
        # Correct the calculus for the log probability density to account for the wider scale
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

        # The reward is -L2_distance in 192-D latent space, which can be large (e.g. 5-50).
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

class StandardReplayBuffer:
    """A highly optimized tensor-based flat replay buffer natively stored on GPU."""
    def __init__(self, latent_dim=192, action_dim=25, capacity=1000000, device="cuda"):
        self.capacity = capacity
        self.device = device

        # Pre-allocate directly on the GPU. At 1M transitions, this is only ~1.5GB of VRAM!
        self.z_curr = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.z_next = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.z_target = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)

        self.position = 0
        self.size = 0

    def store_transitions(self, z_curr, actions, z_next, z_target, rewards, dones):
        batch_size = z_curr.shape[0]
        end_idx = self.position + batch_size

        # Keep everything on device. No .cpu() syncs!
        z_curr_dev = z_curr.detach()
        actions_dev = actions.detach()
        z_next_dev = z_next.detach()
        z_target_dev = z_target.detach()
        rewards_dev = rewards.detach()
        dones_dev = dones.detach()

        # Handle writing to the tensor (including wrap-around if buffer is full)
        if end_idx <= self.capacity:
            self.z_curr[self.position:end_idx] = z_curr_dev
            self.actions[self.position:end_idx] = actions_dev
            self.z_next[self.position:end_idx] = z_next_dev
            self.z_target[self.position:end_idx] = z_target_dev
            self.rewards[self.position:end_idx] = rewards_dev
            self.dones[self.position:end_idx] = dones_dev
        else:
            overflow = end_idx - self.capacity
            valid = batch_size - overflow

            # Fill to the end of the buffer
            self.z_curr[self.position:self.capacity] = z_curr_dev[:valid]
            self.actions[self.position:self.capacity] = actions_dev[:valid]
            self.z_next[self.position:self.capacity] = z_next_dev[:valid]
            self.z_target[self.position:self.capacity] = z_target_dev[:valid]
            self.rewards[self.position:self.capacity] = rewards_dev[:valid]
            self.dones[self.position:self.capacity] = dones_dev[:valid]

            # Wrap around and fill the beginning
            self.z_curr[0:overflow] = z_curr_dev[valid:]
            self.actions[0:overflow] = actions_dev[valid:]
            self.z_next[0:overflow] = z_next_dev[valid:]
            self.z_target[0:overflow] = z_target_dev[valid:]
            self.rewards[0:overflow] = rewards_dev[valid:]
            self.dones[0:overflow] = dones_dev[valid:]

        self.position = end_idx % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample_batch(self, batch_size=256):
        # Sampling is now instantaneous because everything is already on the GPU
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.z_curr[idxs],
            self.actions[idxs],
            self.z_next[idxs],
            self.z_target[idxs],
            self.rewards[idxs],
            self.dones[idxs]
        )

def sample_tasks(all_latents, batch_size, device, mode,
                 fixed_gap=None, gap_schedule=None, current_stage=None):
    """
    Samples (start, goal) pairs according to the training mode.

    Modes:
      'pure_distance' — start=ep[0], goal=ep[-1]; no subgoal structure, reward is dense
                        distance to the true episode endpoint.
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
            z_starts.append(ep[0])
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
        # Stage N only draws from the first 50*(N+1) episodes. This keeps the local
        # dynamics roughly constant until the agent masters them.
        #pool_size = min(num_eps, 50 * (current_stage + 1))
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
               mode='curriculum', fixed_gap=None, tmax_pure_distance=40):

    os.makedirs(save_dir, exist_ok=True)
    mode_str = mode if mode != 'fixed' else f"fixed (gap={fixed_gap})"
    print(f"Starting Training | Mode: {mode_str} | Saving to: {save_dir}")

    # Initialize the CSV Logger for paper statistics
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Stage", "Gap", "Success_Rate", "Buffer_Size",
                         "Actor_Loss", "Critic_Loss", "EnvStep_Time", "BufStore_Time",
                         "Train_Time", "Total_Time"])

    # gap_schedule / tmax_schedule are only active in 'curriculum' mode.
    # Each gap is measured in world-model steps (1 WM step = 5 video frames).
    # T_max = gap gives the agent exactly as many steps as the goal is away,
    # starting trivially tight (gap=1) and growing as the agent succeeds.
    gap_schedule  = [1, 2, 4, 8, 16, 24, 32, 40]
    tmax_schedule = [1, 2, 4, 8, 16, 24, 32, 40]
    stage = 0
    recent_successes = deque(maxlen=2000)  # Tracks success rate over recent attempts

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

        # --- TASK GENERATION ---
        z_curr, z_target, gaps_used = sample_tasks(
            all_latents, env.num_envs, env.device,
            mode=mode, fixed_gap=fixed_gap,
            gap_schedule=gap_schedule, current_stage=stage,
        )

        # Teleport environments to the sampled starting states and set curriculum goal
        env.set_states(z_curr)
        env.z_ultimate_goal = z_target

        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # --- TELEMETRY TRACKERS ---
        rollout_act_mags = []
        avg_start_dist = 0.0
        avg_end_dist = 0.0

        # 1. Rollout for T_max steps (entire loop in no_grad to prevent autograd leaks)
        with torch.no_grad():
            for step in range(T_max):
                if not active_mask.any():
                    break

                if step == 0:
                    avg_start_dist = torch.norm(z_curr - z_target, p=2, dim=-1).mean().item()

                actions, _, _ = actor.sample(z_curr, z_target)
                rollout_act_mags.append(torch.norm(actions, p=2, dim=-1).mean().item())

                # --- TIMER: ENV STEP (Model Inference) ---
                t0 = time.time()
                z_next, rewards, dones, _, _ = env.step(actions)
                env_step_accum += (time.time() - t0)

                if step == T_max - 1:
                    avg_end_dist = torch.norm(z_curr - z_target, p=2, dim=-1).mean().item()

                active_z_curr   = z_curr[active_mask]
                active_actions  = actions[active_mask]
                active_z_next   = z_next[active_mask]
                active_z_target = z_target[active_mask]
                active_dones    = dones[active_mask]

                # --- ACTION REGULARIZATION ---
                # Penalty for large actions keeps the agent inside the expert distribution
                action_penalty = torch.norm(active_actions, p=2, dim=-1) * 0.01
                active_rewards = rewards[active_mask] - action_penalty

                # --- TIMER: BUFFER STORE ---
                t1 = time.time()
                replay_buffer.store_transitions(
                    active_z_curr, active_actions, active_z_next,
                    active_z_target, active_rewards, active_dones
                )
                buf_store_accum += (time.time() - t1)

                just_succeeded = dones & active_mask
                success_mask  |= just_succeeded
                active_mask   &= ~just_succeeded
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)

        # --- CURRICULUM ADVANCEMENT (curriculum mode only) ---
        if mode == 'curriculum':
            # Only track success for the current hardest tasks (ignore the 20% easy replay)
            hard_task_mask = (gaps_used == current_target_gap)
            if hard_task_mask.any():
                recent_successes.extend(success_mask[hard_task_mask].cpu().tolist())

            current_sr = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0

            # Require a full window of 2000 attempts and SR > 85% to advance
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

        # --- 2. SAC Training Phase ---
        iter_train_start = time.time()

        avg_actor_loss = 0.0
        avg_critic_loss = 0.0

        if replay_buffer.size >= 256:
            for _ in range(40):
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(batch_size=256)

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

        train_time_accum += (time.time() - iter_train_start)

        if iteration % 10 == 0:
            actor_val  = avg_actor_loss / 40.0 if replay_buffer.size >= 256 else 0.0
            critic_val = avg_critic_loss / 40.0 if replay_buffer.size >= 256 else 0.0
            elapsed_time = time.time() - start_time
            avg_act_mag  = np.mean(rollout_act_mags) if rollout_act_mags else 0.0

            print(f"Iter {iteration:04d} | SR: {current_sr*100:.1f}% | "
                  f"Act Loss: {actor_val:.1f} | Crit Loss: {critic_val:.1f} | "
                  f"ActMag: {avg_act_mag:.2f} | StartD: {avg_start_dist:.2f} | EndD: {avg_end_dist:.2f}")

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, stage, current_target_gap, current_sr,
                                 replay_buffer.size, actor_val, critic_val,
                                 env_step_accum, buf_store_accum, train_time_accum, elapsed_time])

            start_time = time.time()
            env_step_accum  = 0.0
            buf_store_accum = 0.0
            train_time_accum = 0.0

        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(),  os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            print(f"--> Checkpoint saved at Iteration {iteration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent World Model SAC Training")
    parser.add_argument(
        '--mode', type=str, default='curriculum',
        choices=['pure_distance', 'fixed', 'curriculum'],
        help=(
            "Training mode:\n"
            "  pure_distance — no subgoals; reward is dense distance to the true episode endpoint.\n"
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DERIVE CHECKPOINT DIRECTORY FROM MODE ---
    if args.mode == 'pure_distance':
        save_dir = "./checkpoints_pure_distance"
    elif args.mode == 'fixed':
        save_dir = f"./checkpoints_fixed_gap_{args.gap}"
    else:
        save_dir = "./checkpoints_curriculum"

    DEBUG_MODE = False

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
        num_envs_to_use = 50
        num_iters_to_run = 25000

        ephemeral = os.environ.get("EPHEMERAL")
        if ephemeral is None:
            raise ValueError("EPHEMERAL environment variable is not set")

        data_path = f"{ephemeral}/stable_wm_data/ogbench/cube_single_expert.h5"
        ckpt_path = f"{ephemeral}/stable_wm_data/cube/lejepa_weights.ckpt"

        print(f"Loading Dataset from:    {data_path}")
        print(f"Loading Checkpoint from: {ckpt_path}")

        parent_dir = os.path.abspath(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="eval/cube", overrides=["+policy=cube/lejepa"])

        dataset = swm.data.HDF5Dataset(
            f"{ephemeral}/stable_wm_data/ogbench/cube_single_expert"
        )

        jepa_model = swm.policy.AutoCostModel(cfg.policy)
        jepa_model = jepa_model.to(device)
        jepa_model.eval()
        for param in jepa_model.parameters():
            param.requires_grad = False

        print("Dataset and JEPA Model successfully loaded")

        cache_path = os.path.join(ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
        print(f"Loading latents cache from {cache_path}...")
        cache = torch.load(cache_path)
        all_latents = cache['all_latents']

    # Allow --num_iters to override the default
    if args.num_iters is not None:
        num_iters_to_run = args.num_iters

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use, device=device)

    replay_buffer = StandardReplayBuffer(latent_dim=192, action_dim=25, capacity=1000000, device=device)

    actor         = GoalConditionedActor(action_dim=25).to(device)
    critic        = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features)

    # Start alpha at ~0.13 instead of 1.0 so Q-values matter immediately
    log_alpha      = torch.tensor([-2.0], requires_grad=True, device=device)
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
    )
