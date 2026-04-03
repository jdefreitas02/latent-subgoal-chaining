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
        
        action = y_t * self.action_scale 
        
        log_prob = normal.log_prob(x_t)
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
    """100% GPU-native, highly optimized 3D Tensor Buffer with dynamic HER sampling."""
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
     
    def sample_batch(self, batch_size=256):
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
        
        # --- TRUE VECTORIZED HER FIX ---
        # We must ONLY apply HER if there is a valid future step available
        valid_future = (t_idxs < sampled_lens - 1)
        her_mask = (torch.rand(batch_size, device=self.device) < self.future_p) & valid_future
        
        # Calculate how many steps are strictly AFTER t_idxs
        range_len = sampled_lens - 1 - t_idxs
        safe_range = torch.clamp(range_len, min=1)
        
        # Force offsets to be at least +1 so we never sample the immediate state!
        offsets = (torch.rand(batch_size, device=self.device) * safe_range).long() + 1
        future_t = torch.clamp(t_idxs + offsets, max=sampled_lens - 1)
        
        future_g_b = self.z_curr[ep_idxs, future_t]
        
        # Swap goals based on mask
        g_target_b = torch.where(her_mask.unsqueeze(-1), future_g_b, orig_g_b)
        
        # 4. Dynamically compute the Sparse Rewards based on the active goal
        action_penalty = torch.norm(a_b, p=2, dim=-1) * 0.01
        dist = torch.norm(z_next_b - g_target_b, p=2, dim=-1)
        success = dist < 2.0
        
        r_b = torch.where(success, torch.zeros_like(dist), -torch.ones_like(dist)) - action_penalty
        done_b = success.to(torch.float32)
        
        return z_curr_b, a_b, z_next_b, g_target_b, r_b, done_b

def sample_curriculum_tasks(all_latents, batch_size, gap_schedule, current_stage, device):
    """Dynamically samples (start, goal) pairs based on the Curriculum."""
    num_eps = len(all_latents)
    
    pool_size = min(num_eps, 50 * (current_stage + 1))
    ep_idxs = torch.randint(0, pool_size, (batch_size,))
    
    z_starts = []
    z_targets = []
    gaps_used = []
    
    for ep_idx in ep_idxs:
        ep = all_latents[ep_idx]
        ep_len = ep.shape[0]
        
        if current_stage > 0 and random.random() < 0.2:
            gap = random.choice(gap_schedule[:current_stage])
        else:
            gap = gap_schedule[current_stage]
            
        actual_frame_gap = gap * 5
        actual_frame_gap = min(actual_frame_gap, ep_len - 1)
        
        start_t = torch.randint(0, ep_len - actual_frame_gap, (1,)).item()
        target_t = start_t + actual_frame_gap
        
        z_starts.append(ep[start_t])
        z_targets.append(ep[target_t])
        gaps_used.append(gap)
        
    return torch.stack(z_starts).to(device), torch.stack(z_targets).to(device), torch.tensor(gaps_used, device=device)

def train_loop(env, actor, critic, critic_target, 
               actor_optimizer, critic_optimizer, alpha_optimizer, 
               log_alpha, target_entropy, replay_buffer, all_latents,
               num_iterations=1000, gamma=0.99, tau=0.005, 
               save_dir="./checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting Vectorized Episodic HER Loop. Saving to {save_dir}")
    
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Stage", "Gap", "Success_Rate", "Episodes_in_Buf", "Actor_Loss", "Critic_Loss", "EnvStep_Time", "BufStore_Time", "Train_Time", "Total_Time"])
    
    # --- FIX: BALANCED T_MAX MARGIN OF ERROR ---
    # 1 WM step = 5 physical frames.
    # We now give a tight but forgiving margin: 2-3 extra WM steps (10-15 extra physical frames) 
    # to allow corrective maneuvers without letting the agent wander forever.
    gap_schedule  = [2, 4,  8, 16, 24, 32, 40] 
    tmax_schedule = [4, 6, 10, 20, 30, 40, 50] 
    stage = 0
    recent_successes = deque(maxlen=2000)
    
    start_time = time.time()
    env_step_accum = 0.0
    buf_store_accum = 0.0
    train_time_accum = 0.0
    
    for iteration in range(num_iterations):
        
        current_target_gap = gap_schedule[stage]
        T_max = tmax_schedule[stage]
        
        z_curr, z_target, gaps_used = sample_curriculum_tasks(all_latents, env.num_envs, gap_schedule, stage, env.device)
        
        env.set_states(z_curr)
        env.z_ultimate_goal = z_target
        
        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        rollout_act_mags = []
        avg_start_dist = 0.0
        avg_end_dist = 0.0
        
        # Pre-allocate temporary trajectory memory for blazing fast collection
        rollout_z_curr = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_actions = torch.zeros((env.num_envs, T_max, 25), dtype=torch.float32, device=env.device)
        rollout_z_next = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_z_target = torch.zeros((env.num_envs, T_max, 192), dtype=torch.float32, device=env.device)
        rollout_lengths = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        
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
                
                # Instantly record data ONLY for environments that are still running
                rollout_z_curr[active_mask, step] = z_curr[active_mask]
                rollout_actions[active_mask, step] = actions[active_mask]
                rollout_z_next[active_mask, step] = z_next[active_mask]
                rollout_z_target[active_mask, step] = z_target[active_mask]
                rollout_lengths[active_mask] += 1
                
                buf_store_accum += (time.time() - t1)
                
                just_succeeded = dones & active_mask
                success_mask |= just_succeeded
                
                active_mask &= ~just_succeeded
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)
            
            # Record true ending distance after the loop captures all successes/failures accurately!
            avg_end_dist = torch.norm(z_curr - z_target, p=2, dim=-1).mean().item()

        # Inject the entire rollout batch into the buffer instantly
        t2 = time.time()
        replay_buffer.store_episodes(
            rollout_z_curr, rollout_actions, rollout_z_next, rollout_z_target, rollout_lengths
        )
        buf_store_accum += (time.time() - t2)

        hard_task_mask = (gaps_used == current_target_gap)
        if hard_task_mask.any():
            hard_successes = success_mask[hard_task_mask]
            recent_successes.extend(hard_successes.cpu().tolist())
            
        current_sr = np.mean(recent_successes) if len(recent_successes) > 0 else 0.0
        
        if len(recent_successes) == recent_successes.maxlen and current_sr > 0.85:
            if stage < len(gap_schedule) - 1:
                stage += 1
                recent_successes.clear()
                print(f"\n*** CURRICULUM ADVANCE: Stage {stage} | New Gap: {gap_schedule[stage]} | New T_max: {tmax_schedule[stage]} ***\n")

        # --- SAC Training Phase ---
        iter_train_start = time.time()
        
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        
        # Buffer readiness condition 
        if replay_buffer.size >= 50: 
            for _ in range(40): 
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(batch_size=256)
                
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                # Update Critic
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.sample(z_next_b, g_b)
                    target_q1, target_q2 = critic_target(z_next_b, g_b, next_actions)
                    target_q_min = torch.min(target_q1, target_q2) - alpha * next_log_pi
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
            actor_val = avg_actor_loss / 40.0 if replay_buffer.size >= 50 else 0.0
            critic_val = avg_critic_loss / 40.0 if replay_buffer.size >= 50 else 0.0
            elapsed_time = time.time() - start_time
            avg_act_mag = np.mean(rollout_act_mags) if rollout_act_mags else 0.0
                
            print(f"Iter {iteration:04d} | SR: {current_sr*100:.1f}% | Buf Eps: {replay_buffer.size} | "
                  f"Act Loss: {actor_val:.1f} | Crit Loss: {critic_val:.1f} | "
                  f"ActMag: {avg_act_mag:.2f} | StartD: {avg_start_dist:.2f} | EndD: {avg_end_dist:.2f}")
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, stage, current_target_gap, current_sr, replay_buffer.size, actor_val, critic_val, env_step_accum, buf_store_accum, train_time_accum, elapsed_time])
            
            start_time = time.time()
            env_step_accum = 0.0
            buf_store_accum = 0.0
            train_time_accum = 0.0
                
        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(), os.path.join(save_dir, "actor_policy_baseline.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network_baseline.pth"))
            print(f"--> Checkpoint saved at Iteration {iteration}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        num_iters_to_run = 10000
        
        ephemeral = os.environ.get("EPHEMERAL")
        if ephemeral is None:
            raise ValueError("EPHEMERAL environment variable is not set")
            
        data_path = f"{ephemeral}/stable_wm_data/ogbench/cube_single_expert.h5"
        ckpt_path = f"{ephemeral}/stable_wm_data/cube/lejepa_weights.ckpt"

        print(f"Loading Dataset from: {data_path}")
        print(f"Loading Checkpoint from: {ckpt_path}")

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="eval/cube", overrides=["+policy=cube/lejepa"])

        dataset = swm.data.HDF5Dataset(
            f"{ephemeral}/stable_wm_data/ogbench/cube_single_expert"
        )
        
        jepa_model = swm.policy.AutoCostModel(cfg.policy) 
        
        # Move to GPU and Freeze
        jepa_model = jepa_model.to(device)
        jepa_model.eval()
        for param in jepa_model.parameters():
            param.requires_grad = False
            
        print("Dataset and JEPA Model successfully loaded")
        
        # Load the massive latents cache for Curriculum task sampling
        cache_path = os.path.join(ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
        print(f"Loading latents cache from {cache_path}...")
        cache = torch.load(cache_path)
        all_latents = cache['all_latents']

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use, device=device) 
    
    # Pass max_t=50 to support our max T_max of 40 comfortably!
    replay_buffer = VectorizedEpisodicHERBuffer(latent_dim=192, action_dim=25, capacity_episodes=20000, max_t=50, future_p=0.8, device=device)

    actor = GoalConditionedActor(action_dim=25).to(device)
    critic = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features) 
    
    log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
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
        save_dir="./checkpoints_curriculum_2" 
    )