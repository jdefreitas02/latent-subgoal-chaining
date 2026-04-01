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
    def __init__(self, latent_dim=192, action_dim=5, hidden_dim=256):
        super(GoalConditionedActor, self).__init__()
        
        input_dim = latent_dim * 2
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
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
        action = y_t * 1.0 # Action scale
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

class TwinCritic(nn.Module):
    """The Value Network: Q(z_curr, z_goal, action)"""
    def __init__(self, latent_dim=192, action_dim=5, hidden_dim=256):
        super(TwinCritic, self).__init__()
        
        input_dim = (latent_dim * 2) + action_dim
        
        self.q1_l1 = nn.Linear(input_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)
        
        self.q2_l1 = nn.Linear(input_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)
        
        q1 = F.relu(self.q1_l1(x))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        
        q2 = F.relu(self.q2_l1(x))
        q2 = F.relu(self.q2_l2(q2))
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

def train_loop(env, actor, critic, critic_target, 
               actor_optimizer, critic_optimizer, alpha_optimizer, 
               log_alpha, target_entropy, replay_buffer, 
               num_iterations=1000, T_max=50, gamma=0.99, tau=0.005, 
               save_dir="./checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting Standard Baseline Training Loop. Saving checkpoints to {save_dir}")
    
    # Initialize the CSV Logger for paper statistics
    csv_file = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Buffer_Size", "Actor_Loss", "Critic_Loss", "Reset_Time", "Action_Time", "EnvStep_Time", "BufStore_Time", "Train_Time", "Total_Time"])
    
    start_time = time.time()
    reset_time_accum = 0.0
    act_time_accum = 0.0
    env_step_accum = 0.0
    buf_store_accum = 0.0
    train_time_accum = 0.0
    
    for iteration in range(num_iterations):
        
        # --- TIMER: ENV RESET (Disk I/O and Initial Encoder Pass) ---
        t_reset = time.time()
        z_curr, _ = env.reset()
        reset_time_accum += (time.time() - t_reset)

        if env.z_ultimate_goal.dim() == 3:
            env.z_ultimate_goal = env.z_ultimate_goal.squeeze(1)
            
        z_target = env.z_ultimate_goal.clone()
        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        
        # 1. Rollout for T_max steps (ENTIRE LOOP WRAPPED IN NO_GRAD to prevent Autograd leaks)
        with torch.no_grad():
            for step in range(T_max):
                if not active_mask.any():
                    break 
                
                # --- TIMER: ACTOR SAMPLE ---
                t_act = time.time()
                actions, _, _ = actor.sample(z_curr, z_target)
                act_time_accum += (time.time() - t_act)
                
                # --- TIMER: ENV STEP (Model Inference) ---
                t0 = time.time()
                z_next, rewards, dones, _, _ = env.step(actions)
                env_step_accum += (time.time() - t0)
                
                active_z_curr = z_curr[active_mask]
                active_actions = actions[active_mask]
                active_z_next = z_next[active_mask]
                active_z_target = z_target[active_mask]
                active_rewards = rewards[active_mask]
                active_dones = dones[active_mask]
                
                # --- TIMER: BUFFER STORE ---
                t1 = time.time()
                replay_buffer.store_transitions(
                    active_z_curr, active_actions, active_z_next, 
                    active_z_target, active_rewards, active_dones
                )
                buf_store_accum += (time.time() - t1)
                
                just_succeeded = dones & active_mask
                active_mask &= ~just_succeeded
                z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)

        # --- 2. SAC Training Phase ---
        iter_train_start = time.time()
        
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        
        if replay_buffer.size >= 256: 
            for _ in range(40): 
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(batch_size=256)
                
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.float().unsqueeze(-1)
                alpha = log_alpha.exp().item()

                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.sample(z_next_b, g_b)
                    target_q1, target_q2 = critic_target(z_next_b, g_b, next_actions)
                    target_q_min = torch.min(target_q1, target_q2) - alpha * next_log_pi
                    target_q_value = r_b + (1.0 - d_b) * gamma * target_q_min

                current_q1, current_q2 = critic(z_b, g_b, a_b)
                critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                avg_critic_loss += critic_loss.item()

                new_actions, log_pi, _ = actor.sample(z_b, g_b)
                for p in critic.parameters():
                    p.requires_grad = False
                    
                q1_new, q2_new = critic(z_b, g_b, new_actions)
                q_min_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_pi - q_min_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                avg_actor_loss += actor_loss.item()
                
                for p in critic.parameters():
                    p.requires_grad = True

                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        train_time_accum += (time.time() - iter_train_start)
            
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            
            actor_val = avg_actor_loss/40 if replay_buffer.size >= 256 else 0.0
            critic_val = avg_critic_loss/40 if replay_buffer.size >= 256 else 0.0
            
            print(f"Iter {iteration:04d} | Buf: {replay_buffer.size:06d} | "
                  f"Act Loss: {actor_val:.4f} | Crit Loss: {critic_val:.4f} | "
                  f"Reset: {reset_time_accum:.2f}s | Act: {act_time_accum:.2f}s | "
                  f"EnvStep: {env_step_accum:.2f}s | BufStore: {buf_store_accum:.2f}s | "
                  f"Train: {train_time_accum:.2f}s | Total: {elapsed_time:.2f}s")
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, replay_buffer.size, actor_val, critic_val, reset_time_accum, act_time_accum, env_step_accum, buf_store_accum, train_time_accum, elapsed_time])
            
            start_time = time.time()
            reset_time_accum = 0.0
            act_time_accum = 0.0
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
    else:
        print("--- RUNNING IN PRODUCTION MODE ---")
        num_envs_to_use = 40
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

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use, device=device) 
    
    replay_buffer = StandardReplayBuffer(latent_dim=192, action_dim=25, capacity=1000000, device=device)

    actor = GoalConditionedActor(action_dim=25).to(device)
    critic = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features) 
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
        num_iterations=num_iters_to_run,
        T_max=200, 
        gamma=0.99,
        tau=0.005,
        save_dir="./checkpoints_baseline" 
    )