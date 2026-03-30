import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import numpy as np
import random
import os
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

def compute_reward(z_curr, z_target, threshold=0.5):
    """Computes sparse reward for live env and HER relabeling."""
    distances = torch.norm(z_curr - z_target, p=2, dim=-1)
    rewards = -(distances > threshold).float()
    successes = distances < threshold
    return rewards, successes

# ---------------------------------------------------------
# PHASE 1: SUBGOAL EXTRACTION
# ---------------------------------------------------------
def extract_subgoal_sequences(dataset, model, subgoal_spacing=1, num_episodes=100, device="cuda"):
    """
    Extracts frames from the dataset at fixed temporal intervals 
    and encodes them into sequences of latent subgoals.
    
    Args:
        subgoal_spacing: Number of frames between subgoals (1 = every frame, 5 = every 5th frame).
    """
    print(f"Extracting subgoals with a spacing of {subgoal_spacing} frames...")
    sequences = []
    
    num_episodes_to_extract = min(num_episodes, len(dataset.lengths))
    
    with torch.no_grad():
        for ep_idx in range(num_episodes_to_extract):
            ep_len = dataset.lengths[ep_idx]
            
            # Determine the frame indices we want to extract
            frame_indices = list(range(0, ep_len, subgoal_spacing))
            
            # Ensure the very last frame is always included as the ultimate goal
            if frame_indices[-1] != ep_len - 1:
                frame_indices.append(ep_len - 1)
                
            starts = np.array(frame_indices)
            ends = starts + 1
            
            # Request these specific frames from the HDF5 dataset
            ep_indices_array = np.full(len(starts), ep_idx)
            chunks = dataset.load_chunk(ep_indices_array, starts, ends)
            
            # Stack pixels: shape (num_frames, 1, C, H, W)
            pixels = torch.stack([chunk['pixels'] for chunk in chunks]).to(device)
            
            # Encode the frames into latent space
            encoded = model.encode({'pixels': pixels})['emb'] # Shape: (num_frames, 1, 192)
            
            # Remove the time dimension to get (num_frames, 192)
            seq = encoded.squeeze(1) 
            sequences.append(seq)
            
    return sequences


class DynamicTaskBuffer:
    """
    An active pool of (z_start, z_target) tasks.
    Agents sample from this, and return their results after T_max steps.
    """
    def __init__(self, sequences, max_size=100000, device="cuda"):
        self.sequences = sequences
        self.max_size = max_size
        self.device = device
        self.tasks = []
        
        # Phase 2 Initialization: Add all adjacent (z_i, z_{i+1}) pairs to the buffer
        for seq_idx, seq in enumerate(self.sequences):
            for i in range(len(seq) - 1):
                # Store: (z_start, z_target, seq_idx, target_idx)
                # We keep track of the indices so we know what the *next* subgoal is on success
                self.tasks.append((seq[i], seq[i+1], seq_idx, i+1))
                
    def sample_tasks(self, batch_size):
        """Randomly samples a batch of tasks for the vectorized environments."""
        sampled = random.choices(self.tasks, k=batch_size)
        
        z_starts = torch.stack([t[0] for t in sampled]).to(self.device)
        z_targets = torch.stack([t[1] for t in sampled]).to(self.device)
        seq_indices = [t[2] for t in sampled]
        target_indices = [t[3] for t in sampled]
        
        return z_starts, z_targets, seq_indices, target_indices

    def add_task(self, z_start, z_target, seq_idx, target_idx):
        """Adds a task to the pool, popping the oldest if capacity is reached."""
        if len(self.tasks) >= self.max_size:
            self.tasks.pop(0) 
        self.tasks.append((z_start, z_target, seq_idx, target_idx))

    def process_episode_results(self, z_final, successes, seq_indices, target_indices):
        """
        Evaluates how the agents did and adds 
        the appropriate new tasks back into the pool.
        """
        for i in range(len(successes)):
            seq_idx = seq_indices[i]
            target_idx = target_indices[i]
            current_z = z_final[i].detach() # The state where the agent ended up
            
            if successes[i]:
                # Agent succeeded, advance to the next subgoal in the sequence.
                next_target_idx = target_idx + 1
                if next_target_idx < len(self.sequences[seq_idx]):
                    next_target = self.sequences[seq_idx][next_target_idx]
                    self.add_task(current_z, next_target, seq_idx, next_target_idx)
            else:
                # Agent failed, keep the target and try again from where it got stuck.
                target_z = self.sequences[seq_idx][target_idx]
                self.add_task(current_z, target_z, seq_idx, target_idx)

class EpisodicHERBuffer:
    """Stores episodes and applies HER 80% of the time during sampling."""
    def __init__(self, capacity_episodes=10000, future_p=0.8, device="cuda"):
        self.episodes = []
        self.capacity = capacity_episodes
        self.future_p = future_p
        self.device = device
        self.position = 0

    def store_episodes(self, batched_episodes):
        for ep in batched_episodes:
            if len(ep) > 0: 
                if len(self.episodes) < self.capacity:
                    self.episodes.append(ep)
                else:
                    self.episodes[self.position] = ep
                    self.position = (self.position + 1) % self.capacity
     
    def sample_batch(self, batch_size=256):
        z_batch, a_batch, z_next_batch, g_batch, r_batch, done_batch = [], [], [], [], [], []
        
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            t = random.randint(0, len(ep) - 1)
            z_curr, a, z_next, original_g = ep[t]
            
            if random.random() < self.future_p and t < len(ep) - 1:
                future_t = random.randint(t + 1, len(ep) - 1)
                g_tilde = ep[future_t][2] 
                r, success = compute_reward(z_next, g_tilde)
                g_target = g_tilde
            else:
                r, success = compute_reward(z_next, original_g)
                g_target = original_g
                
            z_batch.append(z_curr)
            a_batch.append(a)
            z_next_batch.append(z_next)
            g_batch.append(g_target)
            r_batch.append(r)
            
            # success is already a boolean tensor from compute_reward, just cast to float
            done_batch.append(success.to(dtype=torch.float32))
            
        return (
            torch.stack(z_batch), torch.stack(a_batch), 
            torch.stack(z_next_batch), torch.stack(g_batch), 
            torch.stack(r_batch), torch.stack(done_batch)
        )

def train_loop(env, actor, critic, critic_target, 
               actor_optimizer, critic_optimizer, alpha_optimizer, 
               log_alpha, target_entropy, task_buffer, replay_buffer, 
               num_iterations=1000, T_max=10, gamma=0.99, tau=0.005, 
               save_dir="./checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting Task-Sampled Training Loop. Saving checkpoints to {save_dir}")
    
    for iteration in range(num_iterations):
        
        z_curr, z_target, seq_idxs, target_idxs = task_buffer.sample_tasks(env.num_envs)
        env.reset()
        # set the environment to start at current latent
        env.set_states(z_curr) 
        
        M_ep = [[] for _ in range(env.num_envs)]
        
        # Track which environments are still trying to reach their goal
        active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        for step in range(T_max):
            if not active_mask.any():
                break # break if all envs have succeeded
                
            with torch.no_grad():
                actions, _, _ = actor.sample(z_curr, z_target)
            
            # Step the environment (World Model)
            z_next, _, _, _, _ = env.step(actions)
            
            # Check for success
            _, successes = compute_reward(z_next, z_target)
            
            # Update masks (Only count successes for envs that were still active)
            just_succeeded = successes & active_mask
            success_mask |= just_succeeded
            
            # Store transitions ONLY for active environments
            for i in range(env.num_envs):
                if active_mask[i]:
                    transition = (z_curr[i], actions[i], z_next[i], z_target[i])
                    M_ep[i].append(transition)
            
            # Turn off environments that just succeeded
            active_mask &= ~just_succeeded
            
            # Update z_curr (If inactive, keep the old z_curr so it doesn't wander)
            z_curr = torch.where(active_mask.unsqueeze(-1), z_next, z_curr)

        # --- 3. Post-Episode Buffer Updates ---
        replay_buffer.store_episodes(M_ep)
        task_buffer.process_episode_results(z_curr, success_mask.cpu().numpy(), seq_idxs, target_idxs)
        #print(len(replay_buffer.episodes))
        # SAC training
        if len(replay_buffer.episodes) >= 10: 
            
            # Variables for logging
            avg_actor_loss = 0.0
            avg_critic_loss = 0.0
            
            for _ in range(40): 
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(batch_size=256)
                
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                #  Update Critic
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

                # Update Actor
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

                # Update Alpha
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Soft Update Target Networks
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:04d} | Buf: {len(replay_buffer.episodes):04d} eps | "
                      f"Act Loss: {avg_actor_loss/40:.4f} | Crit Loss: {avg_critic_loss/40:.4f}")
                
        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(), os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            print(f"--> Checkpoint saved at Iteration {iteration}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DEBUG_MODE = False 

    if DEBUG_MODE:
        print("--- RUNNING IN LOCAL DEBUG MODE ---")
        num_envs_to_use = 2
        num_iters_to_run = 25
        num_episodes_extract = 2
        
        class DummyJEPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_encoder = nn.Linear(5, 192)
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
        num_envs_to_use = 50
        num_iters_to_run = 100
        num_episodes_extract = 100
        

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

    # =====================================================================

    env = LatentEnv(jepa_model=jepa_model, dataset=dataset, num_envs=num_envs_to_use, device=device) 

    spacing_parameter = 5 
    subgoal_sequences = extract_subgoal_sequences(
        dataset=dataset, 
        model=jepa_model, 
        subgoal_spacing=spacing_parameter, 
        num_episodes=num_episodes_extract, 
        device=device
    )
    
    task_buffer = DynamicTaskBuffer(subgoal_sequences, device=device)
    replay_buffer = EpisodicHERBuffer(device=device)

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
        task_buffer=task_buffer, 
        replay_buffer=replay_buffer,
        num_iterations=num_iters_to_run,
        T_max=10,
        gamma=0.99,
        tau=0.005,
        save_dir="./checkpoints" 
    )