import gymnasium as gym
import numpy as np
import torch

class LatentEnv(gym.Env):
    def __init__(self, jepa_model, dataset, num_envs=50, device="cuda", max_steps=50, history_size=3):
        super().__init__()
        
        self.model = jepa_model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.dataset = dataset
        self.num_envs = num_envs
        self.device = device
        self.max_steps = max_steps
        self.history_size = history_size
        
        self.latent_dim = 192
        self.action_dim = 5
        
        self.z_history = None
        self.act_history = None
        self.z_ultimate_goal = None
        self.current_steps = torch.zeros(self.num_envs, device=self.device)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps.zero_()
        
        ep_indices = np.random.randint(0, len(self.dataset.lengths), size=self.num_envs)
        ep_lens = self.dataset.lengths[ep_indices]
        
        starts = np.zeros(self.num_envs, dtype=int)
        ends_for_start = np.ones(self.num_envs, dtype=int)
        starts_for_goal = ep_lens - 1
        ends_for_goal = ep_lens
        
        start_dicts = self.dataset.load_chunk(ep_indices, starts, ends_for_start)
        goal_dicts = self.dataset.load_chunk(ep_indices, starts_for_goal, ends_for_goal)
        
        with torch.no_grad():
            start_pixels = torch.stack([sd['pixels'] for sd in start_dicts]).to(self.device)
            goal_pixels = torch.stack([gd['pixels'] for gd in goal_dicts]).to(self.device)
            
            z_start = self.model.encode({'pixels': start_pixels})['emb']
            self.z_ultimate_goal = self.model.encode({'pixels': goal_pixels})['emb']
            
        self.z_history = z_start
        self.act_history = torch.zeros(
            (self.num_envs, 1, self.model.action_encoder.output_dim), device=self.device
        )
        
        return self._get_obs(), {}

    def set_states(self, z_curr):
        """
        Teleports the environments to the provided latent states and flushes history.
        Expects z_curr shape: [num_envs, 192]
        """
        # Ensure correct shape [num_envs, 1, 192] for the history buffer
        if z_curr.dim() == 2:
            self.z_history = z_curr.unsqueeze(1).clone()
        else:
            self.z_history = z_curr.clone()
            
        # Flush the action history back to zeros
        self.act_history = torch.zeros((self.num_envs, 1, self.model.action_encoder.output_dim), device=self.device)
        
        # Reset the step counter for the T_max loop
        self.current_steps.zero_()

    def step(self, actions):
        self.current_steps += 1
        
        action_tensor = actions.unsqueeze(1)
        
        with torch.no_grad():
            act_emb = self.model.action_encoder(action_tensor)
            
            # Update Action History
            if self.act_history.shape[1] < self.history_size:
                self.act_history = torch.cat([self.act_history, act_emb], dim=1)
            else:
                self.act_history = torch.cat([self.act_history[:, 1:], act_emb], dim=1)
                
            # Predict Next State
            z_next = self.model.predict(self.z_history, self.act_history)[:, -1:] 
            
            # Update State History
            if self.z_history.shape[1] < self.history_size:
                self.z_history = torch.cat([self.z_history, z_next], dim=1)
            else:
                self.z_history = torch.cat([self.z_history[:, 1:], z_next], dim=1)

        current_z = self.z_history[:, -1]
        target_z = self.z_ultimate_goal[:, -1]
        
        distances = torch.norm(current_z - target_z, p=2, dim=-1)
        rewards = -distances
        
        terminated = distances < 0.5 
        truncated = self.current_steps >= self.max_steps
        
        info = {"l2_distance_to_final_goal": distances}
        
        return current_z, rewards, terminated, truncated, info

    def _get_obs(self):
        # Return purely as a tensor
        return self.z_history[:, -1]