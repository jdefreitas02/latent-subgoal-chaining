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
            
        if z_start.dim() == 2:
            z_start = z_start.unsqueeze(1) # Shape: [Batch, 1, Dim]
            
        # We duplicate the starting state 3 times to simulate standing still
        self.z_history = z_start.repeat(1, 3, 1)
        
        latent_dim = z_start.shape[-1]
        
        # Pad the ENCODED action history with 3 steps of zero-actions
        self.act_history = torch.zeros((self.num_envs, 3, latent_dim), device=self.device)
        
        return self._get_obs(), {}

    def set_states(self, z_curr):
        """
        Teleports the environments to the provided latent states and flushes history.
        Expects z_curr shape: [num_envs, 192]
        """
        if not isinstance(z_curr, torch.Tensor):
            z_curr = torch.tensor(z_curr, dtype=torch.float32, device=self.device)
            
        # 1. Enforce correct shape [num_envs, 1, 192] BEFORE repeating
        if z_curr.dim() == 2:
            z_curr = z_curr.unsqueeze(1) 

        # 2. Duplicate the starting state 3 times to simulate standing still
        # Shape safely becomes [num_envs, 3, 192]
        self.z_history = z_curr.repeat(1, 3, 1).to(self.device)
        
        # 3. Pad the action history with 3 steps of zero-actions
        latent_dim = z_curr.shape[-1]
        self.act_history = torch.zeros(
            (self.num_envs, 3, latent_dim), 
            device=self.device
        )
        
        # Reset the step counter for the T_max loop
        self.current_steps.zero_()

    def step(self, actions):
        # Hardcode the paper's architecture requirement for OGBench-Cube
        HS = 3 
        
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
            
        # 1. Encode the current action chunk
        act_emb = self.model.action_encoder(actions)
        if act_emb.dim() == 2: 
            act_emb = act_emb.unsqueeze(1) # [Batch, 1, Dim]
            
        # 2. Append the new action to the history buffer
        self.act_history = torch.cat([self.act_history, act_emb], dim=1)
        
        # 3. THE SLIDING WINDOW (Enforce length = 3)
        z_trunc = self.z_history[:, -HS:]
        act_trunc = self.act_history[:, -HS:]
        
        # 4. Predict the next state
        z_next = self.model.predict(z_trunc, act_trunc)[:, -1:]
        
        # 5. Append the predicted state to the history for the NEXT loop
        self.z_history = torch.cat([self.z_history, z_next], dim=1)
        
        # 6. Compute reward (assuming z_ultimate_goal is set)
        distances = torch.norm(z_next.squeeze(1) - self.z_ultimate_goal, p=2, dim=-1)
        rewards = -(distances > 0.5).float()
        dones = distances < 0.5
        
        return z_next.squeeze(1), rewards, dones, False, {}

    def _get_obs(self):
        # Return purely as a tensor for the Actor
        return self.z_history[:, -1]