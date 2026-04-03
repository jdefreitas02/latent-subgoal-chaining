import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms
import os
import time

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
        self.action_dim = 25
        
        self.z_history = None
        self.act_history = None
        self.z_ultimate_goal = None
        self.current_steps = torch.zeros(self.num_envs, device=self.device)
        
        # --- CACHE LOADING / GENERATION ---
        ephemeral = os.environ.get("EPHEMERAL")
        if ephemeral is None:
            raise ValueError("EPHEMERAL environment variable is not set")
            
        cache_path = os.path.join(ephemeral, "stable_wm_data", "cube_all_latents_cache.pt")
        
        if not os.path.exists(cache_path):
            print(f"Cache not found at {cache_path}. Generating it now...")
            self._precompute(cache_path)
            
        print(f"Loading latents cache from {cache_path}...")
        cache = torch.load(cache_path)
        all_latents = cache['all_latents']
        
        # Pre-slice the starts (first frame) and goals (last frame) for every episode.
        # We push these to the GPU right now so reset() doesn't even need to use PCIe!
        self.all_starts = torch.stack([ep[0] for ep in all_latents]).to(self.device)
        self.all_goals = torch.stack([ep[-1] for ep in all_latents]).to(self.device)

        # --- SHUFFLED SEQUENTIAL QUEUE ---
        self.num_total_episodes = len(all_latents)
        self.ep_order = np.random.permutation(self.num_total_episodes)
        self.ep_ptr = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_steps.zero_()
        
        # 1. Grab the next batch of completely unique episodes
        if self.ep_ptr + self.num_envs > self.num_total_episodes:
            # We reached the end of the dataset. Reshuffle and start over
            self.ep_order = np.random.permutation(self.num_total_episodes)
            self.ep_ptr = 0
            
        ep_indices = self.ep_order[self.ep_ptr : self.ep_ptr + self.num_envs]
        self.ep_ptr += self.num_envs
        
        # 2. INSTANTANEOUS LOAD FROM GPU RAM (Zero Disk I/O, Zero Encoding)
        z_start = self.all_starts[ep_indices]
        self.z_ultimate_goal = self.all_goals[ep_indices].clone()
            
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
            
        if z_curr.dim() == 2:
            z_curr = z_curr.unsqueeze(1) 

        self.z_history = z_curr.repeat(1, 3, 1).to(self.device)
        latent_dim = z_curr.shape[-1]
        self.act_history = torch.zeros((self.num_envs, 3, latent_dim), device=self.device)
        self.current_steps.zero_()

    def step(self, actions):
        HS = 1 
        
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
            
        act_emb = self.model.action_encoder(actions)
        if act_emb.dim() == 2: 
            act_emb = act_emb.unsqueeze(1) 
            
        self.act_history = torch.cat([self.act_history, act_emb], dim=1)
        
        z_trunc = self.z_history[:, -HS:]
        act_trunc = self.act_history[:, -HS:]
        
        z_next = self.model.predict(z_trunc, act_trunc)[:, -1:]
        self.z_history = torch.cat([self.z_history, z_next], dim=1)
        
        distances = torch.norm(z_next.squeeze(1) - self.z_ultimate_goal, p=2, dim=-1)
        
        rewards = -distances
        dones = distances < 2.0
        
        return z_next.squeeze(1), rewards, dones, False, {}

    def _get_obs(self):
        return self.z_history[:, -1]

    def _precompute(self, save_path):
        """
        Automatically generates the 1.5GB Latent Cache if it doesn't exist.
        Re-uses the dataset and model already loaded into the environment.
        """
        print("--- Starting Full Dataset Latent Pre-computation ---")
        num_episodes = len(self.dataset.lengths)
        batch_size = 10 
        
        all_latents = []
        img_transform = tv_transforms.Compose([
            tv_transforms.ToDtype(torch.float32, scale=True),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        total_frames_processed = 0
        t0 = time.time()

        with torch.no_grad():
            for i in range(0, num_episodes, batch_size):
                end_idx = min(i + batch_size, num_episodes)
                ep_indices = np.arange(i, end_idx)
                ep_lens = self.dataset.lengths[ep_indices]

                # Request the ENTIRE episode (from frame 0 to ep_len)
                starts = np.zeros(len(ep_indices), dtype=int)
                ends = ep_lens

                chunks = self.dataset.load_chunk(ep_indices, starts, ends)

                for chunk in chunks:
                    raw_pixels = chunk['pixels'].to(self.device)  # [T, C, H, W] uint8
                    pixels = img_transform(raw_pixels)            # [T, C, H, W] float32, ImageNet-normalised
                    # Add dummy batch dimension for the Vision Transformer
                    pixels_5d = pixels.unsqueeze(0)

                    # Encode and strip the dummy batch dimension
                    z_ep = self.model.encode({'pixels': pixels_5d})['emb'].squeeze(0)
                    
                    all_latents.append(z_ep.cpu())
                    total_frames_processed += len(z_ep)
                
                elapsed = time.time() - t0
                print(f"Processed {end_idx}/{num_episodes} episodes | "
                      f"Total Frames: {total_frames_processed:,} | "
                      f"Time: {elapsed:.2f}s")

        print("\nSaving massive latent cache to disk...")
        torch.save({
            'all_latents': all_latents,
            'total_frames': total_frames_processed
        }, save_path)
        print(f"SUCCESS! Saved cache to {save_path}\n")