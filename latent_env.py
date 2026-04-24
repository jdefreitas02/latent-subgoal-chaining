import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms
import os
import time

class LatentEnv(gym.Env):
    def __init__(self, jepa_model, dataset, num_envs=50, device="cuda", cache_path=None, done_threshold=2.0):
        super().__init__()

        self.model = jepa_model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.dataset = dataset
        self.num_envs = num_envs
        self.device = device

        self.latent_dim = 192
        self.action_dim = 25
        self.done_threshold = done_threshold

        # Single-frame state buffers — the world model is Markovian (HS=1)
        self.z_state = None      # [num_envs, 1, latent_dim]
        self.act_emb = None      # [num_envs, 1, latent_dim]
        self.z_ultimate_goal = None

        # --- CACHE LOADING / GENERATION ---
        if cache_path is None:
            cache_path = os.path.join(os.path.expanduser("~"), "stable_wm_data", "ogbench", "ogbench_latents_cache.pt")
        
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

        # Grab the next batch of completely unique episodes
        if self.ep_ptr + self.num_envs > self.num_total_episodes:
            # Reached the end of the dataset — reshuffle and start over
            self.ep_order = np.random.permutation(self.num_total_episodes)
            self.ep_ptr = 0

        ep_indices = self.ep_order[self.ep_ptr : self.ep_ptr + self.num_envs]
        self.ep_ptr += self.num_envs

        # INSTANTANEOUS LOAD FROM GPU RAM (Zero Disk I/O, Zero Encoding)
        z_start = self.all_starts[ep_indices]           # [B, latent_dim]
        self.z_ultimate_goal = self.all_goals[ep_indices].clone()

        self.z_state  = z_start.unsqueeze(1)            # [B, 1, latent_dim]
        self.act_emb  = torch.zeros_like(self.z_state)  # [B, 1, latent_dim]

        return self._get_obs(), {}

    def set_states(self, z_curr):
        """
        Teleports the environments to the provided latent states and flushes history.
        Expects z_curr shape: [num_envs, latent_dim]
        """
        if not isinstance(z_curr, torch.Tensor):
            z_curr = torch.tensor(z_curr, dtype=torch.float32, device=self.device)

        self.z_state = z_curr.unsqueeze(1).to(self.device)  # [B, 1, latent_dim]
        self.act_emb = torch.zeros_like(self.z_state)

    def step(self, actions):
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        act_emb = self.model.action_encoder(actions)
        if act_emb.dim() == 2:
            act_emb = act_emb.unsqueeze(1)

        # World model is Markovian: predict next state from current state + action only
        z_next = self.model.predict(self.z_state, act_emb)[:, -1:]

        # Overwrite single-frame buffers (no unbounded history growth)
        self.z_state = z_next
        self.act_emb = act_emb

        distances = torch.norm(z_next.squeeze(1) - self.z_ultimate_goal, p=2, dim=-1)
        rewards = -distances
        dones = distances < self.done_threshold

        return z_next.squeeze(1), rewards, dones, False, {}

    def _get_obs(self):
        return self.z_state.squeeze(1)

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