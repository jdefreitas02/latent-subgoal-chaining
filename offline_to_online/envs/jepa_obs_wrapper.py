"""ObservationWrapper that encodes pixel observations to 192-D JEPA latents.

Used by B2 (qc + JEPA encoder, real env online) to keep the agent's
observation space identical to E (192-D latents) while keeping all env
dynamics in the real OGBench env.
"""

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box

from envs.jepa_loader import make_img_transform


LATENT_DIM = 192


class JEPAObsWrapper(gymnasium.ObservationWrapper):
    """Wrap an env whose obs is (H, W, 3) uint8 pixels into (192,) float32 latents."""

    def __init__(self, env, jepa_model, device="cuda"):
        super().__init__(env)
        self.jepa_model = jepa_model
        self.device = device
        self._img_transform = make_img_transform()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(LATENT_DIM,), dtype=np.float32
        )

    def observation(self, observation):
        # observation is (H, W, 3) uint8 from swm/OGBCube-v0
        img = torch.from_numpy(observation).permute(2, 0, 1).contiguous()  # (3, H, W)
        img = self._img_transform(img).to(self.device)  # (3, H, W) float
        info = {"pixels": img.unsqueeze(0).unsqueeze(0)}  # (1, 1, 3, H, W)
        with torch.no_grad():
            info = self.jepa_model.encode(info)
        z = info["emb"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return z
