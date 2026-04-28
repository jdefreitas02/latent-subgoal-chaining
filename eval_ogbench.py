"""
Evaluate a SAC+HER policy using OGBench's native evaluation protocol.

Matches the exact protocol from the OGBench paper (Table 2):
  - swm/OGBCube-v0 environment (native 224×224 pixels, same renderer as eval.py)
  - 5 predefined task configurations (horizontal, vertical×2, diagonal×2)
  - Goal image from env.reset(options=dict(task_id=N)) → info['target']  (swm/OGBCube-v0 key)
  - 50 episodes per task = 250 total
  - Max 200 steps per episode (TimeLimit)
  - Success = info['success'] (cube within 4cm of target, at episode end)

Results are directly comparable to HIQL/GCIVL/GCIQL/CRL/GCBC in Table 2.

Usage:
    # 224x224 pretrained LeWM (default):
    python latent_hindsight_rl/eval_ogbench.py \\
        --ckpt_path ~/.stable_worldmodel/cube/lejepa/weights.pt \\
        --checkpoint_dir ./checkpoints_joint_gap_1_beta3.0_sparse \\
        --dataset_path $STABLEWM_HOME/cube/cube_single_expert \\
        --done_threshold 9.36

    # 64x64 OGBench-trained LeWM:
    python latent_hindsight_rl/eval_ogbench.py \\
        --ckpt_path ./lewm_ogbench_weights.ckpt \\
        --checkpoint_dir ./checkpoints_joint_gap_5_beta3.0_sparse \\
        --dataset_path $STABLEWM_HOME/ogbench/cube_single_play_v0 \\
        --img_size 64 --patch_size 8 --done_threshold 2.887
"""

import os
import re
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.distributions import Normal
from torchvision.transforms import v2 as transforms
from sklearn import preprocessing

# jepa/module live in the parent directory
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import stable_pretraining as spt
import stable_worldmodel as swm

from train_high_level import MLPHighLevel, DiffusionHighLevel


# =============================================================================
# HIQL GaussianActor (mirrors train_hiql_lewm.py — kept inline to avoid
# importing the heavy train script with its top-level side-effects)
# =============================================================================

def _init_weights_gc(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class _GaussianActor(nn.Module):
    """Compact copy of GaussianActor from train_hiql_lewm.py for eval loading."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, latent_dim=192, output_dim=25,
                 hidden_dims=(512, 512, 512),
                 tanh_squash=False, action_scale=1.0):
        super().__init__()
        self.tanh_squash  = tanh_squash
        self.action_scale = action_scale

        in_dim = latent_dim * 2
        backbone = []
        for h in hidden_dims:
            backbone += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.backbone = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.log_stds  = nn.Parameter(torch.zeros(output_dim))
        self.backbone.apply(_init_weights_gc)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias,   0.0)

    def sample(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        h = self.backbone(x)
        mean    = self.mean_head(h)
        log_std = self.log_stds.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        std     = log_std.exp()
        dist    = Normal(mean, std)
        raw     = dist.rsample()
        if self.tanh_squash:
            y      = torch.tanh(raw)
            action = y * self.action_scale
            lp     = dist.log_prob(raw)
            lp    -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
            log_prob = lp.sum(dim=-1)
            deterministic = torch.tanh(mean) * self.action_scale
        else:
            action        = raw
            log_prob      = dist.log_prob(raw).sum(dim=-1)
            deterministic = mean
        return action, log_prob, deterministic


class _HIQLHighLevelWrapper:
    """Wraps a _GaussianActor HL so it exposes the predict() interface
    expected by HierarchicalPolicy."""

    def __init__(self, hl_actor):
        self.hl_actor = hl_actor

    def predict(self, z_curr, z_goal):
        """Return deterministic (mean) subgoal [B, 192]."""
        with torch.no_grad():
            _, _, subgoal = self.hl_actor.sample(z_curr, z_goal)
        return subgoal


class _LewmGaussianActor(nn.Module):
    """LeWM-hybrid HIQL actor matching train_hiql_lewm.py GaussianActor.

    Layer order: Linear → LayerNorm → GELU (state-of-the-art baseline order).
    Has learnable per-dim log_stds (const_std=True style) which the baseline
    actor lacks.

    Used for both HL (state=192, goal=192, output=rep_dim, no squash) and
    LL (state=192, goal=rep_dim, output=25, tanh_squash=True).
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, state_dim, goal_dim, output_dim,
                 hidden_dims=(512, 512, 512), action_scale=1.0, tanh_squash=False):
        super().__init__()
        self.tanh_squash  = tanh_squash
        self.action_scale = action_scale

        in_dim = state_dim + goal_dim
        backbone = []
        for h in hidden_dims:
            backbone += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.backbone  = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.log_stds  = nn.Parameter(torch.zeros(output_dim))

    def sample(self, state, goal):
        x   = torch.cat([state, goal], dim=-1)
        h   = self.backbone(x)
        mean    = self.mean_head(h)
        log_std = self.log_stds.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        std     = log_std.exp()
        from torch.distributions import Normal as _Normal
        dist    = _Normal(mean, std)
        raw     = dist.rsample()
        if self.tanh_squash:
            y             = torch.tanh(raw)
            action        = y * self.action_scale
            lp            = dist.log_prob(raw) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
            deterministic = torch.tanh(mean) * self.action_scale
        else:
            action        = raw
            lp            = dist.log_prob(raw)
            deterministic = mean
        return action, lp.sum(dim=-1), deterministic


class _ActionChunkDecoder(nn.Module):
    """Frozen state-conditioned decoder D_θ(a_first, z) -> 25-D chunk.

    Inline copy of latent_hindsight_rl/train_action_decoder.py:ActionChunkDecoder
    so eval doesn't need to import the heavy training script.
    """

    def __init__(self, in_dim=5, out_dim=25, latent_dim=192,
                 hidden_dims=(256, 256)):
        super().__init__()
        in_d = in_dim + latent_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, a_first, z):
        return self.net(torch.cat([a_first, z], dim=-1))


class _WGSPHierarchicalPolicy:
    """HierarchicalPolicy variant for WGSP checkpoints.

    Identical to HierarchicalPolicy except: when a frozen ActionChunkDecoder
    is provided, the LL actor is expected to output 5-D and the decoder
    inflates it to a 25-D chunk before reshape/inverse_transform — exactly
    matching the training-time data flow in train_hiql_wgsp.py.

    If decoder=None, behaves identically to HierarchicalPolicy (LL outputs 25-D).
    """

    def __init__(self, jepa_model, actor, action_scaler, high_level, gap, device,
                 decoder=None, subgoal_reached_threshold=0.0):
        self.jepa_model    = jepa_model
        self.actor         = actor
        self.action_scaler = action_scaler
        self.high_level    = high_level
        self.gap           = gap
        self.device        = device
        self.decoder       = decoder
        self.subgoal_reached_threshold = subgoal_reached_threshold
        self._buf       = []
        self._wm_steps  = 0
        self._z_subgoal = None

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        diag = None
        if not self._buf:
            z_curr = encode_and_project(self.jepa_model, obs_hwc, self.device)
            if goal_latent is not None:
                z_final_goal = goal_latent.to(self.device)
            else:
                z_final_goal = encode_and_project(self.jepa_model, goal_hwc, self.device)
            with torch.no_grad():
                subgoal_reached = (
                    self._z_subgoal is not None and
                    self._z_subgoal.shape[-1] == z_curr.shape[-1] and
                    torch.norm(z_curr - self._z_subgoal, p=2, dim=-1).item() <
                    self.subgoal_reached_threshold
                )
                switched = (self._z_subgoal is None or
                            self._wm_steps >= self.gap or subgoal_reached)
                if switched:
                    z_subgoal_raw = self.high_level.predict(z_curr, z_final_goal)
                    self._z_subgoal = nn_clamp_subgoal(z_subgoal_raw)
                    self._wm_steps  = 0
                    self._subgoal_switches = getattr(self, '_subgoal_switches', 0) + 1
                _, _, ll_out = self.actor.sample(z_curr, self._z_subgoal)
                # ll_out is (1,5) if decoder present, else (1,25)
                if self.decoder is not None:
                    chunk_25d = self.decoder(ll_out, z_curr)            # (1,25)
                else:
                    chunk_25d = ll_out
                same_dim = self._z_subgoal.shape[-1] == z_curr.shape[-1]
                diag = {
                    'dist_to_goal':         torch.norm(z_curr - z_final_goal, p=2, dim=-1).item(),
                    'dist_to_subgoal':      torch.norm(z_curr - self._z_subgoal, p=2, dim=-1).item() if same_dim else float('nan'),
                    'dist_subgoal_to_goal': torch.norm(self._z_subgoal - z_final_goal, p=2, dim=-1).item() if same_dim else float('nan'),
                    'subgoal_switches':     getattr(self, '_subgoal_switches', 0),
                    'subgoal_switched':     switched,
                }
            self._wm_steps += 1
            raw      = chunk_25d.cpu().numpy().reshape(5, 5)
            physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)
            diag['action_norm'] = float(np.mean(np.abs(physical)))
            self._buf = [physical[t] for t in range(5)]
        return self._buf.pop(0), diag

    def reset(self):
        self._buf = []
        self._wm_steps = 0
        self._z_subgoal = None
        self._subgoal_switches = 0


class _BaselineNativeHierarchicalPolicy:
    """HIQL baseline policy: native 5D actions, encode every env step (no chunking).

    Mirrors ogbench's evaluation: re-encodes the obs every env step, queries HL
    every `gap` env steps for a fresh phi, and outputs one 5D action per step.
    """

    def __init__(self, jepa_model, ll_actor, action_scaler, hl_actor_wrapped,
                 gap, device):
        self.jepa_model = jepa_model
        self.actor = ll_actor
        self.action_scaler = action_scaler
        self.high_level = hl_actor_wrapped
        self.gap = gap
        self.device = device
        self._step = 0
        self._z_subgoal = None
        self._subgoal_switches = 0

    def reset(self):
        self._step = 0
        self._z_subgoal = None
        self._subgoal_switches = 0

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        z_curr = encode_and_project(self.jepa_model, obs_hwc, self.device)
        if goal_latent is not None:
            z_goal = goal_latent.to(self.device)
        else:
            z_goal = encode_and_project(self.jepa_model, goal_hwc, self.device)

        with torch.no_grad():
            switched = self._z_subgoal is None or self._step >= self.gap
            if switched:
                self._z_subgoal = self.high_level.predict(z_curr, z_goal)
                self._step = 0
                self._subgoal_switches += 1
            _, _, mean = self.actor.sample(z_curr, self._z_subgoal)        # [1, 5]

        self._step += 1
        raw = mean.cpu().numpy().reshape(1, 5)
        physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)[0]
        diag = {
            'dist_to_goal': torch.norm(z_curr - z_goal, p=2, dim=-1).item(),
            'dist_to_subgoal': float('nan'),
            'dist_subgoal_to_goal': float('nan'),
            'subgoal_switches': self._subgoal_switches,
            'subgoal_switched': switched,
            'action_norm': float(np.mean(np.abs(physical))),
        }
        return physical, diag


class _BaselineGaussianActor(nn.Module):
    """Baseline HIQL actor with explicit input_dim (vs _GaussianActor's latent_dim*2).

    Matches train_hiql_baseline.py GaussianActor weight layout exactly.
    Used for both HL (input=384, output=rep_dim) and LL (input=192+rep_dim, output=5).
    const_std=True: std is fixed at 1 (matches ogbench GCActor const_std=True).
    """

    def __init__(self, input_dim, output_dim,
                 hidden_dims=(512, 512, 512), action_scale=1.0, tanh_squash=False):
        super().__init__()
        self.tanh_squash  = tanh_squash
        self.action_scale = action_scale

        in_dim = input_dim
        backbone = []
        for h in hidden_dims:
            backbone += [nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)]
            in_dim = h
        self.backbone  = nn.Sequential(*backbone)
        self.mean_head = nn.Linear(in_dim, output_dim)

    def sample(self, state, goal):
        """Accept (state, goal) separately — concatenates internally."""
        x   = torch.cat([state, goal], dim=-1)
        h   = self.backbone(x)
        mean    = self.mean_head(h)
        std     = torch.ones_like(mean)
        from torch.distributions import Normal as _Normal
        dist = _Normal(mean, std)
        raw  = dist.rsample()
        if self.tanh_squash:
            y             = torch.tanh(raw)
            action        = y * self.action_scale
            lp            = dist.log_prob(raw) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
            deterministic = torch.tanh(mean) * self.action_scale
        else:
            action        = raw
            lp            = dist.log_prob(raw)
            deterministic = mean
        return action, lp.sum(dim=-1), deterministic


class _HIQLBaselineHighLevelWrapper:
    """Wraps baseline HL actor (output=rep_dim) for HierarchicalPolicy.

    Predicted phi is length-normalised to match how ogbench's sample_actions
    normalises the HL sample before passing it to the LL (hiql.py line 185):
        goal_reps = goal_reps / ||goal_reps|| * sqrt(rep_dim)
    """

    def __init__(self, hl_actor):
        self.hl_actor = hl_actor

    def predict(self, z_curr, z_goal):
        """Return length-normalised deterministic phi subgoal [B, rep_dim]."""
        with torch.no_grad():
            _, _, phi = self.hl_actor.sample(z_curr, z_goal)
            phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8) * (phi.shape[-1] ** 0.5)
        return phi


class _LewmOneStep(nn.Module):
    """One-step distilled LL policy from train_hiql_flow.py.

    µ_ψ(z, rep, ε) → action  (no tanh squash; actions clipped at inference)
    Layer order: Linear → GELU  (no LayerNorm, matching FQL actor_layer_norm=False)
    """

    def __init__(self, latent_dim, rep_dim, action_dim,
                 hidden_dims=(512, 512, 512, 512)):
        super().__init__()
        self.action_dim = action_dim
        in_dim = latent_dim + rep_dim + action_dim
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def sample(self, state, goal, deterministic=False):
        if deterministic:
            noise = torch.zeros(state.shape[0], self.action_dim, device=state.device)
        else:
            noise = torch.randn(state.shape[0], self.action_dim, device=state.device)
        x   = torch.cat([state, goal, noise], dim=-1)
        a   = self.net(x)
        return a, None, a


# =============================================================================
# Model definitions (must match train.py exactly)
# =============================================================================

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GoalConditionedActor(nn.Module):
    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256, action_scale=3.0):
        super().__init__()
        self.action_scale = action_scale
        self.linear1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.ln1     = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2     = nn.LayerNorm(hidden_dim)
        self.mean_linear    = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)
        nn.init.uniform_(self.mean_linear.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_linear.bias, 0)
        nn.init.uniform_(self.log_std_linear.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std_linear.bias, -1.0)

    def sample(self, state, goal):
        x = F.relu(self.ln1(self.linear1(torch.cat([state, goal], dim=-1))))
        x = F.relu(self.ln2(self.linear2(x)))
        mean    = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)
        normal  = Normal(mean, log_std.exp())
        x_t     = normal.rsample()
        y_t     = torch.tanh(x_t)
        action  = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale


def load_jepa(ckpt_path, device, img_size=64, patch_size=8):
    """Load a JEPA model from checkpoint.

    For 224×224 models: expects a serialized JEPA object (*_object.ckpt),
    loaded directly via torch.load — no manual architecture construction needed.

    For 64×64 OGBench models: constructs the architecture and loads a
    Lightning-style state dict (*_weights.ckpt or bare state dict).

    Args:
        ckpt_path: Path to checkpoint file.
        device: Target device.
        img_size: Image resolution (64 or 224).
        patch_size: ViT patch size (8 for 64x64, 14 for 224x224).
    """
    if img_size == 224:
        # Use the same loading pattern as eval.py: AutoCostModel resolves
        # base_path → base_path_object.ckpt automatically.
        model = swm.policy.AutoCostModel(ckpt_path)
        print(f"  Loaded 224×224 JEPA via AutoCostModel from {ckpt_path}")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # 64×64 OGBench-trained LeWM: construct architecture, load state dict.
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False)
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        epoch = ckpt.get('epoch', '?')
    else:
        raw_sd = dict(ckpt)
        epoch = '?'
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded 64×64 JEPA from {ckpt_path} (epoch {epoch})")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# Observation encoding
# =============================================================================

def _make_img_transform(img_size):
    """Build image transform pipeline. No resize needed — swm/OGBCube-v0 renders natively at img_size."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


# Default (set in main based on --img_size)
_IMG_TRANSFORM = None


def encode_obs(jepa_model, obs_hwc: np.ndarray, device) -> torch.Tensor:
    """Encode a single (H, W, C) uint8 observation → [1, 192] latent."""
    t = _IMG_TRANSFORM(obs_hwc)           # [C, img_size, img_size]
    t = t.unsqueeze(0).unsqueeze(0)       # [1, 1, C, H, W]  (batch=1, T=1)
    with torch.no_grad():
        emb = jepa_model.encode({"pixels": t.to(device)})["emb"]  # [1, 1, 192]
    return emb[:, -1]                                               # [1, 192]


# PCA projection (set in main if --pca_path provided)
_PCA_MEAN   = None   # [192] tensor on device
_PCA_MATRIX = None   # [192, D] tensor on device
_FAISS_INDEX = None  # faiss.Index over projected cache latents (optional)


def encode_and_project(jepa_model, obs_hwc: np.ndarray, device) -> torch.Tensor:
    """Encode obs and optionally project to pca_dim. Returns [1, D] latent."""
    z = encode_obs(jepa_model, obs_hwc, device)  # [1, 192]
    if _PCA_MATRIX is not None:
        z = (z - _PCA_MEAN) @ _PCA_MATRIX  # [1, pca_dim]
    return z


def nn_clamp_subgoal(z_subgoal: torch.Tensor) -> torch.Tensor:
    """Snap a predicted subgoal to the nearest cached latent using FAISS.

    Only active when --faiss_index_path is provided. Otherwise returns z_subgoal unchanged.
    Guarantees the subgoal is a real encoder-space state from the expert dataset.
    """
    if _FAISS_INDEX is None:
        return z_subgoal
    import faiss
    z_np = z_subgoal.cpu().numpy().astype(np.float32)  # [1, D]
    _, I = _FAISS_INDEX.search(z_np, k=1)               # I: [1, 1]
    # Reconstruct nearest neighbor from FAISS (requires storing vectors)
    nn_vec = _FAISS_INDEX.reconstruct(int(I[0, 0]))      # [D] numpy
    return torch.tensor(nn_vec, dtype=torch.float32, device=z_subgoal.device).unsqueeze(0)  # [1, D]


# =============================================================================
# Policy wrappers
# =============================================================================

class FlatPolicy:
    """Flat SAC+HER policy: actor(z_curr, z_goal) → 5 buffered physical actions."""

    def __init__(self, jepa_model, actor, action_scaler, device):
        self.jepa_model    = jepa_model
        self.actor         = actor
        self.action_scaler = action_scaler
        self.device        = device
        self._buf = []

    def reset(self):
        self._buf = []

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        """Return (action, diag_dict). diag_dict is populated on action-block boundaries.

        Args:
            obs_hwc:     Current observation image (H, W, C) uint8.
            goal_hwc:    Goal image (H, W, C) uint8. Used when goal_latent is None.
            goal_latent: Pre-encoded goal [1, D] tensor. Skips goal image encoding.
                         Used for in-distribution lewm goals (--lewm_goals_path).
        """
        diag = None
        if not self._buf:
            z_curr = encode_and_project(self.jepa_model, obs_hwc,  self.device)  # [1, D]
            if goal_latent is not None:
                z_goal = goal_latent.to(self.device)
            else:
                z_goal = encode_and_project(self.jepa_model, goal_hwc, self.device)  # [1, D]
            with torch.no_grad():
                _, _, actions = self.actor.sample(z_curr, z_goal)        # [1, 25]
                dist_to_goal = torch.norm(z_curr - z_goal, p=2, dim=-1).item()
            raw      = actions.cpu().numpy().reshape(5, 5)
            physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)
            diag = {
                'dist_to_goal':    dist_to_goal,
                'dist_to_subgoal': dist_to_goal,  # same as goal for flat policy
                'subgoal_switches': 0,
                'subgoal_switched': False,
                'action_norm': float(np.mean(np.abs(physical))),
            }
            self._buf = [physical[t] for t in range(5)]
        return self._buf.pop(0), diag                                     # [5]


class HierarchicalPolicy:
    """High-level (subgoal proposer) + low-level SAC actor."""

    def __init__(self, jepa_model, actor, action_scaler, high_level, gap, device,
                 subgoal_reached_threshold=2.0):
        self.jepa_model    = jepa_model
        self.actor         = actor
        self.action_scaler = action_scaler
        self.high_level    = high_level
        self.gap           = gap
        self.device        = device
        self.subgoal_reached_threshold = subgoal_reached_threshold
        self._buf              = []
        self._wm_steps         = 0
        self._z_subgoal        = None

    def get_action(self, obs_hwc, goal_hwc=None, goal_latent=None):
        """Return (action, diag_dict | None). diag_dict is populated on WM-step boundaries.

        Args:
            obs_hwc:     Current observation image (H, W, C) uint8.
            goal_hwc:    Goal image (H, W, C) uint8. Used when goal_latent is None.
            goal_latent: Pre-encoded goal [1, D] tensor. Skips goal image encoding.
                         Used for in-distribution lewm goals (--lewm_goals_path).
        """
        diag = None
        if not self._buf:
            # encode_and_project applies PCA if _PCA_MATRIX is set
            z_curr       = encode_and_project(self.jepa_model, obs_hwc,  self.device)  # [1, D]
            if goal_latent is not None:
                z_final_goal = goal_latent.to(self.device)
            else:
                z_final_goal = encode_and_project(self.jepa_model, goal_hwc, self.device)  # [1, D]
            with torch.no_grad():
                subgoal_reached = (
                    self._z_subgoal is not None and
                    self._z_subgoal.shape[-1] == z_curr.shape[-1] and
                    torch.norm(z_curr - self._z_subgoal, p=2, dim=-1).item() < self.subgoal_reached_threshold
                )
                switched = self._z_subgoal is None or self._wm_steps >= self.gap or subgoal_reached
                if switched:
                    z_subgoal_raw = self.high_level.predict(z_curr, z_final_goal)
                    # NN clamping: snap to nearest cached latent if FAISS index available
                    self._z_subgoal = nn_clamp_subgoal(z_subgoal_raw)
                    self._wm_steps  = 0
                    self._subgoal_switches = getattr(self, '_subgoal_switches', 0) + 1
                _, _, actions = self.actor.sample(z_curr, self._z_subgoal)
                same_dim = self._z_subgoal.shape[-1] == z_curr.shape[-1]
                diag = {
                    'dist_to_goal':         torch.norm(z_curr - z_final_goal, p=2, dim=-1).item(),
                    'dist_to_subgoal':      torch.norm(z_curr - self._z_subgoal, p=2, dim=-1).item() if same_dim else float('nan'),
                    'dist_subgoal_to_goal': torch.norm(self._z_subgoal - z_final_goal, p=2, dim=-1).item() if same_dim else float('nan'),
                    'subgoal_switches': getattr(self, '_subgoal_switches', 0),
                    'subgoal_switched': switched,
                }
            self._wm_steps += 1
            raw      = actions.cpu().numpy().reshape(5, 5)
            physical = np.clip(self.action_scaler.inverse_transform(raw), -1.0, 1.0)
            diag['action_norm'] = float(np.mean(np.abs(physical)))
            self._buf = [physical[t] for t in range(5)]
        return self._buf.pop(0), diag

    def reset(self):
        self._buf              = []
        self._wm_steps         = 0
        self._z_subgoal        = None
        self._subgoal_switches = 0


# =============================================================================
# Evaluation loop
# =============================================================================

def run_task_lewm(env, policy, goal_latent, task_name, num_episodes, max_steps,
                  done_threshold, diagnose=False):
    """Evaluate against a pre-encoded in-distribution goal latent.

    Used when --lewm_goals_path is set (the lewm expert dataset doesn't cover
    the 5 standard OGBench task goals, so we use k-means goals from the cache).

    Success criterion: L2(z_final, z_goal) < done_threshold, matching training.
    The env is reset without a task_id constraint — any starting state is valid.
    """
    successes = []
    for ep in range(num_episodes):
        obs, info = env.reset()   # random start, no OGBench task constraint
        policy.reset()
        done  = False
        step  = 0
        diag_log_step = 0
        z_goal_t = goal_latent.to(policy.device)   # [1, D]

        while not done and step < max_steps:
            action, diag = policy.get_action(obs, goal_latent=z_goal_t)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            step += 1
            if diagnose and ep == 0 and diag is not None:
                diag_log_step += 1
                if diag_log_step % 4 == 0:
                    print(
                        f"  [task={task_name} ep={ep} phys_step={step:3d}]"
                        f"  dist_to_goal={diag['dist_to_goal']:.3f}"
                        f"  dist_to_subgoal={diag['dist_to_subgoal']:.3f}"
                        f"  action_norm={diag['action_norm']:.3f}"
                    )

        # Success: final latent within done_threshold of the goal latent
        import torch as _torch
        z_final = encode_and_project(policy.jepa_model, obs, policy.device)
        final_dist = _torch.norm(z_final - z_goal_t, p=2, dim=-1).item()
        success = float(final_dist < done_threshold)
        successes.append(success)

        if diagnose and ep == 0:
            print(
                f"  [task={task_name} ep={ep} DONE  success={bool(success)}]"
                f"  final_dist={final_dist:.3f}  threshold={done_threshold:.3f}  steps={step}"
            )

    mean_sr = np.mean(successes)
    n_ok = int(sum(successes))
    print(f"  {task_name}: {mean_sr*100:5.1f}%  ({n_ok}/{num_episodes})")
    return mean_sr, successes


def run_task(env, policy, task_id, task_name, num_episodes, max_steps,
             diagnose=False, goal_info_key='target'):
    """Evaluate one task.

    Args:
        goal_info_key: Key under which the goal image is stored in info.
            'target' for swm/OGBCube-v0 (224x224), 'goal' for visual-cube-single-v0 (64x64).
    """
    successes = []
    for ep in range(num_episodes):
        obs, info = env.reset(options=dict(task_id=task_id))
        goal = info[goal_info_key]   # (H, W, 3) uint8
        policy.reset()
        done  = False
        step  = 0
        diag_log_step = 0  # counts WM-step boundaries (every 5 physical steps)
        while not done and step < max_steps:
            action, diag = policy.get_action(obs, goal)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            step += 1
            if diagnose and ep == 0 and diag is not None:
                diag_log_step += 1
                if diag_log_step % 4 == 0:   # print every ~20 physical steps
                    d2g  = diag['dist_to_goal']
                    d2sg = diag['dist_to_subgoal']
                    sg2g = diag.get('dist_subgoal_to_goal', float('nan'))
                    print(
                        f"  [task={task_id} ep={ep} phys_step={step:3d}]"
                        f"  dist_to_goal={d2g:.3f}"
                        f"  dist_to_subgoal={d2sg:.3f}"
                        f"  subgoal_to_goal={sg2g:.3f}"
                        f"  subgoal_switches={diag['subgoal_switches']:3d}"
                        f"  action_norm={diag['action_norm']:.3f}"
                    )
        # Read success from the final step's info, matching OGBench evaluation.py
        successes.append(float(info.get('success', 0.0)))
        if diagnose and ep == 0:
            # re-encode final obs to get final distance
            import torch as _torch
            z_f = encode_obs(policy.jepa_model, obs, policy.device)
            z_g = encode_obs(policy.jepa_model, goal, policy.device)
            final_dist = _torch.norm(z_f - z_g, p=2, dim=-1).item()
            print(
                f"  [task={task_id} ep={ep} DONE  success={bool(successes[-1])}]"
                f"  final_dist_to_goal={final_dist:.3f}  steps={step}"
            )

    mean_sr = np.mean(successes)
    n_ok = int(sum(successes))
    print(f"  Task {task_id:d} ({task_name}): {mean_sr*100:5.1f}%  ({n_ok}/{num_episodes})")
    return mean_sr, successes


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OGBench-native evaluation — directly comparable to Table 2.")
    parser.add_argument('--ckpt_path',      default=None,
                        help="Path to JEPA checkpoint. For 224×224: pass base path without "
                             "_object.ckpt suffix (e.g. $STABLEWM_HOME/cube/lejepa), loaded "
                             "via swm.policy.AutoCostModel. For 64×64: pass the Lightning "
                             ".ckpt (state dict). Default auto-selected by --img_size.")
    parser.add_argument('--checkpoint_dir', required=True,
                        help="Directory containing actor_policy.pth")
    parser.add_argument('--dataset_path',   default=None,
                        help="Path to HDF5 dataset without .h5 extension, used to fit action scaler. "
                             "Default: $STABLEWM_HOME/ogbench/cube_single_expert")
    parser.add_argument('--ogbench_dir',    default=None,
                        help="Root of cloned ogbench repo (added to sys.path if provided)")
    parser.add_argument('--num_episodes',   type=int, default=50,
                        help="Episodes per task (paper: 50 → 250 total)")
    parser.add_argument('--max_steps',      type=int, default=200,
                        help="Safety cap on steps per episode (env's TimeLimit also enforces 200)")
    parser.add_argument('--results_dir',    default=None,
                        help="Where to write results.txt (default: eval_native_{exp_name}/)")
    parser.add_argument('--high_level_model_type', default='none',
                        choices=['none', 'mlp', 'diffusion'],
                        help="High-level subgoal model. Use 'none' for flat policy.")
    parser.add_argument('--high_level_ckpt_dir', default='./checkpoints_high_level',
                        help="Root dir containing mlp_gap{N}/ subdirs (fixed-gap hierarchical only)")
    parser.add_argument('--done_threshold', type=float, default=2.41,
                        help="Latent L2 threshold below which a subgoal is considered reached "
                             "(hierarchical only). Should match training done_threshold. "
                             "Default 2.41 = median 1-step predictor drift for the 224x224 model.")
    parser.add_argument('--img_size', type=int, default=224,
                        help="Image resolution for the JEPA encoder. "
                             "64 → uses visual-cube-single-v0 (native 64x64), "
                             "224 → uses swm/OGBCube-v0 (native 224x224).")
    parser.add_argument('--patch_size', type=int, default=14,
                        help="ViT patch size (8 for 64x64, 14 for 224x224).")
    parser.add_argument('--subgoal_steps', type=int, default=None,
                        help="HIQL subgoal horizon. Auto-detected from checkpoint dir name "
                             "(e.g. checkpoints_hiql_lewm_k5_*). Override here if needed.")
    parser.add_argument('--rep_dim', type=int, default=None,
                        help="Baseline HIQL goal-rep dimension. Auto-detected from checkpoint "
                             "dir name (e.g. checkpoints_hiql_baseline_k5_rep10). "
                             "Required when evaluating a baseline checkpoint.")
    parser.add_argument('--decoder_ckpt', type=str, default=None,
                        help="Path to action_decoder.pth. Used by WGSP eval when the "
                             "decoder is not co-located in --checkpoint_dir. Ignored if "
                             "the checkpoint dir already contains action_decoder.pth.")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--diagnose', action='store_true',
                        help="Print per-step diagnostics for episode 0 of each task: "
                             "dist_to_goal, dist_to_subgoal, action_norm, subgoal_switches.")
    parser.add_argument('--pca_path', type=str, default=None,
                        help="Path to PCA projection params (.pt) from build_pca_projection.py. "
                             "If provided, latents are projected to pca_dim before the policy. "
                             "Must match the --pca_path used during training.")
    parser.add_argument('--faiss_index_path', type=str, default=None,
                        help="Path to FAISS index (.index) from build_pca_projection.py. "
                             "If provided, high-level subgoals are snapped to nearest cached latent "
                             "(ensures subgoals are in-distribution). Only used in hierarchical mode.")
    parser.add_argument('--lewm_goals_path', type=str, default=None,
                        help="Path to lewm_eval_goals.pt (from build_lewm_eval_goals.py). "
                             "When set, evaluates against 5 in-distribution goals sampled from "
                             "the lewm cache instead of the 5 hardcoded OGBench task goals. "
                             "Success is measured by L2(z_final, z_goal) < done_threshold.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Resolve default paths from STABLEWM_HOME ─────────────────────────────
    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    if args.ckpt_path is None:
        if args.img_size == 224:
            # base path — AutoCostModel appends _object.ckpt automatically
            args.ckpt_path = os.path.join(stablewm_home, "cube", "lejepa")
        else:
            args.ckpt_path = os.path.join(stablewm_home, "ogbench", "lewm_ogbench_weights.ckpt")
    if args.dataset_path is None:
        args.dataset_path = os.path.join(stablewm_home, "ogbench", "cube_single_expert")
    print(f"Weights:  {args.ckpt_path}")
    print(f"Dataset:  {args.dataset_path}")

    # ── Import ogbench ────────────────────────────────────────────────────────
    if args.ogbench_dir:
        sys.path.insert(0, os.path.abspath(args.ogbench_dir))
    import ogbench   # registers gymnasium environments on import
    import gymnasium

    # ── Environment ───────────────────────────────────────────────────────────
    # For 64x64 models: use native OGBench visual-cube-single-v0 which renders
    # at 64x64 pixels and stores the goal under info['goal'].
    # For 224x224 models: use swm/OGBCube-v0 which renders at 224x224 and
    # stores the goal under info['target'].
    if args.img_size == 64:
        env_name      = 'visual-cube-single-v0'
        goal_info_key = 'goal'
        print(f"Creating {env_name} environment (native 64x64 rendering)...")
        env = gymnasium.make(env_name)
    else:
        env_name      = 'swm/OGBCube-v0'
        goal_info_key = 'target'
        print(f"Creating {env_name} environment (native 224x224 rendering)...")
        env = gymnasium.make(env_name, ob_type='pixels', env_type='single', visualize_info=False)
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks  = len(task_infos)
    print(f"  {num_tasks} predefined tasks: {[t.get('task_name','?') for t in task_infos]}")

    # ── Action scaler ─────────────────────────────────────────────────────────
    print("Fitting action scaler from play dataset...")
    dataset = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=['action'],
        cache_dir=str(Path(args.dataset_path).parent),
    )
    action_data = dataset.get_col_data('action')
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Action scaler fit on {len(action_data):,} steps.")

    # ── Image transform (resize OGBench 64x64 → encoder's expected size) ────
    global _IMG_TRANSFORM
    _IMG_TRANSFORM = _make_img_transform(args.img_size)
    print(f"Image transform: OGBench obs → {args.img_size}x{args.img_size}")

    # ── JEPA encoder ──────────────────────────────────────────────────────────
    print("Loading JEPA vision encoder...")
    jepa_model = load_jepa(args.ckpt_path, device,
                           img_size=args.img_size, patch_size=args.patch_size)

    # ── PCA projection (optional) ─────────────────────────────────────────────
    global _PCA_MEAN, _PCA_MATRIX, _FAISS_INDEX
    latent_dim_rl = 192  # default; overridden below if PCA provided

    if args.pca_path is not None:
        print(f"Loading PCA projection from {args.pca_path} ...")
        pca_data = torch.load(args.pca_path, map_location='cpu')
        _PCA_MEAN   = pca_data['pca_mean'].to(device)    # [192]
        _PCA_MATRIX = pca_data['pca_matrix'].to(device)  # [192, D]
        latent_dim_rl = int(pca_data['pca_dim'])
        top_k_var = pca_data.get('top_k_variance', float('nan'))
        print(f"  PCA: 192D → {latent_dim_rl}D  (top-k variance: {top_k_var*100:.1f}%)")

    if args.faiss_index_path is not None:
        print(f"Loading FAISS index from {args.faiss_index_path} ...")
        import faiss as _faiss
        _FAISS_INDEX = _faiss.read_index(args.faiss_index_path)
        print(f"  FAISS: {_FAISS_INDEX.ntotal:,} vectors, dim={_FAISS_INDEX.d}")
        if not isinstance(_FAISS_INDEX, _faiss.IndexFlat):
            # Wrap in IDMap to support reconstruct() if needed
            pass

    # ── Detect checkpoint format & mode ───────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)

    # Detect HIQL checkpoint variant:
    #   - Baseline  (train_hiql_baseline.py):  ll/hl + goal_rep.pth, dir contains 'baseline'
    #   - LeWM v2   (train_hiql_lewm.py NEW):  ll/hl + goal_rep.pth, dir contains 'lewm'
    #     (HER goal sampling + 10D learned goal_rep, matching ogbench HIQL)
    #   - LeWM v1   (train_hiql_lewm.py OLD):  ll/hl only (192D native, no goal_rep)
    # train_hiql_flow.py saves ll_onestep.pth (+ ll_bc_flow.pth) instead of ll_actor.pth
    is_hiql_flow     = (ckpt_dir / 'll_onestep.pth').exists()
    is_hiql          = (ckpt_dir / 'll_actor.pth').exists() or is_hiql_flow
    has_goal_rep     = is_hiql and (ckpt_dir / 'goal_rep.pth').exists()
    is_hiql_baseline = has_goal_rep and 'baseline' in str(ckpt_dir).lower()
    is_hiql_wgsp     = (has_goal_rep and 'wgsp' in str(ckpt_dir).lower()
                        and not is_hiql_flow and not is_hiql_baseline)
    is_hiql_lewm_v2  = (has_goal_rep and not is_hiql_baseline
                        and not is_hiql_wgsp and not is_hiql_flow)
    is_hiql_v1       = is_hiql and not has_goal_rep and not is_hiql_flow

    # Auto-detect subgoal steps: checkpoints_hiql_{lewm,baseline}_k{N}_*
    hiql_k_match  = re.search(r'_k(\d+)', args.checkpoint_dir)
    subgoal_steps = args.subgoal_steps or (int(hiql_k_match.group(1)) if hiql_k_match else 8)

    # Auto-detect rep_dim for baseline / lewm-v2: checkpoints_hiql_*_rep{D}
    rep_dim_match = re.search(r'_rep(\d+)', args.checkpoint_dir)
    rep_dim = args.rep_dim or (int(rep_dim_match.group(1)) if rep_dim_match else None)
    if (is_hiql_baseline or is_hiql_lewm_v2) and rep_dim is None:
        raise ValueError(
            "Checkpoint with goal_rep.pth detected but rep_dim could not be "
            "auto-detected from the checkpoint dir name. Pass --rep_dim explicitly."
        )

    # Old-format gap detection
    joint_gap_match = re.search(r'joint_gap_(\d+)', args.checkpoint_dir)
    fixed_gap_match = re.search(r'fixed_gap_(\d+)', args.checkpoint_dir)
    if joint_gap_match:
        gap = int(joint_gap_match.group(1))
        mode_tag = 'joint'
    elif fixed_gap_match:
        gap = int(fixed_gap_match.group(1))
        mode_tag = 'fixed'
    else:
        gap = None
        mode_tag = 'flat'

    # ── Load actor(s) & build policy ──────────────────────────────────────────
    if is_hiql_baseline:
        # ── Baseline HIQL checkpoint (train_hiql_baseline.py) ─────────────────
        # HL: input=384 (192+192), output=rep_dim.  LL: input=192+rep_dim, output=5.
        # Native 5D actions, no chunking — encodes every env step.
        ll_ckpt = ckpt_dir / 'll_actor.pth'
        hl_ckpt = ckpt_dir / 'hl_actor.pth'
        print(f"Loading HIQL-baseline actors (k={subgoal_steps}, rep_dim={rep_dim}) ...")

        ll_actor = _BaselineGaussianActor(
            input_dim=192 + rep_dim, output_dim=5,
            hidden_dims=(512, 512, 512),
            tanh_squash=False,
        ).to(device)
        ll_actor.load_state_dict(torch.load(ll_ckpt, map_location=device))
        ll_actor.eval()
        for p in ll_actor.parameters():
            p.requires_grad_(False)

        hl_actor_net = _BaselineGaussianActor(
            input_dim=192 * 2, output_dim=rep_dim,
            hidden_dims=(512, 512, 512),
            tanh_squash=False,
        ).to(device)
        hl_actor_net.load_state_dict(torch.load(hl_ckpt, map_location=device))
        hl_actor_net.eval()
        for p in hl_actor_net.parameters():
            p.requires_grad_(False)

        hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)
        policy = _BaselineNativeHierarchicalPolicy(
            jepa_model, ll_actor, action_scaler, hl_model,
            gap=subgoal_steps, device=device)
        print(f"  Policy: HIQL-baseline Native Hierarchical (k={subgoal_steps}, rep_dim={rep_dim})")

    elif is_hiql_flow:
        # ── Flow LL policy (train_hiql_flow.py) ───────────────────────────────
        # HL: GaussianActor (state=192, goal=192) → rep_dim  (same as lewm_v2)
        # LL: LewmOneStep  (state=192, goal=rep_dim, ε=25)   → action 25 (no squash)
        # At inference: deterministic mode uses ε=0.
        ll_ckpt = ckpt_dir / 'll_onestep.pth'
        hl_ckpt = ckpt_dir / 'hl_actor.pth'
        print(f"Loading HIQL+LeWM+Flow actors (k={subgoal_steps}, rep_dim={rep_dim}) ...")

        ll_actor = _LewmOneStep(
            latent_dim=192, rep_dim=rep_dim, action_dim=25,
            hidden_dims=(512, 512, 512, 512),
        ).to(device)
        ll_actor.load_state_dict(torch.load(ll_ckpt, map_location=device))
        ll_actor.eval()
        for p in ll_actor.parameters():
            p.requires_grad_(False)

        hl_actor_net = _LewmGaussianActor(
            state_dim=192, goal_dim=192, output_dim=rep_dim,
            hidden_dims=(512, 512, 512), tanh_squash=False,
        ).to(device)
        hl_actor_net.load_state_dict(torch.load(hl_ckpt, map_location=device))
        hl_actor_net.eval()
        for p in hl_actor_net.parameters():
            p.requires_grad_(False)

        hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)
        policy = HierarchicalPolicy(
            jepa_model, ll_actor, action_scaler, hl_model,
            gap=subgoal_steps, device=device,
            subgoal_reached_threshold=0.0)
        print(f"  Policy: HIQL+LeWM+Flow Hierarchical (k={subgoal_steps}, rep_dim={rep_dim})")

    elif is_hiql_wgsp:
        # ── WGSP checkpoint (train_hiql_wgsp.py) ─────────────────────────────
        # HL: state=192, goal=192, output=rep_dim, no squash       (same as v2)
        # LL: state=192, goal=rep_dim, output=5 OR 25, tanh_squash=True
        #     5 if action_decoder.pth is present in ckpt_dir, else 25.
        ll_ckpt   = ckpt_dir / 'll_actor.pth'
        hl_ckpt   = ckpt_dir / 'hl_actor.pth'
        dec_ckpt  = ckpt_dir / 'action_decoder.pth'
        # Decoder may live in a sibling dir (the sweep places it there); allow override.
        if not dec_ckpt.exists() and getattr(args, 'decoder_ckpt', None):
            dec_ckpt = Path(args.decoder_ckpt)
        use_decoder = dec_ckpt.exists()
        ll_out_dim  = 5 if use_decoder else 25
        print(f"Loading WGSP actors (k={subgoal_steps}, rep_dim={rep_dim}, "
              f"LL_out={ll_out_dim}, decoder={'yes' if use_decoder else 'no'}) ...")

        ll_actor = _LewmGaussianActor(
            state_dim=192, goal_dim=rep_dim, output_dim=ll_out_dim,
            hidden_dims=(512, 512, 512),
            tanh_squash=True, action_scale=3.0,
        ).to(device)
        ll_actor.load_state_dict(torch.load(ll_ckpt, map_location=device))
        ll_actor.eval()
        for p in ll_actor.parameters():
            p.requires_grad_(False)

        hl_actor_net = _LewmGaussianActor(
            state_dim=192, goal_dim=192, output_dim=rep_dim,
            hidden_dims=(512, 512, 512),
            tanh_squash=False,
        ).to(device)
        hl_actor_net.load_state_dict(torch.load(hl_ckpt, map_location=device))
        hl_actor_net.eval()
        for p in hl_actor_net.parameters():
            p.requires_grad_(False)

        decoder = None
        if use_decoder:
            decoder = _ActionChunkDecoder(in_dim=5, out_dim=25, latent_dim=192,
                                          hidden_dims=(256, 256)).to(device)
            decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad_(False)
            print(f"  Loaded ActionChunkDecoder from {dec_ckpt}")

        hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)
        policy = _WGSPHierarchicalPolicy(
            jepa_model, ll_actor, action_scaler, hl_model,
            gap=subgoal_steps, device=device, decoder=decoder,
            subgoal_reached_threshold=0.0,
        )
        print(f"  Policy: WGSP Hierarchical (k={subgoal_steps}, rep_dim={rep_dim})")

    elif is_hiql_lewm_v2:
        # ── LeWM-hybrid HIQL with HER + 10D goal_rep (train_hiql_lewm.py) ────
        # HL: state=192, goal=192, output=rep_dim, no squash.
        # LL: state=192, goal=rep_dim, output=25, tanh_squash=True (action_scale=3.0).
        # goal_rep.pth is NOT needed at inference — HL predicts phi directly,
        # length-normalised onto sphere of radius √rep_dim (matches ogbench).
        ll_ckpt = ckpt_dir / 'll_actor.pth'
        hl_ckpt = ckpt_dir / 'hl_actor.pth'
        print(f"Loading HIQL+LeWM v2 actors (k={subgoal_steps}, rep_dim={rep_dim}) ...")

        ll_actor = _LewmGaussianActor(
            state_dim=192, goal_dim=rep_dim, output_dim=25,
            hidden_dims=(512, 512, 512),
            tanh_squash=True, action_scale=3.0,
        ).to(device)
        ll_actor.load_state_dict(torch.load(ll_ckpt, map_location=device))
        ll_actor.eval()
        for p in ll_actor.parameters():
            p.requires_grad_(False)

        hl_actor_net = _LewmGaussianActor(
            state_dim=192, goal_dim=192, output_dim=rep_dim,
            hidden_dims=(512, 512, 512),
            tanh_squash=False,
        ).to(device)
        hl_actor_net.load_state_dict(torch.load(hl_ckpt, map_location=device))
        hl_actor_net.eval()
        for p in hl_actor_net.parameters():
            p.requires_grad_(False)

        hl_model = _HIQLBaselineHighLevelWrapper(hl_actor_net)
        # HL output is in rep_dim space (length-normalised) — spatial threshold
        # checking on raw 192D doesn't translate, use time-based switching only.
        policy = HierarchicalPolicy(
            jepa_model, ll_actor, action_scaler, hl_model,
            gap=subgoal_steps, device=device,
            subgoal_reached_threshold=0.0)
        print(f"  Policy: HIQL+LeWM v2 Hierarchical (k={subgoal_steps}, rep_dim={rep_dim})")

    elif is_hiql_v1:
        # ── Legacy hybrid HIQL checkpoint (192D native, no goal_rep) ─────────
        ll_ckpt = ckpt_dir / 'll_actor.pth'
        hl_ckpt = ckpt_dir / 'hl_actor.pth'
        if not ll_ckpt.exists():
            raise FileNotFoundError(f"HIQL ll_actor checkpoint not found: {ll_ckpt}")
        if not hl_ckpt.exists():
            raise FileNotFoundError(f"HIQL hl_actor checkpoint not found: {hl_ckpt}")

        print(f"Loading HIQL actors (k={subgoal_steps}, 192D native) ...")
        ll_actor = _GaussianActor(
            latent_dim=192, output_dim=25,
            hidden_dims=(512, 512, 512),
            tanh_squash=True, action_scale=3.0,
        ).to(device)
        ll_actor.load_state_dict(torch.load(ll_ckpt, map_location=device))
        ll_actor.eval()
        for p in ll_actor.parameters():
            p.requires_grad_(False)

        hl_actor_net = _GaussianActor(
            latent_dim=192, output_dim=192,
            hidden_dims=(512, 512, 512),
            tanh_squash=False,
        ).to(device)
        hl_actor_net.load_state_dict(torch.load(hl_ckpt, map_location=device))
        hl_actor_net.eval()
        for p in hl_actor_net.parameters():
            p.requires_grad_(False)

        hl_model = _HIQLHighLevelWrapper(hl_actor_net)
        policy = HierarchicalPolicy(
            jepa_model, ll_actor, action_scaler, hl_model,
            gap=subgoal_steps, device=device,
            subgoal_reached_threshold=args.done_threshold)
        print(f"  Policy: HIQL Hierarchical (k={subgoal_steps}, done_threshold={args.done_threshold})")

    else:
        # ── Old SAC/joint checkpoint (train_joint.py) ─────────────────────────
        actor_ckpt = ckpt_dir / 'actor_policy.pth'
        if not actor_ckpt.exists():
            raise FileNotFoundError(f"No actor checkpoint at {actor_ckpt}")
        print(f"Loading SAC actor from {actor_ckpt} (latent_dim={latent_dim_rl}) ...")
        actor = GoalConditionedActor(latent_dim=latent_dim_rl, action_dim=25).to(device)
        actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

    if not is_hiql:
        if mode_tag == 'joint':
            # Joint mode: high_actor.pth co-located with actor_policy.pth.
            # HighLevelActor (train_joint.py) and MLPHighLevel share the same
            # nn.Sequential architecture, so the state dict cross-loads directly.
            hl_ckpt = ckpt_dir / "high_actor.pth"
            if not hl_ckpt.exists():
                raise FileNotFoundError(f"Joint high_actor checkpoint not found: {hl_ckpt}")
            print(f"Loading joint-trained high-level actor (gap={gap}, latent_dim={latent_dim_rl}) ...")
            hl_model = MLPHighLevel(latent_dim=latent_dim_rl).to(device)
            hl_model.load_state_dict(torch.load(hl_ckpt, map_location=device))
            hl_model.eval()
            for p in hl_model.parameters():
                p.requires_grad_(False)
            policy = HierarchicalPolicy(
                jepa_model, actor, action_scaler, hl_model, gap, device,
                subgoal_reached_threshold=args.done_threshold)
            print(f"  Policy: Hierarchical (joint-trained, done_threshold={args.done_threshold})")
        elif mode_tag == 'fixed' and args.high_level_model_type != 'none':
            hl_ckpt = (Path(args.high_level_ckpt_dir)
                       / f"{args.high_level_model_type}_gap{gap}"
                       / "best_model.pth")
            if not hl_ckpt.exists():
                raise FileNotFoundError(f"High-level checkpoint not found: {hl_ckpt}")
            print(f"Loading {args.high_level_model_type} high-level (gap={gap}, latent_dim={latent_dim_rl}) ...")
            hl_model = (MLPHighLevel(latent_dim=latent_dim_rl) if args.high_level_model_type == 'mlp'
                        else DiffusionHighLevel(latent_dim=latent_dim_rl)).to(device)
            hl_model.load_state_dict(torch.load(hl_ckpt, map_location=device))
            hl_model.eval()
            for p in hl_model.parameters():
                p.requires_grad_(False)
            policy = HierarchicalPolicy(
                jepa_model, actor, action_scaler, hl_model, gap, device,
                subgoal_reached_threshold=args.done_threshold)
            print("  Policy: Hierarchical")
        else:
            if gap is not None:
                print(f"  Fixed-gap={gap} checkpoint, running flat (no high-level).")
            policy = FlatPolicy(jepa_model, actor, action_scaler, device)
            print("  Policy: Flat")

    # ── Results directory ─────────────────────────────────────────────────────
    exp_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
    if args.high_level_model_type != 'none':
        exp_name += f"_hl_{args.high_level_model_type}"
    results_dir = Path(args.results_dir or f"eval_native_{exp_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OGBench native evaluation: {exp_name}")
    print(f"  {num_tasks} tasks × {args.num_episodes} eps = {num_tasks * args.num_episodes} total")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"{'='*60}")

    t0 = time.time()
    per_task = {}
    all_successes = []

    if args.lewm_goals_path is not None:
        # ── In-distribution lewm goal evaluation ─────────────────────────────
        print(f"Loading in-distribution eval goals from {args.lewm_goals_path} ...")
        goals_data   = torch.load(args.lewm_goals_path, map_location='cpu', weights_only=False)
        goal_latents = goals_data['goal_latents']   # [N, 192]
        task_names   = goals_data['task_names']
        n_goals      = len(task_names)
        protocol_str = f"lewm in-distribution — {n_goals} goals × {args.num_episodes} episodes"
        print(f"  {n_goals} goals: {task_names}")
        print(f"  Success threshold: L2 < {args.done_threshold:.4f}")

        for i, task_name in enumerate(task_names):
            mean_sr, eps = run_task_lewm(
                env, policy,
                goal_latent     = goal_latents[i:i+1],   # [1, 192]
                task_name       = task_name,
                num_episodes    = args.num_episodes,
                max_steps       = args.max_steps,
                done_threshold  = args.done_threshold,
                diagnose        = args.diagnose,
            )
            per_task[task_name] = mean_sr
            all_successes.extend(eps)
    else:
        # ── Standard OGBench 5-task evaluation ───────────────────────────────
        protocol_str = f"OGBench native — 5 tasks × {args.num_episodes} episodes"
        for task_id in range(1, num_tasks + 1):
            task_name = task_infos[task_id - 1].get('task_name', f'task{task_id}')
            mean_sr, eps = run_task(
                env, policy, task_id, task_name, args.num_episodes, args.max_steps,
                diagnose=args.diagnose, goal_info_key=goal_info_key)
            per_task[task_name] = mean_sr
            all_successes.extend(eps)

    overall = np.mean(all_successes)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {exp_name}")
    print(f"{'='*60}")
    for name, sr in per_task.items():
        print(f"  {name:<30s} {sr*100:5.1f}%")
    print(f"  {'─'*36}")
    print(f"  {'overall':<30s} {overall*100:5.1f}%")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")

    out = results_dir / "results.txt"
    with out.open("a") as f:
        f.write(f"\n==== {exp_name} ====\n")
        f.write(f"protocol: {protocol_str}\n")
        f.write(f"checkpoint: {args.checkpoint_dir}\n")
        for name, sr in per_task.items():
            f.write(f"  {name}: {sr*100:.1f}%\n")
        f.write(f"  overall: {overall*100:.1f}%\n")
        f.write(f"elapsed: {elapsed:.0f}s\n")
    print(f"Results saved to {out}")
    env.close()


if __name__ == "__main__":
    main()
