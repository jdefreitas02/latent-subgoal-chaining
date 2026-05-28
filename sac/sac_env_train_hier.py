"""
Hierarchical SAC + HER — real OGBench env rollouts (env-rollout counterpart to sac_train.py).

Algorithm matches sac_train.py exactly except rollouts come from the real OGBench
environment (JEPA-encoded) rather than the LatentEnv WM predictor.  This gives
a direct WM-vs-env comparison for the Hierarchical SAC with Expert-Guided
Stabilisation (thesis §Hierarchical SAC).

Key design decisions (matching sac_train.py):
  - GoalConditionedActor / TwinCritic with action_scale=3.0
  - VectorizedEpisodicHERBuffer with on-the-fly HER future-strategy sampling
  - Sparse reward computed inside the buffer at sample time (not at collection)
  - Optional BC regularisation: --bc_alpha / --bc_model_path
  - log_alpha initialised at 0.0 (α=1.0) + floor clamp at -4.6 (α≥0.01)

Only differences from sac_train.py:
  - No LatentEnv / no curriculum / no task teleportation (can't do with real env)
  - Sequential: 5 episodes/iter (one per OGBench task_id), ~40 actor steps each
  - Each actor step = 1 call to actor.sample → 5 physical env steps (action_block=5)
  - Observation encoded with JEPA at every step

Usage:
    python latent_hindsight_rl/sac/sac_env_train_hier.py \\
        --ckpt_path    $STABLEWM_HOME/cube/lejepa \\
        --dataset_path $STABLEWM_HOME/ogbench/visual-cube-single-play-v0_224 \\
        --num_iters    5000 \\
        --save_dir     ./checkpoints_sac_env_hier_s0
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import csv
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from pathlib import Path
from torchvision.transforms import v2 as transforms
from sklearn import preprocessing

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path: sys.path.insert(0, _THIS_DIR)
if _ROOT     not in sys.path: sys.path.insert(0, _ROOT)

import stable_pretraining as spt
import stable_worldmodel as swm
from bc_policy import BCPolicy


# =============================================================================
# NETWORKS  (identical to sac_train.py — action_scale=3.0)
# =============================================================================

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

def _weights_init(m):
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
        self.apply(_weights_init)
        nn.init.uniform_(self.mean_linear.weight,    -1e-3, 1e-3)
        nn.init.constant_(self.mean_linear.bias,      0)
        nn.init.uniform_(self.log_std_linear.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std_linear.bias,  -1.0)

    def forward(self, state, goal):
        x = F.relu(self.ln1(self.linear1(torch.cat([state, goal], dim=-1))))
        x = F.relu(self.ln2(self.linear2(x)))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std    = log_std.exp()
        x_t    = Normal(mean, std).rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + EPS)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale


class TwinCritic(nn.Module):
    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256):
        super().__init__()
        in_dim = latent_dim * 2 + action_dim
        def _net():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1 = _net()
        self.q2 = _net()
        self.apply(_weights_init)
        for net in (self.q1, self.q2):
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias,   0)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)
        return self.q1(x), self.q2(x)


# =============================================================================
# REPLAY BUFFER  (identical to sac_train.py — VectorizedEpisodicHERBuffer)
# =============================================================================

class VectorizedEpisodicHERBuffer:
    """GPU-native episodic buffer with on-the-fly HER future-strategy sampling.

    Identical to the buffer in sac_train.py so the SAC update code is the same
    between the WM and env variants.
    """
    def __init__(self, latent_dim=192, action_dim=25,
                 capacity_episodes=20000, max_t=50, future_p=0.8, device="cuda"):
        self.capacity = capacity_episodes
        self.max_t    = max_t
        self.future_p = future_p
        self.device   = device
        self.z_curr    = torch.zeros((capacity_episodes, max_t, latent_dim),  device=device)
        self.actions   = torch.zeros((capacity_episodes, max_t, action_dim),  device=device)
        self.z_next    = torch.zeros((capacity_episodes, max_t, latent_dim),  device=device)
        self.original_g = torch.zeros((capacity_episodes, max_t, latent_dim), device=device)
        self.ep_lens   = torch.zeros((capacity_episodes,), dtype=torch.long, device=device)
        self.position  = 0
        self.size      = 0

    @property
    def num_transitions(self):
        return int(self.ep_lens[:self.size].sum().item())

    def store_episodes(self, z_curr_seq, actions_seq, z_next_seq, target_seq, lengths):
        num_new = z_curr_seq.shape[0]
        seq_len = z_curr_seq.shape[1]
        end_idx = self.position + num_new
        if end_idx <= self.capacity:
            s, e = self.position, end_idx
            self.z_curr[s:e, :seq_len]    = z_curr_seq
            self.actions[s:e, :seq_len]   = actions_seq
            self.z_next[s:e, :seq_len]    = z_next_seq
            self.original_g[s:e, :seq_len] = target_seq
            self.ep_lens[s:e]             = lengths
        else:
            overflow = end_idx - self.capacity
            valid    = num_new - overflow
            self.z_curr[self.position:self.capacity, :seq_len]    = z_curr_seq[:valid]
            self.actions[self.position:self.capacity, :seq_len]   = actions_seq[:valid]
            self.z_next[self.position:self.capacity, :seq_len]    = z_next_seq[:valid]
            self.original_g[self.position:self.capacity, :seq_len] = target_seq[:valid]
            self.ep_lens[self.position:self.capacity]             = lengths[:valid]
            self.z_curr[0:overflow, :seq_len]    = z_curr_seq[valid:]
            self.actions[0:overflow, :seq_len]   = actions_seq[valid:]
            self.z_next[0:overflow, :seq_len]    = z_next_seq[valid:]
            self.original_g[0:overflow, :seq_len] = target_seq[valid:]
            self.ep_lens[0:overflow]             = lengths[valid:]
        self.position = end_idx % self.capacity
        self.size = min(self.size + num_new, self.capacity)

    def sample_batch(self, batch_size=256, reward_mode='sparse'):
        ep_idxs     = torch.randint(0, self.size, (batch_size,), device=self.device)
        sampled_lens = self.ep_lens[ep_idxs]
        safe_lens   = torch.clamp(sampled_lens, min=1)
        t_idxs      = (torch.rand(batch_size, device=self.device) * safe_lens).long()
        t_idxs      = torch.clamp(t_idxs, max=safe_lens - 1)

        z_curr_b  = self.z_curr[ep_idxs, t_idxs]
        a_b       = self.actions[ep_idxs, t_idxs]
        z_next_b  = self.z_next[ep_idxs, t_idxs]
        orig_g_b  = self.original_g[ep_idxs, t_idxs]

        valid_future = (t_idxs < sampled_lens - 1)
        her_mask     = (torch.rand(batch_size, device=self.device) < self.future_p) & valid_future
        range_len    = sampled_lens - 1 - t_idxs
        safe_range   = torch.clamp(range_len, min=1)
        offsets      = (torch.rand(batch_size, device=self.device) * safe_range).long() + 1
        future_t     = torch.clamp(t_idxs + offsets, max=sampled_lens - 1)
        future_g_b   = self.z_curr[ep_idxs, future_t]
        g_target_b   = torch.where(her_mask.unsqueeze(-1), future_g_b, orig_g_b)

        action_penalty = torch.norm(a_b, p=2, dim=-1) * 0.01
        dist_next      = torch.norm(z_next_b - g_target_b, p=2, dim=-1)
        success        = dist_next < 2.0
        done_b         = success.float()

        if reward_mode == 'dense':
            dist_curr   = torch.norm(z_curr_b - g_target_b, p=2, dim=-1)
            improvement = (dist_curr - dist_next) / 1.6
            improvement = torch.clamp(improvement, -2.0, 2.0)
            r_b = improvement - action_penalty
        else:
            r_b = torch.where(success,
                              torch.zeros_like(dist_next),
                              -torch.ones_like(dist_next)) - action_penalty

        return z_curr_b, a_b, z_next_b, g_target_b, r_b, done_b


# =============================================================================
# JEPA LOADER + TRANSFORM  (mirrors sac_env_train.py)
# =============================================================================

def load_jepa(ckpt_path, device="cuda", img_size=224, patch_size=14):
    if img_size == 224:
        model = swm.policy.AutoCostModel(ckpt_path)
        print(f"  Loaded 224×224 JEPA via AutoCostModel from {ckpt_path}")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder   = spt.backbone.utils.vit_hf(
        "tiny", patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector  = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                     norm_fn=nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                     norm_fn=nn.BatchNorm1d)
    from jepa import JEPA
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        epoch = ckpt.get("epoch", "?")
    else:
        raw_sd = dict(ckpt)
        epoch  = "?"
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded 64×64 JEPA from {ckpt_path} (epoch {epoch})")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_img_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
    ])


# =============================================================================
# OBSERVATION ENCODING
# =============================================================================

@torch.no_grad()
def encode_obs(obs_hwc, model, transform, device):
    t = transform(obs_hwc).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,C,H,W]
    return model.encode({"pixels": t})["emb"][:, -1].squeeze(0)  # [192]


def _goal_from_info(info):
    for key in ("goal", "target", "desired_goal"):
        if key in info and info[key] is not None:
            return info[key]
    raise KeyError(f"No goal image in env info. Keys: {list(info.keys())}")


# =============================================================================
# EPISODE COLLECTION
# =============================================================================

def collect_env_episode_hier(env, actor, jepa_model, transform, action_scaler,
                              task_id, T_max, device):
    """
    Collect one episode from the real OGBench environment.

    Returns tensors ready to be stored as a single episode in
    VectorizedEpisodicHERBuffer via store_episodes(..., batch_size=1).

    Each actor step (25-D action) expands to 5 physical env steps,
    matching the WM training setup (action_block=5).

    Returns:
        z_curr_seq   : [T, 192] — states at each actor step
        actions_seq  : [T, 25]  — actor actions
        z_next_seq   : [T, 192] — next states
        z_goal       : [192]    — encoded episode goal (task goal from env)
        ep_len       : int      — actual episode length T
        success      : bool
    """
    obs, info  = env.reset(options={"task_id": task_id})
    goal_img   = _goal_from_info(info)
    z_goal     = encode_obs(goal_img, jepa_model, transform, device)
    z_curr     = encode_obs(obs, jepa_model, transform, device)

    z_curr_list  = []
    actions_list = []
    z_next_list  = []
    success      = False

    for _ in range(T_max):
        with torch.no_grad():
            action, _, _ = actor.sample(z_curr.unsqueeze(0), z_goal.unsqueeze(0))
        action = action.squeeze(0)  # [25]

        # Reshape [25] → [5 blocks × 5 joints] → physical actions ∈ [-1, 1]
        action_np = action.detach().cpu().numpy().reshape(5, 5)
        physical  = np.clip(action_scaler.inverse_transform(action_np), -1.0, 1.0)

        terminated = truncated = False
        step_info  = info
        for phys_t in range(5):
            obs_new, _, terminated, truncated, step_info = env.step(physical[phys_t])
            if terminated or truncated:
                break

        z_next  = encode_obs(obs_new, jepa_model, transform, device)
        r_sparse = float(step_info.get("success", 0.0))
        success  = success or bool(r_sparse)

        z_curr_list.append(z_curr)
        actions_list.append(action)
        z_next_list.append(z_next)

        z_curr = z_next
        obs    = obs_new
        info   = step_info

        if terminated or truncated or bool(r_sparse):
            break

    ep_len = len(z_curr_list)
    return (
        torch.stack(z_curr_list),   # [T, 192]
        torch.stack(actions_list),  # [T, 25]
        torch.stack(z_next_list),   # [T, 192]
        z_goal,                     # [192]
        ep_len,
        success,
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_loop_hier(env, actor, critic, critic_target,
                    actor_optimizer, critic_optimizer, alpha_optimizer,
                    log_alpha, target_entropy, replay_buffer,
                    jepa_model, transform, action_scaler,
                    num_iterations=5000, gamma=0.99, tau=0.005,
                    T_max=40, num_task_ids=5,
                    reward_mode='sparse', bc_model=None, bc_alpha=0.0,
                    save_dir="./checkpoints_sac_env_hier_s0"):

    os.makedirs(save_dir, exist_ok=True)
    bc_str = f"BC alpha={bc_alpha}" if bc_model is not None else "no BC"
    print(f"Starting HER Env Training | Reward: {reward_mode} | {bc_str} | "
          f"T_max={T_max} | tasks={num_task_ids} | Saving to: {save_dir}")

    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "Iteration", "Success_Rate", "Buffer_Episodes", "Buffer_Transitions",
            "Actor_Loss", "Critic_Loss", "Alpha", "Total_Time",
        ])

    recent_successes = deque(maxlen=num_task_ids * 20)
    grad_updates     = 20 if reward_mode == "dense" else 40
    start_time       = time.time()

    for iteration in range(num_iterations):

        # ---- Collect one episode per task_id ----
        for task_id in range(1, num_task_ids + 1):
            z_curr_seq, actions_seq, z_next_seq, z_goal, ep_len, success = \
                collect_env_episode_hier(
                    env, actor, jepa_model, transform, action_scaler,
                    task_id=task_id, T_max=T_max, device=next(actor.parameters()).device,
                )
            recent_successes.append(float(success))

            # Build goal tensor [T, 192] (same original goal broadcast across timesteps)
            z_goal_seq = z_goal.unsqueeze(0).expand(ep_len, -1)  # [T, 192]

            # Store as batch of size 1 in episodic buffer
            replay_buffer.store_episodes(
                z_curr_seq.unsqueeze(0),    # [1, T, 192]
                actions_seq.unsqueeze(0),   # [1, T, 25]
                z_next_seq.unsqueeze(0),    # [1, T, 192]
                z_goal_seq.unsqueeze(0),    # [1, T, 192]
                torch.tensor([ep_len], device=z_curr_seq.device),
            )

        # ---- SAC updates ----
        avg_actor_loss = avg_critic_loss = 0.0
        device = next(actor.parameters()).device

        if replay_buffer.num_transitions >= 256:
            for _ in range(grad_updates):
                z_b, a_b, z_next_b, g_b, r_b, d_b = replay_buffer.sample_batch(
                    batch_size=256, reward_mode=reward_mode,
                )
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                # Critic update
                with torch.no_grad():
                    next_a, next_log_pi, _ = actor.sample(z_next_b, g_b)
                    tq1, tq2 = critic_target(z_next_b, g_b, next_a)
                    target_q = r_b + (1.0 - d_b) * gamma * (
                        torch.min(tq1, tq2) - alpha * next_log_pi
                    )

                q1, q2 = critic(z_b, g_b, a_b)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
                avg_critic_loss += critic_loss.item()

                # Actor update
                new_actions, log_pi, _ = actor.sample(z_b, g_b)
                for p in critic.parameters():
                    p.requires_grad = False
                q1_new, q2_new = critic(z_b, g_b, new_actions)
                actor_loss = (alpha * log_pi - torch.min(q1_new, q2_new)).mean()

                if bc_model is not None:
                    with torch.no_grad():
                        bc_actions = bc_model(z_b, g_b)
                    actor_loss = actor_loss + bc_alpha * F.mse_loss(new_actions, bc_actions)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
                avg_actor_loss += actor_loss.item()
                for p in critic.parameters():
                    p.requires_grad = True

                # Alpha update with floor clamp
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=-4.6)   # α floor ≈ 0.01

                # Soft target update
                for tp, p in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        sr = float(np.mean(recent_successes)) if recent_successes else 0.0

        if iteration % 10 == 0:
            elapsed   = time.time() - start_time
            actor_val = avg_actor_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            crit_val  = avg_critic_loss / grad_updates if replay_buffer.num_transitions >= 256 else 0.0
            print(f"Iter {iteration:05d} | SR: {sr*100:.1f}% | "
                  f"Buf: {replay_buffer.size} eps / {replay_buffer.num_transitions} tr | "
                  f"Act: {actor_val:.3f} | Crit: {crit_val:.3f} | "
                  f"α: {log_alpha.exp().item():.3f} | t: {elapsed:.1f}s")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    iteration, sr, replay_buffer.size, replay_buffer.num_transitions,
                    actor_val, crit_val, log_alpha.exp().item(), elapsed,
                ])
            start_time = time.time()

        if (iteration > 0 and iteration % 100 == 0) or iteration == num_iterations - 1:
            torch.save(actor.state_dict(),  os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, "critic_network.pth"))
            print(f"  --> Checkpoint saved at iteration {iteration}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical SAC + HER with real OGBench env rollouts"
    )
    parser.add_argument("--ckpt_path",      required=True,
                        help="Path to JEPA checkpoint (224x model: directory; 64x: .ckpt file)")
    parser.add_argument("--dataset_path",   required=True,
                        help="Path to OGBench HDF5 dataset (for action scaler fitting)")
    parser.add_argument("--save_dir",       default="./checkpoints_sac_env_hier_s0")
    parser.add_argument("--num_iters",      type=int,   default=5000)
    parser.add_argument("--T_max",          type=int,   default=40,
                        help="Max actor steps per episode (each = 5 physical env steps)")
    parser.add_argument("--reward_mode",    type=str,   default="sparse",
                        choices=["sparse", "dense"])
    parser.add_argument("--bc_alpha",       type=float, default=0.0,
                        help="BC regularisation coefficient (0 = disabled)")
    parser.add_argument("--bc_model_path",  type=str,   default=None,
                        help="Path to BCPolicy checkpoint (.pth). Required if --bc_alpha > 0")
    parser.add_argument("--img_size",       type=int,   default=224,
                        help="64 → visual-cube-single-v0; 224 → swm/OGBCube-v0")
    parser.add_argument("--patch_size",     type=int,   default=14)
    args = parser.parse_args()

    if args.bc_alpha > 0 and args.bc_model_path is None:
        parser.error("--bc_model_path is required when --bc_alpha > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- JEPA encoder ---
    print("Loading JEPA encoder...")
    jepa_model = load_jepa(args.ckpt_path, device=device,
                           img_size=args.img_size, patch_size=args.patch_size)
    transform  = make_img_transform()

    # --- Action scaler (fit on dataset, same as sac_env_train.py) ---
    print(f"Loading dataset from {args.dataset_path} to fit action scaler...")
    dataset     = swm.data.HDF5Dataset(args.dataset_path,
                                       keys_to_cache=["action"],
                                       cache_dir=str(Path(args.dataset_path).parent))
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)
    print(f"  Action scaler fitted on {len(action_data)} rows.")

    # --- OGBench environment ---
    import gymnasium
    import ogbench  # noqa: F401

    if args.img_size != 64:
        # stable_worldmodel's CubeEnv has two API mismatches with the ogbench base class:
        # 1. References colors ('yellow', 'magenta', etc.) not in ogbench's _colors dict.
        # 2. compute_reward(self) takes no args but ogbench's step() calls compute_reward(ob, action).
        import ogbench.manipspace.envs.manipspace_env as _ms_env
        import stable_worldmodel.envs.ogbench.cube_env as _swm_cube

        _orig_ms_init = _ms_env.ManipSpaceEnv.__init__
        def _patched_ms_init(self, *a, **kw):
            _orig_ms_init(self, *a, **kw)
            self._colors.setdefault('yellow',        np.array([1.0,  0.93, 0.0,  1.0]))
            self._colors.setdefault('magenta',       np.array([0.9,  0.2,  0.6,  1.0]))
            self._colors.setdefault('lightyellow',   np.array([1.0,  0.98, 0.8,  1.0]))
            self._colors.setdefault('lightmagenta',  np.array([0.98, 0.85, 0.92, 1.0]))
        _ms_env.ManipSpaceEnv.__init__ = _patched_ms_init

        # ogbench's step() calls self.compute_reward(ob, action) but stable_worldmodel's
        # CubeEnv.compute_reward(self) takes no args. Our SAC never uses the env reward
        # (reward is computed in the HER buffer), so returning 0.0 is safe.
        def _patched_compute_reward(self, *a, **kw):
            return 0.0
        _swm_cube.CubeEnv.compute_reward = _patched_compute_reward

    if args.img_size == 64:
        env = gymnasium.make("visual-cube-single-v0")
        print("Created visual-cube-single-v0 (64×64).")
    else:
        env = gymnasium.make("swm/OGBCube-v0",
                             ob_type="pixels", env_type="single", visualize_info=False,
                             disable_env_checker=True)
        print("Created swm/OGBCube-v0 (224×224).")

    # --- Save training config ---
    os.makedirs(args.save_dir, exist_ok=True)
    training_config = {
        "action_scale":  3.0,
        "latent_dim":    192,
        "action_dim":    25,
        "reward_type":   f"sparse_her_env_hier_{args.reward_mode}",
        "bc_alpha":      args.bc_alpha,
        "img_size":      args.img_size,
        "patch_size":    args.patch_size,
    }
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training_config.json to {args.save_dir}")

    # --- SAC components (action_scale=3.0 matching sac_train.py) ---
    actor         = GoalConditionedActor(action_dim=25, action_scale=3.0).to(device)
    critic        = TwinCritic(action_dim=25).to(device)
    critic_target = TwinCritic(action_dim=25).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    target_entropy = -float(actor.mean_linear.out_features)  # -25
    # log_alpha=0 (α=1.0) matching SB3 default + floor clamp at 0.01
    log_alpha      = torch.tensor([0.0], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

    replay_buffer = VectorizedEpisodicHERBuffer(
        latent_dim=192, action_dim=25,
        capacity_episodes=20000, max_t=50, future_p=0.8, device=device,
    )

    # --- Optional BC model ---
    bc_model = None
    if args.bc_alpha > 0:
        ckpt = torch.load(args.bc_model_path, map_location=device, weights_only=False)
        bc_model = BCPolicy(
            latent_dim=ckpt.get("latent_dim", 192),
            action_dim=ckpt.get("action_dim", 25),
            action_scale=ckpt.get("action_scale", 3.0),
        ).to(device)
        bc_model.load_state_dict(ckpt["model_state_dict"])
        bc_model.eval()
        for p in bc_model.parameters():
            p.requires_grad = False
        print(f"BC model loaded from {args.bc_model_path} (bc_alpha={args.bc_alpha})")

    print(f"\nStarting hierarchical env SAC+HER | iters={args.num_iters} | "
          f"T_max={args.T_max} | reward={args.reward_mode} | {bc_model and f'BC={args.bc_alpha}' or 'no BC'}\n")

    train_loop_hier(
        env=env,
        actor=actor, critic=critic, critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        log_alpha=log_alpha,
        target_entropy=target_entropy,
        replay_buffer=replay_buffer,
        jepa_model=jepa_model,
        transform=transform,
        action_scaler=action_scaler,
        num_iterations=args.num_iters,
        gamma=0.99,
        tau=0.005,
        T_max=args.T_max,
        num_task_ids=5,
        reward_mode=args.reward_mode,
        bc_model=bc_model,
        bc_alpha=args.bc_alpha,
        save_dir=args.save_dir,
    )
