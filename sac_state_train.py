"""
Goal-conditioned SAC + HER on OGBench cube-single-v0 (state observations).

Diagnostic baseline — no JEPA, no rendering. Mirrors SB3's HER+SAC design but
implemented directly so we can later port it to latent space:

  * Goal representation: 3-dim cube XYZ (the 'achieved_goal' projection),
    extracted from obs[19:22] (scaled cube position; the same encoding used by
    info['goal']). This matches OGBench's native success criterion exactly:
        success ⇔ ||cube_pos - target_pos|| < 0.04 m (= 0.4 in scaled coords).
    Not the full 28-dim state, which would force the policy to also match
    joint positions / velocities / gripper state that don't affect reward.

  * HER ('future' strategy) sampled at batch-sample time, NOT at episode
    collection time. Each minibatch draws fresh relabelled goals from each
    transition's stored episode and computes the reward with the same rule
    the env uses for info['success'].

  * Bootstrap through TimeLimit truncation: done flag for Q-target is
    dones * (1 - timeouts). The 200-step truncation is NOT treated as task
    failure — the value function still bootstraps from the next state.

  * Actor/critic take (state_28, goal_3) as input; goal_dim ≠ state_dim.

Usage:
    python latent_hindsight_rl/sac_state_train.py \\
        --save_dir ./checkpoints_sac_state_s0 \\
        --num_iters 10000
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


# =============================================================================
# CONSTANTS
# =============================================================================

STATE_DIM       = 28        # cube-single-v0 observation dim
GOAL_DIM        = 3         # cube XYZ (achieved_goal projection)
ACTION_DIM      = 5         # cube-single-v0 action dim
CUBE_OBS_SLICE  = slice(19, 22)   # obs[19:22] = (block_0_pos - center) * 10
GOAL_THRESHOLD  = 0.4       # = 0.04 m × xyz_scaler(10) — env's success radius

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


# =============================================================================
# ACTOR & CRITIC  (state_dim ≠ goal_dim, unlike the original GoalConditionedActor)
# =============================================================================

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """π(a | state, goal_cube_pos). Inputs concatenated, tanh-squashed action."""
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        in_dim = state_dim + goal_dim
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.apply(_weights_init)
        # Small initial action distribution so exploration starts modest
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.mean_head.bias, 0)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale


class TwinCritic(nn.Module):
    """Q1, Q2 over (state, goal_cube_pos, action)."""
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256):
        super().__init__()
        in_dim = state_dim + goal_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_weights_init)
        # Small init on output heads (sparse reward ∈ [0,1] → Q ∈ [0, 100] after clamp)
        for net in (self.q1, self.q2):
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias, 0)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)
        return self.q1(x), self.q2(x)


# =============================================================================
# HER REPLAY BUFFER  (mirrors SB3's HerReplayBuffer behaviour)
# =============================================================================

class HerReplayBuffer:
    """
    Flat tensor replay buffer that records episode boundaries so HER 'future'
    relabelling can be done at sample time (not collection time), with rewards
    computed by the same rule the env uses for info['success']:
        r_her = 1.0 if ||achieved_at_next - relabelled_desired|| < threshold else 0.0

    Each stored transition records:
      states[t]      : 28-dim observation at step t
      actions[t]     : 5-dim action taken at step t
      next_states[t] : 28-dim observation at step t+1
      achieved[t]    : 3-dim cube pos at step t+1 (= obs_{t+1}[19:22])
      desired[t]     : 3-dim target cube pos for the original episode goal
      rewards[t]     : original sparse env reward (info['success'])
      terminations[t]: True if env returned terminated=True (real episode end)
      timeouts[t]    : True if env returned truncated=True (TimeLimit)
      ep_start[t]    : index in the buffer where this episode began
      ep_length[t]   : number of transitions in this episode

    Sampling:
      her_ratio (default 0.8) of each batch is virtual (HER) — for these
      transitions a future timestep within the same episode is sampled (inclusive
      of the current step, matching SB3's 'future' strategy), the achieved_goal
      at that future step is used as the new desired_goal, and the reward is
      recomputed by the threshold check.
    """
    def __init__(self, capacity, state_dim, goal_dim, action_dim, device):
        self.capacity   = capacity
        self.device     = device
        self.states       = torch.zeros((capacity, state_dim),   dtype=torch.float32, device=device)
        self.actions      = torch.zeros((capacity, action_dim),  dtype=torch.float32, device=device)
        self.next_states  = torch.zeros((capacity, state_dim),   dtype=torch.float32, device=device)
        self.achieved     = torch.zeros((capacity, goal_dim),    dtype=torch.float32, device=device)
        self.desired      = torch.zeros((capacity, goal_dim),    dtype=torch.float32, device=device)
        self.rewards      = torch.zeros((capacity,),             dtype=torch.float32, device=device)
        self.terminations = torch.zeros((capacity,),             dtype=torch.float32, device=device)
        self.timeouts     = torch.zeros((capacity,),             dtype=torch.float32, device=device)
        self.ep_start     = torch.zeros((capacity,),             dtype=torch.long,    device=device)
        self.ep_length    = torch.zeros((capacity,),             dtype=torch.long,    device=device)
        self.position = 0
        self.size = 0

    def store_episode(self, ep):
        """ep: dict of np.ndarrays of length T (states, actions, next_states,
        achieved, desired, rewards, terminations, timeouts)."""
        T = ep["states"].shape[0]
        start = self.position
        # Wrap-around storage
        idx = (torch.arange(T, device=self.device) + start) % self.capacity
        self.states[idx]       = torch.as_tensor(ep["states"],       device=self.device)
        self.actions[idx]      = torch.as_tensor(ep["actions"],      device=self.device)
        self.next_states[idx]  = torch.as_tensor(ep["next_states"],  device=self.device)
        self.achieved[idx]     = torch.as_tensor(ep["achieved"],     device=self.device)
        self.desired[idx]      = torch.as_tensor(ep["desired"],      device=self.device)
        self.rewards[idx]      = torch.as_tensor(ep["rewards"],      device=self.device)
        self.terminations[idx] = torch.as_tensor(ep["terminations"], device=self.device)
        self.timeouts[idx]     = torch.as_tensor(ep["timeouts"],     device=self.device)
        # Episode bounds: all transitions in this episode share ep_start and ep_length
        self.ep_start[idx]  = start
        self.ep_length[idx] = T
        self.position = (start + T) % self.capacity
        self.size = min(self.size + T, self.capacity)

    def sample(self, batch_size, her_ratio=0.8, threshold=GOAL_THRESHOLD):
        n_virt = int(batch_size * her_ratio)
        n_real = batch_size - n_virt
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        # ----- HER relabel for the first n_virt entries -----
        virt_idx   = idx[:n_virt]
        ep_starts  = self.ep_start[virt_idx]
        ep_lens    = self.ep_length[virt_idx]
        # position of current transition within its episode (0-indexed)
        pos_in_ep  = (virt_idx - ep_starts) % self.capacity
        # 'future' strategy (inclusive): future_pos ∈ [pos_in_ep, ep_lens - 1]
        rand_u     = torch.rand(n_virt, device=self.device)
        span       = (ep_lens - pos_in_ep).clamp_min(1).float()
        future_pos = pos_in_ep + (rand_u * span).long()
        future_idx = (ep_starts + future_pos) % self.capacity
        # New desired goal = achieved at the future step
        new_desired = self.achieved[future_idx]
        # New reward = env's success criterion applied to relabelled goal
        dist = torch.norm(self.achieved[virt_idx] - new_desired, p=2, dim=-1)
        new_rewards = (dist < threshold).float()

        # ----- Assemble full batch -----
        desired = self.desired[idx].clone()
        rewards = self.rewards[idx].clone()
        desired[:n_virt] = new_desired
        rewards[:n_virt] = new_rewards

        # done flag for Q target: terminations only (timeouts do NOT kill bootstrap)
        # virtual transitions inherit the same stored terminations/timeouts as the
        # underlying real transition (SB3 convention)
        done_for_bootstrap = self.terminations[idx] * (1.0 - self.timeouts[idx])

        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            desired,
            rewards,
            done_for_bootstrap,
        )


# =============================================================================
# EPISODE COLLECTION
# =============================================================================

def collect_episode(env, actor, task_id, T_max, device):
    """
    Run one episode in cube-single-v0 and return arrays for storage.
    Each env.step is one actor call (1-to-1; no action chunking).

    Returns dict of np.ndarrays of length T (actual episode length), and the
    boolean 'real_success' for logging (whether info['success'] was True).
    """
    obs, info = env.reset(options={"task_id": task_id})
    # Use scaled cube target (matches storage encoding for achieved goals)
    desired_goal = np.array(info["goal"][CUBE_OBS_SLICE], dtype=np.float32)

    S, A, S2, AG, R, TERM, TIMEOUT = [], [], [], [], [], [], []
    real_success = False

    for _ in range(T_max):
        s_t = torch.as_tensor(obs,          dtype=torch.float32, device=device).unsqueeze(0)
        g_t = torch.as_tensor(desired_goal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = actor.sample(s_t, g_t)
        a_np = action.squeeze(0).cpu().numpy().astype(np.float32)

        next_obs, _, terminated, truncated, step_info = env.step(a_np)
        achieved_next = next_obs[CUBE_OBS_SLICE].astype(np.float32)
        r_sparse = float(step_info.get("success", 0.0))
        real_success = real_success or bool(r_sparse)

        S.append(obs.astype(np.float32))
        A.append(a_np)
        S2.append(next_obs.astype(np.float32))
        AG.append(achieved_next)
        R.append(r_sparse)
        TERM.append(float(terminated))
        TIMEOUT.append(float(truncated))

        obs = next_obs
        if terminated or truncated:
            break

    T = len(S)
    return {
        "states":       np.stack(S),
        "actions":      np.stack(A),
        "next_states":  np.stack(S2),
        "achieved":     np.stack(AG),
        "desired":      np.tile(desired_goal[None, :], (T, 1)),
        "rewards":      np.array(R,       dtype=np.float32),
        "terminations": np.array(TERM,    dtype=np.float32),
        "timeouts":     np.array(TIMEOUT, dtype=np.float32),
    }, real_success


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_loop(env, actor, critic, critic_target,
               actor_opt, critic_opt, alpha_opt,
               log_alpha, target_entropy, buffer,
               num_iterations, gamma, tau, T_max, num_task_ids,
               batch_size, gradient_steps, her_ratio, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    device = next(actor.parameters()).device

    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["Iteration", "Success_Rate", "Buffer_Size",
                                "Actor_Loss", "Critic_Loss", "Alpha", "Total_Time"])

    recent_successes = deque(maxlen=num_task_ids * 20)
    start_time = time.time()

    for iteration in range(num_iterations):
        # --- Rollout: one episode per task_id ---
        for task_id in range(1, num_task_ids + 1):
            ep, success = collect_episode(env, actor, task_id, T_max=T_max, device=device)
            buffer.store_episode(ep)
            recent_successes.append(float(success))

        # --- SAC updates ---
        avg_actor_loss = avg_critic_loss = 0.0
        if buffer.size >= batch_size:
            for _ in range(gradient_steps):
                s_b, a_b, sn_b, g_b, r_b, d_b = buffer.sample(batch_size, her_ratio=her_ratio)
                r_b = r_b.unsqueeze(-1)
                d_b = d_b.unsqueeze(-1)
                alpha = log_alpha.exp().item()

                # Critic update
                with torch.no_grad():
                    next_a, next_log_pi, _ = actor.sample(sn_b, g_b)
                    tq1, tq2 = critic_target(sn_b, g_b, next_a)
                    target_q = r_b + (1.0 - d_b) * gamma * (
                        torch.min(tq1, tq2) - alpha * next_log_pi
                    )
                    target_q = target_q.clamp(0.0, 1.0 / (1.0 - gamma))

                q1, q2 = critic(s_b, g_b, a_b)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                avg_critic_loss += critic_loss.item()

                # Actor update
                new_a, log_pi, _ = actor.sample(s_b, g_b)
                for p in critic.parameters():
                    p.requires_grad = False
                q1_new, q2_new = critic(s_b, g_b, new_a)
                actor_loss = (alpha * log_pi - torch.min(q1_new, q2_new)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()
                avg_actor_loss += actor_loss.item()
                for p in critic.parameters():
                    p.requires_grad = True

                # Alpha update — clamp log_alpha so α never drops below 0.01
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=-4.6)

                # Target soft update
                for tp, p in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        sr = float(np.mean(recent_successes)) if recent_successes else 0.0

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            a_val = avg_actor_loss / gradient_steps if buffer.size >= batch_size else 0.0
            c_val = avg_critic_loss / gradient_steps if buffer.size >= batch_size else 0.0
            print(f"Iter {iteration:05d} | SR: {sr*100:.1f}% | "
                  f"Act: {a_val:.3f} | Crit: {c_val:.3f} | "
                  f"α: {log_alpha.exp().item():.3f} | "
                  f"Buf: {buffer.size} | t: {elapsed:.1f}s")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([iteration, sr, buffer.size,
                                        a_val, c_val, log_alpha.exp().item(), elapsed])
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
        description="State-based SAC+HER on OGBench cube-single-v0 (no JEPA)"
    )
    parser.add_argument("--save_dir",       default="./checkpoints_sac_state_s0")
    parser.add_argument("--num_iters",      type=int,   default=10000)
    parser.add_argument("--T_max",          type=int,   default=200)
    parser.add_argument("--her_ratio",      type=float, default=0.8,
                        help="Fraction of each batch that is HER-relabelled (SB3: 0.8 for n_sampled_goal=4)")
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--gradient_steps", type=int,   default=40)
    parser.add_argument("--num_task_ids",   type=int,   default=5)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--tau",            type=float, default=0.005)
    args = parser.parse_args()

    import gymnasium
    import ogbench  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gymnasium.make("cube-single-v0")
    print("Created cube-single-v0 (28-dim state, 5-dim action, 3-dim cube goal).")

    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        "state_dim":      STATE_DIM,
        "goal_dim":       GOAL_DIM,
        "action_dim":     ACTION_DIM,
        "action_scale":   1.0,
        "goal_threshold": GOAL_THRESHOLD,
        "cube_obs_slice": [CUBE_OBS_SLICE.start, CUBE_OBS_SLICE.stop],
        "reward_type":    "sparse_her_state_v2",  # bumped to mark the SB3-style rewrite
    }
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved training_config.json to {args.save_dir}")

    actor         = Actor(STATE_DIM, GOAL_DIM, ACTION_DIM, action_scale=1.0).to(device)
    critic        = TwinCritic(STATE_DIM, GOAL_DIM, ACTION_DIM).to(device)
    critic_target = TwinCritic(STATE_DIM, GOAL_DIM, ACTION_DIM).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt   = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_opt  = torch.optim.Adam(critic.parameters(), lr=3e-4)
    target_entropy = -float(ACTION_DIM)   # -5
    # Initial α=1.0 (log_alpha=0) matches SB3's SAC default ent_coef_init=1.0 —
    # higher initial exploration prevents premature alpha collapse under HER.
    log_alpha   = torch.tensor([0.0], requires_grad=True, device=device)
    alpha_opt   = torch.optim.Adam([log_alpha], lr=3e-4)

    buffer = HerReplayBuffer(capacity=1_000_000,
                             state_dim=STATE_DIM, goal_dim=GOAL_DIM,
                             action_dim=ACTION_DIM, device=device)

    print(f"\nStarting state SAC+HER | iters={args.num_iters} | T_max={args.T_max} | "
          f"her_ratio={args.her_ratio} | threshold={GOAL_THRESHOLD} scaled "
          f"(={GOAL_THRESHOLD/10:.2f}m) | save_dir={args.save_dir}\n")

    train_loop(
        env=env, actor=actor, critic=critic, critic_target=critic_target,
        actor_opt=actor_opt, critic_opt=critic_opt, alpha_opt=alpha_opt,
        log_alpha=log_alpha, target_entropy=target_entropy, buffer=buffer,
        num_iterations=args.num_iters, gamma=args.gamma, tau=args.tau,
        T_max=args.T_max, num_task_ids=args.num_task_ids,
        batch_size=args.batch_size, gradient_steps=args.gradient_steps,
        her_ratio=args.her_ratio, save_dir=args.save_dir,
    )

    env.close()
    print("\nTraining complete.")
