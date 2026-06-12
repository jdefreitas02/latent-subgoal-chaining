"""
Hierarchical SAC + AWR — state observations (cube-single-v0).

State-obs equivalent of train_joint.py. Same two-level architecture:
  - HL policy π^HL(s, g_ult) → subgoal_cube_pos  (AWR, distance-based advantage)
  - LL policy π^LL(s, z_sub) → action             (SAC + optional BC, HER relabelling)

Key differences from train_joint.py (latent version):
  - No JEPA encoder / WM predictor → ~50× faster per iteration
  - State obs: 28-dim (cube-single-v0), goal: 3-dim cube_pos (obs[19:22])
  - HL proposes a subgoal *in physical cube_pos space* (3-dim), not latent space
  - LL takes (state_28, subgoal_3) → action_5
  - Success: ||cube_pos_next - subgoal|| < 0.4  (same criterion as sac_state_train.py)
  - No action scaler — cube-single-v0 actions are already in [-1, 1]
  - Collection: 5 sequential cube-single-v0 episodes per iteration (one per task_id)
  - HL re-proposes subgoal every `gap` env steps during an episode

Usage:
    python latent_hindsight_rl/wgsp/train_joint_state.py \\
        --gap 10 --num_iters 5000 --save_dir ./checkpoints_joint_state_gap10
"""

import os, sys, time, csv, json, random, argparse
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from pathlib import Path

# ── Constants (matches sac_state_train.py exactly) ──────────────────────────
STATE_DIM          = 28
GOAL_DIM           = 3            # cube XYZ (subgoal / g_ult both in this space)
ACTION_DIM         = 5
CUBE_OBS_SLICE     = slice(19, 22)
SUCCESS_THRESHOLD  = 0.4          # = 0.04 m × scale(10)
LOG_SIG_MAX, LOG_SIG_MIN, EPS = 2, -20, 1e-6


# =============================================================================
# NETWORKS
# =============================================================================

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class GoalConditionedActor(nn.Module):
    """LL policy π^LL(s, subgoal) → action_5.  Input: state_28 + subgoal_3 = 31-dim."""
    def __init__(self, state_dim=STATE_DIM, goal_dim=GOAL_DIM,
                 action_dim=ACTION_DIM, hidden_dim=256, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        in_dim = state_dim + goal_dim
        self.l1 = nn.Linear(in_dim, hidden_dim);  self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim)
        self.mean_l    = nn.Linear(hidden_dim, action_dim)
        self.log_std_l = nn.Linear(hidden_dim, action_dim)
        self.apply(_weights_init)
        nn.init.uniform_(self.mean_l.weight,    -1e-3, 1e-3); nn.init.constant_(self.mean_l.bias,    0)
        nn.init.uniform_(self.log_std_l.weight, -1e-3, 1e-3); nn.init.constant_(self.log_std_l.bias, -1.0)

    def _forward(self, state, goal):
        x = F.relu(self.ln1(self.l1(torch.cat([state, goal], dim=-1))))
        x = F.relu(self.ln2(self.l2(x)))
        return self.mean_l(x), self.log_std_l(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)

    def sample(self, state, goal):
        mean, log_std = self._forward(state, goal)
        std = log_std.exp()
        x_t = Normal(mean, std).rsample()
        y_t = torch.tanh(x_t)
        action   = y_t * self.action_scale
        log_prob = Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mean) * self.action_scale


class TwinCritic(nn.Module):
    """Twin Q-network Q(s, subgoal, a).  Input: 28+3+5 = 36-dim."""
    def __init__(self, state_dim=STATE_DIM, goal_dim=GOAL_DIM,
                 action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        in_dim = state_dim + goal_dim + action_dim
        def _net():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1, self.q2 = _net(), _net()
        self.apply(_weights_init)
        for net in (self.q1, self.q2):
            nn.init.uniform_(net[-1].weight, -1e-3, 1e-3)
            nn.init.constant_(net[-1].bias, 0)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=-1)
        return self.q1(x), self.q2(x)


class HighLevelActor(nn.Module):
    """HL policy π^HL(s, g_ult) → subgoal_cube_pos_3.
    Trained with AWR: predicts where the cube will be in `gap` steps."""
    def __init__(self, state_dim=STATE_DIM, goal_dim=GOAL_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )
        self.apply(_weights_init)

    def forward(self, state, g_ult):
        return self.net(torch.cat([state, g_ult], dim=-1))


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class StateEpisodicHERBuffer:
    """Episodic buffer for full state trajectories with HER + HL AWR support.

    Stores per-episode:
      states     [B, T, 28], actions  [B, T, 5], next_states [B, T, 28]
      cube_pos   [B, T, 3]  — obs[19:22] at each step (achieved cube position)
      g_ult      [B, 3]     — episode-end cube position (ultimate goal)
      ep_lens    [B]

    sample_batch: HER relabels LL goal with future cube_pos (future strategy).
    sample_hl_triplets: returns (state_t, cube_pos_{t+gap}, g_ult) for AWR.
    """
    def __init__(self, capacity=20000, max_t=200, future_p=0.8, device="cuda"):
        self.capacity, self.max_t, self.future_p = capacity, max_t, future_p
        self.device = device
        self.states     = torch.zeros((capacity, max_t, STATE_DIM),  device=device)
        self.actions    = torch.zeros((capacity, max_t, ACTION_DIM), device=device)
        self.next_states= torch.zeros((capacity, max_t, STATE_DIM),  device=device)
        self.cube_pos   = torch.zeros((capacity, max_t, GOAL_DIM),   device=device)
        self.g_ult      = torch.zeros((capacity, GOAL_DIM),          device=device)
        self.ep_lens    = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.pos, self.size = 0, 0

    @property
    def num_transitions(self):
        return int(self.ep_lens[:self.size].sum().item())

    def store_episode(self, states, actions, next_states, cube_pos, g_ult, ep_len):
        """Store a single episode. All inputs are [T, *] CPU tensors."""
        T = min(ep_len, self.max_t)
        p = self.pos
        self.states[p, :T]      = states[:T].to(self.device)
        self.actions[p, :T]     = actions[:T].to(self.device)
        self.next_states[p, :T] = next_states[:T].to(self.device)
        self.cube_pos[p, :T]    = cube_pos[:T].to(self.device)
        self.g_ult[p]           = g_ult.to(self.device)
        self.ep_lens[p]         = T
        self.pos  = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size=256, reward_mode='sparse'):
        ep_idxs  = torch.randint(0, self.size, (batch_size,), device=self.device)
        ep_lens  = self.ep_lens[ep_idxs]
        safe_len = ep_lens.clamp(min=1)
        t_idxs   = (torch.rand(batch_size, device=self.device) * safe_len).long().clamp(max=safe_len - 1)

        s_b  = self.states[ep_idxs, t_idxs]        # [B, 28]
        a_b  = self.actions[ep_idxs, t_idxs]       # [B, 5]
        sn_b = self.next_states[ep_idxs, t_idxs]   # [B, 28]
        cp_b = self.cube_pos[ep_idxs, t_idxs]      # [B, 3] — achieved goal at next step (via cube_pos of next)
        # Use cube_pos of NEXT state as achieved_next
        cp_next_b = self.cube_pos[ep_idxs, (t_idxs + 1).clamp(max=safe_len - 1)]

        # HER future-strategy goal relabelling
        valid_future = (t_idxs < ep_lens - 1)
        her_mask     = (torch.rand(batch_size, device=self.device) < self.future_p) & valid_future
        range_len    = (ep_lens - 1 - t_idxs).clamp(min=1)
        future_t     = (t_idxs + (torch.rand(batch_size, device=self.device) * range_len).long() + 1).clamp(max=ep_lens - 1)
        future_g     = self.cube_pos[ep_idxs, future_t]
        orig_g       = cp_b   # original LL goal = cube_pos at t (or the stored subgoal — here we use cube_pos)
        g_b          = torch.where(her_mask.unsqueeze(-1), future_g, orig_g)

        dist_next = torch.norm(cp_next_b - g_b, p=2, dim=-1)
        success   = dist_next < SUCCESS_THRESHOLD
        done_b    = success.float()

        if reward_mode == 'dense':
            dist_curr = torch.norm(cp_b - g_b, p=2, dim=-1)
            r_b = ((dist_curr - dist_next) / 0.4).clamp(-2.0, 2.0)
        else:
            r_b = torch.where(success, torch.zeros_like(dist_next), -torch.ones_like(dist_next))

        return s_b, a_b, sn_b, g_b, r_b.unsqueeze(-1), done_b.unsqueeze(-1)

    def sample_hl_triplets(self, batch_size, gap):
        """(state_t, cube_pos_t, cube_pos_{t+gap}, g_ult) for AWR training."""
        ep_idxs  = torch.randint(0, self.size, (batch_size,), device=self.device)
        ep_lens  = self.ep_lens[ep_idxs]
        safe_max = (ep_lens - gap - 1).clamp(min=0)
        t_idxs   = (torch.rand(batch_size, device=self.device) * (safe_max + 1)).long().clamp(max=safe_max)
        t_gap    = (t_idxs + gap).clamp(max=ep_lens - 1)

        state_t   = self.states[ep_idxs, t_idxs]    # [B, 28]
        cp_t      = self.cube_pos[ep_idxs, t_idxs]  # [B, 3]
        cp_t_gap  = self.cube_pos[ep_idxs, t_gap]   # [B, 3]
        g_ult     = self.g_ult[ep_idxs]             # [B, 3]
        return state_t, cp_t, cp_t_gap, g_ult


# =============================================================================
# EPISODE COLLECTION
# =============================================================================

def collect_episode(env, actor, hl_actor, task_id, T_max, gap, device):
    """Collect one episode from cube-single-v0.

    HL re-proposes a subgoal (in cube_pos space) every `gap` env steps.
    LL acts toward the current subgoal.

    Returns (all on CPU):
        states, actions, next_states, cube_pos : [T, *] tensors
        g_ult  : [3]  — episode-end cube_pos (ultimate goal)
        success: bool
    """
    obs, info = env.reset(options={"task_id": task_id})
    g_ult_np  = np.array(info["goal"][CUBE_OBS_SLICE], dtype=np.float32)
    g_ult     = torch.tensor(g_ult_np, dtype=torch.float32)

    states_list, actions_list, next_states_list, cp_list = [], [], [], []
    success = False
    subgoal = g_ult.clone()   # initial subgoal = ultimate goal (HL not yet warm)

    for step in range(T_max):
        # HL re-proposes every `gap` steps
        if step % gap == 0:
            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            g_t = g_ult.to(device).unsqueeze(0)
            with torch.no_grad():
                subgoal = hl_actor(s_t, g_t).squeeze(0).cpu()

        s_tensor = torch.tensor(obs,  dtype=torch.float32, device=device).unsqueeze(0)
        g_tensor = subgoal.to(device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = actor.sample(s_tensor, g_tensor)
        action_np = action.squeeze(0).cpu().numpy().clip(-1.0, 1.0)

        next_obs, _, terminated, truncated, step_info = env.step(action_np)
        cp = np.array(next_obs[CUBE_OBS_SLICE], dtype=np.float32)

        states_list.append(torch.tensor(obs,      dtype=torch.float32))
        actions_list.append(torch.tensor(action_np, dtype=torch.float32))
        next_states_list.append(torch.tensor(next_obs, dtype=torch.float32))
        cp_list.append(torch.tensor(cp, dtype=torch.float32))

        success = success or bool(step_info.get("success", False))
        obs = next_obs
        if terminated or truncated or success:
            break

    return (
        torch.stack(states_list),
        torch.stack(actions_list),
        torch.stack(next_states_list),
        torch.stack(cp_list),
        g_ult,
        success,
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_loop(envs, actor, critic, critic_target, hl_actor,
               actor_opt, critic_opt, alpha_opt, hl_opt,
               log_alpha, target_entropy, replay_buffer,
               num_iterations=5000, gamma=0.99, tau=0.005,
               T_max=200, gap=10,
               reward_mode='sparse', grad_steps=40, hl_grad_steps=10,
               hl_beta=3.0, save_dir="./checkpoints_joint_state"):

    os.makedirs(save_dir, exist_ok=True)
    device = next(actor.parameters()).device
    print(f"Joint-state training | gap={gap} | reward={reward_mode} | "
          f"iters={num_iterations} | save={save_dir}")

    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["Iteration", "Success_Rate", "Buffer_Episodes",
                                 "Buffer_Transitions", "LL_Actor_Loss", "LL_Critic_Loss",
                                 "HL_Loss", "Alpha", "Total_Time"])

    recent = deque(maxlen=len(envs) * 20)
    t0 = time.time()
    num_tasks = len(envs)

    for it in range(num_iterations):

        # ── Collect one episode per task ──────────────────────────────────
        for task_id, env in enumerate(envs, start=1):
            states, actions, next_states, cube_pos, g_ult, suc = collect_episode(
                env, actor, hl_actor, task_id, T_max, gap, device)
            replay_buffer.store_episode(states, actions, next_states, cube_pos, g_ult, len(states))
            recent.append(float(suc))

        # ── SAC (LL) updates ──────────────────────────────────────────────
        ll_act_l = ll_crit_l = 0.0
        if replay_buffer.num_transitions >= 256:
            for _ in range(grad_steps):
                s_b, a_b, sn_b, g_b, r_b, d_b = replay_buffer.sample_batch(256, reward_mode)
                alpha = log_alpha.exp().item()

                with torch.no_grad():
                    na, nlp, _ = actor.sample(sn_b, g_b)
                    tq1, tq2 = critic_target(sn_b, g_b, na)
                    tq = r_b + (1 - d_b) * gamma * (torch.min(tq1, tq2) - alpha * nlp)

                q1, q2 = critic(s_b, g_b, a_b)
                cl = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
                critic_opt.zero_grad(); cl.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step(); ll_crit_l += cl.item()

                new_a, lp, _ = actor.sample(s_b, g_b)
                for p in critic.parameters(): p.requires_grad = False
                q1n, q2n = critic(s_b, g_b, new_a)
                al = (alpha * lp - torch.min(q1n, q2n)).mean()
                actor_opt.zero_grad(); al.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step(); ll_act_l += al.item()
                for p in critic.parameters(): p.requires_grad = True

                aloss = -(log_alpha * (lp + target_entropy).detach()).mean()
                alpha_opt.zero_grad(); aloss.backward(); alpha_opt.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=-4.6)   # α floor ≈ 0.01

                for tp, p in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(tp.data * (1 - tau) + p.data * tau)

            ll_act_l  /= grad_steps
            ll_crit_l /= grad_steps

        # ── AWR (HL) updates ──────────────────────────────────────────────
        hl_l = 0.0
        if replay_buffer.size >= 10 and it >= 5:
            for _ in range(hl_grad_steps):
                s_t, cp_t, cp_tg, g_ult_b = replay_buffer.sample_hl_triplets(256, gap)

                with torch.no_grad():
                    adv = torch.norm(cp_t - g_ult_b, p=2, dim=-1) \
                        - torch.norm(cp_tg - g_ult_b, p=2, dim=-1)
                    w   = torch.exp(hl_beta * adv).clamp(max=100.0)

                pred    = hl_actor(s_t, g_ult_b)
                per_dim = F.mse_loss(pred, cp_tg, reduction='none').mean(dim=-1)
                hl_loss = (w * per_dim).mean()

                hl_opt.zero_grad(); hl_loss.backward()
                nn.utils.clip_grad_norm_(hl_actor.parameters(), 1.0)
                hl_opt.step(); hl_l += hl_loss.item()

            hl_l /= hl_grad_steps

        # ── Logging ───────────────────────────────────────────────────────
        if it % 10 == 0:
            sr      = float(np.mean(recent)) if recent else 0.0
            elapsed = time.time() - t0
            print(f"Iter {it:05d} | SR={sr*100:.1f}% | "
                  f"buf={replay_buffer.size}/{replay_buffer.num_transitions} | "
                  f"LL_act={ll_act_l:.3f} LL_crit={ll_crit_l:.3f} "
                  f"HL={hl_l:.4f} α={log_alpha.exp().item():.4f} | t={elapsed:.1f}s")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([it, sr, replay_buffer.size,
                                        replay_buffer.num_transitions,
                                        ll_act_l, ll_crit_l, hl_l,
                                        log_alpha.exp().item(), elapsed])
            t0 = time.time()

        if (it > 0 and it % 500 == 0) or it == num_iterations - 1:
            torch.save(actor.state_dict(),    os.path.join(save_dir, "actor_policy.pth"))
            torch.save(critic.state_dict(),   os.path.join(save_dir, "critic_network.pth"))
            torch.save(hl_actor.state_dict(), os.path.join(save_dir, "high_actor_state.pth"))
            print(f"  --> Checkpoint saved at iter {it}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical SAC + AWR with state observations (cube-single-v0)"
    )
    parser.add_argument("--gap",         type=int,   default=10,
                        help="HL subgoal horizon in env steps (default 10 ≈ 0.5s)")
    parser.add_argument("--num_iters",   type=int,   default=5000)
    parser.add_argument("--T_max",       type=int,   default=200,
                        help="Max env steps per episode (same as sac_state_train.py)")
    parser.add_argument("--reward_mode", type=str,   default="sparse",
                        choices=["sparse", "dense"])
    parser.add_argument("--hl_beta",     type=float, default=3.0,
                        help="AWR temperature for HL (same default as train_joint.py)")
    parser.add_argument("--grad_steps",  type=int,   default=40,
                        help="LL SAC gradient steps per iteration")
    parser.add_argument("--hl_grad_steps", type=int, default=10,
                        help="HL AWR gradient steps per iteration")
    parser.add_argument("--num_task_ids", type=int,  default=5)
    parser.add_argument("--save_dir",    type=str,   default=None,
                        help="Checkpoint directory (default: auto-derived from gap/reward)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.save_dir is None:
        args.save_dir = f"./checkpoints_joint_state_gap{args.gap}_{args.reward_mode}"

    # ── Environments ─────────────────────────────────────────────────────
    import gymnasium
    import ogbench  # noqa: F401
    envs = [gymnasium.make("cube-single-v0") for _ in range(args.num_task_ids)]
    print(f"Created {len(envs)} cube-single-v0 envs (one per task_id).")

    # ── Save config ───────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {"state_dim": STATE_DIM, "goal_dim": GOAL_DIM, "action_dim": ACTION_DIM,
           "gap": args.gap, "reward_mode": args.reward_mode, "hl_beta": args.hl_beta,
           "success_threshold": SUCCESS_THRESHOLD, "variant": "joint_state"}
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Networks ──────────────────────────────────────────────────────────
    actor         = GoalConditionedActor().to(device)
    critic        = TwinCritic().to(device)
    critic_target = TwinCritic().to(device)
    critic_target.load_state_dict(critic.state_dict())
    hl_actor      = HighLevelActor().to(device)

    actor_opt  = torch.optim.Adam(actor.parameters(),    lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(),   lr=3e-4)
    hl_opt     = torch.optim.Adam(hl_actor.parameters(), lr=3e-4)

    target_entropy = -float(ACTION_DIM)
    log_alpha      = torch.tensor([0.0], requires_grad=True, device=device)
    alpha_opt      = torch.optim.Adam([log_alpha], lr=3e-4)

    # ── Replay buffer ─────────────────────────────────────────────────────
    replay_buffer = StateEpisodicHERBuffer(
        capacity=20000, max_t=args.T_max, future_p=0.8, device=device)

    print(f"\nStarting joint-state training | gap={args.gap} | "
          f"T_max={args.T_max} | reward={args.reward_mode} | "
          f"hl_beta={args.hl_beta} | iters={args.num_iters}\n")

    train_loop(
        envs=envs,
        actor=actor, critic=critic, critic_target=critic_target, hl_actor=hl_actor,
        actor_opt=actor_opt, critic_opt=critic_opt,
        alpha_opt=alpha_opt, hl_opt=hl_opt,
        log_alpha=log_alpha, target_entropy=target_entropy,
        replay_buffer=replay_buffer,
        num_iterations=args.num_iters,
        gamma=0.99, tau=0.005,
        T_max=args.T_max, gap=args.gap,
        reward_mode=args.reward_mode,
        grad_steps=args.grad_steps, hl_grad_steps=args.hl_grad_steps,
        hl_beta=args.hl_beta,
        save_dir=args.save_dir,
    )

    for env in envs:
        env.close()
