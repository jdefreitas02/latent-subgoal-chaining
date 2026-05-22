"""
Diagnose a trained GC TD-MPC2 checkpoint.

Answers the four questions that decide which failure mode we're in:

  1. Is the policy collapsed?       -> log_std at the floor
  2. Is action distribution diverse?-> per-dim std across 512 samples
  3. Is V/Q informative?            -> spread of terminal V across 512 candidates
                                       and spread of Q over data actions
  4. Is the reward signal sane?     -> R̂ at goal vs not-at-goal

We run on a few real OGBench episode resets (5 tasks, first state of each) so
the latents are in-distribution.

Usage:
    python latent_hindsight_rl/diagnose_tdmpc2_policy.py \
        --ckpt_dir checkpoints_tdmpc2_gc_iql_s0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from latent_hindsight_rl.tdmpc2.eval_agent import load_eval_agent  # noqa: E402


def _hwc_to_tensor(obs_hwc, device):
    t = torch.from_numpy(obs_hwc.astype(np.uint8)).to(device).float()
    return t.permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def diagnose(agent, obs_hwc, goal_hwc, task_name):
    """Print policy + value diagnostics at one (obs, goal) pair."""
    model = agent.model
    cfg = agent.cfg
    device = agent.device

    # Encode obs + goal
    obs_t = _hwc_to_tensor(obs_hwc, device)
    goal_t = _hwc_to_tensor(goal_hwc, device)
    z = model.encode(obs_t)        # (1, latent)
    g = model.encode(goal_t)       # (1, latent)

    print(f"\n==== {task_name} ====")
    print(f"||z||={z.norm().item():.3f}  ||g||={g.norm().item():.3f}  "
          f"||z-g||={(z-g).norm().item():.3f}  cos(z,g)="
          f"{torch.nn.functional.cosine_similarity(z, g).item():.3f}")

    # ---------- (1) Policy log_std at this state ----------
    # Inspect the raw log_std the network produces (after the squash to bounds).
    from latent_hindsight_rl.tdmpc2 import tdmpc2_math as math_
    x = torch.cat([z, g], dim=-1)
    mean_raw, log_std_raw = model._pi(x).chunk(2, dim=-1)
    log_std = math_.log_std(log_std_raw, model.log_std_min, model.log_std_dif)
    print(f"  log_std (per dim): {log_std.squeeze(0).cpu().numpy().round(3)}")
    print(f"  log_std min/max bounds: [{cfg.log_std_min}, {cfg.log_std_max}]")
    print(f"  raw mean (pre-squash): {mean_raw.squeeze(0).cpu().numpy().round(3)}")

    # ---------- (2) Sample 512 actions, check diversity ----------
    N = 512
    z_rep = z.repeat(N, 1)
    g_rep = g.repeat(N, 1)
    a_samples, _ = model.pi(z_rep, g_rep)        # (N, action_dim)
    a_np = a_samples.cpu().numpy()
    print(f"  action samples: mean={a_np.mean(0).round(3)}  "
          f"std={a_np.std(0).round(3)}")
    # Check if all 512 samples are essentially the same:
    pairwise_l2 = ((a_samples[0:1] - a_samples).norm(dim=-1)).mean().item()
    print(f"  mean pairwise L2 distance between samples: {pairwise_l2:.4f}")

    # ---------- (3) V/Q informativeness over the planning candidates ----------
    # Roll out N policy sequences for H steps, score endpoints with V.
    H = cfg.horizon
    actions = torch.empty(H, N, cfg.action_dim, device=device)
    _z = z_rep.clone()
    _g = g_rep
    for t in range(H):
        a, _ = model.pi(_z, _g)
        actions[t] = a
        if t < H - 1:
            _z = model.next(_z, a)

    # V at the start state
    v_start_bins = model.value_state(z, g, target=True)
    v_start = math_.two_hot_inv(v_start_bins, cfg).item()
    print(f"  V_target(z, g) at start: {v_start:.3f}")

    # Score 512 candidates with cumulative R + γ^H V_target
    G = torch.zeros(N, 1, device=device)
    discount = 1.0
    _z = z_rep.clone()
    for t in range(H):
        r_bins = model.reward(_z, actions[t], g_rep)
        r = math_.two_hot_inv(r_bins, cfg)
        _z = model.next(_z, actions[t])
        G = G + discount * r
        discount = discount * cfg.gamma
    v_end_bins = model.value_state(_z, g_rep, target=True)
    v_end = math_.two_hot_inv(v_end_bins, cfg)         # (N, 1)
    scores = (G + discount * v_end).squeeze(1)         # (N,)
    s_np = scores.cpu().numpy()
    print(f"  candidate scores: mean={s_np.mean():.3f}  std={s_np.std():.3f}  "
          f"min={s_np.min():.3f}  max={s_np.max():.3f}  "
          f"argmax-vs-argmin gap={s_np.max() - s_np.min():.3f}")
    print(f"  cumulative reward (H steps): mean={G.cpu().numpy().mean():.3f}  "
          f"std={G.cpu().numpy().std():.3f}")
    print(f"  V_target at endpoints: mean={v_end.cpu().numpy().mean():.3f}  "
          f"std={v_end.cpu().numpy().std():.3f}")

    # ---------- (4) Reward signal: R̂ at goal vs at start ----------
    # If reward head is informative, R̂(z=g, a, g) should be ~0 (success); R̂(z, a, g) should be ~-1.
    a_zero = torch.zeros(1, cfg.action_dim, device=device)
    r_at_start_bins = model.reward(z, a_zero, g)
    r_at_start = math_.two_hot_inv(r_at_start_bins, cfg).item()
    r_at_goal_bins = model.reward(g, a_zero, g)
    r_at_goal = math_.two_hot_inv(r_at_goal_bins, cfg).item()
    print(f"  R̂(z_start, 0, g)={r_at_start:.3f}   R̂(z_goal, 0, g)={r_at_goal:.3f}   "
          f"(want ~-1 vs ~0)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--num_tasks', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.ckpt_dir} on {device}...")
    agent = load_eval_agent(args.ckpt_dir, device=str(device))
    cfg = agent.cfg
    print(f"  offline_mode={cfg.offline_mode}  num_pi_trajs={cfg.num_pi_trajs}  "
          f"iterations={cfg.iterations}  horizon={cfg.horizon}")

    print("Setting up OGBench env...")
    import ogbench  # noqa: F401
    import gymnasium
    env = gymnasium.make('visual-cube-single-v0')
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    task_names = [t.get('task_name', f'task{i}') for i, t in enumerate(task_infos)]

    for tid in range(1, args.num_tasks + 1):
        obs, info = env.reset(options=dict(task_id=tid))
        goal = info['goal']
        diagnose(agent, obs, goal, f"task{tid}_{task_names[tid - 1]}")

    print("\n==== Interpretation guide ====")
    print("  log_std hitting -10 floor anywhere -> policy collapsed")
    print("  action samples std ~0 (any dim)    -> samples are clones, MPPI is no-op")
    print("  candidate score gap < 0.05         -> V uninformative; argmax is random")
    print("  R̂(start) ~ R̂(goal)                -> reward head untrained / wrong sign")


if __name__ == '__main__':
    main()
