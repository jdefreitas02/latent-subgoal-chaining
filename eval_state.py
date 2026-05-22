"""
Evaluate a state-based SAC+HER policy on OGBench cube-single-v0.

No JEPA, no renderer. The policy takes the 28-dim state + 3-dim cube target
position (obs[19:22] encoding) and outputs a 5-dim action in [-1, 1].

Usage:
    python latent_hindsight_rl/eval_state.py \\
        --checkpoint_dir ./checkpoints_sac_state_s0
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import json
import argparse
import numpy as np
import torch

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from sac_state_train import Actor, STATE_DIM, GOAL_DIM, ACTION_DIM, CUBE_OBS_SLICE

TASK_NAMES = {
    1: "task1_horizontal",
    2: "task2_vertical1",
    3: "task3_vertical2",
    4: "task4_diagonal1",
    5: "task5_diagonal2",
}


def eval_policy(actor, env, device, num_episodes=50, num_tasks=5):
    """Deterministic eval (actor mean), 5 tasks × num_episodes."""
    results = {}
    actor.eval()
    for task_id in range(1, num_tasks + 1):
        successes = 0
        for _ in range(num_episodes):
            obs, info = env.reset(options={"task_id": task_id})
            desired = np.array(info["goal"][CUBE_OBS_SLICE], dtype=np.float32)
            done = False
            while not done:
                s = torch.as_tensor(obs,     dtype=torch.float32, device=device).unsqueeze(0)
                g = torch.as_tensor(desired, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, _, det_action = actor.sample(s, g)
                obs, _, terminated, truncated, step_info = env.step(
                    det_action.squeeze(0).cpu().numpy()
                )
                done = terminated or truncated
                if step_info.get("success", False):
                    successes += 1
                    done = True
        sr = successes / num_episodes
        results[TASK_NAMES[task_id]] = sr
        print(f"  {TASK_NAMES[task_id]}: {sr*100:.1f}%")

    results["overall"] = float(np.mean(list(results.values())))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir",  required=True)
    parser.add_argument("--num_episodes",    type=int, default=50)
    parser.add_argument("--num_tasks",       type=int, default=5)
    args = parser.parse_args()

    import gymnasium
    import ogbench  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_path = os.path.join(args.checkpoint_dir, "training_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        state_dim    = cfg.get("state_dim",    STATE_DIM)
        goal_dim     = cfg.get("goal_dim",     GOAL_DIM)
        action_dim   = cfg.get("action_dim",   ACTION_DIM)
        action_scale = cfg.get("action_scale", 1.0)
    else:
        state_dim, goal_dim, action_dim, action_scale = STATE_DIM, GOAL_DIM, ACTION_DIM, 1.0

    actor = Actor(state_dim, goal_dim, action_dim, action_scale=action_scale).to(device)
    ckpt_path = os.path.join(args.checkpoint_dir, "actor_policy.pth")
    actor.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded actor from {ckpt_path}")

    env = gymnasium.make("cube-single-v0")
    print(f"Evaluating on cube-single-v0 | {args.num_tasks} tasks × {args.num_episodes} episodes\n")

    t0 = time.time()
    results = eval_policy(actor, env, device,
                          num_episodes=args.num_episodes,
                          num_tasks=args.num_tasks)
    elapsed = time.time() - t0

    print(f"\n  overall: {results['overall']*100:.1f}%")
    print(f"  elapsed: {elapsed:.0f}s")

    out_dir = os.path.join(
        os.path.dirname(args.checkpoint_dir),
        "eval_state_" + os.path.basename(args.checkpoint_dir),
    )
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"==== {os.path.basename(args.checkpoint_dir)} ====\n")
        f.write(f"protocol: OGBench state — {args.num_tasks} tasks × {args.num_episodes} episodes\n")
        f.write(f"checkpoint: {args.checkpoint_dir}\n")
        for k, v in results.items():
            f.write(f"  {k}: {v*100:.1f}%\n")
        f.write(f"elapsed: {elapsed:.0f}s\n")

    print(f"\nResults saved to {results_path}")
    env.close()
