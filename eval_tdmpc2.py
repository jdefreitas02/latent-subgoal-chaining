"""
Eval script for goal-conditioned TD-MPC2 on OGBench cube-single-play (64x64).

Usage:
    python latent_hindsight_rl/eval_tdmpc2.py \
        --ckpt_dir checkpoints_tdmpc2_gc_iql_s0 \
        --num_episodes 50 \
        --results_dir eval_tdmpc2/iql_s0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root on sys.path so we can import sibling modules without install
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from latent_hindsight_rl.tdmpc2.eval_agent import load_eval_agent  # noqa: E402


def run_task(env, agent, task_id, task_name, num_episodes, max_steps,
             diagnose=False, goal_info_key='goal'):
    """Eval a single OGBench task. Mirrors eval_ogbench.run_task.

    Uses info['success'] from the final step (OGBench's native success signal),
    matching the HIQL eval protocol. Do NOT use `terminated` — for these tasks
    the env truncates on step-limit and never sets terminated=True for success.
    """
    successes = []
    for ep in range(num_episodes):
        obs, info = env.reset(options=dict(task_id=task_id))
        goal = info[goal_info_key]   # (H, W, 3) uint8
        agent.reset()
        done = False
        step = 0
        diag_log_step = 0
        while not done and step < max_steps:
            action, diag = agent.get_action(obs, goal)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if diagnose and ep == 0 and diag is not None:
                diag_log_step += 1
                if diag_log_step % 5 == 0:
                    print(
                        f"  [{task_name} ep={ep} step={step:3d}]"
                        f"  dist_to_goal={diag['dist_to_goal']:.3f}"
                        f"  action_norm={diag['action_norm']:.3f}"
                        f"  info_success={info.get('success', 'n/a')}"
                    )
        # Read success from final step's info (matches OGBench's evaluation.py)
        success = float(info.get('success', 0.0))
        successes.append(success)
        if diagnose and ep == 0:
            print(f"  [{task_name} ep={ep} DONE  success={bool(success)}  steps={step}]")
    sr = float(np.mean(successes))
    n_ok = int(sum(successes))
    print(f"  {task_name}: {sr*100:5.1f}%  ({n_ok}/{num_episodes})")
    return sr, successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--mean', action='store_true',
                        help='Use deterministic policy mean instead of MPPI sampling')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # OGBench env (64x64 visual cube-single)
    print("Importing ogbench, creating env...")
    import ogbench  # registers gymnasium envs
    import gymnasium

    env = gymnasium.make('visual-cube-single-v0')
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = len(task_infos)
    task_names = [t.get('task_name', f'task{i}') for i, t in enumerate(task_infos)]
    print(f"  {num_tasks} tasks: {task_names}")

    print(f"Loading TDMPC2 agent from {args.ckpt_dir} ...")
    agent = load_eval_agent(args.ckpt_dir, device=device, use_mean=args.mean)
    print(f"  Loaded. offline_mode={agent.cfg.offline_mode}  use_mean={agent.use_mean}")

    # Run all tasks
    t0 = time.time()
    per_task = {}
    for tid in range(1, num_tasks + 1):  # OGBench task IDs are 1-indexed
        tname = task_names[tid - 1]
        sr, _ = run_task(
            env, agent, tid, tname,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            diagnose=args.diagnose,
            goal_info_key='goal',
        )
        per_task[tname] = sr

    overall = float(np.mean(list(per_task.values())))
    elapsed = time.time() - t0

    # Print + write results
    print()
    print(f'==== {Path(args.ckpt_dir).name} ====')
    print(f"protocol: OGBench native — {num_tasks} tasks × {args.num_episodes} episodes")
    print(f"checkpoint: {Path(args.ckpt_dir).name}")
    for i, (tname, sr) in enumerate(per_task.items(), 1):
        print(f"  task{i}_{tname}: {sr*100:.1f}%")
    print(f"  overall: {overall*100:.1f}%")
    print(f"elapsed: {int(elapsed)}s")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, 'results.txt')
        with open(out_path, 'w') as f:
            f.write(f'==== {Path(args.ckpt_dir).name} ====\n')
            f.write(f"protocol: OGBench native — {num_tasks} tasks × {args.num_episodes} episodes\n")
            f.write(f"checkpoint: {Path(args.ckpt_dir).name}\n")
            for i, (tname, sr) in enumerate(per_task.items(), 1):
                f.write(f"  task{i}_{tname}: {sr*100:.1f}%\n")
            f.write(f"  overall: {overall*100:.1f}%\n")
            f.write(f"elapsed: {int(elapsed)}s\n")
        print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
