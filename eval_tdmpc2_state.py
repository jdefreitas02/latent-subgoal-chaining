"""
Eval script for state-based goal-conditioned TD-MPC2 on OGBench cube-single.

Usage:
    python latent_hindsight_rl/eval_tdmpc2_state.py \
        --ckpt_dir checkpoints_tdmpc2_state_iql_s0 \
        --num_episodes 50 \
        --results_dir eval_tdmpc2_state/iql_s0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from latent_hindsight_rl.tdmpc2.config import GCTDMPC2Config
from latent_hindsight_rl.tdmpc2.gc_world_model import GCWorldModel
from latent_hindsight_rl.tdmpc2.mppi import MPPIPlanner


class StateTDMPC2Agent:
    """Eval agent for state-based GC TD-MPC2."""

    def __init__(self, model: GCWorldModel, cfg: GCTDMPC2Config, device='cuda',
                 use_mean: bool = False):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.use_mean = use_mean
        self.planner = MPPIPlanner(model, cfg, device=device)
        self._goal_latent = None
        self._t = 0

    def reset(self):
        self.planner.reset()
        self._goal_latent = None
        self._t = 0

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, goal: np.ndarray):
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        goal_t = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(self.device)
        z = self.model.encode(obs_t)
        g = self.model.encode(goal_t)
        if self.use_mean:
            action_t = self.planner.plan_mean(z, g)
        else:
            action_t = self.planner.plan(z, g, t0=(self._t == 0), eval_mode=True)
        self._t += 1
        return action_t.detach().cpu().numpy().astype(np.float32), None


def load_state_agent(ckpt_dir: str, device='cuda', use_mean: bool = False):
    cfg = torch.load(os.path.join(ckpt_dir, 'config.pt'), map_location='cpu', weights_only=False)
    cfg.num_pi_trajs = cfg.num_samples
    cfg.iterations = 1
    model = GCWorldModel(cfg)
    state = torch.load(os.path.join(ckpt_dir, 'model.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    return StateTDMPC2Agent(model, cfg, device=device, use_mean=use_mean)


def run_task(env, agent, task_id, task_name, num_episodes, max_steps):
    successes = []
    for ep in range(num_episodes):
        obs, info = env.reset(options=dict(task_id=task_id))
        goal = info['goal']
        agent.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action, _ = agent.get_action(obs, goal)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        success = float(info.get('success', 0.0))
        successes.append(success)
    sr = float(np.mean(successes))
    print(f"  {task_name}: {sr*100:5.1f}%  ({int(sum(successes))}/{num_episodes})")
    return sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--mean', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import ogbench
    import gymnasium
    env = gymnasium.make('cube-single-v0')
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = len(task_infos)
    task_names = [t.get('task_name', f'task{i}') for i, t in enumerate(task_infos)]
    print(f"  {num_tasks} tasks: {task_names}")

    agent = load_state_agent(args.ckpt_dir, device=str(device), use_mean=args.mean)
    cfg = agent.cfg
    print(f"  Loaded. obs={cfg.obs}  offline_mode={cfg.offline_mode}  use_mean={agent.use_mean}")

    t0 = time.time()
    per_task = {}
    for tid in range(1, num_tasks + 1):
        tname = task_names[tid - 1]
        sr = run_task(env, agent, tid, tname,
                      num_episodes=args.num_episodes, max_steps=args.max_steps)
        per_task[tname] = sr

    overall = float(np.mean(list(per_task.values())))
    elapsed = time.time() - t0

    print(f'\n==== {Path(args.ckpt_dir).name} ====')
    print(f"protocol: OGBench state — {num_tasks} tasks × {args.num_episodes} episodes")
    for i, (tname, sr) in enumerate(per_task.items(), 1):
        print(f"  task{i}_{tname}: {sr*100:.1f}%")
    print(f"  overall: {overall*100:.1f}%")
    print(f"elapsed: {int(elapsed)}s")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out = os.path.join(args.results_dir, 'results.txt')
        with open(out, 'w') as f:
            f.write(f'==== {Path(args.ckpt_dir).name} ====\n')
            f.write(f"protocol: OGBench state — {num_tasks} tasks × {args.num_episodes} episodes\n")
            for i, (tname, sr) in enumerate(per_task.items(), 1):
                f.write(f"  task{i}_{tname}: {sr*100:.1f}%\n")
            f.write(f"  overall: {overall*100:.1f}%\n")
            f.write(f"elapsed: {int(elapsed)}s\n")
        print(f"Results written to {out}")


if __name__ == '__main__':
    main()
