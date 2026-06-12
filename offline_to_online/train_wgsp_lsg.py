"""WGSP-LSG trainer. Phase 1 (this file) = hierarchical pretrain on LeJEPA
latents with latent subgoals; Phase 2 (--wgsp_coef>0, added next) = frozen-V
WGSP-MPC distillation. Eval is fully standalone & hierarchical (HL emits a latent
subgoal toward the task goal, LL acts toward it), no WM at test time.

Reuses build_latent_dataset / make_eval_envs from train_hiql_acfql_latent.

  python train_wgsp_lsg.py --save_dir ./ckpt_wgsp_lsg_task1_s0 --task_ids 1
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
_OGB_IMPLS = os.path.expanduser('~/ogbench/impls')
sys.path.insert(0, _OGB_IMPLS)
if _THIS not in sys.path:
    sys.path.append(_THIS)

import jax
import jax.numpy as jnp
import torch

from agents.wgsp_lsg import WGSPLSGAgent, get_config
from utils.datasets import Dataset, HGCDataset
from envs.jepa_loader import load_jepa, encode_pixels_to_latent, make_img_transform
from train_hiql_acfql_latent import build_latent_dataset, make_eval_envs


def evaluate_lsg(agent, real_envs, z_goals, jepa, device, tx, cfg, n_eps, max_steps, seed=0):
    """Standalone hierarchical eval with latent subgoals (no WM at test)."""
    subgoal_steps = cfg['subgoal_steps']
    chunk_len = cfg['action_chunk_len']
    action_dim = cfg['action_dim']
    assert subgoal_steps % chunk_len == 0
    rng = jax.random.PRNGKey(seed)
    per_task = []
    for tid, env in real_envs.items():
        z_goal = jnp.asarray(z_goals[tid])
        succ = []
        for _ep in range(n_eps):
            obs, info = env.reset()
            done, step, w, queue, success_seen = False, 0, None, [], 0.0
            while not done and step < max_steps:
                if len(queue) == 0:
                    z = jnp.asarray(encode_pixels_to_latent(jepa, obs, device, tx))
                    if step % subgoal_steps == 0:
                        rng, k = jax.random.split(rng)
                        w = agent.sample_high_subgoal(z, z_goal, seed=k)
                    rng, k = jax.random.split(rng)
                    chunk = np.array(agent.sample_low_chunk(z, w, seed=k))
                    queue = list(chunk.reshape(-1, action_dim))
                a = np.clip(np.asarray(queue.pop(0)), -1.0, 1.0)
                obs, _, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                step += 1
                success_seen = max(success_seen, float(info.get('success', 0.0)))
            succ.append(float(info.get('success', success_seen)))
        sr = float(np.mean(succ))
        per_task.append(sr)
        print(f'    task {tid}: {sr*100:5.1f}%', flush=True)
    return float(np.mean(per_task)), per_task


def main():
    ap = argparse.ArgumentParser(description='WGSP-LSG Phase-1 trainer')
    ap.add_argument('--wm_ckpt', type=str,
                    default=os.path.expanduser('~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full'))
    ap.add_argument('--cache', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt'))
    ap.add_argument('--hdf5', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5'))
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--task_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--total_steps', type=int, default=1_000_000)
    ap.add_argument('--eval_interval', type=int, default=250_000)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--eval_max_steps', type=int, default=200)
    ap.add_argument('--log_interval', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--subgoal_steps', type=int, default=10)
    ap.add_argument('--action_chunk_len', type=int, default=5)
    ap.add_argument('--flow_steps', type=int, default=10)
    ap.add_argument('--high_alpha', type=float, default=3.0)
    ap.add_argument('--low_alpha', type=float, default=3.0)
    ap.add_argument('--ll_awr', type=lambda s: str(s).lower() in ('1', 'true', 'yes'), default=False)
    ap.add_argument('--subgoal_space', type=str, default='latent', choices=['latent', 'rep'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)

    config = get_config()
    config['batch_size'] = args.batch_size
    config['subgoal_steps'] = args.subgoal_steps
    config['action_chunk_len'] = args.action_chunk_len
    config['flow_steps'] = args.flow_steps
    config['high_alpha'] = args.high_alpha
    config['low_alpha'] = args.low_alpha
    config['ll_awr'] = args.ll_awr
    config['subgoal_space'] = args.subgoal_space
    config['encoder'] = None
    config['frame_stack'] = None

    data = build_latent_dataset(args.cache, args.hdf5)
    train_ds = HGCDataset(Dataset.create(**data), config)

    ex = train_ds.sample(2)
    agent = WGSPLSGAgent.create(args.seed, ex['observations'], ex['actions'], config)
    print(f'Agent created (subgoal_dim={agent.config["subgoal_dim"]}, '
          f'action_dim={agent.config["action_dim"]}, ll_awr={args.ll_awr})', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading LeJEPA encoder for eval: {args.wm_ckpt}', flush=True)
    jepa = load_jepa(args.wm_ckpt, device=device, img_size=224, patch_size=14)
    img_tx = make_img_transform()
    eval_envs, eval_goals = make_eval_envs(jepa, device, img_tx, task_ids=tuple(args.task_ids))

    csv_path = os.path.join(args.save_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'value_loss', 'high_actor_loss', 'low_actor_loss',
                                'overall_success', 'elapsed_s'])

    print('Starting Phase-1 training...', flush=True)
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        batch = train_ds.sample(args.batch_size)
        agent, info = agent.update(batch)
        if step % args.log_interval == 0:
            vl = float(info['value/value_loss']); ha = float(info['high_actor/high_actor_loss'])
            la = float(info['low_actor/low_actor_loss']); hb = float(info['high_actor/high_flow_bc'])
            lb = float(info['low_actor/low_flow_bc'])
            print(f'step {step:>8,} | V {vl:.4f} | HL {ha:.4f}(bc {hb:.4f}) | '
                  f'LL {la:.4f}(bc {lb:.4f}) | {step/max(time.time()-t0,1e-6):.0f} sps', flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, vl, ha, la, '', time.time() - t0])
        if step % args.eval_interval == 0 or step == args.total_steps:
            print(f'=== eval @ step {step:,} ({args.eval_episodes} eps/task) ===', flush=True)
            overall, per_task = evaluate_lsg(agent, eval_envs, eval_goals, jepa, device, img_tx,
                                             config, args.eval_episodes, args.eval_max_steps, args.seed)
            print(f'=== overall @ {step:,}: {overall*100:.1f}% '
                  f'(per-task {[f"{x*100:.0f}" for x in per_task]}) ===', flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, '', '', '', overall, time.time() - t0])

    import pickle
    save_path = os.path.join(args.save_dir, 'agent_final.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'params': jax.device_get(agent.network.params), 'config': dict(config)}, f)
    print(f'Done. Metrics: {csv_path} | Agent: {save_path}', flush=True)


if __name__ == '__main__':
    main()
