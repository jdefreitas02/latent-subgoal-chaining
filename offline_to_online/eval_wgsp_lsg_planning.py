"""Eval-time planning GATE for a frozen WGSP-LSG Phase-1 checkpoint.

Decisive cheap test before committing to a hierarchical online-MPC build: does
adding best-of-N MPC at eval time (the exact thing Phase-2 distills) lift the
frozen Phase-1 policy above its single-sample standalone score, in the 5-task GC
setting?

  - LL best-of-N (n_low): every chunk, sample n_low chunks toward the current
    subgoal w, roll each one frozen-WM step, score by the LL reach metric
    (V(z',w) for 'value'), pick the best. This is `agent.fmq_mpc_low`, whose N is
    config['wgsp_num_samples'] -- swept by copying the (static) config.
  - HL best-of-N (n_high): every subgoal_steps, sample n_high subgoal candidates,
    roll the LL toward each in the WM for subgoal_steps, score V(z_achieved, g),
    commit to the argmax subgoal.

(n_high=1, n_low=1) reproduces the standalone hierarchical eval (~88%). If higher
(n_high, n_low) does not beat it, MPC does not help even at eval time in GC -> the
ceiling is real and the full online build is not worth it.

  python eval_wgsp_lsg_planning.py \
    --ckpt ./ckpt_wgsp_lsg_p1_task12345_latent_s0/agent_final.pkl \
    --n_high 1,4 --n_low 1,8,32 --eval_episodes 20
"""
import argparse
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

from envs.jepa_loader import load_jepa, encode_pixels_to_latent, make_img_transform
from wm_jax import load_wm_jax
from train_hiql_acfql_latent import make_eval_envs
from train_wgsp_lsg_phase2 import load_phase1


def _select_high_bon(agent, zb, gb, n_high, rng):
    """HL best-of-N: sample n_high subgoals, roll the LL toward each in the WM,
    score V(z_achieved, g), return the argmax subgoal (all 1-batch arrays)."""
    if n_high == 1:
        rng, k = jax.random.split(rng)
        return agent.sample_high_subgoal(zb, gb, seed=k), rng
    best_w, best_v = None, -jnp.inf
    for _ in range(n_high):
        rng, k1, k2 = jax.random.split(rng, 3)
        w_i = agent.sample_high_subgoal(zb, gb, seed=k1)
        sg = agent._subgoal_from_latent(zb, w_i)
        z_ach = agent.roll_low_in_wm(zb, sg, k2)
        v1, v2 = agent._value(z_ach, gb)
        v = float(((v1 + v2) / 2.0)[0])
        if v > best_v:
            best_v, best_w = v, w_i
    return best_w, rng


def evaluate_planning(agent, real_envs, z_goals, jepa, device, tx, cfg,
                      n_high, n_low, n_eps, max_steps, seed=0):
    """Hierarchical eval with HL best-of-n_high and LL best-of-n_low MPC."""
    subgoal_steps = cfg['subgoal_steps']
    chunk_len = cfg['action_chunk_len']
    action_dim = cfg['action_dim']
    rng = jax.random.PRNGKey(seed)
    per_task = []
    for tid, env in real_envs.items():
        z_goal = jnp.asarray(z_goals[tid])[None]            # (1, D)
        succ = []
        for _ep in range(n_eps):
            obs, info = env.reset()
            done, step, w, queue, seen = False, 0, None, [], 0.0
            while not done and step < max_steps:
                if len(queue) == 0:
                    z = jnp.asarray(encode_pixels_to_latent(jepa, obs, device, tx))[None]  # (1,D)
                    if step % subgoal_steps == 0:
                        w, rng = _select_high_bon(agent, z, z_goal, n_high, rng)
                    rng, k = jax.random.split(rng)
                    a_mpc, _ = agent.fmq_mpc_low(z, w, k)    # (1, chunk_dim), uses n_low
                    chunk = np.array(a_mpc).reshape(-1, action_dim)
                    queue = list(chunk)
                a = np.clip(np.asarray(queue.pop(0)), -1.0, 1.0)
                obs, _, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                step += 1
                seen = max(seen, float(info.get('success', 0.0)))
            succ.append(float(info.get('success', seen)))
        sr = float(np.mean(succ))
        per_task.append(sr)
        print(f'      task {tid}: {sr*100:5.1f}%', flush=True)
    return float(np.mean(per_task)), per_task


def main():
    ap = argparse.ArgumentParser(description='WGSP-LSG eval-time planning gate')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--wm_ckpt', type=str,
                    default=os.path.expanduser('~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full'))
    ap.add_argument('--task_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--n_high', type=str, default='1,4')
    ap.add_argument('--n_low', type=str, default='1,8,32')
    ap.add_argument('--ll_reach_metric', type=str, default='value', choices=['distance', 'value'])
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--eval_max_steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    n_high_list = [int(x) for x in args.n_high.split(',')]
    n_low_list = [int(x) for x in args.n_low.split(',')]

    print(f'Loading JAX WM: {args.wm_ckpt}', flush=True)
    wm_model, wm_params = load_wm_jax(args.wm_ckpt)
    overrides = dict(wgsp_coef=1.0, ll_reach_metric=args.ll_reach_metric,
                     wgsp_num_samples=max(n_low_list), wgsp_fmq_eta=0.0)
    print(f'Loading Phase-1 checkpoint: {args.ckpt}', flush=True)
    agent, cfg = load_phase1(args.ckpt, wm_model, wm_params, overrides, seed=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jepa = load_jepa(args.wm_ckpt, device=device, img_size=224, patch_size=14)
    tx = make_img_transform()
    eval_envs, eval_goals = make_eval_envs(jepa, device, tx, task_ids=tuple(args.task_ids))

    print(f'\n=== Planning gate: reach={args.ll_reach_metric}  '
          f'n_high={n_high_list}  n_low={n_low_list}  ({args.eval_episodes} eps/task) ===\n', flush=True)
    grid = {}
    for nh in n_high_list:
        for nl in n_low_list:
            agent_nl = agent.replace(config=agent.config.copy({'wgsp_num_samples': nl}))
            t0 = time.time()
            print(f'  --- n_high={nh}  n_low={nl} ---', flush=True)
            overall, per_task = evaluate_planning(
                agent_nl, eval_envs, eval_goals, jepa, device, tx, cfg,
                nh, nl, args.eval_episodes, args.eval_max_steps, args.seed)
            grid[(nh, nl)] = (overall, per_task)
            print(f'  ==> n_high={nh} n_low={nl}: {overall*100:.1f}% '
                  f'(per-task {[f"{x*100:.0f}" for x in per_task]})  [{time.time()-t0:.0f}s]\n', flush=True)

    print('=== SUMMARY GRID (overall success %) ===', flush=True)
    header = 'n_high\\n_low | ' + ' | '.join(f'{nl:>5}' for nl in n_low_list)
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for nh in n_high_list:
        row = f'{nh:>10} | ' + ' | '.join(f'{grid[(nh,nl)][0]*100:5.1f}' for nl in n_low_list)
        print(row, flush=True)
    base = grid[(n_high_list[0], n_low_list[0])][0] * 100
    best = max(v[0] for v in grid.values()) * 100
    print(f'\nbaseline (n_high={n_high_list[0]},n_low={n_low_list[0]}) = {base:.1f}%  |  '
          f'best cell = {best:.1f}%  |  lift = {best-base:+.1f}pp', flush=True)


if __name__ == '__main__':
    main()
