"""Inference-only WGSP planning re-eval on a finished hiql_acfql A2 checkpoint.

Stage A2 only does best-of-N at the LOW level (the WM ranks LL chunks). This
script adds best-of-N at the HIGH level too: every subgoal_steps it samples
n_high candidate subgoal reps from the (flow or gaussian) HL, rolls one LL chunk
per candidate through the frozen WM, scores V(z', g_ult), and commits to the best
subgoal for the next subgoal_steps env steps. The LL then still does its own
best-of-n_low chunk selection within the committed subgoal.

This is the full HL x LL WM planning tree, but applied to an ALREADY-TRAINED
agent (no retraining) -- the cheap gate before Stage B (WM-in-training).

Scoring is kept identical to the LL BoN: one chunk -> one frozen-WM step ->
V(z', g_ult) toward the ultimate goal latent. n_high=1 reproduces the Stage A2
eval exactly; n_low=1 disables LL selection.

Run:
  python eval_wgsp_planning.py \
    --ckpt   ./checkpoints_hiql_acfql_a2_gaussian_s0/agent_final.pkl \
    --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \
    --n_high 1,4 --n_low 1,8 --eval_episodes 20
"""
import argparse
import os
import pickle
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

from agents.hiql_acfql import HIQLACFQLAgent, get_config
from envs.jepa_loader import load_jepa, encode_pixels_to_latent, make_img_transform
from train_hiql_acfql_latent import make_eval_envs, _select_chunk_bon
from wm_jax import load_wm_jax


def load_agent(ckpt_path, seed=0):
    """Rebuild a HIQLACFQLAgent from a saved {'params','config'} pickle."""
    with open(ckpt_path, 'rb') as f:
        blob = pickle.load(f)
    saved_params = blob['params']
    saved_cfg = blob['config']

    config = get_config()
    for k, v in saved_cfg.items():
        config[k] = v
    # Synthetic example inputs to instantiate the network skeleton, then overwrite
    # with the trained params. obs_dim (latent=192) / action_dim are in the config.
    od = int(saved_cfg.get('obs_dim', 192))
    ad = int(saved_cfg['action_dim'])
    ex_obs = np.zeros((2, od), np.float32)
    ex_act = np.zeros((2, ad), np.float32)
    agent = HIQLACFQLAgent.create(seed, ex_obs, ex_act, config)
    agent = agent.replace(network=agent.network.replace(params=saved_params))
    return agent, config


def _select_subgoal_bon(agent, wm_model, wm_params, z_obs_j, z_goal_j, n_high, seed):
    """HL best-of-N: sample n_high subgoal reps, roll one LL chunk per candidate
    through the frozen WM, score V(z', g_ult), return the best subgoal rep.
    n_high==1 falls back to a single deterministic HL sample (Stage A2 behaviour)."""
    if n_high == 1:
        return agent.sample_high_rep(z_obs_j, z_goal_j, seed=seed, temperature=0.0)
    z_tiled = jnp.broadcast_to(z_obs_j, (n_high, z_obs_j.shape[-1]))
    goal_tiled = jnp.broadcast_to(z_goal_j, (n_high, z_goal_j.shape[-1]))
    k1, k2 = jax.random.split(seed)
    subgoals = agent.sample_high_rep(z_tiled, goal_tiled, seed=k1, temperature=0.0)  # (n_high, rep)
    chunks = agent.sample_low_chunk(z_tiled, subgoals, seed=k2, temperature=0.0)     # (n_high, 25)
    z1 = wm_model.apply(wm_params, z_tiled[:, None, :], chunks[:, None, :])[:, -1, :]  # (n_high, 192)
    v1, v2 = agent.network.select('value')(z1, goal_tiled)
    J = (v1 + v2) / 2.0
    best = int(jnp.argmax(J))
    return subgoals[best]


def evaluate_planning(agent, real_envs, z_goals, jepa, device, tx, cfg,
                      n_eps, max_steps, seed, n_high, n_low, wm_model, wm_params, fmq_eta=0.0):
    subgoal_steps = cfg['subgoal_steps']
    chunk_len = cfg['action_chunk_len']
    action_dim = cfg['action_dim']
    assert subgoal_steps % chunk_len == 0
    rng = jax.random.PRNGKey(seed)

    per_task = []
    for tid, env in real_envs.items():
        z_goal_j = jnp.asarray(z_goals[tid])
        succ = []
        for _ep in range(n_eps):
            obs, info = env.reset()
            done, step = False, 0
            subgoal, queue, success_seen = None, [], 0.0
            while not done and step < max_steps:
                if len(queue) == 0:
                    z_obs_j = jnp.asarray(encode_pixels_to_latent(jepa, obs, device, tx))
                    if step % subgoal_steps == 0:
                        rng, k = jax.random.split(rng)
                        subgoal = _select_subgoal_bon(agent, wm_model, wm_params,
                                                      z_obs_j, z_goal_j, n_high, k)
                    rng, k = jax.random.split(rng)
                    chunk = _select_chunk_bon(agent, wm_model, wm_params, z_obs_j,
                                              subgoal, z_goal_j, n_low, k, fmq_eta=fmq_eta)
                    queue = list(chunk.reshape(-1, action_dim))
                a = np.clip(np.asarray(queue.pop(0)), -1.0, 1.0)
                obs, _, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                step += 1
                success_seen = max(success_seen, float(info.get('success', 0.0)))
            succ.append(float(info.get('success', success_seen)))
        sr = float(np.mean(succ))
        per_task.append(sr)
        print(f'    [n_high={n_high} n_low={n_low} fmq={fmq_eta}] task {tid}: {sr*100:5.1f}%', flush=True)
    return float(np.mean(per_task)), per_task


def main():
    ap = argparse.ArgumentParser(description='Inference-only WGSP planning re-eval')
    ap.add_argument('--ckpt', type=str, required=True, help='agent_final.pkl from Stage A2')
    ap.add_argument('--wm_ckpt', type=str,
                    default=os.path.expanduser('~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full'))
    ap.add_argument('--n_high', type=str, default='1,4', help='comma list of HL candidate counts')
    ap.add_argument('--n_low', type=str, default='1,8', help='comma list of LL candidate counts')
    ap.add_argument('--fmq_eta', type=str, default='0', help='comma list of FMQ step sizes (0=off)')
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--eval_max_steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    n_high_list = [int(x) for x in args.n_high.split(',')]
    n_low_list = [int(x) for x in args.n_low.split(',')]
    fmq_list = [float(x) for x in args.fmq_eta.split(',')]

    print(f'Loading agent: {args.ckpt}', flush=True)
    agent, cfg = load_agent(args.ckpt, seed=args.seed)
    print(f'  high_actor_type={cfg["high_actor_type"]} ll_awr={cfg.get("ll_awr")} '
          f'action_dim={cfg["action_dim"]} subgoal_steps={cfg["subgoal_steps"]}', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jepa = load_jepa(args.wm_ckpt, device=device, img_size=224, patch_size=14)
    tx = make_img_transform()
    real_envs, z_goals = make_eval_envs(jepa, device, tx)

    print('Loading JAX WM for planning...', flush=True)
    wm_model, wm_params = load_wm_jax(args.wm_ckpt)

    t0 = time.time()
    print(f'\n=== WGSP planning grid (n_high x n_low x fmq) | {args.eval_episodes} eps/task ===', flush=True)
    results = {}
    for fmq in fmq_list:
        for n_high in n_high_list:
            for n_low in n_low_list:
                overall, per_task = evaluate_planning(
                    agent, real_envs, z_goals, jepa, device, tx, cfg,
                    n_eps=args.eval_episodes, max_steps=args.eval_max_steps, seed=args.seed,
                    n_high=n_high, n_low=n_low, wm_model=wm_model, wm_params=wm_params, fmq_eta=fmq)
                results[(fmq, n_high, n_low)] = (overall, per_task)
                print(f'=== fmq={fmq} n_high={n_high} n_low={n_low}: {overall*100:.1f}% '
                      f'(per-task {[f"{x*100:.0f}" for x in per_task]}) ===\n', flush=True)

    print('\n========== WGSP planning grid summary ==========', flush=True)
    for fmq in fmq_list:
        print(f'-- fmq_eta={fmq} --', flush=True)
        print(f'{"":>10}' + ''.join(f'n_low={nl:<6}' for nl in n_low_list), flush=True)
        for n_high in n_high_list:
            row = f'n_high={n_high:<3}'
            for n_low in n_low_list:
                row += f'{results[(fmq, n_high, n_low)][0]*100:>11.1f}'
            print(row, flush=True)
    print(f'(elapsed {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
