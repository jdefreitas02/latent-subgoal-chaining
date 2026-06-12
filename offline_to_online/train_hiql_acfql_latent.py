"""Stage A2: HIQL + ACFQL flow-chunked LL on LeJEPA 192-D latents.

Same agent as the state-based Stage A1 (ogbench agents/hiql_acfql.py), but the
observations are the finetuned-LeJEPA latents (lejepa_play_ft_full) and eval
encodes pixels->latent online with that same frozen encoder. No world model is
used in training here (the WM only *produced* the latent cache); WM rollouts are
Stage B.

Runs entirely in the leworldmodel .venv (torch LeJEPA encoder + JAX agent).

Train/eval latent consistency: the cache was built by reencode_play_dataset.py
via jepa.encode(...)["emb"]; eval uses encode_pixels_to_latent which is the same
path -> identical 192-D space.

Example:
  python train_hiql_acfql_latent.py \
    --wm_ckpt   ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \
    --cache     ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \
    --hdf5      ~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5 \
    --high_actor_type flow --save_dir ./checkpoints_hiql_acfql_latent_flowhl_s0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_THIS = os.path.abspath(os.path.dirname(__file__))
_OGB_IMPLS = os.path.expanduser('~/ogbench/impls')
# ogbench/impls MUST take priority for the 'agents' and 'utils' packages (both
# repos define them). offline_to_online is appended only for its unique 'envs'
# package (empty __init__, jepa_loader is self-contained) -> no collision.
sys.path.insert(0, _OGB_IMPLS)
if _THIS not in sys.path:
    sys.path.append(_THIS)

import jax
import jax.numpy as jnp
import torch
import h5py

from agents.hiql_acfql import HIQLACFQLAgent, get_config
from utils.datasets import Dataset, HGCDataset
from envs.jepa_loader import load_jepa, encode_pixels_to_latent, make_img_transform


# ---------------------------------------------------------------------------
# Latent goal-conditioned dataset
# ---------------------------------------------------------------------------
def build_latent_dataset(cache_path, hdf5_path):
    """Assemble an ogbench-format (regular) Dataset of LeJEPA latents.

    observations / next_observations : per-episode z[:T] / z[1:T+1]  (T = min(T_z-1, T_a))
    actions                          : per-episode a[:T]  (raw env actions, [-1, 1])
    terminals                        : 1 at each episode's last stored index

    Returns the dict for Dataset.create(**dict).
    """
    print(f'Loading latent cache: {cache_path}', flush=True)
    cache = torch.load(cache_path, map_location='cpu')
    all_latents = cache['all_latents']  # list of (T_i, 192) tensors
    n_ep = len(all_latents)
    print(f'  {n_ep} episodes of latents', flush=True)

    print(f'Loading actions from HDF5: {hdf5_path}', flush=True)
    with h5py.File(hdf5_path, 'r') as hf:
        acts_flat = np.asarray(hf['action'], dtype=np.float32)
        ep_offset = np.asarray(hf['ep_offset'])
        ep_len = np.asarray(hf['ep_len'])
    assert len(ep_offset) == n_ep, f'episode count mismatch: {len(ep_offset)} vs {n_ep}'

    obs_l, nobs_l, act_l, term_l = [], [], [], []
    for i in range(n_ep):
        z = np.asarray(all_latents[i], dtype=np.float32)          # (T_z, 192)
        a = acts_flat[int(ep_offset[i]): int(ep_offset[i]) + int(ep_len[i])]  # (T_a, 5)
        T = min(z.shape[0] - 1, a.shape[0])
        if T <= 0:
            continue
        obs_l.append(z[:T])
        nobs_l.append(z[1:T + 1])
        act_l.append(np.clip(a[:T], -1.0, 1.0))
        term = np.zeros(T, np.float32)
        term[-1] = 1.0
        term_l.append(term)

    observations = np.concatenate(obs_l, axis=0)
    next_observations = np.concatenate(nobs_l, axis=0)
    actions = np.concatenate(act_l, axis=0)
    terminals = np.concatenate(term_l, axis=0)
    print(f'  dataset: {observations.shape[0]:,} transitions, '
          f'obs_dim={observations.shape[1]}, act_dim={actions.shape[1]}', flush=True)
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        terminals=terminals,
    )


# ---------------------------------------------------------------------------
# Proper OGBench eval: per-task visual-cube-single-singletask-task{N}-v0 envs
# at 224px (real ogbench success), matching make_wm_env_and_dataset_multitask /
# evaluate_real_ogbench used by train_online_mpc_gc.py.  NOT swm/OGBCube-v0.
# ---------------------------------------------------------------------------
def make_eval_envs(jepa, device, tx, task_ids=(1, 2, 3, 4, 5), env_family='cube-single'):
    """One singletask env per task (render 224), plus each task's goal latent."""
    import ogbench  # registers gymnasium envs
    import gymnasium as gym
    from envs.env_utils import EpisodeMonitor

    real_envs, z_goals = {}, {}
    for tid in task_ids:
        env = gym.make(f'visual-{env_family}-singletask-task{tid}-v0', width=224, height=224)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        _, info = env.reset(seed=0, options=dict(render_goal=True))
        goal_img = info.get('goal', info.get('target'))
        if goal_img is None:
            raise RuntimeError(f'task {tid} reset info missing goal; keys={list(info.keys())}')
        z_goals[tid] = encode_pixels_to_latent(jepa, goal_img, device, tx)
        real_envs[tid] = env
    print(f'[eval] built {len(real_envs)} singletask envs + encoded goals', flush=True)
    return real_envs, z_goals


def _fmq_refine_chunk(agent, wm_model, wm_params, z_tiled, chunks, goal_tiled, eta):
    """One normalized-gradient FMQ step ascending V(WM(z,chunk), g_ult), backprop'd
    through the differentiable WM. Lets the chunk leave the flow proposal's support
    (which best-of-N cannot). chunks/z_tiled/goal_tiled all (N, .)."""
    def _score(ch):
        z1 = wm_model.apply(wm_params, z_tiled[:, None, :], ch[:, None, :])[:, -1, :]
        v1, v2 = agent.network.select('value')(z1, goal_tiled)
        return ((v1 + v2) / 2.0).sum()
    g = jax.grad(_score)(chunks)
    g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
    return jnp.clip(chunks + eta * g, -1.0, 1.0)


def _select_chunk_bon(agent, wm_model, wm_params, z_obs_j, subgoal, z_goal_j, bon, seed,
                      fmq_eta=0.0):
    """Best-of-N chunk selection: sample `bon` chunks from the flow LL (conditioned
    on the subgoal rep), roll each one step through the frozen WM, and score the
    endpoint with V(z', g_ult) against the ULTIMATE goal latent (ogbench GCValue
    takes a goal observation and computes phi internally). Return the argmax chunk.
    This is the WGSP/ACFQL-style selection the pure AWR-flow LL lacks at inference.
    bon==1 -> a single sample (refined by FMQ if fmq_eta>0, else plain)."""
    z_tiled = jnp.broadcast_to(z_obs_j, (bon, z_obs_j.shape[-1]))
    rep_tiled = jnp.broadcast_to(subgoal, (bon, subgoal.shape[-1]))
    chunks = agent.sample_low_chunk(z_tiled, rep_tiled, seed=seed, temperature=0.0)  # (bon, 25)
    goal_tiled = jnp.broadcast_to(z_goal_j, (bon, z_goal_j.shape[-1]))                # (bon, 192)
    if fmq_eta > 0.0:
        chunks = _fmq_refine_chunk(agent, wm_model, wm_params, z_tiled, chunks, goal_tiled, fmq_eta)
    if bon == 1:
        return np.array(chunks[0])
    z1 = wm_model.apply(wm_params, z_tiled[:, None, :], chunks[:, None, :])[:, -1, :]  # (bon, 192)
    v1, v2 = agent.network.select('value')(z1, goal_tiled)
    J = (v1 + v2) / 2.0  # (bon,)
    best = int(jnp.argmax(J))
    return np.array(chunks[best])


def evaluate_latent(agent, real_envs, z_goals, jepa, device, tx, cfg, n_eps, max_steps,
                    seed=0, bon=1, wm_model=None, wm_params=None, fmq_eta=0.0):
    """Hierarchical chunked eval on the proper per-task ogbench envs. When bon>1,
    each chunk is chosen best-of-N by rolling candidates through the WM and
    scoring V at the endpoint (requires wm_model/wm_params)."""
    subgoal_steps = cfg['subgoal_steps']
    chunk_len = cfg['action_chunk_len']
    action_dim = cfg['action_dim']
    assert subgoal_steps % chunk_len == 0, 'subgoal_steps must be a multiple of action_chunk_len'
    if bon > 1 or fmq_eta > 0.0:
        assert wm_model is not None and wm_params is not None, 'bon>1 / fmq_eta>0 needs a WM'
    rng = jax.random.PRNGKey(seed)

    per_task = []
    for tid, env in real_envs.items():
        z_goal_j = jnp.asarray(z_goals[tid])
        succ = []
        for _ep in range(n_eps):
            obs, info = env.reset()  # task baked into env registration
            done = False
            step = 0
            subgoal = None
            queue = []
            success_seen = 0.0
            while not done and step < max_steps:
                # Re-encode + resample only at chunk boundaries (open-loop chunk);
                # HL subgoal refreshes every subgoal_steps env steps.
                if len(queue) == 0:
                    z_obs_j = jnp.asarray(encode_pixels_to_latent(jepa, obs, device, tx))
                    if step % subgoal_steps == 0:
                        rng, k = jax.random.split(rng)
                        subgoal = agent.sample_high_rep(z_obs_j, z_goal_j, seed=k, temperature=0.0)
                    rng, k = jax.random.split(rng)
                    chunk = _select_chunk_bon(agent, wm_model, wm_params, z_obs_j, subgoal,
                                              z_goal_j, bon, k, fmq_eta=fmq_eta)
                    queue = list(chunk.reshape(-1, action_dim))
                a = np.clip(np.asarray(queue.pop(0)), -1.0, 1.0)
                obs, _, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                step += 1
                success_seen = max(success_seen, float(info.get('success', 0.0)))
            succ.append(float(info.get('success', success_seen)))
        sr = float(np.mean(succ))
        per_task.append(sr)
        print(f'    [bon={bon}] task {tid}: {sr*100:5.1f}%', flush=True)
    return float(np.mean(per_task)), per_task


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Stage A2: HIQL+ACFQL flow chunking on LeJEPA latents')
    ap.add_argument('--wm_ckpt', type=str,
                    default=os.path.expanduser('~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full'))
    ap.add_argument('--cache', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt'))
    ap.add_argument('--hdf5', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5'))
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--high_actor_type', type=str, default='gaussian', choices=['gaussian', 'flow'])
    ap.add_argument('--ll_awr', type=lambda s: str(s).lower() in ('1', 'true', 'yes'),
                    default=True, help='True: AWR-weighted flow LL. False: plain BC flow (q-chunking).')
    ap.add_argument('--total_steps', type=int, default=1_000_000)
    ap.add_argument('--eval_interval', type=int, default=200_000)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--eval_max_steps', type=int, default=200)
    ap.add_argument('--eval_bon', type=int, default=1,
                    help='Best-of-N chunk selection at eval (WM-scored). 1=plain. '
                         'When >1, reports both N=1 and best-of-N.')
    ap.add_argument('--log_interval', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--subgoal_steps', type=int, default=10)
    ap.add_argument('--action_chunk_len', type=int, default=5)
    ap.add_argument('--flow_steps', type=int, default=10)
    ap.add_argument('--high_alpha', type=float, default=3.0)
    ap.add_argument('--low_alpha', type=float, default=3.0)
    ap.add_argument('--wgsp_coef', type=float, default=0.0,
                    help='Stage B: weight of the WM-grounded LL improvement term. 0=off (A2).')
    ap.add_argument('--wgsp_num_samples', type=int, default=4)
    ap.add_argument('--wgsp_batch_size', type=int, default=256)
    ap.add_argument('--wgsp_alpha', type=float, default=3.0)
    ap.add_argument('--wgsp_fmq_eta', type=float, default=0.0,
                    help='Stage B: FMQ refinement step size on WGSP candidate chunks. 0=off.')
    ap.add_argument('--eval_fmq_eta', type=float, default=0.0,
                    help='Eval: FMQ refinement step size on best-of-N candidate chunks. 0=off.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Config (mirror ogbench hiql_acfql defaults, latent obs).
    config = get_config()
    config['batch_size'] = args.batch_size
    config['subgoal_steps'] = args.subgoal_steps
    config['action_chunk_len'] = args.action_chunk_len
    config['flow_steps'] = args.flow_steps
    config['high_alpha'] = args.high_alpha
    config['low_alpha'] = args.low_alpha
    config['high_actor_type'] = args.high_actor_type
    config['ll_awr'] = args.ll_awr
    config['wgsp_coef'] = args.wgsp_coef
    config['wgsp_num_samples'] = args.wgsp_num_samples
    config['wgsp_batch_size'] = args.wgsp_batch_size
    config['wgsp_alpha'] = args.wgsp_alpha
    config['wgsp_fmq_eta'] = args.wgsp_fmq_eta
    config['encoder'] = None          # obs already 192-D latents
    config['frame_stack'] = None

    # Dataset.
    data = build_latent_dataset(args.cache, args.hdf5)
    train_ds = HGCDataset(Dataset.create(**data), config)

    # Frozen JAX WM predictor. Loaded once; used by Stage B training (wgsp_coef>0)
    # and/or best-of-N eval selection (eval_bon>1).
    wm_model = wm_params = None
    if args.wgsp_coef > 0.0 or args.eval_bon > 1 or args.eval_fmq_eta > 0.0:
        from wm_jax import load_wm_jax
        print(f'Loading JAX WM (wgsp_coef={args.wgsp_coef}, eval_bon={args.eval_bon}, '
              f'eval_fmq_eta={args.eval_fmq_eta})...', flush=True)
        wm_model, wm_params = load_wm_jax(args.wm_ckpt)

    # Agent.
    ex = train_ds.sample(2)
    agent = HIQLACFQLAgent.create(args.seed, ex['observations'], ex['actions'], config,
                                  wm_model=wm_model, wm_params=wm_params)
    print(f'Agent created (high_actor_type={args.high_actor_type}, '
          f'obs_dim={ex["observations"].shape[-1]}, action_dim={agent.config["action_dim"]}, '
          f'wgsp_coef={args.wgsp_coef})', flush=True)

    # Frozen LeJEPA encoder (for eval only).
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading LeJEPA encoder for eval: {args.wm_ckpt}', flush=True)
    jepa = load_jepa(args.wm_ckpt, device=device, img_size=224, patch_size=14)
    img_tx = make_img_transform()
    eval_envs, eval_goals = make_eval_envs(jepa, device, img_tx)

    # WM (loaded above) doubles as the best-of-N eval selector.
    bon_list = [1, args.eval_bon] if args.eval_bon > 1 else [1]

    csv_path = os.path.join(args.save_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'value_loss', 'low_actor_loss', 'low_flow_bc',
                                'high_actor_loss', 'overall_success', 'elapsed_s'])

    print('Starting training...', flush=True)
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        batch = train_ds.sample(args.batch_size)
        agent, info = agent.update(batch)

        if step % args.log_interval == 0:
            vl = float(info['value/value_loss'])
            la = float(info['low_actor/actor_loss'])
            lf = float(info['low_actor/flow_bc_loss'])
            ha = float(info['high_actor/actor_loss'])
            print(f'step {step:>8,} | V {vl:.4f} | LL {la:.4f} (flow {lf:.4f}) | '
                  f'HL {ha:.4f} | {step/max(time.time()-t0,1e-6):.0f} sps', flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, vl, la, lf, ha, '', time.time() - t0])

        if step % args.eval_interval == 0 or step == args.total_steps:
            print(f'=== eval @ step {step:,} ({args.eval_episodes} eps/task) ===', flush=True)
            eval_agent = jax.device_put(agent, jax.devices('cpu')[0]) if device == 'cpu' else agent
            for bon in bon_list:
                overall, per_task = evaluate_latent(
                    eval_agent, eval_envs, eval_goals, jepa, device, img_tx, config,
                    n_eps=args.eval_episodes, max_steps=args.eval_max_steps, seed=args.seed,
                    bon=bon, wm_model=wm_model, wm_params=wm_params, fmq_eta=args.eval_fmq_eta)
                print(f'=== overall success @ {step:,} [bon={bon}]: {overall*100:.1f}% '
                      f'(per-task {[f"{x*100:.0f}" for x in per_task]}) ===', flush=True)
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([step, '', '', '', f'bon{bon}', overall, time.time() - t0])

    # Save final agent params (for Stage B reuse / re-eval).
    import pickle
    save_path = os.path.join(args.save_dir, 'agent_final.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'params': jax.device_get(agent.network.params),
                     'config': dict(config)}, f)
    print(f'Done. Metrics: {csv_path} | Agent: {save_path}', flush=True)


if __name__ == '__main__':
    main()
