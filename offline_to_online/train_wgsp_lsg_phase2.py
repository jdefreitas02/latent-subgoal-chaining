"""WGSP-LSG Phase 2: frozen-V WGSP-MPC distillation, started from a Phase-1 ckpt.

Reuses the FMQ-MPC machinery exactly as the working online-in-WM method:
  - LL: maintain a buffer of (z, w=data subgoal, a_MPC) where a_MPC is the
    FMQ-MPC-selected chunk that best REACHES w (V(WM(z,a), w) argmax, optional FMQ
    refine). Distill the LL toward a_MPC with plain BC (actor-only). [your idea]
  - HL: AWR-grounded by the value the LL ACHIEVES when aiming for each candidate
    subgoal -- sample N subgoals, roll the LL toward each in the WM, score
    V(z_achieved, g), softmax-weight, flow-BC the HL toward the winners + an
    on-manifold anchor toward the data subgoal. [your 'separate HL loss']

V (the IQL critic) is FROZEN throughout -> stable OOD gate (the thesis's central
lesson). Eval is standalone hierarchical, no WM at test time.

  python train_wgsp_lsg_phase2.py --phase1_ckpt ./ckpt_wgsp_lsg_p1_task1_s0/agent_final.pkl \
      --save_dir ./ckpt_wgsp_lsg_p2_task1_s0 --task_ids 1
"""
import argparse
import csv
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

from agents.wgsp_lsg import WGSPLSGAgent, get_config
from utils.datasets import Dataset, HGCDataset
from envs.jepa_loader import load_jepa, make_img_transform
from wm_jax import load_wm_jax
from train_hiql_acfql_latent import build_latent_dataset, make_eval_envs
from train_wgsp_lsg import evaluate_lsg


def sample_hl_base(train_ds, batch_size, subgoal_steps, np_rng):
    """Sample a batch of (z, g, w_data, real_chunk) tuples from the dataset.

    w_data is the data subgoal at 1×subgoal_steps on z's own trajectory.
    Candidates cand[1..K-1] are generated in the training loop by sampling from the
    HL flow model — they are in-distribution for the HL at every training step.

    Returns z (B,D), g (B,D), w_data (B,D), real_chunks (B, chunk_dim)."""
    chunk_len = train_ds.config['action_chunk_len']
    idxs = train_ds.dataset.get_random_idxs(batch_size)
    final = train_ds.terminal_locs[np.searchsorted(train_ds.terminal_locs, idxs)]
    z = train_ds.get_observations(idxs)
    d = np_rng.random(batch_size)
    g_idxs = np.round(np.minimum(idxs + 1, final) * d + final * (1 - d)).astype(int)
    g = train_ds.get_observations(g_idxs)
    w_data = train_ds.get_observations(np.minimum(idxs + subgoal_steps, g_idxs))
    chunk_idxs = np.minimum(idxs[:, None] + np.arange(chunk_len)[None, :], final[:, None])
    raw = train_ds.dataset['actions'][chunk_idxs]
    real_chunks = raw.reshape(batch_size, -1)
    return (z.astype(np.float32), g.astype(np.float32),
            w_data.astype(np.float32), real_chunks.astype(np.float32))


def load_phase1(ckpt_path, wm_model, wm_params, overrides, seed=0):
    with open(ckpt_path, 'rb') as f:
        blob = pickle.load(f)
    params, saved_cfg = blob['params'], blob['config']
    config = get_config()
    for k, v in saved_cfg.items():
        config[k] = v
    for k, v in overrides.items():
        config[k] = v
    config['encoder'] = None
    config['frame_stack'] = None
    od = int(saved_cfg.get('subgoal_dim', 192))
    ad = int(saved_cfg['action_dim'])
    agent = WGSPLSGAgent.create(seed, np.zeros((2, od), np.float32), np.zeros((2, ad), np.float32),
                                config, wm_model=wm_model, wm_params=wm_params)
    agent = agent.replace(network=agent.network.replace(params=params))
    return agent, config


def main():
    ap = argparse.ArgumentParser(description='WGSP-LSG Phase 2 (frozen-V WGSP-MPC distillation)')
    ap.add_argument('--phase1_ckpt', type=str, required=True)
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--wm_ckpt', type=str,
                    default=os.path.expanduser('~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full'))
    ap.add_argument('--cache', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt'))
    ap.add_argument('--hdf5', type=str,
                    default=os.path.expanduser('~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5'))
    ap.add_argument('--task_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--total_steps', type=int, default=200_000)
    ap.add_argument('--eval_interval', type=int, default=50_000)
    ap.add_argument('--eval_episodes', type=int, default=20)
    ap.add_argument('--eval_max_steps', type=int, default=200)
    ap.add_argument('--log_interval', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=256)        # LL distill minibatch
    ap.add_argument('--hl_batch', type=int, default=256)
    ap.add_argument('--hl_every', type=int, default=4)            # ground HL every N steps
    ap.add_argument('--relabel_every', type=int, default=2500)    # refresh LL buffer
    ap.add_argument('--relabel_size', type=int, default=8192)     # (z,w,a_MPC) per refresh
    ap.add_argument('--wgsp_num_samples', type=int, default=32)
    ap.add_argument('--wgsp_fmq_eta', type=float, default=0.0)
    ap.add_argument('--ll_reach_metric', type=str, default='distance', choices=['distance', 'value'],
                    help="LL reachability score: 'distance' (latent ||z'-w||) vs 'value' (V(z',w)).")
    ap.add_argument('--hl_num_samples', type=int, default=8)
    ap.add_argument('--hl_anchor_coef', type=float, default=1.0)
    ap.add_argument('--hl_ground_coef', type=float, default=0.3,
                    help='[hl_mode=real] weight of the real-candidate grounding nudge vs the data anchor.')
    ap.add_argument('--ll_mpc_coef', type=float, default=0.3,
                    help='weight of the MPC nudge vs the real-data BC anchor in the LL (0 = pure '
                         'Phase-1 continuation; the anchor is what prevents collapse).')
    ap.add_argument('--winner_ll_mode', type=str, default='anchor',
                    choices=['none', 'mpc_only', 'anchor'],
                    help="How to train LL on winner subgoals after HL grounding. "
                         "'none': skip (isolates HL grounding impact); "
                         "'mpc_only': BC toward a_mpc_win only, no real_chunk anchor; "
                         "'anchor': real_chunk anchor + ll_mpc_coef*a_mpc_win (current default).")
    ap.add_argument('--hl_score_mode', type=str, default='reachability',
                    choices=['reachability', 'rollout'],
                    help="Candidate scoring: 'reachability' = V(z_ach,g)-V(cand,g) (gap, fixes "
                         "value inversion); 'rollout' = V(z_ach,g) only (original).")
    ap.add_argument('--hl_mode', type=str, default='awr',
                    choices=['awr', 'real', 'self', 'frozen', 'phase1awr', 'dpo', 'hrf'],
                    help="HL Phase-2 update: 'awr' (AWR BC on REAL data subgoals weighted by the "
                         "WM-rollout advantage -- no self-samples, CPU-verified non-degrading), "
                         "'real' (argmax over [data + HL self-samples] + BC nudge -- degrades V_roll "
                         "due to self-consuming flow distillation), 'self' (old self-AWR loop -- "
                         "collapses), 'frozen' (no HL update; MPC-distill the LL only -- control), "
                         "'phase1awr' (AWR with Phase-1 value advantage, ablation), "
                         "'dpo' (Flow-DPO: contrastive update with Phase-1 reference + feasibility), "
                         "'hrf' (Hindsight Reality Forcing: AWR toward achieved latent z_ach, not "
                         "proposed subgoal -- blocks value gaming by targeting achievable manifold).")
    ap.add_argument('--hl_awr_alpha', type=float, default=1.0,
                    help="[hl_mode=awr] AWR temperature on the standardised WM-rollout advantage.")
    ap.add_argument('--hl_dpo_beta', type=float, default=0.5,
                    help="[hl_mode=dpo] DPO temperature beta: scales the logits before log-sigmoid.")
    ap.add_argument('--hl_dpo_feasibility', type=float, default=0.1,
                    help="[hl_mode=dpo] Feasibility penalty lambda: J_safe = V(z_ach,g) - lambda*||z_ach-sg||.")
    ap.add_argument('--ll_frozen', action='store_true',
                    help="Keep the LL completely frozen (no buffer updates). Only the HL is updated "
                         "via AWR. The WM rollouts then use the exact same LL as eval, eliminating "
                         "LL drift and the HL/LL distribution mismatch.")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f'Loading JAX WM: {args.wm_ckpt}', flush=True)
    wm_model, wm_params = load_wm_jax(args.wm_ckpt)

    overrides = dict(wgsp_coef=1.0, wgsp_num_samples=args.wgsp_num_samples,
                     wgsp_fmq_eta=args.wgsp_fmq_eta, hl_num_samples=args.hl_num_samples,
                     hl_anchor_coef=args.hl_anchor_coef, hl_ground_coef=args.hl_ground_coef,
                     ll_mpc_coef=args.ll_mpc_coef, ll_reach_metric=args.ll_reach_metric,
                     hl_score_mode=args.hl_score_mode, hl_awr_alpha=args.hl_awr_alpha,
                     hl_dpo_beta=args.hl_dpo_beta, hl_dpo_feasibility=args.hl_dpo_feasibility)
    print(f'Loading Phase-1 checkpoint: {args.phase1_ckpt}', flush=True)
    agent, config = load_phase1(args.phase1_ckpt, wm_model, wm_params, overrides, seed=args.seed)
    print(f'Agent restored. wgsp N={args.wgsp_num_samples} fmq_eta={args.wgsp_fmq_eta} '
          f'hl_N={args.hl_num_samples}', flush=True)

    # Store the Phase-1 HL params as the DPO reference policy (frozen throughout Phase-2).
    # JAX arrays are immutable so this reference is safe; agent updates always create new arrays.
    ref_params = agent.network.params

    data = build_latent_dataset(args.cache, args.hdf5)
    train_ds = HGCDataset(Dataset.create(**data), config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jepa = load_jepa(args.wm_ckpt, device=device, img_size=224, patch_size=14)
    img_tx = make_img_transform()
    eval_envs, eval_goals = make_eval_envs(jepa, device, img_tx, task_ids=tuple(args.task_ids))

    csv_path = os.path.join(args.save_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'distill_bc', 'hl_ground_loss', 'hl_J', 'overall_success', 'elapsed_s'])

    # LL buffer of (z, w, a_MPC, real_chunk). real_chunk is the REAL data action
    # chunk at z -- the manifold anchor that prevents the distillation collapse.
    zb = wb = ab = cb = None

    def refresh_buffer():
        nonlocal zb, wb, ab, cb
        b = train_ds.sample(args.relabel_size)
        z = jnp.asarray(b['observations']); w = jnp.asarray(b['high_actor_targets'])
        a_list = []
        bs = 2048
        for i in range(0, args.relabel_size, bs):
            a_mpc, _ = agent.fmq_mpc_low(z[i:i+bs], w[i:i+bs], jax.random.PRNGKey(args.seed + i))
            a_list.append(np.asarray(a_mpc))
        zb = np.asarray(b['observations']); wb = np.asarray(b['high_actor_targets'])
        ab = np.concatenate(a_list, axis=0)
        cb = np.asarray(b['action_chunks'])

    print('Phase-2: building initial LL buffer...', flush=True)
    refresh_buffer()
    print(f'  buffer: {ab.shape[0]} (z,w,a_MPC,real_chunk) tuples | ll_mpc_coef={config["ll_mpc_coef"]}', flush=True)

    print('Starting Phase-2 distillation...', flush=True)
    t0 = time.time()
    last = {'distill_bc': 0.0, 'anchor_bc': 0.0, 'hl_ground_loss': 0.0, 'hl_J': 0.0,
            'hl_gain': 0.0, 'hl_pick_data': 1.0, 'hl_use_winner': 0.0}

    # Step-0 eval: shows baseline offline (Phase-1) policy before any Phase-2 updates.
    print(f'=== eval @ step 0 ({args.eval_episodes} eps/task) ===', flush=True)
    overall, per_task = evaluate_lsg(agent, eval_envs, eval_goals, jepa, device, img_tx,
                                     config, args.eval_episodes, args.eval_max_steps, args.seed)
    print(f'=== overall @ 0: {overall*100:.1f}% '
          f'(per-task {[f"{x*100:.0f}" for x in per_task]}) ===', flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([0, '', '', '', overall, 0.0])

    for step in range(1, args.total_steps + 1):
        if step % args.relabel_every == 0:
            refresh_buffer()
        # Anchored LL distill on a buffer minibatch: real-data BC anchor + MPC nudge.
        # Ghost-update fix: save HL params before LL update and restore afterwards.
        # The shared Adam optimizer applies a ghost update to HL via accumulated
        # momentum even when the HL gradient is zero (confirmed: ~0.12 norm/step vs
        # ~0.13 for explicit HL update). Restoring isolates LL and HL training.
        hl_key = 'modules_high_actor_flow'
        if not args.ll_frozen:
            hl_params_snap = agent.network.params[hl_key]
            idx = np.random.randint(0, zb.shape[0], args.batch_size)
            agent, di = agent.update_low_anchored(jnp.asarray(zb[idx]), jnp.asarray(wb[idx]),
                                                  jnp.asarray(cb[idx]), jnp.asarray(ab[idx]))
            patched = dict(agent.network.params); patched[hl_key] = hl_params_snap
            agent = agent.replace(network=agent.network.replace(params=patched))
            last['distill_bc'] = float(di['distill_bc']); last['anchor_bc'] = float(di['anchor_bc'])
        # HL grounding every hl_every steps (skipped entirely when hl_mode='frozen').
        if args.hl_mode != 'frozen' and step % args.hl_every == 0:
            if args.hl_mode == 'awr':
                # AWR on REAL data subgoals weighted by the WM-rollout advantage.
                # No self-samples are fed back into the flow, so there is no
                # self-consuming drift; weighting (not unweighted BC) keeps the AWR
                # sharpening. CPU-verified: V_roll stays flat, unlike 'real'.
                zc, gc, wd, rc = sample_hl_base(
                    train_ds, args.hl_batch, config['subgoal_steps'], np_rng)
                # Ghost-Adam fix: save ALL non-HL modules before the AWR update and
                # restore them after. The shared Adam optimizer applies a non-zero
                # ghost step (decaying Phase-1 momentum) to every module -- including
                # the value network -- even when their gradient is zero. Restoring
                # value params prevents the AWR advantage signal (which depends on V)
                # from being corrupted by value drift over the 125k+ AWR calls.
                _frozen_keys = ['modules_low_actor_bc_flow',
                                'modules_value', 'modules_target_value']
                _frozen_snap = {k: agent.network.params[k] for k in _frozen_keys}
                agent, hi = agent.update_high_ground_awr(
                    jnp.asarray(zc), jnp.asarray(gc), jnp.asarray(wd),
                    jax.random.PRNGKey(args.seed + step))
                patched = dict(agent.network.params)
                patched.update(_frozen_snap)
                agent = agent.replace(network=agent.network.replace(params=patched))
                last['hl_ground_loss'] = float(hi['hl_ground_loss']); last['hl_J'] = float(hi['hl_J_mean'])
                last['hl_gain'] = float(hi['hl_adv_mean']); last['hl_pick_data'] = float(hi['hl_w_mean'])
                last['hl_use_winner'] = float(hi['hl_w_max'])
            elif args.hl_mode == 'phase1awr':
                # Ablation: same AWR BC target (real data subgoals) but advantage is
                # V(w_data,g)−V(z,g) — the Phase-1 signal — instead of WM rollout.
                # If this degrades at the same rate as 'awr', the BC target distribution
                # is the root cause; the WM rollout signal is not making things worse.
                zc, gc, wd, rc = sample_hl_base(
                    train_ds, args.hl_batch, config['subgoal_steps'], np_rng)
                _frozen_keys = ['modules_low_actor_bc_flow',
                                'modules_value', 'modules_target_value']
                _frozen_snap = {k: agent.network.params[k] for k in _frozen_keys}
                agent, hi = agent.update_high_ground_phase1awr(
                    jnp.asarray(zc), jnp.asarray(gc), jnp.asarray(wd),
                    jax.random.PRNGKey(args.seed + step))
                patched = dict(agent.network.params)
                patched.update(_frozen_snap)
                agent = agent.replace(network=agent.network.replace(params=patched))
                last['hl_ground_loss'] = float(hi['hl_ground_loss']); last['hl_J'] = float(hi['hl_J_mean'])
                last['hl_gain'] = float(hi['hl_adv_mean']); last['hl_pick_data'] = float(hi['hl_w_mean'])
                last['hl_use_winner'] = float(hi['hl_w_max'])
            elif args.hl_mode == 'real':
                zc, gc, wd, rc = sample_hl_base(
                    train_ds, args.hl_batch, config['subgoal_steps'], np_rng)
                # Build K candidates: cand[0]=data subgoal (anchor), cand[1..K-1]=HL samples.
                # HL samples are always in-distribution (what the HL currently predicts
                # for this z, g pair). Reachability-gap scoring picks the best direction.
                hl_cands = [wd]
                hl_base_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), step)
                for ki in range(1, args.hl_num_samples):
                    sg = agent.sample_high_subgoal(
                        jnp.asarray(zc), jnp.asarray(gc),
                        seed=jax.random.fold_in(hl_base_key, ki))
                    hl_cands.append(np.asarray(sg))
                cand = jnp.stack([jnp.asarray(c) for c in hl_cands], axis=0)  # (K, B, D)
                # Ghost-update fix: restore LL params after HL update.
                ll_key = 'modules_low_actor_bc_flow'
                ll_params_snap = agent.network.params[ll_key]
                agent, hi = agent.update_high_ground_real(
                    jnp.asarray(zc), jnp.asarray(gc), cand, jnp.asarray(wd),
                    jax.random.PRNGKey(args.seed + step))
                patched = dict(agent.network.params); patched[ll_key] = ll_params_snap
                agent = agent.replace(network=agent.network.replace(params=patched))
                last['hl_ground_loss'] = float(hi['hl_ground_loss']); last['hl_J'] = float(hi['hl_J_best'])
                last['hl_gain'] = float(hi['hl_gain']); last['hl_pick_data'] = float(hi['hl_pick_data_frac'])
                last['hl_use_winner'] = float(hi['hl_use_winner_frac'])
                # Distribution-gap fix: train LL toward the winner subgoal so it learns
                # to reach exactly what the grounded HL will output at eval time.
                if args.winner_ll_mode != 'none':
                    win_lat = hi['win_lat']                   # (B, D) — per-element winner
                    a_mpc_win, _ = agent.fmq_mpc_low(
                        jnp.asarray(zc), win_lat, jax.random.PRNGKey(args.seed + step + 1))
                    hl_params_snap2 = agent.network.params[hl_key]
                    if args.winner_ll_mode == 'mpc_only':
                        # Train LL to reach winner via MPC chunk — no data-anchor
                        # (avoids teaching LL to ignore subgoal conditioning)
                        agent, _ = agent.update_low_anchored(
                            jnp.asarray(zc), win_lat, a_mpc_win, a_mpc_win)
                    else:  # 'anchor': real_chunk anchor + mpc nudge
                        agent, _ = agent.update_low_anchored(
                            jnp.asarray(zc), win_lat, jnp.asarray(rc), a_mpc_win)
                    patched = dict(agent.network.params); patched[hl_key] = hl_params_snap2
                    agent = agent.replace(network=agent.network.replace(params=patched))
            elif args.hl_mode == 'hrf':
                # Hindsight Reality Forcing: BC toward the ACHIEVED latent z_ach (not the proposal).
                # Blocks value gaming (HL can't game V by proposing near-g hallucinations because
                # the BC target is constrained to the achievable manifold by the frozen LL+WM).
                # Avoids de-sharpening (achievable manifold is narrower than data distribution).
                zc, gc, wd, rc = sample_hl_base(
                    train_ds, args.hl_batch, config['subgoal_steps'], np_rng)
                _frozen_keys = ['modules_low_actor_bc_flow',
                                'modules_value', 'modules_target_value']
                _frozen_snap = {k: agent.network.params[k] for k in _frozen_keys}
                agent, hi = agent.update_high_ground_hrf(
                    jnp.asarray(zc), jnp.asarray(gc),
                    jax.random.PRNGKey(args.seed + step))
                patched = dict(agent.network.params)
                patched.update(_frozen_snap)
                agent = agent.replace(network=agent.network.replace(params=patched))
                last['hl_ground_loss'] = float(hi['hl_ground_loss'])
                last['hl_J'] = float(hi['hl_J_mean'])
                last['hl_gain'] = float(hi['hl_adv_mean'])
                last['hl_pick_data'] = float(hi['hl_w_mean'])
                last['hl_use_winner'] = float(hi['hl_w_max'])
            elif args.hl_mode == 'dpo':
                # Flow-DPO: contrastive HL update with Phase-1 reference policy.
                # Samples K subgoals from current HL, scores by J_safe = V(z_ach,g) - lambda*||z_ach-sg||,
                # picks percentile winner/loser, applies DPO loss bounded by the reference KL.
                zc, gc, wd, rc = sample_hl_base(
                    train_ds, args.hl_batch, config['subgoal_steps'], np_rng)
                _frozen_keys = ['modules_low_actor_bc_flow',
                                'modules_value', 'modules_target_value']
                _frozen_snap = {k: agent.network.params[k] for k in _frozen_keys}
                agent, hi = agent.update_high_ground_dpo(
                    jnp.asarray(zc), jnp.asarray(gc),
                    ref_params,
                    jax.random.PRNGKey(args.seed + step))
                patched = dict(agent.network.params)
                patched.update(_frozen_snap)
                agent = agent.replace(network=agent.network.replace(params=patched))
                last['hl_ground_loss'] = float(hi['hl_ground_loss'])
                last['hl_J'] = float(hi['hl_dpo_logits'])
                last['hl_gain'] = float(hi['hl_improve_win'])
                last['hl_pick_data'] = float(hi['hl_improve_lose'])
                last['hl_use_winner'] = float(hi['hl_J_win'])
            else:  # 'self' -- legacy self-generated AWR loop (known to collapse)
                hb = train_ds.sample(args.hl_batch)
                agent, hi = agent.update_high_ground(
                    jnp.asarray(hb['observations']), jnp.asarray(hb['high_actor_goals']),
                    jnp.asarray(hb['high_actor_targets']), jax.random.PRNGKey(args.seed + step))
                last['hl_ground_loss'] = float(hi['hl_ground_loss']); last['hl_J'] = float(hi['hl_J_mean'])

        if step % args.log_interval == 0:
            if args.hl_mode == 'real':
                hl_extra = (f' gain {last["hl_gain"]:+.3f} pick_data {last["hl_pick_data"]:.2f} '
                            f'use_winner {last["hl_use_winner"]:.2f}')
            elif args.hl_mode == 'awr':
                hl_extra = (f' adv {last["hl_gain"]:+.3f} w_mean {last["hl_pick_data"]:.2f} '
                            f'w_max {last["hl_use_winner"]:.1f}')
            else:
                hl_extra = ''
            print(f'step {step:>7,} | anchor_bc {last["anchor_bc"]:.4f} mpc_bc {last["distill_bc"]:.4f} | '
                  f'hl_loss {last["hl_ground_loss"]:.3f} hl_J {last["hl_J"]:.3f}{hl_extra} | '
                  f'{step/max(time.time()-t0,1e-6):.0f} sps', flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, last['distill_bc'], last['hl_ground_loss'],
                                        last['hl_J'], '', time.time() - t0])

        if step % args.eval_interval == 0 or step == args.total_steps:
            print(f'=== eval @ step {step:,} ({args.eval_episodes} eps/task) ===', flush=True)
            overall, per_task = evaluate_lsg(agent, eval_envs, eval_goals, jepa, device, img_tx,
                                             config, args.eval_episodes, args.eval_max_steps, args.seed)
            print(f'=== overall @ {step:,}: {overall*100:.1f}% '
                  f'(per-task {[f"{x*100:.0f}" for x in per_task]}) ===', flush=True)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, '', '', '', overall, time.time() - t0])

    save_path = os.path.join(args.save_dir, 'agent_final.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'params': jax.device_get(agent.network.params), 'config': dict(config)}, f)
    print(f'Done. Metrics: {csv_path} | Agent: {save_path}', flush=True)


if __name__ == '__main__':
    main()
