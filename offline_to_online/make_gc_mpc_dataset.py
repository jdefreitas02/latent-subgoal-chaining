"""Generate a goal-conditioned MPC dataset from a frozen offline GC checkpoint.

The single-task generator is the existing train_online_mpc_only.py with --gen_only
(its Variant A/B loop already produces rollout/relabel transitions). The GC online
trainer (train_online_mpc_gc.py) stores a HER-split buffer (192-D obs + goal field),
so for a clean BC-consumable dataset we generate here directly, emitting 384-D
[z, g] observations.

Two modes (the GC analogues of Variant A / B):
  --mode rollout : episodic WM rollout. obs=[z_curr, g], a_mpc=MPC([z_curr,g]),
                   z'=WM(z_curr, a_mpc). Imagined states + MPC actions.
  --mode relabel : sweep real offline states. obs=[z_real, g], a_mpc=MPC([z_real,g]),
                   z'=z'_real (offline). Real states + MPC actions.

Goals are random achieved states (the leakage-free "fair" source, matching E3g), one
per episode (rollout) or one per state (relabel), so the (z, g) space is broadly covered.

Usage:
    python make_gc_mpc_dataset.py --mode rollout \\
        --policy_ckpt ~/stable_wm_data/cube/e3b_offline_gc/params_500000.pkl \\
        --wm_ckpt  ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --out ~/stable_wm_data/ogbench/bc_datasets/gc_rollout.npz
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_env import make_wm_env_and_dataset_multitask
from utils.flax_utils import restore_agent_with_file
from eval_mpc import _load_wm, _make_mpc_fn_gc
from train_interleaved_mpc import MPC_CONFIGS


def main():
    p = argparse.ArgumentParser(description="Generate a goal-conditioned MPC dataset")
    p.add_argument("--policy_ckpt", required=True, help="Offline GC checkpoint (.pkl)")
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--task_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--env_family", default="cube-single")
    p.add_argument("--wm_device", default="cuda")
    p.add_argument("--mode", choices=["rollout", "relabel", "play"], required=True,
                   help="rollout=imagined states+MPC actions; relabel=real states+MPC actions; "
                        "play=real states+PLAY actions (goal-matched baseline for fair comparison).")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--gen_n", type=int, default=0,
                   help="Number of transitions. 0 = one pass over the offline data.")
    p.add_argument("--mpc_config", choices=["cheap", "full"], default="full")
    p.add_argument("--fmq_eta", type=float, default=0.02)
    p.add_argument("--bon", action="store_true",
                   help="Use plain BoN (no FMQ refinement step). Default is FMQ.")
    p.add_argument("--done_threshold", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Dataset + task goals (envs unused for generation)
    # ------------------------------------------------------------------
    print("=== Loading multitask dataset and task goals...", flush=True)
    (train_dataset_dict, _jepa, _real_envs, z_goals_all) = \
        make_wm_env_and_dataset_multitask(
            wm_ckpt_path=args.wm_ckpt,
            latent_cache_path=args.wm_cache,
            hdf5_dataset_path=args.wm_hdf5,
            task_ids=tuple(args.task_ids),
            done_threshold=args.done_threshold,
            wm_device=args.wm_device,
            img_size=224,
            env_family=args.env_family,
        )
    obs_all      = np.asarray(train_dataset_dict["observations"])
    next_obs_all = np.asarray(train_dataset_dict["next_observations"])
    actions_all  = np.asarray(train_dataset_dict["actions"])
    if obs_all.ndim == 3:
        obs_all      = obs_all[:, -1, :]
        next_obs_all = next_obs_all[:, -1, :]
    if actions_all.ndim == 3:
        actions_all = actions_all[:, 0, :]
    N_data = obs_all.shape[0]
    D_obs  = obs_all.shape[-1]   # 192
    print(f"  Dataset: {N_data} transitions, obs_dim={D_obs}", flush=True)

    # ------------------------------------------------------------------
    # 2. WM + agent + GC MPC closure (frozen offline checkpoint)
    # ------------------------------------------------------------------
    wm_model, wm_params = _load_wm(args.wm_ckpt)

    @jax.jit
    def _wm_step_single(z, a):
        return wm_model.apply(wm_params, z[None, None, :], a[None, None, :])[0, -1, :]

    ex_obs = np.zeros((1, 2 * D_obs), dtype=np.float32)
    ex_act = np.zeros((1, 25),        dtype=np.float32)
    config = get_acfql_config()
    config["encoder"]           = "jepa_head"
    config["actor_type"]        = "best-of-n"
    config["actor_num_samples"] = 4
    config["horizon_length"]    = 1
    config["action_chunking"]   = False
    agent = ACFQLAgent.create(
        seed=args.seed, ex_observations=ex_obs, ex_actions=ex_act,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
    )
    agent = restore_agent_with_file(agent, args.policy_ckpt)
    print(f"  Loaded checkpoint: {args.policy_ckpt}", flush=True)

    mpc_cfg = MPC_CONFIGS[args.mpc_config]
    print("=== JIT-compiling GC FMQ MPC closure...", flush=True)
    t_jit = time.time()
    mpc_fn = _make_mpc_fn_gc(
        agent=agent, wm_model=wm_model, wm_params=wm_params, latent_dim=D_obs,
        N=mpc_cfg["N"], H=mpc_cfg["H"], gamma=mpc_cfg["gamma"],
        dense_scale=mpc_cfg["dense_scale"], K_grad=0, lr=mpc_cfg["lr"],
        q_only=False, q_every_step=True, fmq=(not args.bon), fmq_eta=args.fmq_eta,
    )
    _ = mpc_fn(observations=jnp.zeros((2 * D_obs,)),
               rng=jax.random.PRNGKey(0)).block_until_ready()
    print(f"  FMQ JIT compiled in {time.time()-t_jit:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Generation loop
    # ------------------------------------------------------------------
    np_rng  = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed + 100)
    n_steps = args.gen_n if args.gen_n > 0 else N_data

    def _pick_goal():
        return obs_all[np_rng.integers(0, N_data)].astype(np.float32)

    obs_l, act_l, rew_l, term_l, mask_l, nobs_l = [], [], [], [], [], []

    if args.mode == "rollout":
        # GOAL-LESS episodic rollout buffer: store raw (z, z', a_mpc, terminal) only.
        # The goal g_active still DRIVES the MPC but is NOT stored, so the dataset has
        # the exact same form as the play data and is processed by the SAME HER sampler
        # downstream (terminals delimit episodes for HER future-goal sampling).
        z_curr = obs_all[np_rng.integers(0, N_data)].astype(np.float32)
        g_active = _pick_goal()
        ep_step = 0
        for step in range(1, n_steps + 1):
            jax_rng, key = jax.random.split(jax_rng)
            obs_aug = np.concatenate([z_curr, g_active]).astype(np.float32)
            a_mpc = np.asarray(mpc_fn(observations=jnp.asarray(obs_aug), rng=key))
            z_next = np.asarray(_wm_step_single(jnp.asarray(z_curr), jnp.asarray(a_mpc)))
            reached = float(np.linalg.norm(z_next - g_active)) < args.done_threshold
            done = bool(reached or (ep_step + 1 >= args.max_episode_steps))
            obs_l.append(z_curr.astype(np.float32))        # (192,)  -- no goal
            act_l.append(a_mpc.astype(np.float32))         # (25,)
            rew_l.append(np.float32(0.0))                  # unused (HER recomputes)
            term_l.append(np.float32(done))                # episode boundary
            mask_l.append(np.float32(1.0))                 # unused
            nobs_l.append(z_next.astype(np.float32))       # (192,)
            if done:
                z_curr = obs_all[np_rng.integers(0, N_data)].astype(np.float32)
                g_active = _pick_goal()
                ep_step = 0
            else:
                z_curr = z_next
                ep_step += 1
            if step % 5000 == 0:
                print(f"  [gen:rollout] {step}/{n_steps}", flush=True)
    else:  # relabel (real states + MPC actions) / play (real states + PLAY actions)
        for step in range(1, n_steps + 1):
            idx = (step - 1) % N_data
            g_active = _pick_goal()
            z_real = obs_all[idx].astype(np.float32)
            z_next = next_obs_all[idx].astype(np.float32)
            obs_aug = np.concatenate([z_real, g_active]).astype(np.float32)
            if args.mode == "play":
                a_sel = actions_all[idx].astype(np.float32)
            else:
                jax_rng, key = jax.random.split(jax_rng)
                a_sel = np.asarray(mpc_fn(observations=jnp.asarray(obs_aug), rng=key)).astype(np.float32)
            dist = float(np.linalg.norm(z_next - g_active))
            reward = 1.0 if dist < args.done_threshold else 0.0
            obs_l.append(obs_aug)
            act_l.append(a_sel)
            rew_l.append(np.float32(reward))
            term_l.append(np.float32(reward > 0.0))
            mask_l.append(np.float32(1.0 - (reward > 0.0)))
            nobs_l.append(np.concatenate([z_next, g_active]).astype(np.float32))
            if step % 5000 == 0:
                print(f"  [gen:{args.mode}] {step}/{n_steps}", flush=True)

    out = dict(
        observations=np.stack(obs_l).astype(np.float32),
        actions=np.stack(act_l).astype(np.float32),
        rewards=np.stack(rew_l).astype(np.float32),
        terminals=np.stack(term_l).astype(np.float32),
        masks=np.stack(mask_l).astype(np.float32),
        next_observations=np.stack(nobs_l).astype(np.float32),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"=== [gc-gen:{args.mode}] dumped {out['observations'].shape[0]} transitions "
          f"(obs {out['observations'].shape}, act {out['actions'].shape}) to {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
