"""Online goal-conditioned MPC fine-tuning from a GC offline checkpoint (E3b Phase 2).

Mirrors train_online_mpc_only.py Variant A (episodic WM rollouts) but
operates on goal-conditioned 384-D observations. Each WM rollout is bound to
a single active goal sampled round-robin from the 5 task goals.

KEY CHANGES vs E3a:
  1. **HIQL-style indicator reward** (no latent distance): GCHERSampler uses
     index-based reward — r=1 only at the goal step. Removes dependence on
     latent-space geometry which the WM was never trained for.
  2. **Actor-only update**: agent.update_actor_only() trains only the BC flow.
     The critic stays frozen at the offline checkpoint values throughout
     online. This breaks the actor-critic feedback loop that destabilised
     E2a/E2b/E3a and matches D7d's "distill MPC into BC + use frozen Q for
     selection" recipe (D7d: 96.4% MPC eval).

The FMQ-MPC closure is built once from the offline checkpoint and never
refreshed; with actor-only updates the agent's critic is also fixed so the
MPC's Q matches the agent's Q exactly throughout training.

Usage:
    python train_online_mpc_gc.py \\
        --policy_ckpt $STABLEWM_HOME/cube/e3_offline_gc/params_500000.pkl \\
        --wm_ckpt $STABLEWM_HOME/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache $STABLEWM_HOME/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --online_steps 500000 \\
        --save_dir $STABLEWM_HOME/cube/e3_online_mpc_gc_s42
"""
import argparse
import glob
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_env import make_wm_env_and_dataset_multitask
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import restore_agent_with_file, save_agent
from utils.datasets import ReplayBuffer
from utils.gc_her_sampler import GCHERSampler, build_episode_metadata
from eval_mpc import _load_wm, _make_mpc_fn_gc
from train_interleaved_mpc import CsvLogger, MPC_CONFIGS


def main():
    p = argparse.ArgumentParser(
        description="Online goal-conditioned MPC fine-tuning from a GC offline checkpoint")
    # Checkpoint
    p.add_argument("--policy_ckpt", required=True,
                   help="Pre-trained offline GC checkpoint (.pkl)")
    # Env / dataset
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--task_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--env_family", default="cube-single")
    p.add_argument("--wm_device", default="cuda")
    # Training
    p.add_argument("--online_steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=25000)
    p.add_argument("--save_interval", type=int, default=0,
                   help="Save a checkpoint every N steps. 0 = save only the "
                        "final step and whenever a new best overall SR is hit "
                        "(keeps disk usage to ~2 checkpoints/run).")
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--buffer_size", type=int, default=2_000_000)
    # MPC
    p.add_argument("--mpc_config", choices=["cheap", "full"], default="full")
    p.add_argument("--fmq_eta", type=float, default=0.02)
    # WM rollout
    p.add_argument("--done_threshold", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=40)
    # Online sampling strategy (hypotheses E3c/E3d/E3e)
    p.add_argument("--online_sample_mode", choices=["her", "online_only", "mix"],
                   default="her",
                   help="her (E3b/E3e): HER relabel over whole buffer. "
                        "online_only (E3c): train only on online MPC transitions "
                        "conditioned on the pursued goal g_active, no HER. "
                        "mix (E3d): blend offline-HER and online-(g_active) batches.")
    p.add_argument("--online_mix_ratio", type=float, default=0.5,
                   help="[mix] fraction of each batch drawn from online "
                        "(g_active) transitions; the rest is offline HER.")
    p.add_argument("--online_warmup_steps", type=int, default=2000,
                   help="Use offline HER until this many online transitions "
                        "exist (avoids degenerate sampling from a tiny pool).")
    p.add_argument("--online_goal_source", choices=["eval_goals", "random_states"],
                   default="eval_goals",
                   help="What goal each online WM rollout is steered toward. "
                        "eval_goals (E3b/default): round-robin the 5 OGBench task "
                        "goals -- collection peeks at eval goals. random_states "
                        "(E3g, FAIR): sample g_active from random achieved dataset "
                        "states -- eval-agnostic data collection, the clean "
                        "generalization test.")
    p.add_argument("--unfreeze_critic", action="store_true",
                   help="[E3e] update the critic too (standard actor-critic). "
                        "Default: critic frozen at the offline checkpoint.")
    # HER mix (online)
    p.add_argument("--p_curgoal",    type=float, default=0.2)
    p.add_argument("--p_trajgoal",   type=float, default=0.5)
    p.add_argument("--p_randomgoal", type=float, default=0.3)
    p.add_argument("--p_taskgoal",   type=float, default=0.0,
                   help="P(sample uniform task goal). Default 0 under indicator reward "
                        "(no ground-truth reward available for online WM transitions).")
    p.add_argument("--p_geom",       type=float, default=0.1)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Dataset + 5 task goals + per-task eval envs
    # ------------------------------------------------------------------
    print("=== Loading multitask dataset and encoding 5 task goals...", flush=True)
    (train_dataset_dict, jepa_model, real_envs, z_goals_all) = \
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
    terms_all    = np.asarray(train_dataset_dict["terminals"])
    masks_all    = np.asarray(train_dataset_dict["masks"])
    if obs_all.ndim == 3:
        obs_all      = obs_all[:, -1, :]
        next_obs_all = next_obs_all[:, -1, :]
    N_data = obs_all.shape[0]
    D_obs  = obs_all.shape[-1]    # 192
    D_act  = actions_all.shape[-1] # 25
    print(f"  Dataset: {N_data} transitions, obs_dim={D_obs}, act_dim={D_act}", flush=True)

    # ------------------------------------------------------------------
    # 2. Episode metadata (offline portion) — grown dynamically online
    # ------------------------------------------------------------------
    ep_starts_off, ep_lens_off, ep_ids_off, t_within_off = build_episode_metadata(terms_all)
    n_offline_eps = len(ep_starts_off)
    print(f"  Offline episodes: {n_offline_eps}, mean_len={ep_lens_off.mean():.1f}",
          flush=True)

    # ------------------------------------------------------------------
    # 3. Replay buffer seeded with offline data (including ep_id, t_within)
    # ------------------------------------------------------------------
    print("=== Building replay buffer seeded from offline data...", flush=True)
    init_dataset = dict(
        observations=obs_all.astype(np.float32),
        next_observations=next_obs_all.astype(np.float32),
        actions=actions_all.astype(np.float32),
        rewards=np.asarray(train_dataset_dict["rewards"], dtype=np.float32),  # unused; placeholder
        terminals=terms_all.astype(np.float32),
        masks=masks_all.astype(np.float32),
        ep_id=ep_ids_off.astype(np.int32),
        t_within=t_within_off.astype(np.int32),
        # g_active per transition; offline rows are zeros (never read — offline
        # is always sampled via HER). Online rows store the pursued task goal.
        goal=np.zeros_like(obs_all, dtype=np.float32),
    )
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        init_dataset,
        size=max(args.buffer_size, N_data + 1),
    )
    print(f"  Buffer size: {replay_buffer.size} (initial), max={replay_buffer.max_size}",
          flush=True)

    # Externally-maintained episode metadata. We pre-allocate enough room for
    # all online episodes (worst case: every step is its own episode).
    max_eps = n_offline_eps + args.online_steps + 1
    ep_starts = np.zeros(max_eps, dtype=np.int64)
    ep_lens   = np.zeros(max_eps, dtype=np.int64)
    ep_starts[:n_offline_eps] = ep_starts_off
    ep_lens[:n_offline_eps]   = ep_lens_off
    n_eps_total = n_offline_eps  # current count of episodes registered

    # ------------------------------------------------------------------
    # 4. JAX WM (for per-step rollout in Variant A)
    # ------------------------------------------------------------------
    print("=== Loading JAX WM...", flush=True)
    wm_model, wm_params = _load_wm(args.wm_ckpt)
    z_goals_jax = jnp.asarray(z_goals_all.astype(np.float32))  # (K, 192)
    print("  WM loaded.", flush=True)

    @jax.jit
    def _wm_step_single(z, a):
        return wm_model.apply(wm_params, z[None, None, :], a[None, None, :])[0, -1, :]

    # ------------------------------------------------------------------
    # 5. Create GC ACFQL agent and restore offline checkpoint
    # ------------------------------------------------------------------
    print("=== Creating GC ACFQL agent and loading offline checkpoint...", flush=True)
    ex_obs = np.zeros((1, 2 * D_obs), dtype=np.float32)
    ex_act = np.zeros((1, D_act),     dtype=np.float32)

    config = get_acfql_config()
    config["encoder"]           = "jepa_head"
    config["actor_type"]        = "best-of-n"
    config["actor_num_samples"] = 4
    config["horizon_length"]    = 1
    config["action_chunking"]   = False
    config["lr"]                = args.lr
    config["discount"]          = args.discount

    agent = ACFQLAgent.create(
        seed=args.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
    )
    agent = restore_agent_with_file(agent, args.policy_ckpt)
    print(f"  Loaded checkpoint: {args.policy_ckpt}", flush=True)

    # ------------------------------------------------------------------
    # 6. Build GC MPC closure (built once, never refreshed)
    # ------------------------------------------------------------------
    mpc_cfg = MPC_CONFIGS[args.mpc_config]
    print("=== JIT-compiling GC FMQ MPC closure...", flush=True)
    t_jit = time.time()
    mpc_fn = _make_mpc_fn_gc(
        agent=agent,
        wm_model=wm_model,
        wm_params=wm_params,
        latent_dim=D_obs,
        N=mpc_cfg["N"],
        H=mpc_cfg["H"],
        gamma=mpc_cfg["gamma"],
        dense_scale=mpc_cfg["dense_scale"],
        K_grad=0,
        lr=mpc_cfg["lr"],
        q_only=False,
        q_every_step=True,
        fmq=True,
        fmq_eta=args.fmq_eta,
    )
    _ = mpc_fn(observations=jnp.zeros((2 * D_obs,)),
               rng=jax.random.PRNGKey(0)).block_until_ready()
    print(f"  FMQ JIT compiled in {time.time()-t_jit:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 7. HER sampler + logging
    # ------------------------------------------------------------------
    sampler = GCHERSampler(
        task_goals=z_goals_all,
        done_threshold=args.done_threshold,
        p_curgoal=args.p_curgoal,
        p_trajgoal=args.p_trajgoal,
        p_randomgoal=args.p_randomgoal,
        p_taskgoal=args.p_taskgoal,
        p_geom=args.p_geom,
    )
    np_rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed + 100)

    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))

    task_id_to_idx = {tid: i for i, tid in enumerate(args.task_ids)}
    def obs_augment(z, task_id):
        return np.concatenate([z, z_goals_all[task_id_to_idx[task_id]]]).astype(np.float32)

    crit_mode = "UNFROZEN (actor-critic)" if args.unfreeze_critic else "FROZEN (actor-only)"
    print(f"\n=== Online GC MPC training: {args.online_steps} steps ===", flush=True)
    print(f"    mpc_config={args.mpc_config}  fmq_eta={args.fmq_eta}", flush=True)
    print(f"    sample_mode={args.online_sample_mode}"
          + (f" (mix_ratio={args.online_mix_ratio})" if args.online_sample_mode == "mix" else "")
          + f"  critic={crit_mode}", flush=True)
    print(f"    online_goal_source={args.online_goal_source}", flush=True)
    print(f"    HER mix: cur={args.p_curgoal}  traj={args.p_trajgoal}  "
          f"rand={args.p_randomgoal}  task={args.p_taskgoal}", flush=True)
    print(f"    lr={args.lr}  batch_size={args.batch_size}", flush=True)
    t_start = time.time()
    best_sr = -1.0   # track best overall SR for checkpoint policy

    # ------------------------------------------------------------------
    # 8. Online loop — episodic WM rollouts with round-robin task goals
    # ------------------------------------------------------------------
    K = len(args.task_ids)
    online_ep_counter = 0
    new_episode = True
    z_curr = obs_all[np_rng.integers(0, N_data)].astype(np.float32)
    ep_step = 0

    def _pick_goal(ep_idx):
        # eval_goals: round-robin the 5 task goals (peeks at eval; E3b).
        # random_states: a random achieved state from the offline data (FAIR; E3g).
        if args.online_goal_source == "random_states":
            return obs_all[np_rng.integers(0, N_data)].astype(np.float32)
        return z_goals_all[ep_idx % K]

    g_active = _pick_goal(online_ep_counter)  # initial goal

    for step in range(1, args.online_steps + 1):
        # ---- Episode start bookkeeping ----
        if new_episode:
            g_active = _pick_goal(online_ep_counter)
            # Register a new episode in metadata
            ep_starts[n_eps_total] = replay_buffer.pointer
            ep_lens[n_eps_total]   = 0
            current_ep_id = n_eps_total
            n_eps_total += 1
            new_episode = False

        # ---- One MPC step in the WM ----
        jax_rng, key = jax.random.split(jax_rng)
        obs_aug = np.concatenate([z_curr, g_active]).astype(np.float32)
        a_mpc = np.asarray(mpc_fn(observations=jnp.asarray(obs_aug), rng=key))
        z_next = np.asarray(_wm_step_single(jnp.asarray(z_curr), jnp.asarray(a_mpc)))
        dist = float(np.linalg.norm(z_next - g_active))
        reward = 1.0 if dist < args.done_threshold else 0.0
        done   = (reward > 0.0) or (ep_step + 1 >= args.max_episode_steps)

        transition = dict(
            observations=z_curr.astype(np.float32),
            next_observations=z_next.astype(np.float32),
            actions=a_mpc.astype(np.float32),
            rewards=np.float32(reward),
            terminals=np.float32(done),
            masks=np.float32(1.0 - (reward > 0.0)),
            ep_id=np.int32(current_ep_id),
            t_within=np.int32(ep_step),
            goal=g_active.astype(np.float32),
        )
        replay_buffer.add_transition(transition)
        # Update episode length tracking
        ep_lens[current_ep_id] += 1

        # ---- Episode end ----
        if done:
            z_curr = obs_all[np_rng.integers(0, N_data)].astype(np.float32)
            ep_step = 0
            online_ep_counter += 1
            new_episode = True
        else:
            z_curr = z_next
            ep_step += 1

        # ---- Build training batch per online_sample_mode ----
        n_online = replay_buffer.size - N_data   # online transitions added so far
        if args.online_sample_mode == "her" or n_online < args.online_warmup_steps:
            # E3b/E3e, or warmup: HER relabel over the whole buffer.
            batch = sampler.sample(
                buffer_dict=replay_buffer._dict,
                current_size=replay_buffer.size,
                ep_starts=ep_starts, ep_lens=ep_lens,
                batch_size=args.batch_size, np_rng=np_rng,
            )
        elif args.online_sample_mode == "online_only":
            # E3c: only online MPC transitions, conditioned on stored g_active.
            batch = sampler.sample_stored_goal(
                buffer_dict=replay_buffer._dict,
                batch_size=args.batch_size, np_rng=np_rng,
                idx_low=N_data, idx_high=replay_buffer.size,
            )
        else:  # "mix" — E3d
            n_on  = int(round(args.batch_size * args.online_mix_ratio))
            n_off = args.batch_size - n_on
            b_off = sampler.sample(
                buffer_dict=replay_buffer._dict,
                current_size=replay_buffer.size,
                ep_starts=ep_starts, ep_lens=ep_lens,
                batch_size=n_off, np_rng=np_rng,
                idx_low=0, idx_high=N_data,
            )
            b_on = sampler.sample_stored_goal(
                buffer_dict=replay_buffer._dict,
                batch_size=n_on, np_rng=np_rng,
                idx_low=N_data, idx_high=replay_buffer.size,
            )
            batch = {k: np.concatenate([b_off[k], b_on[k]], axis=0) for k in b_off}

        # ---- Update: actor-only (frozen critic) unless --unfreeze_critic ----
        if args.unfreeze_critic:
            agent, info = agent.update(batch)
        else:
            agent, info = agent.update_actor_only(batch)

        # ---- Logging ----
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}
            log_data["elapsed_s"]     = elapsed
            log_data["buffer_size"]   = float(replay_buffer.size)
            log_data["online_eps"]    = float(online_ep_counter)
            logger_train.log(log_data, step=step)
            print(
                f"  step {step:7d}/{args.online_steps}  "
                f"critic={info.get('critic/critic_loss', float('nan')):.4f}  "
                f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                f"buf={replay_buffer.size}  online_eps={online_ep_counter}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        # ---- Eval + checkpoint ----
        if step % args.eval_interval == 0 or step == args.online_steps:
            print(f"\n[step {step}] Evaluating on {len(args.task_ids)} tasks...",
                  flush=True)
            metrics = evaluate_real_ogbench(
                agent=agent,
                real_env=real_envs,
                jepa_model=jepa_model,
                device=args.wm_device,
                task_ids=tuple(args.task_ids),
                num_episodes_per_task=args.eval_episodes,
                action_dispatch="chunk25",
                pass_task_id_on_reset=False,
                obs_augment=obs_augment,
            )
            sr_overall = float(metrics.get("overall/success_rate", float("nan")))
            print(f"  [eval] overall_success_rate = {sr_overall:.4f} "
                  f"({sr_overall*100:.1f}%)", flush=True)
            logger_eval.log(metrics, step=step)
            # Checkpointing policy: with save_interval=0 (default) keep disk
            # small — save only on a new best SR, plus the final step. Otherwise
            # save every save_interval steps.
            is_final = (step == args.online_steps)
            is_periodic = (args.save_interval > 0 and step % args.save_interval == 0)
            is_new_best = (sr_overall > best_sr + 1e-9)
            if is_new_best:
                best_sr = sr_overall
            if is_periodic or is_final or (args.save_interval == 0 and is_new_best):
                tag = step if (is_periodic or is_final) else f"best_{step}"
                save_agent(agent, args.save_dir, tag)
                # Keep only one "best_*" file: remove older best_* checkpoints
                if args.save_interval == 0 and is_new_best and not is_final:
                    for old in glob.glob(os.path.join(args.save_dir, "params_best_*.pkl")):
                        if old != os.path.join(args.save_dir, f"params_best_{step}.pkl"):
                            os.remove(old)
                print(f"  [ckpt] saved params_{tag}.pkl  (best={best_sr:.3f})", flush=True)

    print(f"\n=== Done. Checkpoints in {args.save_dir}/  best_sr={best_sr:.4f}", flush=True)


if __name__ == "__main__":
    main()
