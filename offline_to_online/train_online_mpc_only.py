"""Online MPC-driven fine-tuning from a pre-trained offline checkpoint.

Mirrors the structure of main.py's online phase but replaces real-env
interactions with FMQ-MPC-generated transitions. Same single-buffer
flow: one MPC call per step → one transition added to the replay buffer
→ one agent update.

Two variants:
  Variant B (default): (z, a_mpc, z'_offline, r_offline)
      For each step, sample a random offline transition (z, a_old, z', r)
      and replace the action with a_mpc. z'/r come from the offline data.

  Variant A (--wm_consistent_targets): (z, a_mpc, z'_wm, r_wm)
      Maintain an episodic rollout in the WM. z' = WM(z, a_mpc);
      r = 1{||z' - z_goal|| < done_threshold}. Resets on success or
      max_episode_steps.

The FMQ closure is refreshed every --fmq_refresh_interval steps using
current agent params (re-JIT cost ~5s, amortized over the interval).

Uses D7r-matched FMQ params: full config (N=32, H=2), FMQ η=0.02.

Usage:
    python train_online_mpc_only.py \\
        --policy_ckpt ~/stable_wm_data/cube/e2_offline/params_500000.pkl \\
        --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --online_steps 500000 --save_dir ~/stable_wm_data/cube/e2b_online_actions
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
from envs.wm_env import make_wm_env_and_dataset
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import restore_agent_with_file, save_agent
from utils.datasets import Dataset, ReplayBuffer
from eval_mpc import _load_wm, _make_mpc_fn
from train_interleaved_mpc import CsvLogger, MPC_CONFIGS


def main():
    p = argparse.ArgumentParser(description="Online MPC-only fine-tuning from offline checkpoint")
    # Checkpoint
    p.add_argument("--policy_ckpt", required=True,
                   help="Pre-trained offline checkpoint (.pkl) to load")
    # Env / dataset
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--task_id", type=int, default=1)
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
    p.add_argument("--eval_episodes", type=int, default=250)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--buffer_size", type=int, default=2_000_000)
    # MPC
    p.add_argument("--mpc_config", choices=["cheap", "full"], default="full")
    p.add_argument("--fmq_eta", type=float, default=0.02,
                   help="FMQ trust-region radius (D7q/D7r eval best: 0.02)")
    # Variant A: WM-consistent transitions (episodic WM rollouts)
    p.add_argument("--wm_consistent_targets", action="store_true",
                   help="Variant A: z' = WM(z, a_mpc), r = 1{||z' - z_goal|| < threshold}, "
                        "episodic rollouts in the WM. Otherwise (Variant B): pair "
                        "a_mpc with offline (z, z', r).")
    p.add_argument("--done_threshold", type=float, default=2.0,
                   help="[Variant A] L2 latent distance threshold for sparse reward.")
    p.add_argument("--max_episode_steps", type=int, default=40,
                   help="[Variant A] Max WM rollout length before truncation.")
    p.add_argument("--freeze_critic", action="store_true",
                   help="If set, skip critic_loss during online updates. Only the BC "
                        "flow (actor) is trained on MPC transitions; the critic stays "
                        "at the offline checkpoint values throughout. Breaks the "
                        "actor-critic feedback loop that destabilised E2a/E2b.")
    # Dataset-generation mode (no agent training): dump generated transitions.
    p.add_argument("--bon", action="store_true",
                   help="Use plain BoN (no FMQ refinement). Default is FMQ.")
    p.add_argument("--gen_only", action="store_true",
                   help="Generation mode: run the transition loop only (no agent "
                        "update, no eval) and dump the generated transitions to "
                        "--dump_dataset. The mode is set by --wm_consistent_targets "
                        "(ON=rollout / imagined states; OFF=relabel / real states).")
    p.add_argument("--dump_dataset", default=None,
                   help="[--gen_only] Output .npz path for generated transitions.")
    p.add_argument("--gen_n", type=int, default=0,
                   help="[--gen_only] Number of transitions to generate. 0 (default) "
                        "= one pass over the offline dataset (size-matched).")
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset + real eval env
    # ------------------------------------------------------------------
    print("=== Loading WM env and dataset...", flush=True)
    (_, _, train_dataset_dict, _, jepa_model,
     real_env, z_goal_task) = make_wm_env_and_dataset(
        wm_ckpt_path=args.wm_ckpt,
        latent_cache_path=args.wm_cache,
        hdf5_dataset_path=args.wm_hdf5,
        task_id=args.task_id,
        done_threshold=args.done_threshold,
        max_episode_steps=args.max_episode_steps,
        wm_device=args.wm_device,
        img_size=224,
        env_family=args.env_family,
    )
    train_dataset = Dataset.create(**train_dataset_dict)
    obs_all       = np.asarray(train_dataset["observations"])
    next_obs_all  = np.asarray(train_dataset["next_observations"])
    rewards_all   = np.asarray(train_dataset["rewards"])
    terminals_all = np.asarray(train_dataset["terminals"])
    masks_all     = np.asarray(train_dataset["masks"])
    if obs_all.ndim == 3:
        obs_all      = obs_all[:, -1, :]
        next_obs_all = next_obs_all[:, -1, :]
    N_data = obs_all.shape[0]
    print(f"  Dataset: {N_data} transitions, obs shape {obs_all.shape}", flush=True)

    # ------------------------------------------------------------------
    # 2. Load JAX WM
    # ------------------------------------------------------------------
    print("=== Loading JAX WM...", flush=True)
    wm_model, wm_params = _load_wm(args.wm_ckpt)
    z_goal = jnp.asarray(z_goal_task.astype(np.float32))
    print("  WM loaded.", flush=True)

    # ------------------------------------------------------------------
    # 3. Create agent and restore offline checkpoint
    # ------------------------------------------------------------------
    print("=== Creating agent and loading offline checkpoint...", flush=True)
    example_batch = train_dataset.sample(1)
    ex_obs = example_batch["observations"][:1]
    ex_act = example_batch["actions"][:1]
    if ex_act.ndim == 3:
        ex_act = ex_act[:, 0, :]

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
    # 4. Replay buffer seeded with offline data (mirrors main.py)
    # ------------------------------------------------------------------
    print("=== Building replay buffer from offline data...", flush=True)
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset),
        size=max(args.buffer_size, train_dataset.size + 1),
    )
    print(f"  Buffer size: {replay_buffer.size} (initial)", flush=True)
    offline_size = replay_buffer.size  # online transitions are appended past this index

    # ------------------------------------------------------------------
    # 5. Set up WM forward step for Variant A
    # ------------------------------------------------------------------
    @jax.jit
    def _wm_step_single(z, a):
        # z: (192,), a: (25,) → z': (192,)
        return wm_model.apply(wm_params, z[None, None, :], a[None, None, :])[0, -1, :]

    # ------------------------------------------------------------------
    # 6. Set up FMQ MPC function — built once from the offline checkpoint,
    #    same as eval_mpc.py. Params are fixed throughout online training;
    #    this is fine because the Q changes slowly and the distillation
    #    approach (D7d) showed 50k-stale MPC params still achieves 96.4%.
    # ------------------------------------------------------------------
    mpc_cfg = MPC_CONFIGS[args.mpc_config]

    print("=== JIT-compiling FMQ closure...", flush=True)
    t_jit = time.time()
    mpc_fn = _make_mpc_fn(
        agent=agent,
        wm_model=wm_model,
        wm_params=wm_params,
        z_goal=z_goal,
        N=mpc_cfg["N"],
        H=mpc_cfg["H"],
        gamma=mpc_cfg["gamma"],
        dense_scale=mpc_cfg["dense_scale"],
        K_grad=0,
        lr=mpc_cfg["lr"],
        q_only=False,
        q_every_step=True,
        fmq=(not args.bon),
        fmq_eta=args.fmq_eta,
    )
    _ = mpc_fn(observations=jnp.zeros((192,)),
               rng=jax.random.PRNGKey(0)).block_until_ready()
    print(f"  FMQ JIT compiled in {time.time()-t_jit:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 7. Logging
    # ------------------------------------------------------------------
    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))

    variant = "A (WM-consistent, episodic)" if args.wm_consistent_targets else "B (actions-only, offline z'/r)"
    print(f"\n=== Online MPC training: {args.online_steps} steps ===", flush=True)
    print(f"    variant={variant}", flush=True)
    print(f"    mpc_config={args.mpc_config}  fmq_eta={args.fmq_eta}", flush=True)
    print(f"    lr={args.lr}  batch_size={args.batch_size}", flush=True)
    t_start = time.time()

    # State for Variant A episodic rollouts
    rng_np = np.random.default_rng(args.seed)
    z_curr = obs_all[rng_np.integers(0, N_data)].astype(np.float32)
    ep_step = 0

    jax_rng = jax.random.PRNGKey(args.seed + 100)

    n_steps = (args.gen_n if args.gen_n > 0 else N_data) if args.gen_only else args.online_steps
    for step in range(1, n_steps + 1):

        # ---- Build one transition via FMQ-MPC ----
        jax_rng, key = jax.random.split(jax_rng)

        if args.wm_consistent_targets:
            # Variant A: episodic WM rollout
            z_j = jnp.asarray(z_curr)
            a_mpc = np.asarray(mpc_fn(observations=z_j, rng=key))   # (25,)
            z_next = np.asarray(_wm_step_single(z_j, jnp.asarray(a_mpc)))   # (192,)
            dist = float(np.linalg.norm(z_next - np.asarray(z_goal)))
            reward = 1.0 if dist < args.done_threshold else 0.0
            done   = (reward > 0.0) or (ep_step + 1 >= args.max_episode_steps)
            transition = dict(
                observations=z_curr,
                actions=a_mpc,
                rewards=np.float32(reward),
                terminals=np.float32(done),
                masks=np.float32(1.0 - (reward > 0.0)),
                next_observations=z_next,
            )
            if done:
                z_curr = obs_all[rng_np.integers(0, N_data)].astype(np.float32)
                ep_step = 0
            else:
                z_curr = z_next
                ep_step += 1
        else:
            # Variant B: pair a_mpc with offline (z, z', r). Same as sampling a
            # random offline transition and replacing only the action.
            idx = (step - 1) % N_data if args.gen_only else int(rng_np.integers(0, N_data))
            z_j = jnp.asarray(obs_all[idx])
            a_mpc = np.asarray(mpc_fn(observations=z_j, rng=key))   # (25,)
            transition = dict(
                observations=obs_all[idx],
                actions=a_mpc,
                rewards=rewards_all[idx],
                terminals=terminals_all[idx],
                masks=masks_all[idx],
                next_observations=next_obs_all[idx],
            )

        replay_buffer.add_transition(transition)

        # ---- Generation mode: no training/eval, just accumulate ----
        if args.gen_only:
            if step % 5000 == 0:
                print(f"  [gen] {step}/{n_steps} transitions generated", flush=True)
            continue

        # ---- Train ----
        batch = replay_buffer.sample_sequence(
            args.batch_size, sequence_length=1, discount=args.discount
        )
        if args.freeze_critic:
            agent, info = agent.update_actor_only(batch)
        else:
            agent, info = agent.update(batch)

        # ---- Logging ----
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}
            log_data["elapsed_s"] = elapsed
            log_data["buffer_size"] = float(replay_buffer.size)
            logger_train.log(log_data, step=step)
            print(
                f"  step {step:7d}/{args.online_steps}  "
                f"critic={info.get('critic/critic_loss', float('nan')):.4f}  "
                f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                f"buf={replay_buffer.size}  elapsed={elapsed:.0f}s",
                flush=True
            )

        # ---- Eval + checkpoint ----
        if step % args.eval_interval == 0 or step == args.online_steps:
            print(f"\n[step {step}] Evaluating...", flush=True)
            metrics = evaluate_real_ogbench(
                agent=agent,
                real_env=real_env,
                jepa_model=jepa_model,
                device=args.wm_device,
                task_ids=(args.task_id,),
                num_episodes_per_task=args.eval_episodes,
                action_dispatch="chunk25",
                pass_task_id_on_reset=False,
            )
            sr = metrics.get(f"task_{args.task_id}/success_rate",
                             metrics.get("overall/success_rate", float("nan")))
            print(f"  [eval] success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
            logger_eval.log(metrics, step=step)
            save_agent(agent, args.save_dir, step)
            print(f"  [ckpt] saved params_{step}.pkl", flush=True)

    if args.gen_only:
        keys = ["observations", "actions", "rewards", "terminals", "masks", "next_observations"]
        out = {k: np.asarray(replay_buffer[k][offline_size:replay_buffer.size]) for k in keys}
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_dataset)), exist_ok=True)
        np.savez_compressed(args.dump_dataset, **out)
        mode = "rollout" if args.wm_consistent_targets else "relabel"
        print(f"=== [gen:{mode}] dumped {out['observations'].shape[0]} transitions "
              f"(obs {out['observations'].shape}, act {out['actions'].shape}) "
              f"to {args.dump_dataset}", flush=True)
        return

    print(f"\n=== Done. Checkpoints in {args.save_dir}/", flush=True)


if __name__ == "__main__":
    main()
