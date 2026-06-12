"""Pure flow-matching behavioural cloning, from scratch, on a chosen dataset.

Trains ONLY the BC-flow actor of ACFQL (via update_actor_only — the critic is
created but never updated and never used), then evaluates the policy standalone
on the real OGBench env (single flow sample per decision, 5 actions dispatched
per 25-D chunk, no planner). This is the downstream learner in the
"MPC-in-WM as a dataset generator" experiment.

Conditions (the dataset the BC policy imitates):
  original : the real play dataset (single: build_for_E; GC: HER over offline data)
  relabel  : real states + MPC actions  (npz from the generators)
  rollout  : imagined states + MPC actions  (npz from the generators)

The actor architecture matches the rest of the thesis (encoder=jepa_head,
action_chunking=False so the 25-D chunk is the action, flow_steps=10), created
fresh and with actor_num_samples=1 so eval is the pure BC sample (critic-free).

Usage (single, rollout):
    python train_bc_flow.py --condition rollout \\
        --dataset_path ~/stable_wm_data/ogbench/bc_datasets/single_rollout.npz \\
        --wm_ckpt  ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --save_dir ~/stable_wm_data/cube/bc_single_rollout_s42 --seed 42
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
from envs.wm_env import make_wm_env_and_dataset, make_wm_env_and_dataset_multitask
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import save_agent
from utils.datasets import Dataset
from utils.gc_her_sampler import GCHERSampler, build_episode_metadata
from train_interleaved_mpc import CsvLogger


def main():
    p = argparse.ArgumentParser(description="Pure flow-matching BC trainer")
    p.add_argument("--gc", action="store_true", help="Goal-conditioned (384-D obs, 5 tasks).")
    p.add_argument("--condition", choices=["original", "relabel", "rollout"], required=True)
    p.add_argument("--dataset_path", default=None,
                   help="npz of generated transitions (required for relabel/rollout).")
    # Env / WM (for the eval encoder + real env)
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--task_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--env_family", default="cube-single")
    p.add_argument("--wm_device", default="cuda")
    # Training
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=25000)
    p.add_argument("--eval_episodes", type=int, default=250)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_interval", type=int, default=0,
                   help="0 = save final checkpoint only (best/final SR come from eval.csv).")
    # HER params (gc + condition=original only)
    p.add_argument("--done_threshold", type=float, default=2.0)
    p.add_argument("--p_curgoal",    type=float, default=0.2)
    p.add_argument("--p_trajgoal",   type=float, default=0.5)
    p.add_argument("--p_randomgoal", type=float, default=0.3)
    p.add_argument("--p_taskgoal",   type=float, default=0.0)
    p.add_argument("--p_geom",       type=float, default=0.1)
    args = p.parse_args()

    if args.condition in ("relabel", "rollout") and not args.dataset_path:
        p.error("--dataset_path is required for condition relabel/rollout")

    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load env + encoder + (offline dataset for the 'original' condition)
    # ------------------------------------------------------------------
    print("=== Loading WM env / dataset...", flush=True)
    if args.gc:
        (train_dataset_dict, jepa_model, real_envs, z_goals_all) = \
            make_wm_env_and_dataset_multitask(
                wm_ckpt_path=args.wm_ckpt, latent_cache_path=args.wm_cache,
                hdf5_dataset_path=args.wm_hdf5, task_ids=tuple(args.task_ids),
                done_threshold=args.done_threshold, wm_device=args.wm_device,
                img_size=224, env_family=args.env_family,
            )
        real_env_eval = real_envs
        task_id_to_idx = {tid: i for i, tid in enumerate(args.task_ids)}
        def obs_augment(z, task_id):
            return np.concatenate([z, z_goals_all[task_id_to_idx[task_id]]]).astype(np.float32)
        obs_dim = 2 * np.asarray(train_dataset_dict["observations"]).shape[-1]
    else:
        (_, _, train_dataset_dict, _, jepa_model, real_env, _z_goal) = \
            make_wm_env_and_dataset(
                wm_ckpt_path=args.wm_ckpt, latent_cache_path=args.wm_cache,
                hdf5_dataset_path=args.wm_hdf5, task_id=args.task_id,
                done_threshold=args.done_threshold, wm_device=args.wm_device,
                img_size=224, env_family=args.env_family,
            )
        real_env_eval = real_env
        obs_augment = None
        obs_dim = np.asarray(train_dataset_dict["observations"]).shape[-1]
        if np.asarray(train_dataset_dict["observations"]).ndim == 3:
            obs_dim = np.asarray(train_dataset_dict["observations"]).shape[-1]

    # ------------------------------------------------------------------
    # 2. Build the per-step batch sampler for this condition
    # ------------------------------------------------------------------
    np_rng = np.random.default_rng(args.seed)

    # GC original/rollout -> SAME HER sampler over play or the goal-less rollout buffer
    # (identical processing, different trajectories). GC relabel stays a fixed-goal npz
    # (its MPC action was computed for one specific goal -> not HER-compatible).
    # single-task -> uniform (no goals).
    if args.gc and args.condition in ("original", "rollout"):
        if args.condition == "original":
            obs_all      = np.asarray(train_dataset_dict["observations"], np.float32)
            next_obs_all = np.asarray(train_dataset_dict["next_observations"], np.float32)
            actions_all  = np.asarray(train_dataset_dict["actions"], np.float32)
            terms_all    = np.asarray(train_dataset_dict["terminals"], np.float32)
            masks_all    = np.asarray(train_dataset_dict["masks"], np.float32)
            if obs_all.ndim == 3:
                obs_all, next_obs_all = obs_all[:, -1, :], next_obs_all[:, -1, :]
            if actions_all.ndim == 3:
                actions_all = actions_all[:, 0, :]
        else:  # rollout: goal-less generated buffer
            d = np.load(args.dataset_path)
            obs_all      = np.asarray(d["observations"], np.float32)
            next_obs_all = np.asarray(d["next_observations"], np.float32)
            actions_all  = np.asarray(d["actions"], np.float32)
            terms_all    = np.asarray(d["terminals"], np.float32)
            masks_all    = np.asarray(d["masks"], np.float32) if "masks" in d else (1.0 - terms_all)
        N_data = obs_all.shape[0]
        ep_starts, ep_lens, ep_ids, t_within = build_episode_metadata(terms_all)
        buffer_dict = dict(observations=obs_all, next_observations=next_obs_all,
                           actions=actions_all, terminals=terms_all, masks=masks_all,
                           ep_id=ep_ids, t_within=t_within)
        sampler = GCHERSampler(task_goals=z_goals_all, done_threshold=args.done_threshold,
                               p_curgoal=args.p_curgoal, p_trajgoal=args.p_trajgoal,
                               p_randomgoal=args.p_randomgoal, p_taskgoal=args.p_taskgoal,
                               p_geom=args.p_geom)
        n_train = N_data
        def sample_batch():
            b = sampler.sample(buffer_dict=buffer_dict, current_size=N_data,
                               ep_starts=ep_starts, ep_lens=ep_lens,
                               batch_size=args.batch_size, np_rng=np_rng)
            return {"observations": np.asarray(b["observations"], np.float32),
                    "actions": np.asarray(b["actions"], np.float32)}   # (B, 1, 25)
    elif not args.gc and args.condition == "original":
        obs_all = np.asarray(train_dataset_dict["observations"], dtype=np.float32)
        act_all = np.asarray(train_dataset_dict["actions"], dtype=np.float32)
        if obs_all.ndim == 3:
            obs_all = obs_all[:, -1, :]
        if act_all.ndim == 3:
            act_all = act_all[:, 0, :]
        n_train = obs_all.shape[0]
        def sample_batch():
            idxs = np_rng.integers(0, n_train, size=args.batch_size)
            return {"observations": obs_all[idxs], "actions": act_all[idxs][:, None, :]}
    else:  # single relabel/rollout, or gc relabel: uniform over the npz
        data = np.load(args.dataset_path)
        obs_np = np.asarray(data["observations"], dtype=np.float32)
        act_np = np.asarray(data["actions"], dtype=np.float32)
        n_train = obs_np.shape[0]
        def sample_batch():
            idxs = np_rng.integers(0, n_train, size=args.batch_size)
            return {"observations": obs_np[idxs], "actions": act_np[idxs][:, None, :]}

    # ------------------------------------------------------------------
    # 3. Fresh ACFQL agent — pure BC (num_samples=1), train actor only
    # ------------------------------------------------------------------
    print(f"=== Creating fresh BC agent (obs_dim={obs_dim}, gc={args.gc})...", flush=True)
    ex_obs = np.zeros((1, obs_dim), dtype=np.float32)
    ex_act = np.zeros((1, 25),      dtype=np.float32)
    config = get_acfql_config()
    config["encoder"]           = "jepa_head"
    config["actor_type"]        = "best-of-n"
    config["actor_num_samples"] = 1          # pure BC: single flow sample, critic-free
    config["horizon_length"]    = 1
    config["action_chunking"]   = False
    config["lr"]                = args.lr
    agent = ACFQLAgent.create(
        seed=args.seed, ex_observations=ex_obs, ex_actions=ex_act,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
    )
    print(f"  Agent created. Training data: {n_train} transitions "
          f"(condition={args.condition}).", flush=True)

    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))

    print(f"\n=== BC training: {args.steps} steps  (gc={args.gc}, "
          f"condition={args.condition}, seed={args.seed}) ===", flush=True)
    t_start = time.time()
    best_sr = -1.0

    def _do_eval(step):
        if args.gc:
            metrics = evaluate_real_ogbench(
                agent=agent, real_env=real_env_eval, jepa_model=jepa_model,
                device=args.wm_device, task_ids=tuple(args.task_ids),
                num_episodes_per_task=args.eval_episodes,
                action_dispatch="chunk25", pass_task_id_on_reset=False,
                obs_augment=obs_augment,
            )
            sr = float(metrics.get("overall/success_rate", float("nan")))
            print(f"  [eval] overall_success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
        else:
            metrics = evaluate_real_ogbench(
                agent=agent, real_env=real_env_eval, jepa_model=jepa_model,
                device=args.wm_device, task_ids=(args.task_id,),
                num_episodes_per_task=args.eval_episodes,
                action_dispatch="chunk25", pass_task_id_on_reset=False,
            )
            sr = metrics.get(f"task_{args.task_id}/success_rate",
                             metrics.get("overall/success_rate", float("nan")))
            print(f"  [eval] success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
        logger_eval.log(metrics, step=step)
        return float(sr)

    # ------------------------------------------------------------------
    # 4. Train loop (actor-only updates)
    # ------------------------------------------------------------------
    for step in range(1, args.steps + 1):
        batch = sample_batch()
        agent, info = agent.update_actor_only(batch)

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}
            log_data["elapsed_s"] = elapsed
            logger_train.log(log_data, step=step)
            print(f"  step {step:7d}/{args.steps}  "
                  f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n[step {step}] Evaluating...", flush=True)
            sr = _do_eval(step)
            if sr > best_sr:
                best_sr = sr
            if args.save_interval > 0 and step % args.save_interval == 0:
                save_agent(agent, args.save_dir, step)

    save_agent(agent, args.save_dir, args.steps)  # final checkpoint
    print(f"\n=== Done. best_SR={best_sr:.4f} ({best_sr*100:.1f}%). "
          f"Final checkpoint params_{args.steps}.pkl in {args.save_dir}/", flush=True)


if __name__ == "__main__":
    main()
