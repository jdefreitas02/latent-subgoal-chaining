"""IQL (value-based offline RL) on the latent chunked-action datasets.

The decisive "does the WM-generated data help a *value-based* learner, not just BC?"
test. Only valid on datasets with DYNAMICALLY-CONSISTENT transitions:
  - original : real (z, a_play, z'_real)            -> consistent (baseline)
  - rollout  : imagined (z, a_mpc, z'_wm)           -> consistent (the test)
The `relabel` dataset is (z, a_mpc, z'_real): a_mpc did NOT produce z'_real, so its
transitions are inconsistent and it is NOT a valid value-learning input -> blocked here.

Reward is recomputed uniformly as a latent goal-reaching indicator so the convention is
identical across conditions (only the (state, action, next_state) differs):
  r = 1{ || z'_z - g || < done_threshold } ,  mask = 1 - r
where for GC, g = obs[:, D:2D] and z'_z = next_obs[:, :D]; for single, g = z_goal_task.

Eval: standalone, deterministic IQL/AWR actor (mode), chunk25 dispatch, real env.
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np

from agents.iql_chunk import IQLAgent, get_config as get_iql_config
from envs.wm_env import make_wm_env_and_dataset, make_wm_env_and_dataset_multitask
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import save_agent
from utils.gc_her_sampler import GCHERSampler, build_episode_metadata
from train_interleaved_mpc import CsvLogger


def main():
    p = argparse.ArgumentParser(description="IQL on latent chunked-action datasets")
    p.add_argument("--gc", action="store_true")
    p.add_argument("--condition", choices=["original", "rollout"], required=True,
                   help="relabel is intentionally unsupported (inconsistent transitions).")
    p.add_argument("--dataset_path", default=None, help="npz (required for rollout).")
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--task_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--env_family", default="cube-single")
    p.add_argument("--wm_device", default="cuda")
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--expectile", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=10.0, help="AWR temperature.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=25000)
    p.add_argument("--eval_episodes", type=int, default=250)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_interval", type=int, default=0)
    p.add_argument("--done_threshold", type=float, default=2.0)
    p.add_argument("--p_curgoal",    type=float, default=0.2)
    p.add_argument("--p_trajgoal",   type=float, default=0.5)
    p.add_argument("--p_randomgoal", type=float, default=0.3)
    p.add_argument("--p_taskgoal",   type=float, default=0.0)
    p.add_argument("--p_geom",       type=float, default=0.1)
    args = p.parse_args()

    if args.condition == "rollout" and not args.dataset_path:
        p.error("--dataset_path is required for condition rollout")
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Env + encoder (+ offline dataset / goals)
    # ------------------------------------------------------------------
    print("=== Loading WM env / dataset...", flush=True)
    z_goal_task = None
    if args.gc:
        (train_dataset_dict, jepa_model, real_envs, z_goals_all) = \
            make_wm_env_and_dataset_multitask(
                wm_ckpt_path=args.wm_ckpt, latent_cache_path=args.wm_cache,
                hdf5_dataset_path=args.wm_hdf5, task_ids=tuple(args.task_ids),
                done_threshold=args.done_threshold, wm_device=args.wm_device,
                img_size=224, env_family=args.env_family)
        real_env_eval = real_envs
        task_id_to_idx = {tid: i for i, tid in enumerate(args.task_ids)}
        def obs_augment(z, task_id):
            return np.concatenate([z, z_goals_all[task_id_to_idx[task_id]]]).astype(np.float32)
        D = np.asarray(train_dataset_dict["observations"]).shape[-1]  # 192
        obs_dim = 2 * D
    else:
        (_, _, train_dataset_dict, _, jepa_model, real_env, z_goal_task) = \
            make_wm_env_and_dataset(
                wm_ckpt_path=args.wm_ckpt, latent_cache_path=args.wm_cache,
                hdf5_dataset_path=args.wm_hdf5, task_id=args.task_id,
                done_threshold=args.done_threshold, wm_device=args.wm_device,
                img_size=224, env_family=args.env_family)
        real_env_eval = real_env
        obs_augment = None
        z_goal_task = np.asarray(z_goal_task, dtype=np.float32)
        D = np.asarray(train_dataset_dict["observations"]).shape[-1]
        obs_dim = D

    np_rng = np.random.default_rng(args.seed)

    def to_iql_batch(obs, act, next_obs):
        """Attach a uniform latent goal-reaching reward; mask = 1 - reward."""
        if args.gc:
            g  = obs[:, D:]
            nz = next_obs[:, :D]
        else:
            g  = z_goal_task[None, :]
            nz = next_obs
        r = (np.linalg.norm(nz - g, axis=-1) < args.done_threshold).astype(np.float32)
        return dict(observations=obs.astype(np.float32), actions=act.astype(np.float32),
                    rewards=r, masks=(1.0 - r).astype(np.float32),
                    next_observations=next_obs.astype(np.float32))

    # ------------------------------------------------------------------
    # 2. Per-condition batch sampler.
    #    GC: HER sampler over a buffer that is play (original) or the goal-LESS
    #        generated rollout (rollout) -> identical processing, different data.
    #        Reward/masks come from the HER sampler (indicator), same for both arms.
    #    single: uniform sampling + recomputed latent indicator reward (no goals).
    # ------------------------------------------------------------------
    if args.gc:
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
        else:  # rollout: goal-less generated buffer, same form as play
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
            return dict(observations=np.asarray(b["observations"], np.float32),
                        actions=np.asarray(b["actions"])[:, 0, :].astype(np.float32),
                        rewards=np.asarray(b["rewards"])[:, 0].astype(np.float32),
                        masks=np.asarray(b["masks"])[:, 0].astype(np.float32),
                        next_observations=np.asarray(b["next_observations"])[:, 0, :].astype(np.float32))
    else:  # single-task
        if args.condition == "original":
            obs_all      = np.asarray(train_dataset_dict["observations"], np.float32)
            act_all      = np.asarray(train_dataset_dict["actions"], np.float32)
            next_obs_all = np.asarray(train_dataset_dict["next_observations"], np.float32)
            if obs_all.ndim == 3:
                obs_all, next_obs_all = obs_all[:, -1, :], next_obs_all[:, -1, :]
            if act_all.ndim == 3:
                act_all = act_all[:, 0, :]
        else:  # rollout npz (single-task, goal-less)
            d = np.load(args.dataset_path)
            obs_all = np.asarray(d["observations"], np.float32)
            act_all = np.asarray(d["actions"], np.float32)
            next_obs_all = np.asarray(d["next_observations"], np.float32)
        n_train = obs_all.shape[0]
        def sample_batch():
            i = np_rng.integers(0, n_train, size=args.batch_size)
            return to_iql_batch(obs_all[i], act_all[i], next_obs_all[i])

    # ------------------------------------------------------------------
    # 3. IQL agent
    # ------------------------------------------------------------------
    print(f"=== Creating IQL agent (obs_dim={obs_dim}, gc={args.gc}, cond={args.condition})...", flush=True)
    ex_obs = np.zeros((1, obs_dim), dtype=np.float32)
    ex_act = np.zeros((1, 25),      dtype=np.float32)
    config = get_iql_config()
    config["encoder"]    = "jepa_head"
    config["lr"]         = args.lr
    config["expectile"]  = args.expectile
    config["alpha"]      = args.alpha
    agent = IQLAgent.create(seed=args.seed, ex_observations=ex_obs, ex_actions=ex_act,
                            config=config.to_dict() if hasattr(config, "to_dict") else dict(config))
    print(f"  Agent created. Training data: {n_train} transitions.", flush=True)

    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))
    print(f"\n=== IQL training: {args.steps} steps (gc={args.gc}, {args.condition}, seed={args.seed}) ===",
          flush=True)
    t_start = time.time(); best_sr = -1.0

    def _do_eval(step):
        if args.gc:
            m = evaluate_real_ogbench(agent=agent, real_env=real_env_eval, jepa_model=jepa_model,
                                      device=args.wm_device, task_ids=tuple(args.task_ids),
                                      num_episodes_per_task=args.eval_episodes,
                                      action_dispatch="chunk25", pass_task_id_on_reset=False,
                                      obs_augment=obs_augment)
            sr = float(m.get("overall/success_rate", float("nan")))
            print(f"  [eval] overall_success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
        else:
            m = evaluate_real_ogbench(agent=agent, real_env=real_env_eval, jepa_model=jepa_model,
                                      device=args.wm_device, task_ids=(args.task_id,),
                                      num_episodes_per_task=args.eval_episodes,
                                      action_dispatch="chunk25", pass_task_id_on_reset=False)
            sr = m.get(f"task_{args.task_id}/success_rate", m.get("overall/success_rate", float("nan")))
            print(f"  [eval] success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
        logger_eval.log(m, step=step)
        return float(sr)

    # ------------------------------------------------------------------
    # 4. Train loop
    # ------------------------------------------------------------------
    for step in range(1, args.steps + 1):
        agent, info = agent.update(sample_batch())
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}; log_data["elapsed_s"] = elapsed
            logger_train.log(log_data, step=step)
            print(f"  step {step:7d}/{args.steps}  "
                  f"critic={info.get('critic/critic_loss', float('nan')):.3f}  "
                  f"v={info.get('value/v_mean', float('nan')):.3f}  "
                  f"adv={info.get('actor/adv', float('nan')):.3f}  elapsed={elapsed:.0f}s", flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n[step {step}] Evaluating...", flush=True)
            sr = _do_eval(step); best_sr = max(best_sr, sr)
            if args.save_interval > 0 and step % args.save_interval == 0:
                save_agent(agent, args.save_dir, step)

    save_agent(agent, args.save_dir, args.steps)
    print(f"\n=== Done. best_SR={best_sr*100:.1f}%. Final params_{args.steps}.pkl in {args.save_dir}/",
          flush=True)


if __name__ == "__main__":
    main()
