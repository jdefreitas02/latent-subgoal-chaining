"""Offline goal-conditioned pretraining (E3b Phase 1).

Trains ACFQL with goal-conditioned observations (z, g) -> 384-D obs.
HIQL-style HER sampling: 20% current / 50% trajectory-future / 30% random.

Reward is index-based indicator (HIQL convention):
  - cur goal (g = obs[i]):     r=1, terminal -- V(s, s) = 1
  - traj goal (g = future obs): r=0          -- TD propagation back to goal
  - random goal:                r=0          -- no signal
No latent distance is used anywhere; reward is purely index-driven so it
does not depend on the WM's latent geometry.

Usage:
    python train_offline_gc.py \\
        --wm_ckpt $STABLEWM_HOME/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache $STABLEWM_HOME/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --offline_steps 500000 \\
        --save_dir $STABLEWM_HOME/cube/e3_offline_gc
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_env import make_wm_env_and_dataset_multitask
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import save_agent
from utils.gc_her_sampler import GCHERSampler, build_episode_metadata
from train_interleaved_mpc import CsvLogger


def main():
    p = argparse.ArgumentParser(description="Offline goal-conditioned pretraining (E3)")
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--wm_cache", required=True)
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--env_family", default="cube-single")
    p.add_argument("--wm_device", default="cuda")
    p.add_argument("--task_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    # Training
    p.add_argument("--offline_steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=25000)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--save_dir", required=True)
    # GC / HER
    p.add_argument("--done_threshold", type=float, default=2.0)
    p.add_argument("--p_curgoal",   type=float, default=0.2)
    p.add_argument("--p_trajgoal",  type=float, default=0.5)
    p.add_argument("--p_randomgoal", type=float, default=0.3)
    p.add_argument("--p_taskgoal", type=float, default=0.0,
                   help="P(sample uniform task goal). Default 0 under indicator-reward "
                        "mode (no per-(transition, task) ground-truth reward precomputed; "
                        "task-goal samples would all get r=0 and contribute no signal).")
    p.add_argument("--p_geom", type=float, default=0.1,
                   help="Geometric param for future-offset; mean = 1/p_geom = 10.")
    # Data source: 'original' = play+HER (the baseline); 'rollout' = a generated
    # npz of consistent (z,a_mpc,z'_wm) transitions (does the strong learner do
    # better on the WM-generated data than on play?).
    p.add_argument("--condition", choices=["original", "rollout"], default="original")
    p.add_argument("--dataset_path", default=None, help="npz for --condition rollout")
    p.add_argument("--save_interval", type=int, default=0,
                   help="0 = save final checkpoint only (SR comes from eval.csv).")
    args = p.parse_args()
    if args.condition == "rollout" and not args.dataset_path:
        p.error("--dataset_path required for --condition rollout")

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
    obs_all      = np.asarray(train_dataset_dict["observations"])      # (N, 192)
    next_obs_all = np.asarray(train_dataset_dict["next_observations"]) # (N, 192)
    actions_all  = np.asarray(train_dataset_dict["actions"])           # (N, 25)
    terms_all    = np.asarray(train_dataset_dict["terminals"])         # (N,)
    masks_all    = np.asarray(train_dataset_dict["masks"])             # (N,)
    if obs_all.ndim == 3:
        obs_all      = obs_all[:, -1, :]
        next_obs_all = next_obs_all[:, -1, :]
    if actions_all.ndim == 3:
        actions_all = actions_all[:, 0, :]

    # rollout condition: swap in the goal-LESS generated buffer. It has the exact same
    # form as the play data (192-D z, episodic via terminals) and is processed by the
    # SAME HER sampler below, so original vs rollout differ ONLY in the trajectories.
    if args.condition == "rollout":
        _d = np.load(args.dataset_path)
        obs_all      = np.asarray(_d["observations"], np.float32)
        next_obs_all = np.asarray(_d["next_observations"], np.float32)
        actions_all  = np.asarray(_d["actions"], np.float32)
        terms_all    = np.asarray(_d["terminals"], np.float32)
        masks_all    = np.asarray(_d["masks"], np.float32) if "masks" in _d else (1.0 - terms_all)
        print(f"  [rollout] using generated goal-less buffer {args.dataset_path}", flush=True)

    N_data = obs_all.shape[0]
    D_obs  = obs_all.shape[-1]    # 192
    D_act  = actions_all.shape[-1] # 25
    print(f"  Dataset ({args.condition}): {N_data} transitions, obs_dim={D_obs}, act_dim={D_act}",
          flush=True)

    # ------------------------------------------------------------------
    # 2. Episode metadata for HER
    # ------------------------------------------------------------------
    ep_starts, ep_lens, ep_ids, t_within = build_episode_metadata(terms_all)
    print(f"  Episodes: {len(ep_starts)}, mean_len={ep_lens.mean():.1f}, "
          f"min={ep_lens.min()}, max={ep_lens.max()}", flush=True)

    # Build the buffer-dict that the sampler reads from
    buffer_dict = dict(
        observations=obs_all,
        next_observations=next_obs_all,
        actions=actions_all,
        terminals=terms_all.astype(np.float32),
        masks=masks_all.astype(np.float32),
        ep_id=ep_ids,
        t_within=t_within,
    )

    # ------------------------------------------------------------------
    # 3. Create goal-conditioned ACFQL agent (384-D obs)
    # ------------------------------------------------------------------
    print("=== Creating GC ACFQL agent (obs_dim = 2 * 192 = 384)...", flush=True)
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
    print("  Agent created.", flush=True)

    # ------------------------------------------------------------------
    # 4. HER sampler
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

    # Both conditions use the IDENTICAL HER sampler over buffer_dict; the only
    # difference is whether buffer_dict came from play or the generated rollout.
    def sample_batch():
        return sampler.sample(buffer_dict=buffer_dict, current_size=N_data,
                              ep_starts=ep_starts, ep_lens=ep_lens,
                              batch_size=args.batch_size, np_rng=np_rng)

    # Lambda for eval obs augmentation: maps task_id to goal index in z_goals_all
    task_id_to_idx = {tid: i for i, tid in enumerate(args.task_ids)}
    def obs_augment(z, task_id):
        return np.concatenate([z, z_goals_all[task_id_to_idx[task_id]]]).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Logging
    # ------------------------------------------------------------------
    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))

    print(f"\n=== Offline GC training: {args.offline_steps} steps ===", flush=True)
    print(f"    HER mix: cur={args.p_curgoal}  traj={args.p_trajgoal}  "
          f"rand={args.p_randomgoal}  task={args.p_taskgoal}", flush=True)
    print(f"    lr={args.lr}  batch_size={args.batch_size}  discount={args.discount}",
          flush=True)
    t_start = time.time()

    for step in range(1, args.offline_steps + 1):
        batch = sample_batch()
        agent, info = agent.update(batch)

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}
            log_data["elapsed_s"] = elapsed
            logger_train.log(log_data, step=step)
            print(
                f"  step {step:7d}/{args.offline_steps}  "
                f"critic={info.get('critic/critic_loss', float('nan')):.4f}  "
                f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % args.eval_interval == 0 or step == args.offline_steps:
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
            if args.save_interval > 0 and step % args.save_interval == 0:
                save_agent(agent, args.save_dir, step)
                print(f"  [ckpt] saved params_{step}.pkl", flush=True)

    save_agent(agent, args.save_dir, args.offline_steps)
    print(f"\n=== Done. Final checkpoint in {args.save_dir}/", flush=True)


if __name__ == "__main__":
    main()
