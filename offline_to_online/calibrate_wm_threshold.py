"""Calibrate WMEnv's done_threshold.

Question: what does ||z - z_goal|| look like for real success states?
The WMEnv treats d < done_threshold as success+reward; if real successes
sit at d ~ 8-12, then thresholds in {1.5, 2.0, 2.5, 3.0} are unreachable
and the online phase trains on all-zero reward.

Procedure:
1. Build E's offline dataset + z_goal_task via make_wm_env_and_dataset.
   The dataset has 192-D JEPA latents already cached.
2. Look at OGBench's relabeled rewards (0 = success, -1 = not-yet).
3. Compute ||z - z_goal|| for transitions tagged as success vs. not.
"""
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--wm_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--wm_hdf5", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from envs.wm_env import make_wm_env_and_dataset

    print("=== Building E env+dataset (this loads the cached latents + z_goal)")
    train_env, eval_env, train_dataset, val_dataset, jepa, real_env, z_goal = \
        make_wm_env_and_dataset(
            wm_ckpt_path=args.wm_ckpt,
            latent_cache_path=args.wm_cache,
            hdf5_dataset_path=args.wm_hdf5,
            task_id=args.task_id,
            done_threshold=2.0,         # irrelevant for this analysis
            max_episode_steps=40,
            wm_device=args.device,
            img_size=224,
        )

    obs = np.asarray(train_dataset["observations"])         # (N, 192)
    next_obs = np.asarray(train_dataset["next_observations"])  # (N, 192)
    rewards = np.asarray(train_dataset["rewards"])             # (N,)
    masks = np.asarray(train_dataset.get("masks", np.ones_like(rewards)))
    print(f"  dataset: N={obs.shape[0]}, obs_dim={obs.shape[1]}")
    print(f"  z_goal shape: {z_goal.shape}, dtype: {z_goal.dtype}")
    print(f"  ||z_goal||_2 = {np.linalg.norm(z_goal):.3f}")

    # OGBench task-relabeled rewards: 0 if at goal, -1 otherwise (for cube tasks).
    print(f"\n=== Reward distribution ===")
    uniq, counts = np.unique(rewards, return_counts=True)
    for u, c in zip(uniq, counts):
        print(f"  reward={u:.3f}: count={c} ({100*c/len(rewards):.2f}%)")

    z_goal_np = z_goal.astype(np.float32)

    def dist_stats(name, z_arr):
        if z_arr.shape[0] == 0:
            print(f"  {name}: EMPTY")
            return
        d = np.linalg.norm(z_arr - z_goal_np[None, :], axis=-1)
        pct = np.percentile(d, [0, 5, 25, 50, 75, 95, 100])
        print(f"  {name:32s} N={z_arr.shape[0]:>8d}  "
              f"min={pct[0]:6.2f}  p5={pct[1]:6.2f}  p25={pct[2]:6.2f}  "
              f"med={pct[3]:6.2f}  p75={pct[4]:6.2f}  p95={pct[5]:6.2f}  max={pct[6]:6.2f}  "
              f"mean={d.mean():6.2f}")

    # OGBench cube tasks: reward == 0 iff task is solved at that state.
    # next_observations[t] is the state AFTER action a[t]; so success-state latents
    # are those where rewards[t] == 0 (reward is for arriving at next_obs[t]).
    success_mask = (rewards == 0.0)
    fail_mask = ~success_mask

    print(f"\n=== Distance ||next_obs - z_goal||_2 by reward label ===")
    dist_stats("ALL transitions (next_obs)", next_obs)
    dist_stats("SUCCESS (reward=0, next_obs)", next_obs[success_mask])
    dist_stats("FAIL    (reward<0, next_obs)", next_obs[fail_mask])

    print(f"\n=== Distance ||obs - z_goal||_2 by reward label ===")
    dist_stats("ALL transitions (obs)", obs)
    dist_stats("SUCCESS (reward=0, obs)", obs[success_mask])
    dist_stats("FAIL    (reward<0, obs)", obs[fail_mask])

    # Practical threshold recommendation: pick a threshold T such that
    # fraction of true-success states with d < T is high (say >=50%) and
    # fraction of fail states with d < T is low (say <=5%). Sweep T.
    print(f"\n=== Threshold sweep: P(d<T | success) vs P(d<T | fail) ===")
    d_succ = np.linalg.norm(next_obs[success_mask] - z_goal_np[None, :], axis=-1)
    d_fail = np.linalg.norm(next_obs[fail_mask] - z_goal_np[None, :], axis=-1)
    for T in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
              11.0, 12.0, 14.0, 16.0, 18.0, 20.0]:
        p_s = (d_succ < T).mean() if d_succ.size else 0.0
        p_f = (d_fail < T).mean() if d_fail.size else 0.0
        # Want p_s high, p_f low. F1-ish: precision = p_s*Ns / (p_s*Ns + p_f*Nf)
        Ns, Nf = d_succ.size, d_fail.size
        tp = p_s * Ns
        fp = p_f * Nf
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = p_s
        print(f"  T={T:5.1f}  recall(succ)={p_s:.4f}  FPR(fail)={p_f:.4f}  "
              f"precision={prec:.4f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
