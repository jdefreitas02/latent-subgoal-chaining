"""Diagnostic A: does the JEPA latent distance to z_goal_task correlate with the
OGBench task-success criterion?

The dense reward we use in WMEnv (and in the relabeled offline buffer) is
   r = -||z_t - z_goal_task||_2 / scale
The implicit assumption is that low latent distance correlates with high task
success. If this is false, the critic is fitting a proxy that disagrees with
the real evaluation metric, and the policy will optimize the wrong thing.

Method: for each offline transition we have
  - z_t (cached JEPA latent of the rendered scene)
  - OGBench's qpos-derived success label (reward >= 0 in the relabel)
We compute d_t = ||z_t - z_goal_task||, then check:
  - Histogram of d_t split by success label
  - Best-threshold accuracy and ROC-AUC of d_t as a success classifier
  - Confusion matrix at the threshold we actually used (wm_done_threshold)

If AUC is high (>0.9) the dense reward is well-aligned.
If AUC is near 0.5 the reward signal is essentially uninformative.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from envs.jepa_loader import load_jepa, encode_pixels_to_latent
from envs.wm_dataset_builder import _load_ogbench_singletask_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--latent_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--hdf5_dataset", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"))
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="The wm_done_threshold actually used in training; we report the "
                        "confusion matrix at this threshold for context.")
    p.add_argument("--out_dir", default="diagnostics_reward")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Get z_goal_task the same way wm_env.py does it (encode rendered goal image)
    import gymnasium as gym
    import ogbench  # noqa
    print(f"[load] JEPA from {args.wm_ckpt}", flush=True)
    jepa = load_jepa(args.wm_ckpt, device=args.device, img_size=224, patch_size=14)
    print(f"[load] real env to render goal image", flush=True)
    real_env = gym.make(
        f"visual-cube-single-singletask-task{args.task_id}-v0",
        width=224, height=224,
    )
    obs, info = real_env.reset(seed=0, options=dict(render_goal=True))
    goal_img = info.get("goal", info.get("target", None))
    if goal_img is None:
        raise RuntimeError(f"missing goal in reset info; keys={list(info.keys())}")
    z_goal_task = encode_pixels_to_latent(jepa, goal_img, args.device)  # (192,)
    real_env.close()
    print(f"  z_goal_task: shape={z_goal_task.shape}, norm={np.linalg.norm(z_goal_task):.3f}",
          flush=True)

    # 2. Load offline latents (flat) and align with OGBench-relabeled rewards
    print(f"[load] latent cache and OGBench rewards", flush=True)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"] if isinstance(cache, dict) and "all_latents" in cache else cache
    latents_per_ep = []
    for ep in all_latents:
        latents_per_ep.append(ep.cpu().numpy().astype(np.float32) if torch.is_tensor(ep) else np.asarray(ep, dtype=np.float32))

    actions, rewards, terminals, masks, episode_lens = _load_ogbench_singletask_dataset(
        task_id=args.task_id, hdf5_path=args.hdf5_dataset,
    )

    # In OGBench's relabel, success = reward >= 0 (typically reward in {-1, 0}; 0 means success).
    success_flat = (rewards >= 0).astype(np.int32)

    # Align latents to rewards: same total length expected (or off by 1)
    # rewards are flat over all timesteps. latents_per_ep is a list of (T_i, 192).
    z_flat = np.concatenate(latents_per_ep, axis=0)
    n_min = min(z_flat.shape[0], rewards.shape[0])
    print(f"  latents flat: {z_flat.shape}, rewards: {rewards.shape}, success_count={success_flat.sum()}",
          flush=True)
    z_flat = z_flat[:n_min]
    success = success_flat[:n_min]
    rewards = rewards[:n_min]

    # 3. Compute d_t = ||z_t - z_goal_task||
    print(f"[compute] L2 distance to z_goal_task", flush=True)
    d = np.linalg.norm(z_flat - z_goal_task[None, :], axis=1).astype(np.float32)
    print(f"  d shape: {d.shape}")

    # 4. Histogram & summary
    d_succ = d[success == 1]
    d_fail = d[success == 0]
    print(f"\n  success states: n={d_succ.size}, "
          f"mean d={d_succ.mean():.3f}, "
          f"median d={np.median(d_succ):.3f}, "
          f"p10={np.percentile(d_succ, 10):.3f}, "
          f"p90={np.percentile(d_succ, 90):.3f}")
    print(f"  failure states: n={d_fail.size}, "
          f"mean d={d_fail.mean():.3f}, "
          f"median d={np.median(d_fail):.3f}, "
          f"p10={np.percentile(d_fail, 10):.3f}, "
          f"p90={np.percentile(d_fail, 90):.3f}")

    # 5. ROC-AUC: low d should predict success → use -d as score
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        auc_roc = float(roc_auc_score(success, -d))
        prec, rec, _ = precision_recall_curve(success, -d)
        auc_pr = float(auc(rec, prec))
    except Exception as e:
        print(f"  sklearn unavailable: {e}")
        auc_roc, auc_pr = float("nan"), float("nan")

    # 6. Best-threshold accuracy by simple sweep
    thresholds = np.linspace(d.min(), d.max(), 200)
    accs = []
    for t in thresholds:
        pred = (d < t).astype(np.int32)
        accs.append(float((pred == success).mean()))
    accs = np.asarray(accs)
    best_idx = int(np.argmax(accs))
    best_thr = float(thresholds[best_idx])
    best_acc = float(accs[best_idx])

    # 7. Confusion matrix at the threshold we used in training
    pred_used = (d < args.threshold).astype(np.int32)
    tp = int(((pred_used == 1) & (success == 1)).sum())
    tn = int(((pred_used == 0) & (success == 0)).sum())
    fp = int(((pred_used == 1) & (success == 0)).sum())
    fn = int(((pred_used == 0) & (success == 1)).sum())
    n = tp + tn + fp + fn
    print(f"\n=== Reward-success correlation ===")
    print(f"  AUC-ROC          : {auc_roc:.4f}   (1.0 = perfect; 0.5 = noise)")
    print(f"  AUC-PR           : {auc_pr:.4f}")
    print(f"  best threshold   : {best_thr:.3f}  (acc={best_acc:.4f})")
    print(f"  used  threshold  : {args.threshold:.3f}")
    print(f"  confusion at used thr:")
    print(f"      TP={tp}  FP={fp}")
    print(f"      FN={fn}  TN={tn}")
    print(f"    precision={tp/max(1,tp+fp):.4f}  recall={tp/max(1,tp+fn):.4f}  "
          f"specificity={tn/max(1,tn+fp):.4f}  acc={(tp+tn)/n:.4f}")

    # 8. Per-decile latent-distance: what fraction of states in each d-decile are successes?
    print(f"\n  d-decile  | range            | n     | success_rate")
    deciles = np.percentile(d, np.linspace(0, 100, 11))
    for k in range(10):
        lo, hi = deciles[k], deciles[k + 1]
        m = (d >= lo) & (d < hi if k < 9 else d <= hi)
        sr = float(success[m].mean()) if m.any() else 0.0
        print(f"  {k:2d}        | [{lo:6.2f}, {hi:6.2f}] | {m.sum():6d} | {sr:.4f}")

    # 9. Save
    report = dict(
        auc_roc=auc_roc, auc_pr=auc_pr,
        best_threshold=best_thr, best_acc=best_acc,
        used_threshold=float(args.threshold),
        confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
        d_succ=dict(mean=float(d_succ.mean()), median=float(np.median(d_succ)),
                    p10=float(np.percentile(d_succ, 10)),
                    p90=float(np.percentile(d_succ, 90)), n=int(d_succ.size)),
        d_fail=dict(mean=float(d_fail.mean()), median=float(np.median(d_fail)),
                    p10=float(np.percentile(d_fail, 10)),
                    p90=float(np.percentile(d_fail, 90)), n=int(d_fail.size)),
    )
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savez(os.path.join(args.out_dir, "raw.npz"),
             d=d, success=success, rewards=rewards, z_goal=z_goal_task)
    print(f"\nWritten to {args.out_dir}/report.json and raw.npz")


if __name__ == "__main__":
    main()
