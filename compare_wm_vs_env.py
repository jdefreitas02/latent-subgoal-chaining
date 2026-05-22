"""
Compare WM-based vs env-based SAC training using the eval_ogbench.py protocol.

Runs eval_ogbench.py for two checkpoint directories and prints a side-by-side
performance table.  Both evaluations use the same OGBench native protocol
(visual-cube-single-v0, 5 tasks × num_episodes episodes, success from info['success']).

Usage:
    python latent_hindsight_rl/compare_wm_vs_env.py \\
        --wm_dir  ./checkpoints_wm_pure_distance \\
        --env_dir ./checkpoints_env_pure_distance \\
        --ckpt_path ./lewm_ogbench_weights.ckpt \\
        --dataset_path $STABLEWM_HOME/ogbench/cube_single_play_v0

    # Optional overrides:
        --num_episodes 50      # episodes per task (default: 50)
        --img_size 64          # 64 or 224 (default: 64)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run_eval(checkpoint_dir, ckpt_path, dataset_path, num_episodes, img_size, patch_size):
    """
    Call eval_ogbench.py for one checkpoint directory.
    Streams output live; returns (results_dir Path, elapsed seconds).
    """
    script = Path(__file__).parent / "eval_ogbench.py"
    cmd = [
        sys.executable, str(script),
        "--checkpoint_dir", os.path.abspath(checkpoint_dir),
        "--ckpt_path",      os.path.abspath(ckpt_path),
        "--dataset_path",   os.path.abspath(dataset_path),
        "--num_episodes",   str(num_episodes),
        "--img_size",       str(img_size),
        "--patch_size",     str(patch_size),
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"\n  Evaluation finished in {elapsed:.1f}s")

    exp_name    = os.path.basename(checkpoint_dir.rstrip("/"))
    results_dir = Path(f"eval_native_{exp_name}")
    return results_dir, elapsed


def parse_results(results_dir):
    """
    Parse metrics from eval_ogbench.py results files.
    Looks for per-task success lines and the overall summary.
    Returns dict with 'overall_sr', 'per_task', and raw text.
    """
    results_file = results_dir / "results.txt"
    if not results_file.exists():
        return {"error": f"results file not found: {results_file}"}

    text = results_file.read_text()
    metrics = {"raw": text}

    # Per-task lines: "  Task N (name): XX.X%  (k/N)"
    task_pattern = re.compile(r'Task\s+(\d+)[^:]*:\s+([\d.]+)%\s+\((\d+)/(\d+)\)')
    task_results = {}
    for m in task_pattern.finditer(text):
        task_id = int(m.group(1))
        sr      = float(m.group(2)) / 100.0
        task_results[f"task_{task_id}"] = sr
    if task_results:
        metrics["per_task"] = task_results
        metrics["overall_sr"] = float(np.mean(list(task_results.values())))

    # Overall line: "Overall: XX.X%"
    ov = re.search(r'Overall[^:]*:\s+([\d.]+)%', text)
    if ov:
        metrics["overall_sr"] = float(ov.group(1)) / 100.0

    return metrics


def print_comparison(wm_metrics, env_metrics, wm_dir, env_dir,
                     wm_elapsed, env_elapsed):
    import numpy as np

    print("\n" + "=" * 70)
    print("  SAC COMPARISON: WM rollouts vs Real-Env rollouts")
    print("=" * 70)

    col_w = 18
    print(f"  {'Metric':<28} {'WM rollouts':>{col_w}} {'Env rollouts':>{col_w}}")
    print(f"  {'-'*28} {'-'*col_w} {'-'*col_w}")

    def fmt(v):
        if isinstance(v, float):
            return f"{v*100:.1f}%"
        return str(v)

    # Overall success rate
    wm_sr  = wm_metrics.get("overall_sr",  "N/A")
    env_sr = env_metrics.get("overall_sr", "N/A")
    print(f"  {'Overall success rate':<28} {fmt(wm_sr):>{col_w}} {fmt(env_sr):>{col_w}}")

    # Per-task breakdown
    all_tasks = sorted(
        set(list(wm_metrics.get("per_task", {}).keys()) +
            list(env_metrics.get("per_task", {}).keys()))
    )
    for task_key in all_tasks:
        wm_v  = wm_metrics.get("per_task", {}).get(task_key, "N/A")
        env_v = env_metrics.get("per_task", {}).get(task_key, "N/A")
        print(f"  {task_key:<28} {fmt(wm_v):>{col_w}} {fmt(env_v):>{col_w}}")

    print(f"  {'Eval time (s)':<28} {wm_elapsed:>{col_w}.1f} {env_elapsed:>{col_w}.1f}")
    print("=" * 70)
    print(f"  WM checkpoint : {wm_dir}")
    print(f"  Env checkpoint: {env_dir}")
    print("=" * 70 + "\n")

    # Write to file
    out = Path("comparison_results.txt")
    with out.open("w") as f:
        f.write("SAC COMPARISON: WM rollouts vs Real-Env rollouts\n")
        f.write(f"WM checkpoint : {wm_dir}\n")
        f.write(f"Env checkpoint: {env_dir}\n\n")
        f.write(f"{'Metric':<28} {'WM rollouts':>18} {'Env rollouts':>18}\n")
        f.write(f"{'-'*64}\n")
        f.write(f"{'Overall success rate':<28} {fmt(wm_sr):>18} {fmt(env_sr):>18}\n")
        for task_key in all_tasks:
            wm_v  = wm_metrics.get("per_task", {}).get(task_key, "N/A")
            env_v = env_metrics.get("per_task", {}).get(task_key, "N/A")
            f.write(f"{task_key:<28} {fmt(wm_v):>18} {fmt(env_v):>18}\n")
        f.write(f"{'Eval time (s)':<28} {wm_elapsed:>18.1f} {env_elapsed:>18.1f}\n")
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Compare WM-based and env-based SAC via eval_ogbench.py"
    )
    parser.add_argument("--wm_dir",      required=True,
                        help="Checkpoint dir for the WM-trained actor (sac_wm_train.py)")
    parser.add_argument("--env_dir",     required=True,
                        help="Checkpoint dir for the env-trained actor (sac_env_train.py)")
    parser.add_argument("--ckpt_path",   required=True,
                        help="Path to JEPA encoder checkpoint (.ckpt)")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to OGBench HDF5 dataset (for action scaler)")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes per task (default 50 = 250 total)")
    parser.add_argument("--img_size",     type=int, default=64)
    parser.add_argument("--patch_size",   type=int, default=8)
    args = parser.parse_args()

    wm_results_dir,  wm_elapsed  = run_eval(
        args.wm_dir,  args.ckpt_path, args.dataset_path,
        args.num_episodes, args.img_size, args.patch_size
    )
    env_results_dir, env_elapsed = run_eval(
        args.env_dir, args.ckpt_path, args.dataset_path,
        args.num_episodes, args.img_size, args.patch_size
    )

    wm_metrics  = parse_results(wm_results_dir)
    env_metrics = parse_results(env_results_dir)

    print_comparison(wm_metrics, env_metrics,
                     args.wm_dir, args.env_dir,
                     wm_elapsed, env_elapsed)
