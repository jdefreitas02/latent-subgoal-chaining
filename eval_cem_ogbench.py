"""
Evaluate the LeWM world model as a CEM/MPC planner on the 5 OGBench Cube tasks.

This is the fair, out-of-distribution complement to eval.py: instead of sampling
goals from within training episodes, it uses the 5 hardcoded OGBench task goals
(horizontal, vertical1/2, diagonal1/2) — the exact protocol that HIQL/GCIQL/CRL
report in the OGBench paper.

The policy is swm.policy.WorldModelPolicy with CEMSolver + AutoCostModel(LeWM),
replanning every `receding_horizon` planner-steps. Success is OGBench's own
info['success'] at episode end (cube within 4cm of target).

Usage (224x224 LeWM, defaults match config/eval/cube.yaml):

    python latent_hindsight_rl/eval_cem_ogbench.py \\
        --ckpt_path $STABLEWM_HOME/cube/lejepa \\
        --dataset_path $STABLEWM_HOME/cube/cube_single_expert
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver


def make_img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def fit_process(dataset, keys):
    """Fit StandardScaler for each non-pixel key. Mirrors eval.py."""
    process = {}
    for col in keys:
        if col in ("pixels",):
            continue
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def reset_policy_state(policy):
    """Clear the WorldModelPolicy's action buffer and warm-start between tasks."""
    if policy._action_buffer is not None:
        policy._action_buffer.clear()
    policy._next_init = None


def main():
    parser = argparse.ArgumentParser(
        description="CEM/MPC evaluation of LeWM on the 5 OGBench Cube tasks."
    )
    parser.add_argument("--ckpt_path", default=None,
                        help="Base path to LeWM ckpt (AutoCostModel appends _object.ckpt). "
                             "Default: $STABLEWM_HOME/cube/lejepa")
    parser.add_argument("--dataset_path", default=None,
                        help="HDF5 dataset (without .h5) used only to fit action scaler. "
                             "Default: $STABLEWM_HOME/cube/cube_single_expert")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes per task. Standard OGBench protocol is 50.")
    parser.add_argument("--eval_budget",  type=int, default=50,
                        help="Planner steps per episode. Each step runs CEM and "
                             "executes receding_horizon * action_block env steps.")
    parser.add_argument("--horizon",          type=int, default=5)
    parser.add_argument("--receding_horizon", type=int, default=5)
    parser.add_argument("--action_block",     type=int, default=5,
                        help="Frameskip: each planned action is repeated this many env steps.")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--cem_steps",   type=int, default=30)
    parser.add_argument("--topk",        type=int, default=30)
    parser.add_argument("--var_scale",   type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stablewm_home = os.environ.get("STABLEWM_HOME",
                                   os.path.join(os.path.expanduser("~"), "stable_wm_data"))
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(stablewm_home, "cube", "lejepa")
    if args.dataset_path is None:
        args.dataset_path = os.path.join(stablewm_home, "cube", "cube_single_expert")
    print(f"Weights:  {args.ckpt_path}")
    print(f"Dataset:  {args.dataset_path}  (action-scaler only)")

    # ── ogbench must be imported to register swm/OGBCube-v0's task_infos ────
    import ogbench  # noqa: F401

    # ── World ───────────────────────────────────────────────────────────────
    plan_len_env_steps = args.horizon * args.action_block
    max_env_steps = args.eval_budget + 10
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=args.num_episodes,
        image_shape=(args.img_size, args.img_size),
        max_episode_steps=max_env_steps,
        history_size=1,
        frame_skip=1,
        env_type="single",
        ob_type="states",
        visualize_info=False,
        terminate_at_goal=False,   # keep episodes running; success tracked separately
        multiview=False,
        width=args.img_size,
        height=args.img_size,
    )

    task_infos = world.envs.envs[0].unwrapped.task_infos
    num_tasks = len(task_infos)
    print(f"  {num_tasks} predefined tasks: {[t.get('task_name','?') for t in task_infos]}")
    print(f"  num_envs (parallel episodes): {world.num_envs}")

    # ── Pre-render goal images for each task ────────────────────────────────
    # World uses ob_type='states', so world.infos['target'] is a state vector.
    # Open a separate pixel env to get rendered goal images, exactly as
    # eval_ogbench.py does.
    import gymnasium
    print("Pre-rendering OGBench task goal images...")
    goal_env = gymnasium.make(
        "swm/OGBCube-v0", ob_type="pixels", env_type="single", visualize_info=False
    )
    task_goal_images = {}
    for task_id in range(1, num_tasks + 1):
        _, info = goal_env.reset(options={"task_id": task_id})
        task_goal_images[task_id] = np.asarray(info["target"])  # (H, W, C) uint8
        print(f"  task {task_id}: goal image shape {task_goal_images[task_id].shape}")
    goal_env.close()

    # ── Action scaler + process dict ────────────────────────────────────────
    print("Fitting action scaler from dataset...")
    dataset = swm.data.HDF5Dataset(
        args.dataset_path,
        keys_to_cache=["action"],
        cache_dir=str(Path(args.dataset_path).parent),
    )
    process = fit_process(dataset, keys=["action"])
    print(f"  Fit action scaler on {len(dataset):,} steps.")

    transform = {
        "pixels": make_img_transform(args.img_size),
        "goal":   make_img_transform(args.img_size),
    }

    # ── LeWM cost model + CEM policy ────────────────────────────────────────
    print("Loading LeWM cost model via AutoCostModel...")
    model = swm.policy.AutoCostModel(args.ckpt_path).to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    plan_cfg = swm.policy.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
        history_len=1,
        warm_start=True,
    )
    assert plan_cfg.horizon * plan_cfg.action_block <= args.eval_budget * plan_cfg.receding_horizon * plan_cfg.action_block

    solver = CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        n_steps=args.cem_steps,
        topk=args.topk,
        device=device,
        seed=args.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_cfg, process=process, transform=transform
    )
    world.set_policy(policy)

    # ── Results dir ─────────────────────────────────────────────────────────
    exp_name = os.path.basename(args.ckpt_path.rstrip("/")) or "lewm"
    results_dir = Path(args.results_dir or f"eval_cem_ogbench_{exp_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # eval_budget is the total env steps per episode, matching cube.yaml / eval.py
    # convention. WorldModelPolicy internally replans when its action buffer drains
    # (every receding_horizon * action_block env steps).
    env_steps_per_episode = args.eval_budget

    print(f"\n{'='*60}", flush=True)
    print(f"CEM/MPC OGBench evaluation: {exp_name}", flush=True)
    print(f"  protocol: 5 tasks × {args.num_episodes} episodes", flush=True)
    print(f"  planner:  horizon={args.horizon} recede={args.receding_horizon} "
          f"action_block={args.action_block} → {env_steps_per_episode} env steps/ep", flush=True)
    print(f"  CEM:      samples={args.num_samples} iters={args.cem_steps} "
          f"topk={args.topk} var={args.var_scale}", flush=True)
    print(f"{'='*60}", flush=True)

    per_task = {}
    all_successes = []
    t0 = time.time()

    for task_id in range(1, num_tasks + 1):
        task_name = task_infos[task_id - 1].get("task_name", f"task{task_id}")
        print(f"\n─── Task {task_id}: {task_name} ───", flush=True)

        # Reset all num_envs envs with this task_id.
        world.reset(options=[{"task_id": task_id} for _ in range(world.num_envs)])

        # Build goal array shaped (num_envs, history_size=1, H, W, C) for
        # WorldModelPolicy._prepare_info, which expects the same (e, t, ...) layout
        # as the rest of infos. The goal image is the same for every env in this task.
        goal_hwc = task_goal_images[task_id]   # (H, W, C) uint8
        goal = np.broadcast_to(
            goal_hwc[None, None],              # (1, 1, H, W, C)
            (world.num_envs, 1, *goal_hwc.shape),  # (num_envs, 1, H, W, C)
        ).copy()

        # Fresh plan for each task (no warm-start carry-over across tasks).
        reset_policy_state(policy)

        episode_successes = np.zeros(world.num_envs, dtype=bool)

        for step in range(env_steps_per_episode):
            if step % 10 == 0:
                print(f"  step {step}/{env_steps_per_episode}  "
                      f"successes so far: {episode_successes.sum()}", flush=True)
            # The env never emits 'goal', so it's absent from infos after each
            # envs.step(). Inject it before the policy reads infos.
            world.infos["goal"] = goal.copy()
            world.step()

            # Disable auto-reset (mirrors evaluate_from_dataset pattern).
            world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

            # Track success. info['success'] is broadcast into history dim.
            if "success" in world.infos:
                cur = np.asarray(world.infos["success"])
                if cur.ndim > 1:
                    cur = cur[:, -1]
                episode_successes = np.logical_or(episode_successes, cur.astype(bool))

        sr = float(episode_successes.mean())
        n_ok = int(episode_successes.sum())
        print(f"  {task_name}: {sr*100:5.1f}%  ({n_ok}/{world.num_envs})", flush=True)
        per_task[task_name] = sr
        all_successes.extend(episode_successes.tolist())

    overall = float(np.mean(all_successes))
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (CEM/MPC on OGBench 5 tasks): {exp_name}")
    print(f"{'='*60}")
    for name, sr in per_task.items():
        print(f"  {name:<30s} {sr*100:5.1f}%")
    print(f"  {'─'*36}")
    print(f"  {'overall':<30s} {overall*100:5.1f}%")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")

    out = results_dir / "results.txt"
    with out.open("a") as f:
        f.write(f"\n==== CEM {exp_name} ====\n")
        f.write(f"protocol: OGBench native 5-task CEM — {args.num_episodes} episodes/task\n")
        f.write(f"checkpoint: {args.ckpt_path}\n")
        f.write(f"planner: horizon={args.horizon} recede={args.receding_horizon} "
                f"action_block={args.action_block} eval_budget={args.eval_budget}\n")
        f.write(f"cem: samples={args.num_samples} iters={args.cem_steps} "
                f"topk={args.topk} var={args.var_scale}\n")
        for name, sr in per_task.items():
            f.write(f"  {name}: {sr*100:.1f}%\n")
        f.write(f"  overall: {overall*100:.1f}%\n")
        f.write(f"elapsed: {elapsed:.0f}s\n")
    print(f"Results saved to {out}")
    world.close()


if __name__ == "__main__":
    main()
