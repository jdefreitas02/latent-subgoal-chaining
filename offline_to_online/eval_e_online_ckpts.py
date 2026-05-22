"""Post-hoc real-env eval for E's online_final checkpoints.

The 4 threshold-sweep jobs (55278-55281) saved their online_final/params_*.pkl
but failed to write online_only_eval.json because main.py's logger didn't
have an 'online_only' csv prefix registered. This script does the missing
eval against the real OGBench env directly, without re-training.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch  # noqa
import ml_collections

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from utils.flax_utils import restore_agent_with_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="List of online_final/params_*.pkl paths to eval")
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--n_episodes", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wm_ckpt", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--wm_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--wm_hdf5", default=os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from envs.wm_env import make_wm_env_and_dataset
    from envs.real_ogbench_eval import evaluate_real_ogbench

    print(f"=== Building E env (loads JEPA + real env for eval)", flush=True)
    train_env, eval_env, train_dataset, val_dataset, jepa, real_env, z_goal = \
        make_wm_env_and_dataset(
            wm_ckpt_path=args.wm_ckpt,
            latent_cache_path=args.wm_cache,
            hdf5_dataset_path=args.wm_hdf5,
            task_id=args.task_id,
            done_threshold=2.0,             # irrelevant for real-env eval
            max_episode_steps=40,
            wm_device=args.device,
            img_size=224,
        )

    ex_obs = train_dataset["observations"][:1]
    ex_act = train_dataset["actions"][:1]
    print(f"  ex_obs.shape={ex_obs.shape}, ex_act.shape={ex_act.shape}", flush=True)

    # Match training config: jepa_head encoder, best-of-n actor, horizon=1,
    # action_chunking=False (so action_dim=25, not chunked-from-5).
    config = get_acfql_config()
    config["encoder"] = "jepa_head"
    config["actor_type"] = "best-of-n"
    config["actor_num_samples"] = 4
    config["horizon_length"] = 1
    config["action_chunking"] = False

    results = {}
    for ckpt in args.ckpts:
        tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        print(f"\n=== Evaluating {tag} :: {ckpt}", flush=True)

        agent = ACFQLAgent.create(
            seed=args.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        )
        agent = restore_agent_with_file(agent, ckpt)

        metrics = evaluate_real_ogbench(
            agent=agent,
            real_env=real_env,
            jepa_model=jepa,
            device=args.device,
            task_ids=(args.task_id,),
            num_episodes_per_task=args.n_episodes,
            action_dispatch="chunk25",
            pass_task_id_on_reset=False,
        )
        succ = float(metrics.get(f"task_{args.task_id}/success_rate",
                                 metrics.get("overall/success_rate", -1.0)))
        results[tag] = {"success_rate": succ, "all_metrics": {k: float(v) for k, v in metrics.items()}}
        print(f"  >>> success_rate={succ:.4f}", flush=True)

        # Also write the json next to the checkpoint
        out = os.path.join(os.path.dirname(ckpt), "online_only_eval.json")
        with open(out, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print(f"  -> {out}", flush=True)

    print("\n=== Summary ===")
    for tag, r in results.items():
        print(f"  {tag:60s}  success={r['success_rate']:.4f}")


if __name__ == "__main__":
    main()
