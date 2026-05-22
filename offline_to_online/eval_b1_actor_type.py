"""B1 eval-only: load a trained B1 (qc + IMPALA encoder, raw 64x64 pixels)
checkpoint and evaluate with multiple actor_type / N values without retraining.

Same pattern as eval_b2_actor_type.py but for B1's env path (no JEPA wrapper).
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch  # noqa

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from utils.flax_utils import restore_agent_with_file
from envs.env_utils import make_env_and_datasets
from evaluation import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env_name", default="visual-cube-single-play-singletask-task1-v0")
    p.add_argument("--horizon_length", type=int, default=5)
    p.add_argument("--encoder", default="impala_small")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+",
                   default=["distill-ddpg", "best-of-n-8", "best-of-n-32", "best-of-n-128"])
    args = p.parse_args()

    print(f"=== Building env {args.env_name} (B1 path)", flush=True)
    env, eval_env, train_dataset, _ = make_env_and_datasets(args.env_name)
    ex_obs = train_dataset["observations"][:1]
    ex_act = train_dataset["actions"][:1]
    print(f"  ex_obs.shape={ex_obs.shape}, ex_act.shape={ex_act.shape}")

    action_dim = ex_act.shape[-1]
    results = {}

    for variant in args.variants:
        if variant.startswith("best-of-n-"):
            actor_type = "best-of-n"
            num_samples = int(variant.rsplit("-", 1)[-1])
        else:
            actor_type = variant
            num_samples = 32

        print(f"\n=== Variant: {variant} (actor_type={actor_type}, "
              f"num_samples={num_samples}) ===", flush=True)

        config = get_acfql_config()
        config["actor_type"] = actor_type
        config["actor_num_samples"] = num_samples
        config["horizon_length"] = args.horizon_length
        config["encoder"] = args.encoder
        config["action_chunking"] = True

        agent = ACFQLAgent.create(
            seed=args.seed,
            ex_observations=ex_obs,
            ex_actions=ex_act,
            config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        )
        agent = restore_agent_with_file(agent, args.ckpt)

        eval_info, _, _ = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=action_dim,
            num_eval_episodes=args.n_episodes,
            num_video_episodes=0,
            video_frame_skip=1,
            max_episode_steps=args.max_episode_steps,
        )
        succ = float(eval_info.get("success", -1.0))
        ret  = float(eval_info.get("episode.return", float("nan")))
        elen = float(eval_info.get("episode.length", float("nan")))
        results[variant] = {"success": succ, "return": ret, "length": elen}
        print(f"  >>> success={succ:.3f}  return={ret:.2f}  length={elen:.1f}", flush=True)

    print("\n=== Summary (B1) ===")
    for v, r in results.items():
        print(f"  {v:18s} success={r['success']:.3f}  return={r['return']:8.2f}  length={r['length']:.1f}")

    out = os.path.join(os.path.dirname(args.ckpt), "actor_type_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
