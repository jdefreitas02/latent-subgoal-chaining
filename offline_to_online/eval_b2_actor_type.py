"""Eval-only script: load a trained B2 (qc + JEPA encoder) checkpoint and
evaluate with a chosen actor_type WITHOUT retraining.

The checkpoint contains weights for both the BC-flow actor and the one-step
distilled actor. The two actor_types use different code paths at action time:
    distill-ddpg: one-step actor (trained by q_loss + distill_loss). May exploit
                  spurious Q-peaks if the encoder is frozen and Q is unreliable.
    best-of-n:    sample N candidates from BC flow, pick highest-Q. Stays inside
                  the dataset action manifold.

If best-of-n recovers performance from the same checkpoint that distill-ddpg
gets ~3% on, the bottleneck is the one-step actor exploiting Q. If both fail,
the issue is upstream (critic / encoder representation).
"""
import argparse
import json
import os
import sys

# Need this BEFORE the heavy imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import numpy as np
import torch  # noqa
import gymnasium as gym
import ml_collections

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from utils.flax_utils import restore_agent_with_file
from utils.datasets import Dataset
from envs.env_utils import EpisodeMonitor
from envs.jepa_loader import load_jepa
from envs.jepa_obs_wrapper import JEPAObsWrapper
from envs.wm_dataset_builder import build_for_B2
from evaluation import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--wm_ckpt_path", default=os.path.expanduser("~/stable_wm_data/cube/lejepa"))
    p.add_argument("--wm_latent_cache", default=os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"))
    p.add_argument("--env_name", default="visual-cube-single-singletask-task1-v0")
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--horizon_length", type=int, default=5)
    p.add_argument("--encoder", default="jepa_head")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+",
                   default=["distill-ddpg", "best-of-n-8", "best-of-n-32", "best-of-n-128"])
    args = p.parse_args()

    print(f"=== Loading JEPA from {args.wm_ckpt_path}", flush=True)
    jepa = load_jepa(args.wm_ckpt_path, device=args.device, img_size=224, patch_size=14)

    def _make_env(seed):
        e = gym.make(args.env_name, width=224, height=224)
        e = EpisodeMonitor(e, filter_regexes=[".*privileged.*", ".*proprio.*"])
        e.reset(seed=seed)
        return e

    print(f"=== Building eval env {args.env_name} @224 with JEPA wrapper", flush=True)
    eval_env = JEPAObsWrapper(_make_env(args.seed + 2), jepa, device=args.device)

    # Build example batch so we know obs/action dims for agent creation
    cache = torch.load(args.wm_latent_cache, map_location="cpu", weights_only=False)
    all_latents = cache["all_latents"] if isinstance(cache, dict) and "all_latents" in cache else cache
    ds_dict = build_for_B2(all_latents, task_id=args.task_id)
    ds = Dataset.create(**ds_dict)
    # Match qc's sample_sequence -> we just need example shapes
    ex_obs = ds_dict["observations"][:1]
    ex_act = ds_dict["actions"][:1]
    print(f"  ex_obs.shape={ex_obs.shape}, ex_act.shape={ex_act.shape}")

    action_dim = ex_act.shape[-1]
    results = {}

    for variant in args.variants:
        # Parse e.g. "best-of-n-32" -> actor_type="best-of-n", num_samples=32
        if variant.startswith("best-of-n-"):
            actor_type = "best-of-n"
            num_samples = int(variant.rsplit("-", 1)[-1])
        else:
            actor_type = variant
            num_samples = 32

        print(f"\n=== Variant: {variant} (actor_type={actor_type}, "
              f"num_samples={num_samples}) ===", flush=True)

        # Build config matching what train was; only change actor_type / num_samples
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

        # During eval, action chunks of length=horizon_length are executed fully.
        # Mirror what main.py does: feed full chunk via env queue.
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

    print("\n=== Summary ===")
    for v, r in results.items():
        print(f"  {v:18s} success={r['success']:.3f}  return={r['return']:8.2f}  length={r['length']:.1f}")

    out = os.path.join(os.path.dirname(args.ckpt), "actor_type_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
