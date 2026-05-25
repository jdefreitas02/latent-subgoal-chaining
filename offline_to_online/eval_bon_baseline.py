"""Pure best-of-N baseline evaluation (H=0, no world model).

The ACFQL agent's built-in sample_actions with actor_type=best-of-n already
scores N flow-policy proposals by Q(z_0, a_i) and returns the argmax. This
gives us a fair BoN baseline at the same N as the MPC experiments but with
zero WM lookahead.

Usage:
    python eval_bon_baseline.py \
        --policy_ckpt <path> \
        --wm_ckpt <path>        \   # needed only for JEPA pixel encoding + z_goal
        --wm_cache <path>       \
        --bon_n 32              \   # number of action proposals (N=1 = greedy)
        --n_episodes 250 --task_id 1
"""
import argparse
import json
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from envs.wm_env import make_wm_env_and_dataset
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import restore_agent_with_file

# ---------------------------------------------------------------------------
# Agent setup (mirrors eval_mpc.py)
# ---------------------------------------------------------------------------

def _load_agent(policy_ckpt, bon_n, seed=0):
    from agents.acfql import ACFQLAgent
    from agents.acfql_config import get_acfql_config

    ex_obs = np.zeros((1, 192), dtype=np.float32)
    ex_act = np.zeros((1, 25),  dtype=np.float32)

    config = get_acfql_config()
    config["encoder"]           = "jepa_head"
    config["actor_type"]        = "best-of-n"
    config["actor_num_samples"] = bon_n          # <-- controls N at eval time
    config["horizon_length"]    = 1
    config["action_chunking"]   = False

    agent = ACFQLAgent.create(seed, ex_obs, ex_act, config.to_dict())
    agent = restore_agent_with_file(agent, policy_ckpt)
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_ckpt", required=True)
    p.add_argument("--wm_ckpt",     required=True,
                   help="JEPA ckpt used for pixel encoding and z_goal extraction")
    p.add_argument("--wm_cache",    required=True)
    p.add_argument("--bon_n",       type=int, default=32,
                   help="Number of action proposals (N=1 = greedy single sample)")
    p.add_argument("--n_episodes",  type=int, default=250)
    p.add_argument("--task_id",     type=int, default=1)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--out_json",    default=None)
    args = p.parse_args()

    label = f"BoN N={args.bon_n}, H=0 (no WM)"
    print(f"=== {label} ===", flush=True)
    print(f"    policy_ckpt: {args.policy_ckpt}", flush=True)
    print(f"    wm_ckpt:     {args.wm_ckpt}", flush=True)

    # --- Build env + JEPA encoder (same path as eval_mpc.py) ---
    print("=== Building env stack...", flush=True)
    (_, _, _, _, jepa, real_env, _) = make_wm_env_and_dataset(
        wm_ckpt_path=args.wm_ckpt,
        wm_cache_path=args.wm_cache,
        task_id=args.task_id,
        max_episode_steps=40,
        wm_device=args.device,
    )

    # --- Load agent ---
    print(f"=== Loading offline policy (N={args.bon_n})...", flush=True)
    agent = _load_agent(args.policy_ckpt, args.bon_n, seed=args.seed)
    print("  Agent loaded.", flush=True)

    # --- Evaluate ---
    print(f"=== Starting eval: {args.n_episodes} episodes on task {args.task_id}", flush=True)
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

    sr = float(metrics.get(f"task_{args.task_id}/success_rate",
                           metrics.get("overall/success_rate", -1.0)))
    print(f"\n=== RESULT: success_rate={sr:.4f} ({sr*100:.1f}%)", flush=True)

    # --- Save ---
    out_json = args.out_json
    if out_json is None:
        ckpt_dir = os.path.dirname(args.policy_ckpt)
        out_json = os.path.join(ckpt_dir, f"bon_n{args.bon_n}_h0_task{args.task_id}.json")

    result = {
        "bon_n": args.bon_n,
        "mpc_h": 0,
        "label": label,
        "success_rate": sr,
        "task_id": args.task_id,
        "n_episodes": args.n_episodes,
        "policy_ckpt": args.policy_ckpt,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
