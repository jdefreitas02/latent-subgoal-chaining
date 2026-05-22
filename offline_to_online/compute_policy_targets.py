"""Compute policy distillation targets from a trained JAX qc agent.

For each offline transition (z_cached, action_data), evaluates:
  - target_Q = critic(z_cached, action_data)   -- "how good does the policy
    think this action is from this state"
  - target_action = actor(z_cached)            -- "what action does the policy
    prefer from this state" (deterministic, noise=0)

Saves these as .pt for use by finetune_jepa_distill.py.

Usage:
    python compute_policy_targets.py \\
        --ckpt PATH \\
        --cache PATH \\
        --task_id 1 \\
        --out PATH
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import torch

from utils.flax_utils import restore_agent_with_file
from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_dataset_builder import build_for_B2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='JAX policy checkpoint .pkl')
    parser.add_argument('--cache', required=True, help='Latent cache .pt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--encoder', default='jepa_head_deep')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_transitions', type=int, default=None,
                       help='If set, subsample to this many transitions')
    args = parser.parse_args()

    # ---- Build offline dataset (same as B2 path) ----
    print(f"Loading cache from {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    all_latents = cache['all_latents']
    print(f"Cache: {len(all_latents)} episodes, {sum(ep.shape[0] for ep in all_latents)} total frames")
    train_dataset = build_for_B2(all_latents, task_id=args.task_id)
    obs = np.asarray(train_dataset['observations'], dtype=np.float32)  # (N, 192)
    actions = np.asarray(train_dataset['actions'], dtype=np.float32)     # (N, 5)
    N = obs.shape[0]
    print(f"Dataset: N={N}, obs.shape={obs.shape}, actions.shape={actions.shape}")

    if args.max_transitions and args.max_transitions < N:
        idx = np.random.default_rng(0).choice(N, args.max_transitions, replace=False)
        obs = obs[idx]
        actions = actions[idx]
        N = obs.shape[0]
        print(f"Subsampled to N={N}")

    # ---- Build agent identically to main.py for use_jepa_obs ----
    config = get_acfql_config()
    config["encoder"] = args.encoder
    config["action_chunking"] = True
    config["horizon_length"] = 5
    config["actor_type"] = "distill-ddpg"

    # Init batch (single example, matching dataset shape)
    ex_obs = obs[0]            # (192,)
    ex_actions = actions[0]    # (5,)
    print(f"Creating agent with ex_obs={ex_obs.shape}, ex_actions={ex_actions.shape}")
    agent = ACFQLAgent.create(0, ex_obs, ex_actions, config)
    agent = restore_agent_with_file(agent, args.ckpt)
    print(f"Loaded agent from {args.ckpt}")

    # Construct the 25-D actions matrix used by the critic:
    # for each transition t, the critic was trained with action chunk [a_t, a_{t+1}, ..., a_{t+4}]
    # We need that 25-D chunk for each starting transition. Build it.
    print("Building chunk-actions (25-D) for each transition...")
    H = 5
    chunk_actions = np.zeros((N, H * 5), dtype=np.float32)
    for j in range(H):
        # Indices shifted by j, clamped to N-1
        idx = np.minimum(np.arange(N) + j, N - 1)
        chunk_actions[:, j * 5:(j + 1) * 5] = actions[idx]

    # ---- Compute targets in batches ----
    @jax.jit
    def compute_q(obs_batch, action_batch):
        # critic returns shape (num_qs, B); take mean
        qs = agent.network.select('critic')(obs_batch, actions=action_batch)
        return qs.mean(axis=0)  # (B,)

    @jax.jit
    def compute_actor_deterministic(obs_batch):
        # Deterministic action: feed zero noise into the one-step flow
        action_dim_chunked = 25  # action_chunking=True, horizon=5
        noises = jnp.zeros((obs_batch.shape[0], action_dim_chunked), dtype=jnp.float32)
        actions_out = agent.network.select('actor_onestep_flow')(obs_batch, noises)
        actions_out = jnp.clip(actions_out, -1, 1)
        return actions_out  # (B, 25)

    print(f"Computing Q and actor targets in batches of {args.batch_size}...")
    all_q = []
    all_actor = []
    for s in range(0, N, args.batch_size):
        e = min(s + args.batch_size, N)
        obs_b = jnp.asarray(obs[s:e])
        act_b = jnp.asarray(chunk_actions[s:e])
        q_b = compute_q(obs_b, act_b)
        a_b = compute_actor_deterministic(obs_b)
        all_q.append(np.asarray(q_b))
        all_actor.append(np.asarray(a_b))
        if s % (args.batch_size * 100) == 0:
            print(f"  {s}/{N}")

    q_targets = np.concatenate(all_q, axis=0)             # (N,)
    actor_targets = np.concatenate(all_actor, axis=0)     # (N, 25)
    print(f"Q targets: shape={q_targets.shape}, "
          f"mean={q_targets.mean():.3f}, std={q_targets.std():.3f}, "
          f"min={q_targets.min():.3f}, max={q_targets.max():.3f}")
    print(f"Actor targets: shape={actor_targets.shape}, "
          f"mean={actor_targets.mean():.3f}, std={actor_targets.std():.3f}, "
          f"per-dim std mean={actor_targets.std(0).mean():.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'q_targets': torch.from_numpy(q_targets),
        'actor_targets': torch.from_numpy(actor_targets),
        'num_transitions': int(N),
    }, args.out)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
