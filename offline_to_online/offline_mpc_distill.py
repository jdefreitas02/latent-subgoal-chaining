"""Offline MPC Distillation (Exp C).

Phase A — Relabeling:
    For each z_0 in the offline latent dataset, run MPC to get a*_mpc (N=32, H=2, K=5, Q-only).
    This creates a new dataset of (z_0, a*_mpc) pairs — actions selected by MPC rather than
    by the human demonstrator.

Phase B — Distillation:
    Fine-tune only the BC flow actor on (z_0, a*_mpc) for N steps using flow matching loss.
    Critic and Q-network are frozen throughout.
    Result: a flow actor that imitates MPC (better action proposals when scoring under Q).

Phase C — Evaluation:
    Evaluate the distilled agent with standard BoN (no WM needed after distillation).
    Can optionally also run with MPC on top of the distilled actor.

Usage:
    python offline_mpc_distill.py \\
        --policy_ckpt exp/qc/e_qc_jepa_wm_ftfull/.../offline_final/params_500000.pkl \\
        --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --out_dir ~/stable_wm_data/cube/distilled_policy/ \\
        --mpc_n 32 --mpc_h 2 --mpc_k_grad 5 \\
        --finetune_steps 100000 --finetune_lr 1e-4
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_env import make_wm_env_and_dataset
from utils.flax_utils import restore_agent_with_file, save_agent
from eval_mpc import _load_wm, _make_mpc_fn


# ---------------------------------------------------------------------------
# Phase A: Relabeling — run MPC on all latent dataset states
# ---------------------------------------------------------------------------

def relabel_dataset(obs_all, mpc_fn, batch_size=64, seed=0):
    """Run MPC on every z_0 in the dataset to collect a*_mpc labels.

    Args:
        obs_all: (N, 192) all latent observations.
        mpc_fn: jitted fn(observations, rng) -> (25,) best action.
        batch_size: States processed per RNG key split (purely for progress display).
        seed: RNG seed.

    Returns:
        a_mpc: (N, 25) MPC-selected action chunks.
    """
    N_data = obs_all.shape[0]
    a_mpc = np.zeros((N_data, 25), dtype=np.float32)
    rng = jax.random.PRNGKey(seed)

    t0 = time.time()
    for i in range(0, N_data, batch_size):
        end = min(i + batch_size, N_data)
        for j in range(i, end):
            rng, key = jax.random.split(rng)
            z_j = jnp.asarray(obs_all[j])  # (192,)
            a_j = mpc_fn(observations=z_j, rng=key)
            a_mpc[j] = np.asarray(a_j)

        if (i // batch_size) % 20 == 0 or end == N_data:
            elapsed = time.time() - t0
            pct = 100.0 * end / N_data
            eta = elapsed / (end / N_data) - elapsed if end > 0 else 0
            print(f"  Relabeling: {end}/{N_data} ({pct:.1f}%)  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"  Relabeling done: {N_data} transitions in {time.time()-t0:.0f}s", flush=True)
    return a_mpc


# ---------------------------------------------------------------------------
# Phase B: Distillation — fine-tune BC flow actor on (z_0, a*_mpc)
# ---------------------------------------------------------------------------

@jax.jit
def _bc_flow_loss_fn(actor_params, network, z_batch, a_mpc_batch, x_0, t):
    """Flow matching BC loss on MPC-labelled actions.

    Loss = ||predict_vel(z, x_t, t) - (a_mpc - x_0)||²

    Args:
        actor_params: dict of params for the 'actor_bc_flow' module.
        network: the full TrainState (apply_fn used, other params NOT differentiated).
        z_batch:     (B, 192) current latent observations.
        a_mpc_batch: (B, 25)  MPC-selected action chunks (targets).
        x_0:         (B, 25)  noise samples.
        t:           (B, 1)   interpolation times in [0,1].

    Returns:
        scalar loss.
    """
    x_t = (1.0 - t) * x_0 + t * a_mpc_batch   # (B, 25)
    vel = a_mpc_batch - x_0                      # (B, 25) target velocity

    # Merge actor_params into the full network params so we can call select().
    # We only differentiate actor_params; the rest are constants captured by closure.
    full_params = {**network.params, 'modules_actor_bc_flow': actor_params}

    # Forward pass through the flow network.
    pred = network.apply_fn(
        {'params': full_params},
        z_batch, x_t, t,
        name='actor_bc_flow',
    )  # (B, 25)

    return jnp.mean((pred - vel) ** 2)


def finetune_actor(agent, obs_all, a_mpc, finetune_steps, lr, weight_decay,
                   batch_size, log_interval, seed):
    """Fine-tune only actor_bc_flow params on (z_0, a*_mpc) pairs.

    Args:
        agent: ACFQLAgent (full network; critic frozen throughout).
        obs_all: (N, 192) latent observations.
        a_mpc: (N, 25) MPC-selected action chunks.
        finetune_steps: Number of gradient steps.
        lr, weight_decay: AdamW hyperparams.
        batch_size: Mini-batch size.
        log_interval: Steps between log prints.
        seed: RNG seed.

    Returns:
        agent with updated actor_bc_flow params.
    """
    N_data = obs_all.shape[0]

    # Extract only the actor_bc_flow params (what we're fine-tuning).
    actor_params = agent.network.params['modules_actor_bc_flow']

    # Create a separate optimizer for actor_bc_flow only.
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = tx.init(actor_params)

    grad_fn = jax.jit(jax.value_and_grad(_bc_flow_loss_fn))

    rng = jax.random.PRNGKey(seed + 1)
    t0 = time.time()

    for step in range(1, finetune_steps + 1):
        rng, idx_key, noise_key, t_key = jax.random.split(rng, 4)

        # Sample mini-batch
        idx = np.random.randint(0, N_data, size=batch_size)
        z_b = jnp.asarray(obs_all[idx])    # (B, 192)
        a_b = jnp.asarray(a_mpc[idx])      # (B, 25)
        x_0 = jax.random.normal(noise_key, (batch_size, 25))
        t   = jax.random.uniform(t_key, (batch_size, 1))

        loss, grads = grad_fn(actor_params, agent.network, z_b, a_b, x_0, t)
        updates, opt_state = tx.update(grads, opt_state, actor_params)
        actor_params = optax.apply_updates(actor_params, updates)

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:6d}/{finetune_steps}  "
                  f"bc_flow_loss={float(loss):.6f}  ({elapsed:.0f}s)", flush=True)

    print(f"  Fine-tuning done in {time.time()-t0:.0f}s", flush=True)

    # Inject updated actor_bc_flow params back into the full network.
    new_network_params = {**agent.network.params,
                          'modules_actor_bc_flow': actor_params}
    new_network = agent.network.replace(params=new_network_params)
    agent = agent.replace(network=new_network)
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Offline MPC Distillation of BC flow actor")
    p.add_argument("--policy_ckpt", required=True,
                   help="Path to offline_final/params_*.pkl checkpoint")
    p.add_argument("--wm_ckpt", required=True,
                   help="WM predictor ckpt for MPC rollout (JEPA ckpt dir or VAML .pkl)")
    p.add_argument("--wm_cache", required=True,
                   help="Latent cache .pt path (e.g. lewm_224_latents_cache_ftfull.pt)")
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"),
                   help="HDF5 play dataset path")
    p.add_argument("--out_dir", required=True,
                   help="Directory to save the distilled agent checkpoint")
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    # MPC hyper-parameters (for relabeling)
    p.add_argument("--mpc_n", type=int, default=32)
    p.add_argument("--mpc_h", type=int, default=2)
    p.add_argument("--mpc_k_grad", type=int, default=5)
    p.add_argument("--mpc_lr", type=float, default=0.01)
    p.add_argument("--mpc_gamma", type=float, default=0.99)
    p.add_argument("--mpc_dense_scale", type=float, default=10.0)
    # Fine-tuning hyper-parameters
    p.add_argument("--finetune_steps", type=int, default=100000)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--finetune_weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_interval", type=int, default=1000)
    # Skip relabeling if a saved a_mpc.npy already exists
    p.add_argument("--skip_relabel", action="store_true",
                   help="Skip Phase A and load a_mpc.npy from --out_dir directly")
    args = p.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    a_mpc_path = os.path.join(args.out_dir, "a_mpc_labels.npy")

    # ------------------------------------------------------------------
    # 1. Load latent dataset (offline cache)
    # ------------------------------------------------------------------
    print("=== Loading offline latent dataset...", flush=True)
    (_, _, train_dataset, _, jepa, real_env, z_goal) = make_wm_env_and_dataset(
        wm_ckpt_path=args.wm_ckpt,
        latent_cache_path=args.wm_cache,
        hdf5_dataset_path=args.wm_hdf5,
        task_id=args.task_id,
        done_threshold=2.0,
        max_episode_steps=40,
        wm_device=args.device,
        img_size=224,
    )

    obs_all  = np.asarray(train_dataset["observations"])     # (N, 192)
    act_all  = np.asarray(train_dataset["actions"])          # (N, 25)
    if obs_all.ndim == 3:
        obs_all = obs_all[:, -1, :]
    if act_all.ndim == 3:
        act_all = act_all[:, 0, :]

    N_data = obs_all.shape[0]
    print(f"  Dataset: {N_data} transitions, obs={obs_all.shape}, act={act_all.shape}",
          flush=True)

    ex_obs = obs_all[:1]
    ex_act = act_all[:1]

    # ------------------------------------------------------------------
    # 2. Load agent
    # ------------------------------------------------------------------
    print("=== Loading offline policy agent...", flush=True)
    config = get_acfql_config()
    config["encoder"] = "jepa_head"
    config["actor_type"] = "best-of-n"
    config["actor_num_samples"] = 4
    config["horizon_length"] = 1
    config["action_chunking"] = False

    agent = ACFQLAgent.create(
        seed=args.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
    )
    agent = restore_agent_with_file(agent, args.policy_ckpt)
    print("  Agent loaded.", flush=True)

    # ------------------------------------------------------------------
    # Phase A: Relabeling
    # ------------------------------------------------------------------
    if args.skip_relabel and os.path.exists(a_mpc_path):
        print(f"=== Skipping Phase A — loading a_mpc from {a_mpc_path}", flush=True)
        a_mpc = np.load(a_mpc_path)
        print(f"  Loaded a_mpc: {a_mpc.shape}", flush=True)
    else:
        print("=== Phase A: Relabeling dataset with MPC actions...", flush=True)
        wm_model, wm_params = _load_wm(args.wm_ckpt)
        z_goal_jax = jnp.asarray(z_goal.astype(np.float32))

        mpc_fn = _make_mpc_fn(
            agent=agent,
            wm_model=wm_model,
            wm_params=wm_params,
            z_goal=z_goal_jax,
            N=args.mpc_n,
            H=args.mpc_h,
            gamma=args.mpc_gamma,
            dense_scale=args.mpc_dense_scale,
            K_grad=args.mpc_k_grad,
            lr=args.mpc_lr,
            q_only=True,
            q_every_step=False,
        )

        # JIT warm-up
        print("  JIT warm-up...", flush=True)
        _dummy_z = jnp.zeros((192,))
        _dummy_rng = jax.random.PRNGKey(0)
        mpc_fn(observations=_dummy_z, rng=_dummy_rng).block_until_ready()
        print("  JIT compiled.", flush=True)

        a_mpc = relabel_dataset(obs_all, mpc_fn, batch_size=64, seed=args.seed)

        np.save(a_mpc_path, a_mpc)
        print(f"  Saved a_mpc labels to {a_mpc_path}", flush=True)

    print(f"  a_mpc stats: mean={a_mpc.mean():.4f} std={a_mpc.std():.4f} "
          f"min={a_mpc.min():.4f} max={a_mpc.max():.4f}", flush=True)

    # ------------------------------------------------------------------
    # Phase B: Fine-tune BC flow actor
    # ------------------------------------------------------------------
    print(f"\n=== Phase B: Fine-tuning BC flow actor for {args.finetune_steps} steps...",
          flush=True)
    print(f"    lr={args.finetune_lr}  wd={args.finetune_weight_decay}  "
          f"batch={args.batch_size}", flush=True)

    agent = finetune_actor(
        agent=agent,
        obs_all=obs_all,
        a_mpc=a_mpc,
        finetune_steps=args.finetune_steps,
        lr=args.finetune_lr,
        weight_decay=args.finetune_weight_decay,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Save distilled agent
    # ------------------------------------------------------------------
    save_epoch = args.finetune_steps
    save_agent(agent, args.out_dir, save_epoch)
    print(f"\n=== Distilled agent saved to {args.out_dir}/params_{save_epoch}.pkl",
          flush=True)
    print(f"\nTo evaluate, run:")
    print(f"  python eval_mpc.py \\")
    print(f"    --policy_ckpt {args.out_dir}/params_{save_epoch}.pkl \\")
    print(f"    --wm_ckpt {args.wm_ckpt} \\")
    print(f"    --wm_cache {args.wm_cache} \\")
    print(f"    --mpc_n 32 --mpc_h 2 --mpc_k_grad 5 --mpc_q_only \\")
    print(f"    --n_episodes 250 --task_id {args.task_id}")
    print(f"\nOr for plain BoN eval (no WM needed):")
    print(f"  python eval_offline.py --policy_ckpt {args.out_dir}/params_{save_epoch}.pkl \\")
    print(f"    --actor_type best-of-n --actor_num_samples 32 --n_episodes 250")


if __name__ == "__main__":
    main()
