"""Exp D: Interleaved MPC distillation during offline RL training.

Fixes the diversity-collapse failure of post-hoc distillation (Exp C):
  - BC loss preserved throughout → actor retains proposal diversity
  - Q co-evolves with actor → MPC labels improve progressively (curriculum)
  - More training signal: 500k mixed steps vs 100k post-hoc steps

Two ablation axes, run all four combinations:
  --mpc_config cheap   N=16, H=1, K_grad=3   (~35% overhead)
  --mpc_config full    N=32, H=2, K_grad=5   (~100% overhead)

  --mpc_update_mode separate   extra actor-only grad step after each main update
                               (Q never sees MPC actions → cleaner Q signal)
  --mpc_update_mode mixed      replace mpc_mix_ratio of batch actions with MPC
                               labels before the main update
                               (simpler, Q sees some MPC actions)

Usage:
    python train_interleaved_mpc.py \\
        --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --task_id 1 --mpc_config cheap --mpc_update_mode separate \\
        --save_dir ~/stable_wm_data/cube/interleaved_d1/
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
from envs.real_ogbench_eval import evaluate_real_ogbench
from utils.flax_utils import save_agent
from eval_mpc import _load_wm, _make_mpc_fn

# ---------------------------------------------------------------------------
# MPC config presets
# ---------------------------------------------------------------------------

MPC_CONFIGS = {
    "cheap": dict(N=16, H=1, K_grad=3, lr=0.01, gamma=0.99,
                  dense_scale=10.0, q_only=True, q_every_step=False),
    "full":  dict(N=32, H=2, K_grad=5, lr=0.01, gamma=0.99,
                  dense_scale=10.0, q_only=True, q_every_step=False),
}

# ---------------------------------------------------------------------------
# MPC Label Buffer
# ---------------------------------------------------------------------------

class MPCLabelBuffer:
    """Fixed-size buffer of (z_obs, a_mpc) pairs. Replaced entirely each refresh."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.obs = None   # (max_size, 192)
        self.acts = None  # (max_size, 25)
        self.size = 0

    def add(self, obs: np.ndarray, acts: np.ndarray):
        """Replace buffer contents entirely with new (obs, acts) arrays."""
        assert obs.shape[0] == acts.shape[0], "obs/acts must have same leading dim"
        self.obs = obs.astype(np.float32)
        self.acts = acts.astype(np.float32)
        self.size = obs.shape[0]

    def sample(self, n: int):
        """Sample n pairs uniformly at random."""
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=n)
        return self.obs[idxs], self.acts[idxs]

    def is_ready(self, min_size: int = 1) -> bool:
        return self.size >= min_size


# ---------------------------------------------------------------------------
# MPC Relabeling
# ---------------------------------------------------------------------------

def run_mpc_relabeling(agent, wm_model, wm_params, z_goal, obs_all,
                       mpc_config_dict, n_states, seed, verbose=True):
    """Sample n_states latents, run MPC on each, return filled MPCLabelBuffer.

    Args:
        agent: Current ACFQLAgent (updated Q/actor used for MPC scoring).
        wm_model: Frozen LeJEPAJaxForward module.
        wm_params: Frozen WM params dict.
        z_goal: (192,) goal latent as JAX array.
        obs_all: (N_dataset, 192) all latent observations from the dataset.
        mpc_config_dict: Dict with keys N, H, K_grad, lr, gamma, dense_scale, q_only.
        n_states: Number of states to label this round.
        seed: RNG seed for state sampling and MPC.
        verbose: Whether to print progress.
    Returns:
        MPCLabelBuffer filled with (z, a_mpc) pairs.
    """
    cfg = mpc_config_dict
    t0 = time.time()

    # Build mpc_fn with current agent params baked in (will JIT-compile)
    if verbose:
        print(f"  [relabel] Building mpc_fn (N={cfg['N']}, H={cfg['H']}, "
              f"K={cfg['K_grad']})...", flush=True)
    mpc_fn = _make_mpc_fn(
        agent=agent,
        wm_model=wm_model,
        wm_params=wm_params,
        z_goal=z_goal,
        N=cfg["N"],
        H=cfg["H"],
        gamma=cfg["gamma"],
        dense_scale=cfg["dense_scale"],
        K_grad=cfg["K_grad"],
        lr=cfg["lr"],
        q_only=cfg["q_only"],
        q_every_step=cfg["q_every_step"],
    )

    # JIT warm-up (avoids cold-start latency in the labeling loop)
    _dummy_z = jnp.zeros((192,))
    _dummy_rng = jax.random.PRNGKey(seed)
    _ = mpc_fn(observations=_dummy_z, rng=_dummy_rng).block_until_ready()
    if verbose:
        print(f"  [relabel] JIT compiled in {time.time()-t0:.1f}s", flush=True)

    # Sample random states from the dataset
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, obs_all.shape[0], size=n_states)
    states = obs_all[idxs]  # (n_states, 192)

    # Run MPC on each state
    a_mpc = np.zeros((n_states, 25), dtype=np.float32)
    jax_rng = jax.random.PRNGKey(seed + 1)
    t_label = time.time()
    for i, z_i in enumerate(states):
        jax_rng, key = jax.random.split(jax_rng)
        a_mpc[i] = np.asarray(mpc_fn(observations=jnp.asarray(z_i), rng=key))
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - t_label
            eta = elapsed / (i + 1) * (n_states - i - 1)
            print(f"  [relabel] {i+1}/{n_states}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    buf = MPCLabelBuffer(max_size=n_states)
    buf.add(states, a_mpc)
    if verbose:
        print(f"  [relabel] done: {n_states} labels in {time.time()-t0:.0f}s  "
              f"a_mpc stats: mean={a_mpc.mean():.3f} std={a_mpc.std():.3f}", flush=True)
    return buf


# ---------------------------------------------------------------------------
# Actor-only MPC update (separate mode)
# ---------------------------------------------------------------------------

@jax.jit
def _mpc_flow_loss(actor_params, network, z_batch, a_mpc_batch, x_0, t):
    """Flow matching loss on MPC-labeled actions (actor_bc_flow only).

    Loss = ||predict_vel(z, x_t, t) - (a_mpc - x_0)||²

    Args:
        actor_params: 'modules_actor_bc_flow' sub-dict of network params.
        network: Full TrainState (apply_fn used; other params not differentiated).
        z_batch:     (B, 192) latent observations.
        a_mpc_batch: (B, 25)  MPC-selected action chunks (targets).
        x_0:         (B, 25)  noise.
        t:           (B, 1)   flow time in [0, 1].
    Returns:
        scalar loss.
    """
    x_t = (1.0 - t) * x_0 + t * a_mpc_batch   # (B, 25)
    vel = a_mpc_batch - x_0                      # (B, 25)
    full_params = {**network.params, "modules_actor_bc_flow": actor_params}
    pred = network.apply_fn(
        {"params": full_params},
        z_batch, x_t, t,
        name="actor_bc_flow",
    )  # (B, 25)
    return jnp.mean((pred - vel) ** 2)


def make_actor_optimizer(lr: float, weight_decay: float = 1e-4):
    return optax.adamw(learning_rate=lr, weight_decay=weight_decay)


def actor_only_update_step(agent, mpc_buffer, batch_size, actor_tx, actor_opt_state,
                           rng: jax.random.PRNGKey):
    """One gradient step on actor_bc_flow using MPC-labeled batch. Critic frozen.

    Args:
        agent: Current ACFQLAgent.
        mpc_buffer: MPCLabelBuffer with (z, a_mpc) pairs.
        batch_size: Mini-batch size.
        actor_tx: optax optimizer for actor_bc_flow.
        actor_opt_state: Current optimizer state.
        rng: JAX RNG key.
    Returns:
        (agent_updated, new_opt_state, float loss)
    """
    actor_params = agent.network.params["modules_actor_bc_flow"]

    z_b, a_b = mpc_buffer.sample(batch_size)
    z_b = jnp.asarray(z_b)       # (B, 192)
    a_b = jnp.asarray(a_b)       # (B, 25)

    rng, noise_key, t_key = jax.random.split(rng, 3)
    x_0 = jax.random.normal(noise_key, (batch_size, 25))
    t   = jax.random.uniform(t_key, (batch_size, 1))

    grad_fn = jax.value_and_grad(_mpc_flow_loss)
    loss, grads = grad_fn(actor_params, agent.network, z_b, a_b, x_0, t)

    updates, new_opt_state = actor_tx.update(grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)

    # Inject updated actor_bc_flow params back into the agent
    new_network_params = {**agent.network.params,
                          "modules_actor_bc_flow": new_actor_params}
    new_network = agent.network.replace(params=new_network_params)
    agent = agent.replace(network=new_network)

    return agent, new_opt_state, float(loss)


# ---------------------------------------------------------------------------
# Batch injection (mixed mode)
# ---------------------------------------------------------------------------

def inject_mpc_into_batch(batch, mpc_buffer, mix_ratio: float, rng: np.random.Generator):
    """Replace mix_ratio fraction of batch actions with MPC-labeled actions.

    batch['actions'] shape: (B, seq_len, action_dim) with seq_len=1.
    Replaces first n_mpc rows' actions with MPC-labeled ones.
    Observations are also replaced so the actor_loss BC targets match.

    Args:
        batch: Dict from sample_sequence (modified in place).
        mpc_buffer: MPCLabelBuffer with (z, a_mpc) pairs.
        mix_ratio: Fraction of batch to replace (e.g. 0.3).
        rng: NumPy random generator for shuffling.
    Returns:
        Modified batch dict.
    """
    B = batch["observations"].shape[0]
    n_mpc = max(1, int(B * mix_ratio))
    z_mpc, a_mpc = mpc_buffer.sample(n_mpc)     # (n_mpc, 192), (n_mpc, 25)

    # Replace observations and first-sequence-slot actions
    batch["observations"][:n_mpc] = z_mpc
    # batch["actions"] shape: (B, 1, 25) — replace first seq slot
    batch["actions"][:n_mpc, 0, :] = a_mpc
    # Also keep valid mask consistent (MPC-labeled entries are always valid)
    batch["valid"][:n_mpc] = 1.0
    return batch


# ---------------------------------------------------------------------------
# Simple CSV Logger
# ---------------------------------------------------------------------------

class CsvLogger:
    def __init__(self, path):
        self.path = path
        self._header_written = False

    def log(self, data: dict, step: int):
        row = {"step": step, **{k: float(v) for k, v in data.items()}}
        import csv
        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Exp D: Interleaved MPC distillation")
    # Environment / dataset
    p.add_argument("--wm_ckpt", required=True,
                   help="WM checkpoint path (lejepa_play_ft_full base dir)")
    p.add_argument("--wm_cache", required=True,
                   help="Latent cache .pt path")
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"),
                   help="HDF5 play dataset path (no .h5 extension)")
    p.add_argument("--task_id", type=int, default=1,
                   help="OGBench cube-single task id (1..5)")
    p.add_argument("--wm_device", default="cuda",
                   help="Device for JEPA PyTorch model")
    # Training
    p.add_argument("--offline_steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate for main Q+actor optimizer")
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=100000)
    p.add_argument("--eval_episodes", type=int, default=250)
    p.add_argument("--save_dir", required=True)
    # MPC distillation
    p.add_argument("--mpc_config", choices=["cheap", "full"], default="cheap",
                   help="MPC config preset: cheap=N16/H1/K3, full=N32/H2/K5")
    p.add_argument("--mpc_update_mode", choices=["separate", "mixed"], default="separate",
                   help="How to integrate MPC labels: separate actor update or mixed batch")
    p.add_argument("--mpc_warmup_steps", type=int, default=150000,
                   help="Steps of normal training before MPC labeling starts")
    p.add_argument("--mpc_relabel_interval", type=int, default=25000,
                   help="Refresh MPC label buffer every N training steps")
    p.add_argument("--mpc_buffer_size", type=int, default=2000,
                   help="Number of (z, a_mpc) pairs per refresh")
    p.add_argument("--mpc_mix_ratio", type=float, default=0.3,
                   help="Fraction of batch replaced with MPC labels (mixed mode only)")
    p.add_argument("--mpc_actor_lr", type=float, default=1e-4,
                   help="Learning rate for the separate actor-only optimizer (separate mode)")
    p.add_argument("--mpc_actor_wd", type=float, default=1e-4,
                   help="Weight decay for the actor-only optimizer (separate mode)")
    p.add_argument("--mpc_actor_steps_every", type=int, default=50,
                   help="Do 1 actor-only MPC step every N main training steps. "
                        "Controls the BC:MPC update ratio. Default 50 → ~128 epochs "
                        "per 2k-state buffer refresh (50k steps / 50 × 256/2000).")
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset + real env
    # ------------------------------------------------------------------
    print("=== Loading WM env and dataset...", flush=True)
    (_, _, train_dataset_dict, _, jepa_model,
     real_env, z_goal_task) = make_wm_env_and_dataset(
        wm_ckpt_path=args.wm_ckpt,
        latent_cache_path=args.wm_cache,
        hdf5_dataset_path=args.wm_hdf5,
        task_id=args.task_id,
        done_threshold=2.0,
        max_episode_steps=40,
        wm_device=args.wm_device,
        img_size=224,
    )
    from utils.datasets import Dataset
    train_dataset = Dataset.create(**train_dataset_dict)

    obs_all  = np.asarray(train_dataset["observations"])   # (N, 192)
    print(f"  Dataset size: {obs_all.shape[0]} transitions", flush=True)

    # ------------------------------------------------------------------
    # 2. Load frozen JAX WM
    # ------------------------------------------------------------------
    print("=== Loading JAX WM...", flush=True)
    wm_model, wm_params = _load_wm(args.wm_ckpt)
    z_goal = jnp.asarray(z_goal_task.astype(np.float32))  # (192,)
    print("  WM loaded.", flush=True)

    # ------------------------------------------------------------------
    # 3. Create agent (same config as original E pipeline training)
    # ------------------------------------------------------------------
    print("=== Creating agent...", flush=True)
    example_batch = train_dataset.sample(1)
    ex_obs = example_batch["observations"][:1]   # (1, 192)
    ex_act = example_batch["actions"][:1]        # (1, 25)
    if ex_act.ndim == 3:
        ex_act = ex_act[:, 0, :]                 # (1, 25) if shape was (1, 1, 25)

    config = get_acfql_config()
    config["encoder"]            = "jepa_head"
    config["actor_type"]         = "best-of-n"
    config["actor_num_samples"]  = 4
    config["horizon_length"]     = 1
    config["action_chunking"]    = False
    config["lr"]                 = args.lr
    config["discount"]           = args.discount

    agent = ACFQLAgent.create(
        seed=args.seed,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=config.to_dict() if hasattr(config, "to_dict") else dict(config),
    )
    print("  Agent created.", flush=True)

    # ------------------------------------------------------------------
    # 4. Set up actor-only optimizer (separate mode)
    # ------------------------------------------------------------------
    actor_tx = make_actor_optimizer(lr=args.mpc_actor_lr, weight_decay=args.mpc_actor_wd)
    actor_opt_state = actor_tx.init(
        agent.network.params["modules_actor_bc_flow"]
    )

    # ------------------------------------------------------------------
    # 5. Logging
    # ------------------------------------------------------------------
    logger_train = CsvLogger(os.path.join(args.save_dir, "train.csv"))
    logger_eval  = CsvLogger(os.path.join(args.save_dir, "eval.csv"))

    mpc_cfg = MPC_CONFIGS[args.mpc_config]
    mpc_buffer = None
    jax_rng = jax.random.PRNGKey(args.seed + 100)

    print(f"\n=== Starting training: {args.offline_steps} steps ===", flush=True)
    print(f"    mpc_config={args.mpc_config}  mpc_update_mode={args.mpc_update_mode}", flush=True)
    print(f"    warmup={args.mpc_warmup_steps}  relabel_interval={args.mpc_relabel_interval}", flush=True)
    print(f"    buffer_size={args.mpc_buffer_size}  mix_ratio={args.mpc_mix_ratio}", flush=True)
    t_start = time.time()

    for step in range(1, args.offline_steps + 1):

        # ---- MPC relabeling (after warmup, every mpc_relabel_interval steps) ----
        if (step >= args.mpc_warmup_steps and
                step % args.mpc_relabel_interval == 0):
            print(f"\n[step {step}] Starting MPC relabeling...", flush=True)
            mpc_buffer = run_mpc_relabeling(
                agent=agent,
                wm_model=wm_model,
                wm_params=wm_params,
                z_goal=z_goal,
                obs_all=obs_all,
                mpc_config_dict=mpc_cfg,
                n_states=args.mpc_buffer_size,
                seed=args.seed + step,
                verbose=True,
            )
            # Re-init actor optimizer so its momentum doesn't fight MPC labels
            actor_opt_state = actor_tx.init(
                agent.network.params["modules_actor_bc_flow"]
            )

        # ---- Sample main training batch ----
        batch = train_dataset.sample_sequence(
            args.batch_size, sequence_length=1, discount=args.discount
        )

        # ---- Mixed mode: inject MPC labels before main update ----
        if (args.mpc_update_mode == "mixed" and
                mpc_buffer is not None and mpc_buffer.is_ready(args.batch_size)):
            batch = inject_mpc_into_batch(batch, mpc_buffer, args.mpc_mix_ratio,
                                          np.random.default_rng(step))

        # ---- Main update: Q + actor on demo batch ----
        agent, info = agent.update(batch)

        # ---- Separate mode: actor-only MPC gradient step ----
        # One MPC update every mpc_actor_steps_every main steps to control
        # the BC:MPC ratio. Default every=50 → ~128 epochs per buffer refresh,
        # which matches post-hoc distillation convergence without overfitting.
        mpc_loss = float("nan")
        if (args.mpc_update_mode == "separate" and
                mpc_buffer is not None and mpc_buffer.is_ready(args.batch_size) and
                step % args.mpc_actor_steps_every == 0):
            jax_rng, key = jax.random.split(jax_rng)
            agent, actor_opt_state, mpc_loss = actor_only_update_step(
                agent, mpc_buffer, args.batch_size, actor_tx, actor_opt_state, key
            )
            info["mpc_flow_loss"] = mpc_loss

        # ---- Logging ----
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            log_data = {k: float(v) for k, v in info.items()}
            log_data["mpc_flow_loss"] = mpc_loss
            log_data["elapsed_s"] = elapsed
            logger_train.log(log_data, step=step)
            print(
                f"  step {step:7d}/{args.offline_steps}  "
                f"critic={info.get('critic/critic_loss', float('nan')):.4f}  "
                f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                f"mpc_flow={mpc_loss:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True
            )

        # ---- Periodic eval ----
        if step % args.eval_interval == 0 or step == args.offline_steps:
            print(f"\n[step {step}] Evaluating...", flush=True)
            from envs.real_ogbench_eval import evaluate_real_ogbench
            metrics = evaluate_real_ogbench(
                agent=agent,
                real_env=real_env,
                jepa_model=jepa_model,
                device=args.wm_device,
                task_ids=(args.task_id,),
                num_episodes_per_task=args.eval_episodes,
                action_dispatch="chunk25",
                pass_task_id_on_reset=False,
            )
            sr = metrics.get(f"task_{args.task_id}/success_rate",
                             metrics.get("overall/success_rate", float("nan")))
            print(f"  [eval] success_rate = {sr:.4f} ({sr*100:.1f}%)", flush=True)
            logger_eval.log(metrics, step=step)

    # ------------------------------------------------------------------
    # 6. Save checkpoint
    # ------------------------------------------------------------------
    save_agent(agent, args.save_dir, args.offline_steps)
    print(f"\n=== Training done. Checkpoint: {args.save_dir}/params_{args.offline_steps}.pkl",
          flush=True)
    print(f"\nTo evaluate with MPC on top of the trained actor:")
    print(f"  python eval_mpc.py \\")
    print(f"    --policy_ckpt {args.save_dir}/params_{args.offline_steps}.pkl \\")
    print(f"    --wm_ckpt {args.wm_ckpt} \\")
    print(f"    --wm_cache {args.wm_cache} \\")
    print(f"    --mpc_n 32 --mpc_h 2 --mpc_k_grad 5 --mpc_q_only \\")
    print(f"    --n_episodes 250 --task_id {args.task_id}")


if __name__ == "__main__":
    main()
