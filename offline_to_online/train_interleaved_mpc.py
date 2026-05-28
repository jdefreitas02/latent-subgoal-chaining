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
    """Fixed-size buffer of (z_obs, a_mpc) pairs. Replaced entirely each refresh.

    Optionally stores pre-computed advantages so mixed_awr injection is a cheap
    numpy swap rather than re-scoring every training step.

    Optionally stores WM-consistent next observations and rewards so the Q
    receives a consistent Bellman target (z, a_mpc, r_wm, z'_wm) rather than
    the mismatched offline (r, z') from an unrelated transition.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.obs = None       # (max_size, 192)
        self.acts = None      # (max_size, 25)
        self.advantages = None  # (max_size,) or None
        self.next_obs = None  # (max_size, 192) WM-predicted z' or None
        self.rewards = None   # (max_size,)     WM-computed r  or None
        self.size = 0

    def add(self, obs: np.ndarray, acts: np.ndarray,
            advantages: np.ndarray = None,
            next_obs: np.ndarray = None,
            rewards: np.ndarray = None):
        """Replace buffer contents entirely with new arrays."""
        assert obs.shape[0] == acts.shape[0], "obs/acts must have same leading dim"
        self.obs = obs.astype(np.float32)
        self.acts = acts.astype(np.float32)
        self.advantages = advantages.astype(np.float32) if advantages is not None else None
        self.next_obs = next_obs.astype(np.float32) if next_obs is not None else None
        self.rewards = rewards.astype(np.float32) if rewards is not None else None
        self.size = obs.shape[0]

    def has_wm_targets(self) -> bool:
        return self.next_obs is not None and self.rewards is not None

    def sample(self, n: int):
        """Sample n pairs uniformly at random.

        Returns:
            (obs, acts, next_obs, rewards) — next_obs/rewards are None if not stored.
        """
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=n)
        next_obs = self.next_obs[idxs] if self.next_obs is not None else None
        rewards  = self.rewards[idxs]  if self.rewards  is not None else None
        return self.obs[idxs], self.acts[idxs], next_obs, rewards

    def sample_top_advantage(self, n: int, threshold: float = 0.0):
        """Return up to n pairs with highest pre-computed advantage above threshold.

        Pairs are already sorted descending by advantage (set during add()).
        Returns (obs, acts, next_obs, rewards, n_returned, mean_advantage_of_returned).
        next_obs/rewards are None if not stored.
        """
        assert self.size > 0, "Buffer is empty"
        assert self.advantages is not None, "No advantages stored — call add() with advantages"
        passing = np.where(self.advantages > threshold)[0]
        n_take = min(len(passing), n)
        if n_take == 0:
            return None, None, None, None, 0, float(self.advantages.mean())
        # Buffer is pre-sorted descending — take the first n_take passing indices
        chosen = passing[:n_take]
        next_obs = self.next_obs[chosen] if self.next_obs is not None else None
        rewards  = self.rewards[chosen]  if self.rewards  is not None else None
        return (self.obs[chosen], self.acts[chosen], next_obs, rewards,
                n_take, float(self.advantages[chosen].mean()))

    def is_ready(self, min_size: int = 1) -> bool:
        return self.size >= min_size


# ---------------------------------------------------------------------------
# MPC Relabeling
# ---------------------------------------------------------------------------

def run_mpc_relabeling(agent, wm_model, wm_params, z_goal, obs_all,
                       mpc_config_dict, n_states, seed, verbose=True,
                       compute_advantages=False, awr_threshold=0.0,
                       fmq=False, fmq_eta=0.1,
                       wm_consistent_targets=False, done_threshold=2.0):
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
        compute_advantages: If True, compute Q(a_mpc) - Q(a_actor) for each label
            and store in buffer sorted descending. Used by mixed_awr injection so
            per-step injection is a cheap numpy swap rather than re-scoring live.
        awr_threshold: Advantage threshold reported in verbose output (informational).
        fmq: If True, use FMQ trust-region update instead of K Grad-MPC steps.
        fmq_eta: FMQ trust-region radius η (default 0.1).
        wm_consistent_targets: If True, run one WM forward step per label to produce
            a consistent (z, a_mpc, r_wm, z'_wm) tuple. The next observation z'_wm =
            WM(z, a_mpc) and reward r_wm = 1{||z'_wm - z_goal|| < done_threshold}.
            These replace the mismatched offline (r, z') during Q-training.
        done_threshold: L2 distance in latent space below which the goal is reached
            (default 2.0, matching make_wm_env_and_dataset).
    Returns:
        MPCLabelBuffer filled with (z, a_mpc) pairs, plus WM targets if requested.
    """
    cfg = mpc_config_dict
    t0 = time.time()

    # Build mpc_fn with current agent params baked in (will JIT-compile)
    if verbose:
        mode_str = f"FMQ η={fmq_eta}" if fmq else f"K={cfg['K_grad']}"
        print(f"  [relabel] Building mpc_fn (N={cfg['N']}, H={cfg['H']}, "
              f"{mode_str})...", flush=True)
    mpc_fn = _make_mpc_fn(
        agent=agent,
        wm_model=wm_model,
        wm_params=wm_params,
        z_goal=z_goal,
        N=cfg["N"],
        H=cfg["H"],
        gamma=cfg["gamma"],
        dense_scale=cfg["dense_scale"],
        K_grad=0 if fmq else cfg["K_grad"],
        lr=cfg["lr"],
        q_only=False if fmq else cfg["q_only"],
        q_every_step=fmq,   # FMQ uses J = Σ_t γ^t Q(z_t, a_t) via BPTT
        fmq=fmq,
        fmq_eta=fmq_eta,
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

    # Optionally compute advantages once for all labels (used by mixed_awr)
    advantages = None
    if compute_advantages:
        if verbose:
            print(f"  [relabel] Computing advantages for {n_states} labels...", flush=True)
        t_adv = time.time()
        z_jax = jnp.asarray(states)
        a_mpc_jax = jnp.asarray(a_mpc)
        # Q(z, a_mpc)
        q_mpc = agent.network.select("critic")(z_jax, a_mpc_jax).mean(axis=0)  # (n_states,)
        # Q(z, a_base) — one actor sample per state
        jax_rng, noise_key = jax.random.split(jax_rng)
        noises = jax.random.normal(noise_key, (n_states, 25))
        a_base = agent.compute_flow_actions(z_jax, noises)
        q_base = agent.network.select("critic")(z_jax, a_base).mean(axis=0)  # (n_states,)
        advantages = np.asarray(q_mpc - q_base)  # (n_states,)
        n_pass = int((advantages > awr_threshold).sum())
        if verbose:
            print(f"  [relabel] advantages done in {time.time()-t_adv:.1f}s  "
                  f"mean={advantages.mean():.3f}  pass>{awr_threshold}: {n_pass}/{n_states}",
                  flush=True)
        # Sort descending by advantage so sample_top_advantage is O(1)
        sort_idx = np.argsort(-advantages)
        states   = states[sort_idx]
        a_mpc    = a_mpc[sort_idx]
        advantages = advantages[sort_idx]

    # ------------------------------------------------------------------
    # Optionally compute WM-consistent next states and rewards
    # ------------------------------------------------------------------
    next_obs_wm = None
    rewards_wm = None
    if wm_consistent_targets:
        if verbose:
            print(f"  [relabel] Computing WM-consistent next states "
                  f"(batch size={n_states}, chunk=256)...", flush=True)
        t_wm = time.time()
        chunk_size = 256
        z_prime_chunks = []
        # JIT-compile a vectorised 1-step WM forward on first chunk
        @jax.jit
        def _wm_step_batch(z_b, a_b):
            # z_b: (B, 192), a_b: (B, 25) → z': (B, 192)
            return wm_model.apply(wm_params, z_b[:, None, :], a_b[:, None, :])[:, -1, :]

        for i in range(0, n_states, chunk_size):
            z_chunk = jnp.asarray(states[i:i + chunk_size])
            a_chunk = jnp.asarray(a_mpc[i:i + chunk_size])
            z_prime_chunk = _wm_step_batch(z_chunk, a_chunk).block_until_ready()
            z_prime_chunks.append(np.asarray(z_prime_chunk))

        next_obs_wm = np.concatenate(z_prime_chunks, axis=0)  # (n_states, 192)
        # Sparse goal-reaching reward: 1 if within done_threshold in latent L2
        z_goal_np = np.asarray(z_goal)
        dists = np.linalg.norm(next_obs_wm - z_goal_np[None, :], axis=-1)  # (n_states,)
        rewards_wm = (dists < done_threshold).astype(np.float32)
        if verbose:
            pct_rewarded = rewards_wm.mean() * 100
            print(f"  [relabel] WM targets done in {time.time()-t_wm:.1f}s  "
                  f"rewarded={pct_rewarded:.1f}%  "
                  f"dist mean={dists.mean():.3f} min={dists.min():.3f}", flush=True)

    buf = MPCLabelBuffer(max_size=n_states)
    buf.add(states, a_mpc, advantages=advantages,
            next_obs=next_obs_wm, rewards=rewards_wm)
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


@jax.jit
def _mpc_flow_loss_weighted(actor_params, network, z_batch, a_mpc_batch, x_0, t, weights):
    """AWR-weighted flow matching loss.

    Args:
        weights: (B,) non-negative importance weights (e.g. exp(advantage/beta) * mask).
                 Normalised internally so the loss scale stays stable.
    Returns:
        scalar weighted loss.
    """
    x_t = (1.0 - t) * x_0 + t * a_mpc_batch
    vel = a_mpc_batch - x_0
    full_params = {**network.params, "modules_actor_bc_flow": actor_params}
    pred = network.apply_fn(
        {"params": full_params},
        z_batch, x_t, t,
        name="actor_bc_flow",
    )  # (B, 25)
    per_sample = jnp.mean((pred - vel) ** 2, axis=-1)   # (B,)
    total_weight = jnp.sum(weights) + 1e-8
    return jnp.sum(weights * per_sample) / total_weight


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

    z_b, a_b, _, _ = mpc_buffer.sample(batch_size)  # next_obs/rewards unused in actor update
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


def actor_only_update_step_awr(agent, mpc_buffer, batch_size, actor_tx, actor_opt_state,
                                rng: jax.random.PRNGKey,
                                awr_threshold: float = 0.0,
                                awr_beta: float = 1.0):
    """AWR-gated actor update: only learn from MPC actions that beat the current actor.

    For each sampled (z, a_mpc):
      1. Compute Q(z, a_mpc) via the current critic ensemble (mean).
      2. Draw one flow sample a_base from the current actor.
      3. Compute Q(z, a_base).
      4. advantage = Q(z, a_mpc) - Q(z, a_base).
      5. Weight = exp(advantage / beta) if advantage > threshold, else 0.
      6. Weighted flow-matching loss on (z, a_mpc) pairs only.

    This prevents actor drift: when MPC labels are not better than the actor
    (noisy Q, OOD states), the filter masks them out automatically.

    Args:
        agent: Current ACFQLAgent.
        mpc_buffer: MPCLabelBuffer with (z, a_mpc) pairs.
        batch_size: Mini-batch size.
        actor_tx: optax optimizer for actor_bc_flow.
        actor_opt_state: Current optimizer state.
        rng: JAX RNG key.
        awr_threshold: Minimum advantage to allow an update (default 0.0).
        awr_beta: Temperature; lower = sharper selection toward best MPC actions.
    Returns:
        (agent_updated, new_opt_state, float loss, int n_active, float mean_advantage)
    """
    actor_params = agent.network.params["modules_actor_bc_flow"]

    z_b, a_mpc_b, _, _ = mpc_buffer.sample(batch_size)  # next_obs/rewards unused in actor update
    z_b_jax    = jnp.asarray(z_b)       # (B, 192)
    a_mpc_jax  = jnp.asarray(a_mpc_b)   # (B, 25)

    # Q(z, a_mpc) — mean over critic ensemble; raw output is (n_critics, B)
    q_mpc = agent.network.select("critic")(z_b_jax, a_mpc_jax).mean(axis=0)  # (B,)

    # Q(z, a_base) — one flow sample from current actor as baseline
    rng, noise_key = jax.random.split(rng)
    noises_base = jax.random.normal(noise_key, (batch_size, 25))
    a_base = agent.compute_flow_actions(z_b_jax, noises_base)                 # (B, 25)
    q_base = agent.network.select("critic")(z_b_jax, a_base).mean(axis=0)    # (B,)

    # Advantage and AWR weights
    advantage = q_mpc - q_base                                                 # (B,)
    mask      = (advantage > awr_threshold).astype(jnp.float32)               # (B,)
    weights   = jnp.exp(jnp.clip(advantage / awr_beta, -10.0, 5.0)) * mask   # (B,)

    n_active  = int(jnp.sum(mask))
    mean_adv  = float(jnp.mean(advantage))

    # Flow matching grad step with AWR weights
    rng, noise_key2, t_key = jax.random.split(rng, 3)
    x_0 = jax.random.normal(noise_key2, (batch_size, 25))
    t   = jax.random.uniform(t_key,     (batch_size, 1))

    grad_fn = jax.value_and_grad(_mpc_flow_loss_weighted)
    loss, grads = grad_fn(actor_params, agent.network, z_b_jax, a_mpc_jax, x_0, t, weights)

    updates, new_opt_state = actor_tx.update(grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)

    new_network_params = {**agent.network.params,
                          "modules_actor_bc_flow": new_actor_params}
    new_network = agent.network.replace(params=new_network_params)
    agent = agent.replace(network=new_network)

    return agent, new_opt_state, float(loss), n_active, mean_adv


# ---------------------------------------------------------------------------
# Batch injection (mixed mode)
# ---------------------------------------------------------------------------

def inject_mpc_into_batch(batch, mpc_buffer, mix_ratio: float, rng: np.random.Generator):
    """Replace mix_ratio fraction of batch actions with MPC-labeled actions.

    batch['actions'] shape: (B, seq_len, action_dim) with seq_len=1.
    Replaces first n_mpc rows' actions with MPC-labeled ones.
    Observations are also replaced so the actor_loss BC targets match.

    If the buffer contains WM-consistent targets (next_obs, rewards), these
    replace the corresponding offline fields so the Q receives a consistent
    Bellman target: Q(z, a_mpc) ← r_wm + γ·V(WM(z, a_mpc)).

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
    z_mpc, a_mpc, next_obs_mpc, rewards_mpc = mpc_buffer.sample(n_mpc)

    # Replace observations and first-sequence-slot actions
    batch["observations"][:n_mpc] = z_mpc
    # batch["actions"] shape: (B, 1, 25) — replace first seq slot
    batch["actions"][:n_mpc, 0, :] = a_mpc
    # Also keep valid mask consistent (MPC-labeled entries are always valid)
    batch["valid"][:n_mpc] = 1.0

    # If WM-consistent targets available, replace next_obs and rewards too.
    # Reshape to match batch dims (e.g. next_obs may be (B,1,192) with seq dim).
    if next_obs_mpc is not None:
        batch["next_observations"][:n_mpc] = next_obs_mpc.reshape(
            n_mpc, *batch["next_observations"].shape[1:])
    if rewards_mpc is not None:
        batch["rewards"][:n_mpc] = rewards_mpc.reshape(
            n_mpc, *batch["rewards"].shape[1:])

    return batch


def inject_mpc_into_batch_awr(batch, mpc_buffer, mix_ratio: float,
                               awr_threshold: float = 0.0):
    """AWR-filtered mixed injection using pre-computed advantages.

    The buffer is pre-sorted descending by advantage at relabeling time, so this
    function is a cheap numpy swap — no JAX calls, same cost as plain injection.

    Only injects labels whose pre-computed advantage > awr_threshold.
    Takes the highest-advantage labels first (buffer already sorted).

    Args:
        batch: Dict from sample_sequence (modified in place).
        mpc_buffer: MPCLabelBuffer with pre-sorted advantages (from run_mpc_relabeling
                    with compute_advantages=True).
        mix_ratio: Target fraction of batch to replace (e.g. 0.3).
        awr_threshold: Minimum advantage to inject (default 0.0).
    Returns:
        (modified_batch, n_injected, mean_advantage_of_injected)
    """
    B = batch["observations"].shape[0]
    n_mpc_target = max(1, int(B * mix_ratio))

    z_inject, a_inject, next_obs_inject, rewards_inject, n_inject, mean_adv = \
        mpc_buffer.sample_top_advantage(n_mpc_target, threshold=awr_threshold)

    if n_inject == 0:
        return batch, 0, mean_adv

    batch["observations"][:n_inject] = z_inject
    batch["actions"][:n_inject, 0, :] = a_inject
    batch["valid"][:n_inject] = 1.0

    if next_obs_inject is not None:
        batch["next_observations"][:n_inject] = next_obs_inject.reshape(
            n_inject, *batch["next_observations"].shape[1:])
    if rewards_inject is not None:
        batch["rewards"][:n_inject] = rewards_inject.reshape(
            n_inject, *batch["rewards"].shape[1:])

    return batch, n_inject, mean_adv


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
                   help="OGBench task id (1..5)")
    p.add_argument("--env_family", default="cube-single",
                   help="OGBench env family: 'cube-single' or 'cube-double' (default: cube-single)")
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
    p.add_argument("--mpc_update_mode", choices=["separate", "mixed", "awr", "mixed_awr"],
                   default="separate",
                   help="How to integrate MPC labels:\n"
                        "  separate   — extra actor-only flow step (Q clean)\n"
                        "  mixed      — inject all MPC labels into main batch (Q shaped)\n"
                        "  awr        — separate + advantage-weighted gating (Q clean)\n"
                        "  mixed_awr  — inject only high-advantage labels (Q shaped cleanly)")
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
    # AWR-specific flags
    p.add_argument("--awr_threshold", type=float, default=0.0,
                   help="[AWR mode] Minimum Q-advantage to allow an actor update. "
                        "0.0 = update whenever MPC > current actor.")
    p.add_argument("--awr_beta", type=float, default=1.0,
                   help="[AWR mode] Temperature for exp(advantage / beta) weighting. "
                        "Lower = sharper selection; higher = more uniform.")
    # FMQ flags
    p.add_argument("--mpc_fmq", action="store_true",
                   help="Use FMQ trust-region update (1 normalised grad step, Theorem 3.2) "
                        "instead of K Grad-MPC gradient steps for label generation.")
    p.add_argument("--fmq_eta", type=float, default=0.1,
                   help="FMQ trust-region radius η (default: 0.1, best from Phase 1 eval)")
    # WM-consistent Bellman target flag
    p.add_argument("--wm_consistent_targets", action="store_true",
                   help="Replace offline (r, z') in MPC-injected batch rows with "
                        "WM-predicted (r_wm, z'_wm=WM(z,a_mpc)). Gives the Q a "
                        "consistent Bellman target, reducing critic loss spikes "
                        "after relabeling. Cost: one extra WM forward pass per "
                        "relabeling event.")
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
        env_family=args.env_family,
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
    if args.mpc_fmq:
        print(f"    label_gen=FMQ  fmq_eta={args.fmq_eta}  (replaces Grad-MPC K={mpc_cfg['K_grad']})", flush=True)
    else:
        print(f"    label_gen=GradMPC  K={mpc_cfg['K_grad']}  lr={mpc_cfg['lr']}", flush=True)
    if args.wm_consistent_targets:
        print(f"    wm_consistent_targets=True  (Q target: r_wm+γV(WM(z,a_mpc)))", flush=True)
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
                compute_advantages=(args.mpc_update_mode == "mixed_awr"),
                awr_threshold=args.awr_threshold,
                fmq=args.mpc_fmq,
                fmq_eta=args.fmq_eta,
                wm_consistent_targets=args.wm_consistent_targets,
            )
            # Re-init actor optimizer so its momentum doesn't fight MPC labels
            actor_opt_state = actor_tx.init(
                agent.network.params["modules_actor_bc_flow"]
            )

        # ---- Sample main training batch ----
        batch = train_dataset.sample_sequence(
            args.batch_size, sequence_length=1, discount=args.discount
        )

        # ---- Injection modes: inject MPC labels before main update ----
        mpc_loss = float("nan")
        awr_n_active = -1
        awr_mean_adv = float("nan")

        if (args.mpc_update_mode == "mixed" and
                mpc_buffer is not None and mpc_buffer.is_ready(args.batch_size)):
            batch = inject_mpc_into_batch(batch, mpc_buffer, args.mpc_mix_ratio,
                                          np.random.default_rng(step))

        elif (args.mpc_update_mode == "mixed_awr" and
                mpc_buffer is not None and mpc_buffer.is_ready(args.batch_size)):
            batch, awr_n_active, awr_mean_adv = inject_mpc_into_batch_awr(
                batch, mpc_buffer, args.mpc_mix_ratio,
                awr_threshold=args.awr_threshold,
            )

        # ---- Main update: Q + actor on demo batch ----
        agent, info = agent.update(batch)

        # ---- Separate / AWR mode: actor-only MPC gradient step ----
        # One MPC update every mpc_actor_steps_every main steps to control
        # the BC:MPC ratio. Default every=50 → ~128 epochs per buffer refresh,
        # which matches post-hoc distillation convergence without overfitting.
        # AWR mode adds advantage-weighted gating on top of the separate update.
        if (args.mpc_update_mode in ("separate", "awr") and
                mpc_buffer is not None and mpc_buffer.is_ready(args.batch_size) and
                step % args.mpc_actor_steps_every == 0):
            jax_rng, key = jax.random.split(jax_rng)
            if args.mpc_update_mode == "awr":
                agent, actor_opt_state, mpc_loss, awr_n_active, awr_mean_adv = (
                    actor_only_update_step_awr(
                        agent, mpc_buffer, args.batch_size,
                        actor_tx, actor_opt_state, key,
                        awr_threshold=args.awr_threshold,
                        awr_beta=args.awr_beta,
                    )
                )
                info["mpc_flow_loss"]  = mpc_loss
                info["awr_n_active"]   = awr_n_active
                info["awr_mean_adv"]   = awr_mean_adv
            else:  # separate
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
            if args.mpc_update_mode in ("awr", "mixed_awr"):
                log_data["awr_n_active"] = awr_n_active
                log_data["awr_mean_adv"] = awr_mean_adv
            logger_train.log(log_data, step=step)
            awr_str = (f"  awr={awr_n_active}/{args.batch_size}  adv={awr_mean_adv:.3f}"
                       if args.mpc_update_mode in ("awr", "mixed_awr") else "")
            print(
                f"  step {step:7d}/{args.offline_steps}  "
                f"critic={info.get('critic/critic_loss', float('nan')):.4f}  "
                f"bc_flow={info.get('actor/bc_flow_loss', float('nan')):.4f}  "
                f"mpc_flow={mpc_loss:.4f}"
                f"{awr_str}  elapsed={elapsed:.0f}s",
                flush=True
            )

        # ---- Periodic eval + checkpoint ----
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
            # Save checkpoint at every eval so intermediate peaks are recoverable
            save_agent(agent, args.save_dir, step)
            print(f"  [ckpt] saved params_{step}.pkl", flush=True)

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
