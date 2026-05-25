"""MPC eval: frozen offline policy + frozen JAX WM as planners at test time.

For each chunk decision (5 real env steps):
  1. Encode current pixel obs → z_t (192-D) via JEPA
  2. mpc_select_action: sample N proposals from BC flow at z_t, roll out H WM
     steps with policy-resampled continuations, score via dense reward +
     terminal Q, return best 25-D chunk
  3. evaluate_real_ogbench dispatches it as 5×5-D sub-actions

Flat rollout (not tree): N×H WM evaluations total. Each branch re-samples
continuation actions from the flow policy at each predicted WM state.

Reward: r_t = -||z_t - z_goal||_2 / dense_scale
Score:  Σ_{h=0}^{H-1} γ^h * r_{h+1} + γ^H * Q(z_H, a_H).mean(axis=0)

Usage:
    python eval_mpc.py \\
        --policy_ckpt exp/qc/e_qc_jepa_wm_ftfull/.../offline_final/params_500000.pkl \\
        --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --mpc_n 32 --mpc_h 3 --n_episodes 250 --task_id 1

Exp 1: lejepa WM + lejepa policy (sd000s_55258, 61.6%)
Exp 2: lejepa_play_ft_full WM + ftfull policy (sd000s_55445, 81.6%)
Exp 3: Exp 2 + gradient ascent (--mpc_k_grad 5)
Exp 4: VAML-tuned WM (produced by finetune_wm_vaml.py) + ftfull policy
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import torch

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.real_ogbench_eval import evaluate_real_ogbench
from envs.wm_env import make_wm_env_and_dataset
from utils.flax_utils import restore_agent_with_file
from wm_jax import LeJEPAJaxForward, load_wm_jax


def _load_wm(ckpt_path):
    """Load (model, params) from either a raw WM ckpt or a VAML params .pkl."""
    if ckpt_path.endswith(".pkl"):
        import pickle
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        # wm_state.params is the raw {action_encoder, predictor, pred_proj} dict;
        # Flax's .apply() expects {"params": ...} as the variables argument.
        return LeJEPAJaxForward(), {"params": data["params"]}
    return load_wm_jax(ckpt_path)


# ---------------------------------------------------------------------------
# MPC action selector
# ---------------------------------------------------------------------------

def _make_mpc_fn(agent, wm_model, wm_params, z_goal, N, H, gamma,
                 dense_scale, K_grad, lr, q_only=False, q_every_step=False):
    """Return a jitted function with the same signature as agent.sample_actions.

    Args:
        agent: Restored ACFQLAgent (JAX pytree).
        wm_model: LeJEPAJaxForward Flax module (static).
        wm_params: Frozen JAX params for wm_model.
        z_goal: (192,) goal latent as a JAX array.
        N: Number of proposals.
        H: WM lookahead steps (each = 5 real env steps).
        gamma: Discount factor.
        dense_scale: Dense reward scale (r = -dist/scale).
        K_grad: Gradient ascent steps on first-chunk actions (0 = disabled).
        lr: Gradient ascent step size.
        q_only: If True, score = Q(z_H, a_H) only — no intermediate dense rewards.
            WM rollout still happens to get z_H; z_goal is not used for scoring.
        q_every_step: If True, score = Σ_{h=0}^{H-1} γ^h * Q(z_{h+1}, a_{h+1}).
            Uses Q at every WM step (not just terminal). Supersedes q_only when True.
            For H=1 this is identical to q_only=True.

    Returns:
        fn(observations, rng) -> (25,) best action chunk.
    """
    z_goal_j = jnp.asarray(z_goal)  # (192,)

    def _wm_step(z_t, a_t):
        """One WM forward pass: (N,192) × (N,25) → (N,192)."""
        return wm_model.apply(wm_params, z_t[:, None, :], a_t[:, None, :])[:, -1, :]

    def _score_proposals(a_first, z_batch, rng):
        """Score N proposals via H-step flat rollout.

        a_first: (N, 25) initial chunks (possibly grad-refined)
        z_batch: (N, 192) tiled current latent
        Returns: (N,) scores, (N, 192) terminal z, (N, 25) terminal a
        """
        total_score = jnp.zeros(N)
        z_t = z_batch
        a_t = a_first
        if q_every_step:
            # Score = Σ_{h=0}^{H-1} γ^h * Q(z_{h+1}, a_{h+1})
            # Each step contributes its own Q estimate; early steps are less WM-corrupted.
            for h in range(H):
                z_next = _wm_step(z_t, a_t)
                rng, key_h = jax.random.split(rng)
                noises_h = jax.random.normal(key_h, (N, 25))
                a_next = agent.compute_flow_actions(z_next, noises_h)
                q_h = agent.network.select('critic')(z_next, a_next).mean(axis=0)  # (N,)
                total_score = total_score + (gamma ** h) * q_h
                a_t = a_next
                z_t = z_next
        else:
            for h in range(H):
                z_next = _wm_step(z_t, a_t)
                if not q_only:
                    r = -jnp.linalg.norm(z_next - z_goal_j[None], axis=-1) / dense_scale
                    total_score = total_score + (gamma ** h) * r
                rng, key_h = jax.random.split(rng)
                noises_h = jax.random.normal(key_h, (N, 25))
                a_t = agent.compute_flow_actions(z_next, noises_h)
                z_t = z_next
            q_term = agent.network.select('critic')(z_t, a_t).mean(axis=0)  # (N,)
            total_score = total_score + q_term  # no gamma^H discount when q_only (cleaner)
        return total_score, z_t, a_t

    if K_grad > 0:
        # Pre-declare: gradient ascent variant (Exp 3)
        @jax.jit
        def mpc_fn(observations, rng):
            z_0 = observations  # (192,)
            z_batch = jnp.tile(z_0[None], (N, 1))  # (N, 192)

            # Sample N initial proposals
            rng, key0 = jax.random.split(rng)
            noises0 = jax.random.normal(key0, (N, 25))
            a_first = agent.compute_flow_actions(z_batch, noises0)  # (N, 25)

            # Pre-sample continuation noises so they are constants wrt a_first
            cont_keys = []
            for _ in range(H):
                rng, kk = jax.random.split(rng)
                cont_keys.append(kk)

            def score_for_grad(a_0):
                """Score function wrt a_0 only; continuations are stop_gradient.

                Respects q_only and q_every_step flags for consistency with _score_proposals.
                """
                total = jnp.zeros(N)
                z_t = z_batch
                a_t = a_0
                if q_every_step:
                    for h in range(H):
                        z_next = _wm_step(z_t, a_t)
                        noises_h = jax.random.normal(cont_keys[h], (N, 25))
                        a_next = jax.lax.stop_gradient(agent.compute_flow_actions(z_next, noises_h))
                        q_h = agent.network.select('critic')(z_next, a_next).mean(axis=0)
                        total = total + (gamma ** h) * q_h
                        a_t = a_next
                        z_t = z_next
                else:
                    for h in range(H):
                        z_next = _wm_step(z_t, a_t)
                        if not q_only:
                            r = -jnp.linalg.norm(z_next - z_goal_j[None], axis=-1) / dense_scale
                            total = total + (gamma ** h) * r
                        noises_h = jax.random.normal(cont_keys[h], (N, 25))
                        a_t = jax.lax.stop_gradient(agent.compute_flow_actions(z_next, noises_h))
                        z_t = z_next
                    q_t = agent.network.select('critic')(z_t, a_t).mean(axis=0)
                    total = total + (gamma ** H) * q_t
                return total.sum()

            grad_fn = jax.grad(score_for_grad)
            for _ in range(K_grad):
                grads = grad_fn(a_first)
                a_first = jnp.clip(a_first + lr * grads, -1.0, 1.0)

            # Final scoring pass (fresh continuation samples after grad refinement)
            rng, score_rng = jax.random.split(rng)
            total_score, _, _ = _score_proposals(a_first, z_batch, score_rng)

            best_i = jnp.argmax(total_score)
            return a_first[best_i]  # (25,)

    else:
        # Plain BoN-MPC (Exps 1, 2, 4)
        @jax.jit
        def mpc_fn(observations, rng):
            z_0 = observations  # (192,)
            z_batch = jnp.tile(z_0[None], (N, 1))  # (N, 192)

            # Sample N initial proposals from BC flow at z_0
            rng, key0 = jax.random.split(rng)
            noises0 = jax.random.normal(key0, (N, 25))
            a_first = agent.compute_flow_actions(z_batch, noises0)  # (N, 25)

            rng, score_rng = jax.random.split(rng)
            total_score, _, _ = _score_proposals(a_first, z_batch, score_rng)

            best_i = jnp.argmax(total_score)
            return a_first[best_i]  # (25,)

    return mpc_fn


# ---------------------------------------------------------------------------
# Thin wrapper so evaluate_real_ogbench can call .sample_actions
# ---------------------------------------------------------------------------

class _MPCAgent:
    """Minimal duck-type wrapper exposing sample_actions for evaluate_real_ogbench."""

    def __init__(self, mpc_fn):
        self.sample_actions = mpc_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="MPC eval with frozen offline policy + JAX WM")
    p.add_argument("--policy_ckpt", required=True,
                   help="Path to offline_final/params_*.pkl checkpoint")
    p.add_argument("--wm_ckpt", required=True,
                   help="WM predictor ckpt for MPC rollout. Pass a .pkl for VAML-tuned WM "
                        "(see finetune_wm_vaml.py). For stock WMs, pass the JEPA ckpt dir.")
    p.add_argument("--jepa_ckpt", default=None,
                   help="JEPA encoder ckpt for make_wm_env_and_dataset (pixel encoding + z_goal). "
                        "Required when --wm_ckpt is a VAML .pkl; otherwise defaults to --wm_ckpt.")
    p.add_argument("--wm_cache", required=True,
                   help="Latent cache .pt path (e.g. lewm_224_latents_cache_ftfull.pt)")
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"),
                   help="HDF5 play dataset path")
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--n_episodes", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    # MPC hyper-parameters
    p.add_argument("--mpc_n", type=int, default=32, help="Number of proposals")
    p.add_argument("--mpc_h", type=int, default=3, help="WM lookahead steps (H>=1)")
    p.add_argument("--mpc_gamma", type=float, default=0.99, help="Discount")
    p.add_argument("--mpc_dense_scale", type=float, default=10.0, help="Dense reward scale")
    p.add_argument("--mpc_k_grad", type=int, default=0,
                   help="Gradient ascent steps on first chunk (0=disabled, Exp 3=5)")
    p.add_argument("--mpc_lr", type=float, default=0.01, help="Gradient ascent step size")
    p.add_argument("--mpc_q_only", action="store_true",
                   help="Score = Q(z_H, a_H) only, no intermediate dense rewards. "
                        "WM rollout still runs to get z_H; isolates WM lookahead value.")
    p.add_argument("--mpc_q_every_step", action="store_true",
                   help="Score = Σ γ^h * Q(z_{h+1}, a_{h+1}) for h=0..H-1. "
                        "Uses Q at every WM step, not just terminal. "
                        "For H=1 this is identical to --mpc_q_only. Supersedes --mpc_q_only.")
    # Output
    p.add_argument("--out_json", default=None,
                   help="Optional path to write metrics JSON (defaults to next to policy_ckpt)")
    args = p.parse_args()

    assert args.mpc_h >= 1, "--mpc_h must be >= 1 (use the offline best-of-n for H=0 baseline)"

    jepa_ckpt = args.jepa_ckpt if args.jepa_ckpt is not None else args.wm_ckpt
    if args.wm_ckpt.endswith(".pkl") and args.jepa_ckpt is None:
        raise ValueError(
            "--wm_ckpt is a .pkl (VAML WM) but --jepa_ckpt is not set. "
            "Pass the original JEPA ckpt dir via --jepa_ckpt for pixel encoding."
        )

    print(f"=== MPC Eval: N={args.mpc_n}, H={args.mpc_h}, K_grad={args.mpc_k_grad}", flush=True)
    print(f"    policy_ckpt: {args.policy_ckpt}", flush=True)
    print(f"    wm_ckpt:     {args.wm_ckpt}", flush=True)
    print(f"    jepa_ckpt:   {jepa_ckpt}", flush=True)

    # ------------------------------------------------------------------
    # 1. Build env stack (JEPA + real env + z_goal) via the E pipeline
    # ------------------------------------------------------------------
    print("=== Building env stack...", flush=True)
    (_, _, train_dataset, _, jepa, real_env, z_goal) = make_wm_env_and_dataset(
        wm_ckpt_path=jepa_ckpt,
        latent_cache_path=args.wm_cache,
        hdf5_dataset_path=args.wm_hdf5,
        task_id=args.task_id,
        done_threshold=2.0,   # irrelevant for real-env eval
        max_episode_steps=40,
        wm_device=args.device,
        img_size=224,
    )

    ex_obs = train_dataset["observations"][:1]
    ex_act = train_dataset["actions"][:1]
    print(f"  Dataset sample shapes: obs={ex_obs.shape}, act={ex_act.shape}", flush=True)

    # ------------------------------------------------------------------
    # 2. Reconstruct and restore the offline policy agent
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
    # 3. Load JAX WM
    # ------------------------------------------------------------------
    print(f"=== Loading JAX WM from {args.wm_ckpt}...", flush=True)
    wm_model, wm_params = _load_wm(args.wm_ckpt)
    z_goal_jax = jnp.asarray(z_goal.astype(np.float32))
    print("  WM loaded.", flush=True)

    # ------------------------------------------------------------------
    # 4. Build and warm-up the JIT-compiled MPC function
    # ------------------------------------------------------------------
    print("=== Building MPC function...", flush=True)
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
        q_only=args.mpc_q_only,
        q_every_step=args.mpc_q_every_step,
    )

    # Warm-up JIT compile before eval timing
    print("  JIT warm-up...", flush=True)
    _dummy_z = jnp.zeros((192,))
    _dummy_rng = jax.random.PRNGKey(0)
    _dummy_out = mpc_fn(observations=_dummy_z, rng=_dummy_rng)
    _dummy_out.block_until_ready()
    print("  JIT compiled.", flush=True)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print(f"=== Starting eval: {args.n_episodes} episodes on task {args.task_id}", flush=True)
    mpc_agent = _MPCAgent(mpc_fn)
    metrics = evaluate_real_ogbench(
        agent=mpc_agent,
        real_env=real_env,
        jepa_model=jepa,
        device=args.device,
        task_ids=(args.task_id,),
        num_episodes_per_task=args.n_episodes,
        action_dispatch="chunk25",
        pass_task_id_on_reset=False,
    )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    sr = float(metrics.get(f"task_{args.task_id}/success_rate",
                           metrics.get("overall/success_rate", -1.0)))
    print(f"\n=== RESULT: success_rate={sr:.4f} ({sr*100:.1f}%)", flush=True)

    if args.out_json is None:
        ckpt_dir = os.path.dirname(args.policy_ckpt)
        tag = f"mpc_n{args.mpc_n}_h{args.mpc_h}"
        if args.mpc_q_every_step:
            tag += "_qes"
        elif args.mpc_q_only:
            tag += "_qonly"
        if args.mpc_k_grad > 0:
            tag += f"_k{args.mpc_k_grad}"
        args.out_json = os.path.join(ckpt_dir, f"mpc_eval_{tag}.json")

    with open(args.out_json, "w") as f:
        payload = {
            "mpc_n": args.mpc_n,
            "mpc_h": args.mpc_h,
            "mpc_k_grad": args.mpc_k_grad,
            "mpc_lr": args.mpc_lr,
            "mpc_q_only": args.mpc_q_only,
            "mpc_q_every_step": args.mpc_q_every_step,
            "policy_ckpt": args.policy_ckpt,
            "wm_ckpt": args.wm_ckpt,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        json.dump(payload, f, indent=2)
    print(f"  Metrics saved to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
