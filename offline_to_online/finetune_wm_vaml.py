"""VAML offline fine-tuning of the JAX WM (Exp 4, Phase A).

Loss: L = L_latent + lambda_vaml * L_VAML
  L_latent = mean(||z_hat - z_next||²)            one-step prediction MSE
  L_VAML   = mean(||Q(z_hat, a) - Q(z_next, a)||²)   value-aware alignment

Q-network (from the offline policy checkpoint) is fully frozen via stop_gradient.
Only WM params are updated via AdamW.

After training, the updated WM params are exported as a new checkpoint that can
be loaded by eval_mpc.py via --wm_ckpt.

Usage:
    python finetune_wm_vaml.py \\
        --policy_ckpt exp/qc/e_qc_jepa_wm_ftfull/.../offline_final/params_500000.pkl \\
        --wm_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full/lejepa_play_ft_full \\
        --wm_cache ~/stable_wm_data/ogbench/lewm_224_latents_cache_ftfull.pt \\
        --out_ckpt ~/stable_wm_data/cube/lejepa_play_ft_full_vaml/ \\
        --n_steps 50000 --lambda_vaml 1.0 --lr 1e-5
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
import torch

from agents.acfql import ACFQLAgent, get_config as get_acfql_config
from envs.wm_env import make_wm_env_and_dataset
from utils.flax_utils import restore_agent_with_file
from wm_jax import load_wm_jax, make_wm_trainstate, jax_params_to_torch_state_dict


# ---------------------------------------------------------------------------
# VAML loss
# ---------------------------------------------------------------------------

def make_vaml_loss(wm_state, agent, lambda_vaml):
    """Return a JIT-compiled loss function.

    wm_state.apply_fn({"params": wm_grad_params}, z[:,None,:], a[:,None,:]) → (B,1,192)
    Q: agent.network.select('critic')(z, a) → (num_qs, B)
    """

    @jax.jit
    def loss_fn(wm_grad_params, z, a, z_next):
        """
        z:      (B, 192) current latent
        a:      (B, 25)  action chunk
        z_next: (B, 192) true next latent

        Gradient flows through z_hat (from WM) into wm_grad_params.
        Q network params inside agent.network are accessed but NOT differentiated
        (they have no grad since this fn only receives wm_grad_params).
        """
        # WM one-step prediction — gradient flows through this into wm_grad_params
        z_hat = wm_state.apply_fn(
            {"params": wm_grad_params}, z[:, None, :], a[:, None, :]
        )[:, -1, :]  # (B, 192)

        # L_latent: standard MSE
        l_latent = jnp.mean(jnp.sum((z_hat - z_next) ** 2, axis=-1))

        # L_VAML: ||Q(z_hat, a) - Q(z_next, a)||^2
        # Gradient flows from Q(z_hat) → z_hat → wm_grad_params.
        # Q(z_next) is a constant target (stop_gradient on z_next ensures this).
        q_hat  = agent.network.select('critic')(z_hat,  a).mean(axis=0)   # (B,) differentiable
        q_true = agent.network.select('critic')(
            jax.lax.stop_gradient(z_next), a
        ).mean(axis=0)                                                      # (B,) target
        l_vaml = jnp.mean((q_hat - jax.lax.stop_gradient(q_true)) ** 2)

        loss = l_latent + lambda_vaml * l_vaml
        return loss, {"l_latent": l_latent, "l_vaml": l_vaml, "loss": loss}

    return loss_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="VAML offline fine-tuning of JAX WM")
    p.add_argument("--policy_ckpt", required=True,
                   help="Offline final policy checkpoint (provides frozen Q-network)")
    p.add_argument("--wm_ckpt", required=True,
                   help="Initial WM checkpoint path (e.g. lejepa_play_ft_full/lejepa_play_ft_full)")
    p.add_argument("--wm_cache", required=True,
                   help="Latent cache .pt path")
    p.add_argument("--wm_hdf5",
                   default=os.path.expanduser(
                       "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"),
                   help="HDF5 play dataset")
    p.add_argument("--out_ckpt", required=True,
                   help="Directory to save the VAML-tuned WM checkpoint")
    p.add_argument("--task_id", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    # Training hyper-parameters
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lambda_vaml", type=float, default=1.0,
                   help="Weight on L_VAML relative to L_latent")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--log_interval", type=int, default=500)
    args = p.parse_args()

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # ------------------------------------------------------------------
    # 1. Load offline latent dataset
    # ------------------------------------------------------------------
    print("=== Loading offline latent dataset...", flush=True)
    (_, _, train_dataset, _, _, _, z_goal) = make_wm_env_and_dataset(
        wm_ckpt_path=args.wm_ckpt,
        latent_cache_path=args.wm_cache,
        hdf5_dataset_path=args.wm_hdf5,
        task_id=args.task_id,
        done_threshold=2.0,
        max_episode_steps=40,
        wm_device=args.device,
        img_size=224,
    )

    obs_all   = np.asarray(train_dataset["observations"])       # (N, 192)
    act_all   = np.asarray(train_dataset["actions"])            # (N, 25)
    nobs_all  = np.asarray(train_dataset["next_observations"])  # (N, 192)
    # Handle possible trailing time dimension
    if obs_all.ndim == 3:
        obs_all  = obs_all[:, -1, :]
    if nobs_all.ndim == 3:
        nobs_all = nobs_all[:, -1, :]
    if act_all.ndim == 3:
        act_all  = act_all[:, 0, :]

    N_data = obs_all.shape[0]
    print(f"  Dataset: {N_data} transitions, obs={obs_all.shape}, act={act_all.shape}",
          flush=True)

    ex_obs = obs_all[:1]
    ex_act = act_all[:1]

    # ------------------------------------------------------------------
    # 2. Load frozen Q-network (policy agent)
    # ------------------------------------------------------------------
    print("=== Loading frozen Q-network from policy checkpoint...", flush=True)
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
    print("  Q-network loaded (frozen throughout).", flush=True)

    # ------------------------------------------------------------------
    # 3. Build trainable WM state
    # ------------------------------------------------------------------
    print(f"=== Building WM TrainState (lr={args.lr})...", flush=True)
    wm_state = make_wm_trainstate(args.wm_ckpt, lr=args.lr, weight_decay=args.weight_decay)
    print("  WM TrainState ready.", flush=True)

    # ------------------------------------------------------------------
    # 4. Build VAML loss and gradient function
    # ------------------------------------------------------------------
    loss_fn = make_vaml_loss(wm_state, agent, args.lambda_vaml)
    # grad wrt first arg (wm params); extract scalar loss from (loss, aux) tuple
    grad_fn = jax.jit(jax.value_and_grad(
        lambda p, z, a, zn: loss_fn(p, z, a, zn)[0]
    ))

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"=== VAML training: {args.n_steps} steps, batch={args.batch_size}, "
          f"lambda={args.lambda_vaml}", flush=True)

    t0 = time.time()
    wm_params = wm_state.params

    for step in range(1, args.n_steps + 1):
        idx = np.random.randint(0, N_data, size=args.batch_size)
        z_b  = jnp.asarray(obs_all[idx])   # (B, 192)
        a_b  = jnp.asarray(act_all[idx])   # (B, 25)
        zn_b = jnp.asarray(nobs_all[idx])  # (B, 192)

        loss_val, grads = grad_fn(wm_params, z_b, a_b, zn_b)
        wm_state = wm_state.apply_gradients(grads=grads)
        wm_params = wm_state.params

        if step % args.log_interval == 0 or step == 1:
            # Get aux info (re-run loss for logging — cheap)
            _, aux = loss_fn(wm_params, z_b, a_b, zn_b)
            elapsed = time.time() - t0
            print(f"  step {step:6d}/{args.n_steps}  "
                  f"loss={float(aux['loss']):.6f}  "
                  f"l_latent={float(aux['l_latent']):.6f}  "
                  f"l_vaml={float(aux['l_vaml']):.6f}  "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"=== Training done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # 6. Save VAML-tuned WM checkpoint
    # ------------------------------------------------------------------
    os.makedirs(args.out_ckpt, exist_ok=True)

    # Convert JAX params → PyTorch state_dict and save via torch.save so
    # load_wm_jax / load_jepa can reload it exactly as any other WM ckpt.
    torch_sd = jax_params_to_torch_state_dict(wm_params)
    out_path = os.path.join(args.out_ckpt, "vaml_wm.pt")
    torch.save(torch_sd, out_path)
    print(f"  Saved PyTorch state_dict → {out_path}", flush=True)

    # Save JAX params as the primary eval_mpc.py input
    import pickle
    jax_path = os.path.join(args.out_ckpt, "vaml_wm_params.pkl")
    with open(jax_path, "wb") as f:
        pickle.dump({"params": jax.device_get(wm_params)}, f)
    print(f"  Saved JAX params → {jax_path}", flush=True)

    print(f"\n=== VAML fine-tune complete.")
    print(f"    Pass to eval_mpc.py via:  --wm_ckpt {jax_path}")


if __name__ == "__main__":
    main()
