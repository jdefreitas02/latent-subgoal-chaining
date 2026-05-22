"""Numerical parity test between the PyTorch WM and the JAX port.

We load the same checkpoint into both implementations, feed identical
inputs, and require the outputs to match to a tight tolerance. If this
passes, the JAX port can be used for differentiable rollouts in QC training
with confidence that the dynamics it predicts are bit-equivalent (modulo
fp arithmetic noise) to the WMEnv used in our other experiments.
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import jax
import jax.numpy as jnp

from envs.jepa_loader import load_jepa
from wm_jax import (
    LeJEPAJaxForward,
    torch_state_dict_to_jax_params,
    LATENT_DIM, ACTION_RAW_DIM, NUM_FRAMES,
)


def _torch_predict(jepa, emb_np, action_raw_np):
    """Mirror wm_env.py's step(): apply action_encoder to actions, then
    predictor, then pred_proj. Returns the FULL output sequence (B, T, D)."""
    with torch.no_grad():
        emb = torch.from_numpy(emb_np).float()
        a = torch.from_numpy(action_raw_np).float()
        # action_encoder takes (B, T, action_dim) in our load_jepa convention
        act_emb = jepa.action_encoder(a)
        # predictor takes (emb, act_emb)
        preds = jepa.predictor(emb, act_emb)            # (B, T, hidden)
        # pred_proj after rearrange to (B*T, D)
        B, T, D = preds.shape
        preds = jepa.pred_proj(preds.reshape(B * T, D)).reshape(B, T, -1)
    return preds.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt",
                   default="/home/jrd21/stable_wm_data/cube/lejepa",
                   help="Torch WM ckpt (or `_state_dict.pt` for fine-tuned).")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="Max-abs error tolerance (bf16/fp32 mix gives ~1e-4).")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    B, T = args.batch, NUM_FRAMES

    # 1. Load Torch WM
    print(f"[load] Torch WM from {args.wm_ckpt}", flush=True)
    jepa_t = load_jepa(args.wm_ckpt, device="cpu", img_size=224, patch_size=14)
    jepa_t.eval()

    # 2. Convert state_dict to JAX params
    print(f"[convert] Torch state_dict -> Flax params", flush=True)
    sd = jepa_t.state_dict() if hasattr(jepa_t, "state_dict") else jepa_t.jepa.state_dict()
    # In case the loaded shim wraps a JEPA at .jepa, normalise keys:
    if not any(k.startswith("predictor.") for k in sd):
        # Try stripping a top-level prefix like "jepa." or "model."
        for prefix in ("jepa.", "model."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                break
    params = torch_state_dict_to_jax_params(sd)

    # 3. Run both on random inputs
    emb_np = rng.standard_normal((B, T, LATENT_DIM)).astype(np.float32) * 5.0
    a_np = rng.uniform(-1, 1, size=(B, T, ACTION_RAW_DIM)).astype(np.float32)

    print(f"[run] Torch predict", flush=True)
    t0 = time.time()
    torch_out = _torch_predict(jepa_t, emb_np, a_np)
    print(f"  {time.time()-t0:.3f}s  shape={torch_out.shape}", flush=True)

    print(f"[run] JAX predict", flush=True)
    t0 = time.time()
    jax_model = LeJEPAJaxForward()
    jax_out = jax_model.apply(params, jnp.asarray(emb_np), jnp.asarray(a_np))
    jax_out = np.asarray(jax_out)
    print(f"  {time.time()-t0:.3f}s  shape={jax_out.shape}", flush=True)

    # 4. Compare
    abs_err = np.abs(torch_out - jax_out)
    rel_err = abs_err / (np.abs(torch_out) + 1e-8)
    print(f"\n=== Parity ===")
    print(f"  torch out: mean={torch_out.mean():.6f}  std={torch_out.std():.6f}  "
          f"min={torch_out.min():.6f}  max={torch_out.max():.6f}")
    print(f"  jax out:   mean={jax_out.mean():.6f}  std={jax_out.std():.6f}  "
          f"min={jax_out.min():.6f}  max={jax_out.max():.6f}")
    print(f"  abs error: mean={abs_err.mean():.2e}  max={abs_err.max():.2e}")
    print(f"  rel error: mean={rel_err.mean():.2e}  max={rel_err.max():.2e}")
    print(f"  cosine sim (per-token mean): "
          f"{(torch_out * jax_out).sum(-1).mean() / (np.linalg.norm(torch_out, axis=-1) * np.linalg.norm(jax_out, axis=-1)).mean():.6f}")

    ok = abs_err.max() < args.tolerance
    print(f"\n{'OK' if ok else 'FAIL'}: max-abs error "
          f"{abs_err.max():.2e} vs tolerance {args.tolerance:.0e}")

    # 5. Also verify gradients flow (sanity check for the "differentiable" claim)
    print(f"\n[grad] verifying gradients flow through the JAX model")
    def loss_fn(emb, act, params):
        out = jax_model.apply(params, emb, act)
        return out.sum()
    grads = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.asarray(emb_np), jnp.asarray(a_np), params,
    )
    print(f"  d/d(emb) shape={grads[0].shape}  abs_max={float(jnp.abs(grads[0]).max()):.4e}")
    print(f"  d/d(act) shape={grads[1].shape}  abs_max={float(jnp.abs(grads[1]).max()):.4e}")
    print("  (non-zero gradients = JAX is set up to support analytic policy gradient)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
