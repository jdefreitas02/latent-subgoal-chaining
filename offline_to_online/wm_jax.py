"""JAX/Flax port of the JEPA world model's forward path.

What this enables: differentiable rollouts in the QC (JAX) training loop.
The Torch JEPA's predictor + action_encoder + pred_proj are reimplemented in
Flax such that, given (z_t, raw_action), we can compute z_{t+1} as part of
a JAX computation graph. This means the policy gradient can flow backward
through the world model, enabling analytic actor / critic training signals
in addition to the bootstrap-only signal currently available.

What this does NOT include: the JEPA visual encoder (HF ViTModel). Rollouts
during RL training only operate on already-encoded latents, so we only need
the predictor side. Eval-time pixel encoding stays in Torch via the existing
`load_jepa` helper.

The architecture mirrors `module.py` and `jepa.py` exactly:

    ARPredictor(z, c)
      z = z + pos_embedding[:, :T]
      for _ in range(depth):
          ConditionalBlock(z, c)   # AdaLN-zero + causal attn + MLP
      z = LayerNorm(z)
      return z

    ActionEncoder(a_raw)            # Embedder in Torch
      a = Linear(25 -> 10)(a)       # Conv1d k=1 is equivalent
      a = Linear(10 -> 768)(a)
      a = SiLU(a)
      a = Linear(768 -> 192)(a)

    PredProj(z)                     # MLP with frozen BN
      z = Linear(192 -> 2048)(z)
      z = (z - rm) / sqrt(rv + 1e-5) * gamma + beta
      z = GELU(z)
      z = Linear(2048 -> 192)(z)

A combined module ``LeJEPAJaxForward`` exposes
``forward(emb, action_raw)`` which composes them the same way Torch's
``JEPA.predict(emb, action_encoder(action))`` does, and returns the
next-step embedding (B, T, D). Causal masking in attention means only the
last position depends on the full history; this matches Torch's behaviour
exactly.

Numerical parity with the Torch model is verified by
``test_wm_jax_parity.py``.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as fnn
import numpy as np


# ============================================================
# Constants matching the trained WM at 224x14
# ============================================================
LATENT_DIM = 192
ACTION_RAW_DIM = 25
ACTION_SMOOTHED_DIM = 10
ACTION_MLP_HIDDEN = 768   # mlp_scale=4 * emb_dim=192
NUM_FRAMES = 3
DEPTH = 6
HEADS = 16
DIM_HEAD = 64
INNER_DIM = HEADS * DIM_HEAD       # 1024
MLP_DIM = 2048
LN_EPS_AFFINE = 1e-5
LN_EPS_NOAFFINE = 1e-6


# ============================================================
# Atomic ops
# ============================================================

def _silu(x):
    return x * jax.nn.sigmoid(x)


def _layernorm_noaffine(x, eps=LN_EPS_NOAFFINE):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


# ============================================================
# Causal attention (matches PyTorch F.scaled_dot_product_attention is_causal=True)
# ============================================================

class CausalAttention(fnn.Module):
    """Self-attention with internal pre-LN (affine) and causal masking.

    Matches PyTorch ``Attention`` in module.py: LN-affine input, single QKV
    projection (no bias), reshape to (B, H, T, D_h), softmax with causal mask,
    output projection back to ``dim``.
    """

    dim: int
    heads: int
    dim_head: int

    @fnn.compact
    def __call__(self, x):
        inner = self.heads * self.dim_head
        # Pre-norm (with affine), matching `Attention.norm`
        x = fnn.LayerNorm(epsilon=LN_EPS_AFFINE, name="norm")(x)
        # Combined QKV proj (no bias)
        qkv = fnn.Dense(inner * 3, use_bias=False, name="to_qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        B, T, _ = x.shape
        # (B, T, H, D_h) -> (B, H, T, D_h)
        q = q.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        scale = jnp.float32(self.dim_head) ** -0.5
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        # Causal mask: position q attends to k <= q
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(causal_mask, attn, -jnp.inf)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, inner)
        out = fnn.Dense(self.dim, name="to_out_0")(out)
        return out


# ============================================================
# Feed-forward (matches PyTorch ``FeedForward``)
# ============================================================

class FeedForward(fnn.Module):
    dim: int
    hidden_dim: int

    @fnn.compact
    def __call__(self, x):
        x = fnn.LayerNorm(epsilon=LN_EPS_AFFINE, name="net_0_ln")(x)
        x = fnn.Dense(self.hidden_dim, name="net_1")(x)
        x = jax.nn.gelu(x, approximate=False)  # match PyTorch nn.GELU default
        x = fnn.Dense(self.dim, name="net_4")(x)
        return x


# ============================================================
# AdaLN-zero conditional block (matches ``ConditionalBlock``)
# ============================================================

class ConditionalBlock(fnn.Module):
    dim: int
    heads: int
    dim_head: int
    mlp_dim: int

    @fnn.compact
    def __call__(self, x, c):
        # SiLU(c) -> Linear(192 -> 6*192)
        cond = _silu(c)
        cond = fnn.Dense(6 * self.dim, name="adaLN_mod_1")(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            cond, 6, axis=-1
        )
        # Attention branch
        h = _layernorm_noaffine(x)
        h = _modulate(h, shift_msa, scale_msa)
        h = CausalAttention(
            dim=self.dim, heads=self.heads, dim_head=self.dim_head, name="attn"
        )(h)
        x = x + gate_msa * h
        # MLP branch
        h = _layernorm_noaffine(x)
        h = _modulate(h, shift_mlp, scale_mlp)
        h = FeedForward(dim=self.dim, hidden_dim=self.mlp_dim, name="mlp")(h)
        x = x + gate_mlp * h
        return x


# ============================================================
# AR predictor (pos embedding + N ConditionalBlocks + final LN)
# ============================================================

class ARPredictor(fnn.Module):
    num_frames: int = NUM_FRAMES
    dim: int = LATENT_DIM
    depth: int = DEPTH
    heads: int = HEADS
    dim_head: int = DIM_HEAD
    mlp_dim: int = MLP_DIM

    @fnn.compact
    def __call__(self, x, c):
        pos = self.param(
            "pos_embedding",
            lambda key: jnp.zeros((1, self.num_frames, self.dim), dtype=jnp.float32),
        )
        T = x.shape[1]
        x = x + pos[:, :T]
        for i in range(self.depth):
            x = ConditionalBlock(
                dim=self.dim, heads=self.heads, dim_head=self.dim_head,
                mlp_dim=self.mlp_dim, name=f"block_{i}",
            )(x, c)
        x = fnn.LayerNorm(epsilon=LN_EPS_AFFINE, name="final_ln")(x)
        return x


# ============================================================
# Action encoder (``Embedder`` in PyTorch)
# ============================================================

class ActionEncoder(fnn.Module):
    """Conv1d k=1 is equivalent to a Linear applied along the last dim."""

    input_dim: int = ACTION_RAW_DIM
    smoothed_dim: int = ACTION_SMOOTHED_DIM
    emb_dim: int = LATENT_DIM
    mlp_hidden: int = ACTION_MLP_HIDDEN

    @fnn.compact
    def __call__(self, a):
        a = fnn.Dense(self.smoothed_dim, name="patch_embed")(a)
        a = fnn.Dense(self.mlp_hidden, name="embed_0")(a)
        a = _silu(a)
        a = fnn.Dense(self.emb_dim, name="embed_2")(a)
        return a


# ============================================================
# pred_proj: Linear -> BatchNorm(frozen) -> GELU -> Linear
# ============================================================

class PredProjFrozenBN(fnn.Module):
    """MLP whose middle BatchNorm uses pre-baked running stats.

    At training time PyTorch's BatchNorm updated its stats; we copy those
    final running_mean / running_var / weight / bias into params and apply
    deterministic normalisation. This matches BatchNorm in *eval* mode.
    """

    in_dim: int = LATENT_DIM
    hidden_dim: int = MLP_DIM
    out_dim: int = LATENT_DIM
    bn_eps: float = 1e-5

    @fnn.compact
    def __call__(self, x):
        # x: (B*T, in_dim)
        x = fnn.Dense(self.hidden_dim, name="net_0")(x)
        rm = self.param("bn_running_mean",
                        lambda key: jnp.zeros(self.hidden_dim))
        rv = self.param("bn_running_var",
                        lambda key: jnp.ones(self.hidden_dim))
        gamma = self.param("bn_weight",
                           lambda key: jnp.ones(self.hidden_dim))
        beta = self.param("bn_bias",
                          lambda key: jnp.zeros(self.hidden_dim))
        x = (x - rm) * jax.lax.rsqrt(rv + self.bn_eps) * gamma + beta
        x = jax.nn.gelu(x, approximate=False)
        x = fnn.Dense(self.out_dim, name="net_3")(x)
        return x


# ============================================================
# Composite: takes (emb, raw_action), returns next-step emb
# ============================================================

class LeJEPAJaxForward(fnn.Module):
    """One WM-step prediction in JAX.

    Inputs:
      emb:        (B, T, LATENT_DIM)  -- current latent context
      action_raw: (B, T, ACTION_RAW_DIM)  -- raw 25-D action chunks

    Returns: (B, T, LATENT_DIM)
        Same convention as Torch ``JEPA.predict(emb, action_encoder(action))``.
        The last position is the predicted *next* latent given the full context.
    """

    @fnn.compact
    def __call__(self, emb, action_raw):
        c = ActionEncoder(name="action_encoder")(action_raw)  # (B, T, 192)
        z = ARPredictor(name="predictor")(emb, c)             # (B, T, 192)
        B, T, D = z.shape
        z = PredProjFrozenBN(name="pred_proj")(z.reshape(B * T, D))
        z = z.reshape(B, T, -1)
        return z


# ============================================================
# Torch -> JAX param-tree converter
# ============================================================

def torch_state_dict_to_jax_params(sd):
    """Convert a torch state_dict (from load_jepa) into a Flax FrozenDict
    matching ``LeJEPAJaxForward``'s init structure.

    Only consumes keys with prefix ``predictor.``, ``action_encoder.``,
    ``pred_proj.``. The visual encoder + projector are ignored (not used by
    the JAX forward path).
    """
    p = {}

    # ---- action_encoder ----
    # patch_embed is Conv1d k=1 with weight (out=10, in=25, k=1).
    # Drop the kernel dim -> (out, in). Flax Dense kernel is (in, out).
    pe_w = _to_np(sd["action_encoder.patch_embed.weight"]).squeeze(-1)  # (10, 25)
    pe_b = _to_np(sd["action_encoder.patch_embed.bias"])                # (10,)
    e0_w = _to_np(sd["action_encoder.embed.0.weight"])                  # (768, 10)
    e0_b = _to_np(sd["action_encoder.embed.0.bias"])                    # (768,)
    e2_w = _to_np(sd["action_encoder.embed.2.weight"])                  # (192, 768)
    e2_b = _to_np(sd["action_encoder.embed.2.bias"])                    # (192,)
    p["action_encoder"] = {
        "patch_embed": {"kernel": pe_w.T, "bias": pe_b},
        "embed_0":     {"kernel": e0_w.T, "bias": e0_b},
        "embed_2":     {"kernel": e2_w.T, "bias": e2_b},
    }

    # ---- predictor ----
    pred = {"pos_embedding": _to_np(sd["predictor.pos_embedding"])}     # (1, 3, 192)
    for i in range(DEPTH):
        prefix = f"predictor.transformer.layers.{i}"
        block = {
            "adaLN_mod_1": {
                "kernel": _to_np(sd[f"{prefix}.adaLN_modulation.1.weight"]).T,
                "bias":   _to_np(sd[f"{prefix}.adaLN_modulation.1.bias"]),
            },
            "attn": {
                "norm": {
                    "scale": _to_np(sd[f"{prefix}.attn.norm.weight"]),
                    "bias":  _to_np(sd[f"{prefix}.attn.norm.bias"]),
                },
                "to_qkv": {
                    "kernel": _to_np(sd[f"{prefix}.attn.to_qkv.weight"]).T,
                },
                "to_out_0": {
                    "kernel": _to_np(sd[f"{prefix}.attn.to_out.0.weight"]).T,
                    "bias":   _to_np(sd[f"{prefix}.attn.to_out.0.bias"]),
                },
            },
            "mlp": {
                "net_0_ln": {
                    "scale": _to_np(sd[f"{prefix}.mlp.net.0.weight"]),
                    "bias":  _to_np(sd[f"{prefix}.mlp.net.0.bias"]),
                },
                "net_1": {
                    "kernel": _to_np(sd[f"{prefix}.mlp.net.1.weight"]).T,
                    "bias":   _to_np(sd[f"{prefix}.mlp.net.1.bias"]),
                },
                "net_4": {
                    "kernel": _to_np(sd[f"{prefix}.mlp.net.4.weight"]).T,
                    "bias":   _to_np(sd[f"{prefix}.mlp.net.4.bias"]),
                },
            },
        }
        pred[f"block_{i}"] = block
    pred["final_ln"] = {
        "scale": _to_np(sd["predictor.transformer.norm.weight"]),
        "bias":  _to_np(sd["predictor.transformer.norm.bias"]),
    }
    p["predictor"] = pred

    # ---- pred_proj ----
    p["pred_proj"] = {
        "net_0": {
            "kernel": _to_np(sd["pred_proj.net.0.weight"]).T,
            "bias":   _to_np(sd["pred_proj.net.0.bias"]),
        },
        "net_3": {
            "kernel": _to_np(sd["pred_proj.net.3.weight"]).T,
            "bias":   _to_np(sd["pred_proj.net.3.bias"]),
        },
        "bn_weight":       _to_np(sd["pred_proj.net.1.weight"]),
        "bn_bias":         _to_np(sd["pred_proj.net.1.bias"]),
        "bn_running_mean": _to_np(sd["pred_proj.net.1.running_mean"]),
        "bn_running_var":  _to_np(sd["pred_proj.net.1.running_var"]),
    }
    return {"params": p}


def _to_np(t):
    """Convert torch.Tensor or numpy / jnp array to a contiguous float32 ndarray."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(t).astype(np.float32)


def _to_torch_tensor(arr):
    """Convert a numpy / jnp array to a torch.Tensor (cpu, float32)."""
    import torch
    return torch.from_numpy(np.asarray(arr).astype(np.float32))


def jax_params_to_torch_state_dict(params):
    """Reverse of ``torch_state_dict_to_jax_params``.

    Takes a Flax param tree (with keys ``action_encoder``, ``predictor``,
    ``pred_proj`` as produced by ``LeJEPAJaxForward``'s ``init``) and returns
    a PyTorch state_dict whose keys mirror the original ``jepa.state_dict()``
    output (prefixed ``action_encoder.``, ``predictor.``, ``pred_proj.``).

    The torch BatchNorm running stats (which were folded into named params on
    the JAX side) are written back to ``pred_proj.net.1.running_mean`` etc.
    as plain tensors. ``num_batches_tracked`` is set to a placeholder of 1.
    """
    import torch

    sd = {}

    # ---- action_encoder ----
    ae = params["action_encoder"]
    # JAX patch_embed kernel: (in=25, out=10). Torch: (out=10, in=25, k=1).
    sd["action_encoder.patch_embed.weight"] = _to_torch_tensor(ae["patch_embed"]["kernel"]).t().unsqueeze(-1)
    sd["action_encoder.patch_embed.bias"]   = _to_torch_tensor(ae["patch_embed"]["bias"])
    sd["action_encoder.embed.0.weight"] = _to_torch_tensor(ae["embed_0"]["kernel"]).t()
    sd["action_encoder.embed.0.bias"]   = _to_torch_tensor(ae["embed_0"]["bias"])
    sd["action_encoder.embed.2.weight"] = _to_torch_tensor(ae["embed_2"]["kernel"]).t()
    sd["action_encoder.embed.2.bias"]   = _to_torch_tensor(ae["embed_2"]["bias"])

    # ---- predictor ----
    pred = params["predictor"]
    sd["predictor.pos_embedding"] = _to_torch_tensor(pred["pos_embedding"])
    for i in range(DEPTH):
        b = pred[f"block_{i}"]
        sd[f"predictor.transformer.layers.{i}.adaLN_modulation.1.weight"] = _to_torch_tensor(b["adaLN_mod_1"]["kernel"]).t()
        sd[f"predictor.transformer.layers.{i}.adaLN_modulation.1.bias"]   = _to_torch_tensor(b["adaLN_mod_1"]["bias"])
        sd[f"predictor.transformer.layers.{i}.attn.norm.weight"] = _to_torch_tensor(b["attn"]["norm"]["scale"])
        sd[f"predictor.transformer.layers.{i}.attn.norm.bias"]   = _to_torch_tensor(b["attn"]["norm"]["bias"])
        sd[f"predictor.transformer.layers.{i}.attn.to_qkv.weight"] = _to_torch_tensor(b["attn"]["to_qkv"]["kernel"]).t()
        sd[f"predictor.transformer.layers.{i}.attn.to_out.0.weight"] = _to_torch_tensor(b["attn"]["to_out_0"]["kernel"]).t()
        sd[f"predictor.transformer.layers.{i}.attn.to_out.0.bias"]   = _to_torch_tensor(b["attn"]["to_out_0"]["bias"])
        sd[f"predictor.transformer.layers.{i}.mlp.net.0.weight"] = _to_torch_tensor(b["mlp"]["net_0_ln"]["scale"])
        sd[f"predictor.transformer.layers.{i}.mlp.net.0.bias"]   = _to_torch_tensor(b["mlp"]["net_0_ln"]["bias"])
        sd[f"predictor.transformer.layers.{i}.mlp.net.1.weight"] = _to_torch_tensor(b["mlp"]["net_1"]["kernel"]).t()
        sd[f"predictor.transformer.layers.{i}.mlp.net.1.bias"]   = _to_torch_tensor(b["mlp"]["net_1"]["bias"])
        sd[f"predictor.transformer.layers.{i}.mlp.net.4.weight"] = _to_torch_tensor(b["mlp"]["net_4"]["kernel"]).t()
        sd[f"predictor.transformer.layers.{i}.mlp.net.4.bias"]   = _to_torch_tensor(b["mlp"]["net_4"]["bias"])
    sd["predictor.transformer.norm.weight"] = _to_torch_tensor(pred["final_ln"]["scale"])
    sd["predictor.transformer.norm.bias"]   = _to_torch_tensor(pred["final_ln"]["bias"])

    # ---- pred_proj ----
    pp = params["pred_proj"]
    sd["pred_proj.net.0.weight"] = _to_torch_tensor(pp["net_0"]["kernel"]).t()
    sd["pred_proj.net.0.bias"]   = _to_torch_tensor(pp["net_0"]["bias"])
    sd["pred_proj.net.1.weight"]              = _to_torch_tensor(pp["bn_weight"])
    sd["pred_proj.net.1.bias"]                = _to_torch_tensor(pp["bn_bias"])
    sd["pred_proj.net.1.running_mean"]        = _to_torch_tensor(pp["bn_running_mean"])
    sd["pred_proj.net.1.running_var"]         = _to_torch_tensor(pp["bn_running_var"])
    sd["pred_proj.net.1.num_batches_tracked"] = torch.tensor(1, dtype=torch.long)
    sd["pred_proj.net.3.weight"] = _to_torch_tensor(pp["net_3"]["kernel"]).t()
    sd["pred_proj.net.3.bias"]   = _to_torch_tensor(pp["net_3"]["bias"])

    return sd


# ============================================================
# Differentiable multi-step rollout
# ============================================================

def jax_rollout(wm_model, wm_params, z_0, action_chunks):
    """Autoregressive multi-step rollout in JAX.

    Args:
        wm_model:      An instance of ``LeJEPAJaxForward``.
        wm_params:     Its frozen params dict.
        z_0:           (B, LATENT_DIM) starting latent.
        action_chunks: (B, H, ACTION_RAW_DIM) action chunks; one per WM step.

    Returns:
        (B, H, LATENT_DIM) -- predicted latents at steps 1, 2, ..., H. Fully
        differentiable w.r.t. ``action_chunks`` (and ``wm_params``, though we
        never take gradients w.r.t. WM in the frozen-WM use case).
    """

    def body(z, a):
        # WM expects (B, T, ...) shape; do one step with T=1
        z_in = z[:, None, :]
        a_in = a[:, None, :]
        z_next = wm_model.apply(wm_params, z_in, a_in)[:, -1, :]
        return z_next, z_next

    # scan iterates over the H axis. Move it to position 0 for scan.
    chunks_h_first = jnp.transpose(action_chunks, (1, 0, 2))  # (H, B, 25)
    _, zs = jax.lax.scan(body, z_0, chunks_h_first)            # zs: (H, B, 192)
    return jnp.transpose(zs, (1, 0, 2))                        # (B, H, 192)


# ============================================================
# Convenience loader: returns (model, params) ready for .apply
# ============================================================

def make_wm_trainstate(ckpt_path, lr=1e-5, weight_decay=1e-3):
    """Build a JAX TrainState wrapping the WM forward pass and optimizer.

    Use this when you want to *update* the WM (joint training). For a frozen
    WM use ``load_wm_jax`` instead. The optimizer matches the original WM
    fine-tune recipe: AdamW with the given LR (default 1e-5) and weight decay.

    Returns:
        ``utils.flax_utils.TrainState`` whose ``params`` are initialised from
        the supplied checkpoint and whose ``apply_fn`` is the ``LeJEPAJaxForward``
        forward pass. Use as: ``wm_state.apply_fn({"params": wm_state.params},
        emb, action_raw)``.
    """
    import optax
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.append(here)
    from utils.flax_utils import TrainState

    _, params_pack = _load_wm_jax_raw(ckpt_path)
    params = params_pack["params"]
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    wm_model = LeJEPAJaxForward()
    return TrainState.create(wm_model, params, tx=tx)


def _load_wm_jax_raw(ckpt_path):
    """Internal helper that returns (model, {'params': ...}) -- same as
    ``load_wm_jax`` but factored out so ``make_wm_trainstate`` can reuse it."""
    return load_wm_jax(ckpt_path)


def load_wm_jax(ckpt_path):
    """One-stop helper.

    Loads the WM in PyTorch (via the existing load_jepa helper, which already
    handles the state_dict-vs-pickle fallback), extracts its state_dict, and
    returns an initialised ``LeJEPAJaxForward`` model paired with its Flax
    params dict. Use as:

        model, params = load_wm_jax(ckpt)
        z_next = model.apply(params, z, action_raw)

    The returned object is a frozen forward model. The caller is responsible
    for keeping ``params`` around (e.g., passing through Flax pytrees,
    ``jax.jit``-ing the apply, etc.).
    """
    # Local imports to avoid forcing torch on JAX-only callers
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.append(here)
    from envs.jepa_loader import load_jepa

    jepa_t = load_jepa(ckpt_path, device="cpu", img_size=224, patch_size=14)
    sd = jepa_t.state_dict() if hasattr(jepa_t, "state_dict") else jepa_t.jepa.state_dict()
    # Handle shim wrappers that re-prefix everything
    if not any(k.startswith("predictor.") for k in sd):
        for prefix in ("jepa.", "model."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                break
    params = torch_state_dict_to_jax_params(sd)
    model = LeJEPAJaxForward()
    return model, params
