"""
eval_wgsp_cem_ogbench.py
WGSP-scored CEM planning on OGBench.

Follows eval.py exactly (Hydra config, swm.World, WorldModelPolicy,
evaluate_from_dataset) so results are directly comparable to the
LeWorldModel paper baseline.

The only difference: instead of AutoCostModel's default pixel-reconstruction
cost, we score CEM endpoints with the WGSP objective

    J(z_k, g) = V(z_k, φ(z_k, g)) − β‖z_k − g‖₂

where V + φ come from a value_offline.pt checkpoint trained by
train_value_offline.py.

Usage:
    python latent_hindsight_rl/eval_wgsp_cem_ogbench.py \\
        ++policy=cube/lejepa \\
        ++value_ckpt=/path/to/value_offline.pt \\
        ++beta_geom=0.1

    # Optionally override to use_geometric_term only (no V):
        ++use_v_in_J=False ++beta_geom=0.1

    # Or use V only (no geometric term):
        ++use_geometric_term=False
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path: sys.path.insert(0, _THIS_DIR)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from train_hiql_wgsp import (
    GoalRep,
    EnsembleValue,
    _score_endpoints,
    _wm_predict,
)


# =============================================================================
# Image transform — identical to eval.py
# =============================================================================

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


# =============================================================================
# WGSPCostModel: wraps AutoCostModel, overrides endpoint scoring with WGSP J
# =============================================================================

class WGSPCostModel(nn.Module):
    """AutoCostModel wrapper that replaces get_cost with WGSP endpoint scoring.

    Implements the Costable protocol expected by CEMSolver:
        get_cost(info_dict, action_candidates) -> Tensor[batch, num_samples]

    At planning time the LeWM encoder converts pixel observations to latents,
    the WM predictor rolls out the candidate action sequences, and the WGSP
    scoring function J = V(z_k, φ(z_k, g)) − β‖z_k − g‖₂ ranks the
    rollout endpoints.  The cost returned is -J so that CEM minimises cost.

    Args:
        base_model  : AutoCostModel — provides encode(), action_encoder, predict.
        value_net   : EnsembleValue — V(z, rep) → [B, n_heads].
        goal_rep    : GoalRep — φ(z, z_goal) → rep.
        beta_geom   : Weight on the geometric term (default 0.1).
        use_v_in_J  : If False, V term is dropped (geometric-only scoring).
        use_geom    : If False, geometric term is dropped (V-only scoring).
        device      : torch device.
    """

    def __init__(
        self,
        base_model,
        value_net: EnsembleValue,
        goal_rep: GoalRep,
        beta_geom: float = 0.1,
        use_v_in_J: bool = True,
        use_geom: bool = True,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.base_model = base_model
        self.value_net  = value_net
        self.goal_rep   = goal_rep
        self.beta_geom  = beta_geom
        self.use_v_in_J = use_v_in_J
        self.use_geom   = use_geom
        self.device     = device

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute WGSP endpoint cost for CEM candidates.

        Args:
            info_dict         : Each value has shape (batch, num_samples, ...).
                                Must contain 'pixels' and 'goal' as normalised
                                float tensors (applied by WorldModelPolicy._prepare_info).
            action_candidates : (batch, num_samples, horizon, action_dim)
                                action_dim = raw_action_dim * action_block.
                                For OGBench cube: 5 * 5 = 25.

        Returns:
            costs : (batch, num_samples) — lower is better (= -J).
        """
        pixels = info_dict.get('pixels')
        goal   = info_dict.get('goal')

        if pixels is None or goal is None:
            raise ValueError("info_dict must contain 'pixels' and 'goal'.")

        # pixels/goal shape: (B, N, T, C, H, W) — T is history_size
        B, N = action_candidates.shape[:2]
        horizon = action_candidates.shape[2]

        # Use the last history frame for encoding
        # Reshape to (B*N, 1, C, H, W) for the WM encoder
        def _encode_last_frame(img_tensor):
            # img_tensor: (B, N, T, C, H, W)
            last = img_tensor[:, :, -1]              # (B, N, C, H, W)
            flat = last.reshape(B * N, *last.shape[2:]).unsqueeze(1)  # (B*N, 1, C, H, W)
            emb = self.base_model.encode({'pixels': flat})['emb']     # (B*N, 1, 192)
            return emb[:, -1]                                          # (B*N, 192)

        z = _encode_last_frame(pixels)           # (B*N, 192) — current latent
        z_goal = _encode_last_frame(goal)        # (B*N, 192) — goal latent

        # Roll out the WM for each horizon step
        # action_candidates: (B, N, horizon, 25)
        for h in range(horizon):
            chunk = action_candidates[:, :, h, :]            # (B, N, 25)
            chunk_flat = chunk.reshape(B * N, -1)             # (B*N, 25)
            z = _wm_predict(self.base_model, z, chunk_flat)   # (B*N, 192)

        # Score endpoints with WGSP objective
        scores = _score_endpoints(
            z_k=z,
            g_ult=z_goal,
            value_net=self.value_net,
            goal_rep=self.goal_rep,
            beta_geom=self.beta_geom,
            use_geometric_term=self.use_geom,
            use_v_in_J=self.use_v_in_J,
            lambda_mopo=0.0,
        )                                                      # (B*N,)

        # CEM minimises cost, so return -J
        cost = -scores.reshape(B, N)
        return cost

    # Expose the WM attributes and forward for compatibility with other
    # stable_worldmodel infrastructure (e.g., WorldModelPolicy internals).
    def encode(self, *args, **kwargs):
        return self.base_model.encode(*args, **kwargs)

    @property
    def action_encoder(self):
        return self.base_model.action_encoder

    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)


# =============================================================================
# Dataset helpers — verbatim from eval.py
# =============================================================================

def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


# =============================================================================
# Main
# =============================================================================

@hydra.main(version_base=None, config_path="../config/eval", config_name="cube")
def run(cfg: DictConfig):
    """WGSP-CEM evaluation — directly comparable to eval.py."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be <= eval_budget"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── WGSP-specific overrides ──────────────────────────────────────────────
    value_ckpt       = cfg.get("value_ckpt", None)
    beta_geom        = float(cfg.get("beta_geom",        0.1))
    use_v_in_J       = bool(cfg.get("use_v_in_J",        True))
    use_geom         = bool(cfg.get("use_geometric_term", True))

    if value_ckpt is None:
        raise ValueError(
            "Must provide ++value_ckpt=/path/to/value_offline.pt "
            "(produced by train_value_offline.py).")

    # ── World environment ────────────────────────────────────────────────────
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # ── Image transforms ─────────────────────────────────────────────────────
    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # ── Load V + φ from offline checkpoint ───────────────────────────────────
    print(f"Loading V + φ from {value_ckpt} ...")
    vc = torch.load(value_ckpt, map_location='cpu', weights_only=False)
    rep_dim = vc['rep_dim']
    n_heads = vc['n_heads']

    goal_rep = GoalRep(
        latent_dim=192, rep_dim=rep_dim,
        hidden_dims=(512, 512, 512), layer_norm=True).to(device).eval()
    goal_rep.load_state_dict(vc['goal_rep'])
    goal_rep.requires_grad_(False)

    value_net = EnsembleValue(
        latent_dim=192, rep_dim=rep_dim,
        hidden_dims=(512, 512, 512), n_heads=n_heads).to(device).eval()
    value_net.load_state_dict(vc['value'])
    value_net.requires_grad_(False)
    print(f"  rep_dim={rep_dim}, n_heads={n_heads}, "
          f"trained for {vc.get('step','?')} steps")

    # ── LeWM cost model ───────────────────────────────────────────────────────
    base_model = swm.policy.AutoCostModel(cfg.policy)
    base_model = base_model.to(device).eval()
    base_model.requires_grad_(False)
    if hasattr(base_model, "interpolate_pos_encoding"):
        base_model.interpolate_pos_encoding = True

    # ── WGSP cost model ───────────────────────────────────────────────────────
    wgsp_model = WGSPCostModel(
        base_model=base_model,
        value_net=value_net,
        goal_rep=goal_rep,
        beta_geom=beta_geom,
        use_v_in_J=use_v_in_J,
        use_geom=use_geom,
        device=device,
    ).to(device).eval()

    config = swm.policy.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=wgsp_model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # ── Results path ─────────────────────────────────────────────────────────
    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # ── Episode sampling — identical to eval.py ───────────────────────────────
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()

    print(metrics)

    # ── Write results — same output filename as eval.py ───────────────────────
    wgsp_suffix = (
        f"_wgsp_v{int(use_v_in_J)}_g{int(use_geom)}_b{beta_geom}"
        f"_step{vc.get('step','?')}"
    )
    stem = Path(cfg.output.filename).stem
    ext  = Path(cfg.output.filename).suffix
    out_filename = stem + wgsp_suffix + ext

    out_path = results_path / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"wgsp: beta_geom={beta_geom}  use_v={use_v_in_J}  "
                f"use_geom={use_geom}  value_ckpt={value_ckpt}\n")
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run()
