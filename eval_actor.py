import os
os.environ.setdefault("MUJOCO_GL", "egl")

import re
import sys

# jepa.py lives in the parent directory. torch.load() unpickles the checkpoint
# and needs to resolve the 'jepa' module class, so the parent must be on sys.path
# before any model is loaded.
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import time
from pathlib import Path
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms
from sklearn import preprocessing

import stable_pretraining as spt
import stable_worldmodel as swm

# Import high-level policy architectures from their training module
from train_high_level import MLPHighLevel, DiffusionHighLevel


# ==========================================================
# 0. OGBench WM LOADER (for state_dict .ckpt checkpoints)
# ==========================================================
def load_ogbench_jepa(ckpt_path, device="cuda"):
    """Load an OGBench-style pytorch-lightning state_dict checkpoint into a bare JEPA model.

    The checkpoint stores weights under 'state_dict' with a 'model.' prefix added by
    the spt.Module wrapper. Architecture: ViT-Tiny, patch_size=8, img_size=64.
    This mirrors the loading logic in analyse_ogbench_wm.py.
    """
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=8, image_size=64, pretrained=False, use_mask_token=False
    )
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj  = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder, projector=projector, pred_proj=pred_proj)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("model.")}
    model.load_state_dict(raw_sd, strict=True)
    print(f"  Loaded OGBench JEPA from {ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')})")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ==========================================================
# 1. ARCHITECTURE DEFINITIONS (must match train.py exactly)
# ==========================================================
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GoalConditionedActor(nn.Module):
    """The Policy Network: pi(action | z_curr, z_goal) — must match train.py exactly."""
    def __init__(self, latent_dim=192, action_dim=25, hidden_dim=256, action_scale=3.0):
        super(GoalConditionedActor, self).__init__()
        self.action_scale = action_scale

        input_dim = latent_dim * 2
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

        torch.nn.init.uniform_(self.mean_linear.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.mean_linear.bias, 0)
        torch.nn.init.uniform_(self.log_std_linear.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.log_std_linear.bias, -1.0)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, goal):
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean) * self.action_scale


# ==========================================================
# 2. THE BRIDGE POLICY
# ==========================================================
def img_transform(img_size):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])

class LatentActorPolicy(swm.policy.BasePolicy):
    """
    Bridges the physical swm.World simulator and our 192-D Latent SAC Actor.

    Inherits BasePolicy so _prepare_info handles HWC→CHW transposition and
    ImageNet normalisation automatically for both 'pixels' and 'goal' keys.

    The actor outputs 25-dim actions (5 action_block × 5 joints). swm.World
    calls get_action once per physical step and expects a [num_envs, 5] slice,
    so we buffer all 5 timesteps and pop one per call.
    """
    def __init__(self, cfg, actor_ckpt_path, action_scaler, device="cuda"):
        super().__init__()
        self.device = device
        self.action_scaler = action_scaler
        self._action_buffer = []

        # Setting self.transform makes BasePolicy._prepare_info apply these
        # transforms to 'pixels' and 'goal' in info_dict automatically,
        # including the HWC→CHW permutation that precedes the transform call.
        t = img_transform(cfg.eval.img_size)
        self.transform = {"pixels": t, "goal": t}

        print("Loading JEPA Vision Encoder...")
        if cfg.policy.endswith(".ckpt"):
            self.jepa_model = load_ogbench_jepa(cfg.policy, device)
        else:
            self.jepa_model = swm.policy.AutoCostModel(cfg.policy).to(device)
            self.jepa_model.eval()
            self.jepa_model.requires_grad_(False)
            self.jepa_model.interpolate_pos_encoding = True

        # Load action_scale from training_config.json if present (env-trained checkpoints use 1.0)
        import json as _json
        _config_path = os.path.join(os.path.dirname(actor_ckpt_path), "training_config.json")
        action_scale = 3.0
        if os.path.exists(_config_path):
            with open(_config_path) as _f:
                action_scale = _json.load(_f).get("action_scale", 3.0)
            print(f"  training_config.json found: action_scale={action_scale}")

        print(f"Loading Trained Actor from: {actor_ckpt_path}")
        self.actor = GoalConditionedActor(latent_dim=192, action_dim=25, action_scale=action_scale).to(device)
        self.actor.load_state_dict(torch.load(actor_ckpt_path, map_location=device))
        self.actor.eval()

    def reset(self):
        self._action_buffer = []

    def get_action(self, info_dict):
        """
        Called by swm.World.step() with batched observations for all envs.

        info_dict contains (at minimum):
          'pixels': [num_envs, H, W, C] uint8 numpy — current rendered frame
          'goal':   [num_envs, H, W, C] uint8 numpy — goal rendered frame

        Returns: np.ndarray [num_envs, 5] physical joint actions for this step.
        """
        if not self._action_buffer:
            # _prepare_info: HWC→CHW, applies img_transform, converts to torch
            info_dict = self._prepare_info(info_dict)

            # _prepare_info already restores the time dimension, giving [N, T, C, H, W].
            # jepa.encode() does rearrange("b t ... -> (b t) ...") internally before
            # the ViT, so we must pass [N, T, C, H, W] directly — no unsqueeze needed.
            pixels = info_dict["pixels"].to(self.device)  # [N, T, C, H, W]
            goal   = info_dict["goal"].to(self.device)    # [N, T, C, H, W]

            with torch.no_grad():
                z_curr = self.jepa_model.encode({"pixels": pixels})["emb"][:, -1]  # [N, 192]
                z_goal = self.jepa_model.encode({"pixels": goal})["emb"][:, -1]    # [N, 192]

                # Deterministic mean action (no exploration noise during eval)
                _, _, actions = self.actor.sample(z_curr, z_goal)  # [N, 25]

            actions_np = actions.cpu().numpy()  # [N, 25]
            N = actions_np.shape[0]
            # The scaler was fit on 5-dim per-timestep actions (scale_.shape=(5,)).
            # The 25-dim actor output is 5 consecutive 5-joint actions (action_block=5).
            # Reshape to [N*5, 5] so inverse_transform broadcasts correctly, then restore.
            actions_physical = self.action_scaler.inverse_transform(
                actions_np.reshape(N * 5, 5)
            ).reshape(N, 5, 5)  # [N, action_block=5, joints=5]

            # Buffer all 5 timesteps; each entry is [N, 5]
            for t in range(5):
                self._action_buffer.append(actions_physical[:, t, :])

        return self._action_buffer.pop(0)  # [num_envs, 5]


# ==========================================================
# 3. HIERARCHICAL POLICY (high-level subgoal + low-level actor)
# ==========================================================
def _load_high_level(model_type, gap, ckpt_dir, device):
    """
    Loads MLP or Diffusion high-level policy for a given gap.
    ckpt_dir is the root checkpoints_high_level/ directory.
    Returns the loaded model in eval mode.
    """
    ckpt_path = Path(ckpt_dir) / f"{model_type}_gap{gap}" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"High-level checkpoint not found at '{ckpt_path}'.\n"
            f"Run train_high_level.py --model_type {model_type} --gap {gap} first."
        )
    if model_type == "mlp":
        model = MLPHighLevel().to(device)
    elif model_type == "diffusion":
        model = DiffusionHighLevel().to(device)
    else:
        raise ValueError(f"Unknown high-level model type: '{model_type}'")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.requires_grad_(False)
    print(f"Loaded {model_type} high-level policy (gap={gap}) from {ckpt_path}")
    return model


class HierarchicalLatentActorPolicy(swm.policy.BasePolicy):
    """
    Two-level hierarchical policy.

    High level  — proposes a latent subgoal z_subgoal every `gap` WM steps
                  by calling the trained MLP or Diffusion model on (z_curr, z_final_goal).
                  Uses only the learned model at runtime — no expert data.

    Low level   — the frozen gap-trained SAC actor runs toward z_subgoal for up
                  to `gap` WM steps, then the high level replans.

    One WM step = one action block = one refill of _action_buffer (5 physical steps).
    """
    def __init__(self, cfg, actor_ckpt_path, action_scaler,
                 high_level_model, gap, device="cuda"):
        super().__init__()
        self.device        = device
        self.action_scaler = action_scaler
        self.gap           = gap
        self.high_level    = high_level_model

        t = img_transform(cfg.eval.img_size)
        self.transform = {"pixels": t, "goal": t}

        print("Loading JEPA Vision Encoder...")
        if cfg.policy.endswith(".ckpt"):
            self.jepa_model = load_ogbench_jepa(cfg.policy, device)
        else:
            self.jepa_model = swm.policy.AutoCostModel(cfg.policy).to(device)
            self.jepa_model.eval()
            self.jepa_model.requires_grad_(False)
            self.jepa_model.interpolate_pos_encoding = True

        import json as _json
        _config_path = os.path.join(os.path.dirname(actor_ckpt_path), "training_config.json")
        action_scale = 3.0
        if os.path.exists(_config_path):
            with open(_config_path) as _f:
                action_scale = _json.load(_f).get("action_scale", 3.0)
            print(f"  training_config.json found: action_scale={action_scale}")

        print(f"Loading Trained Actor from: {actor_ckpt_path}")
        self.actor = GoalConditionedActor(latent_dim=192, action_dim=25, action_scale=action_scale).to(device)
        self.actor.load_state_dict(torch.load(actor_ckpt_path, map_location=device))
        self.actor.eval()

        self._action_buffer      = []
        self._wm_steps_to_subgoal = 0   # WM steps taken since last high-level call
        self._z_subgoal           = None

    def reset(self):
        self._action_buffer       = []
        self._wm_steps_to_subgoal = 0
        self._z_subgoal           = None

    def get_action(self, info_dict):
        if not self._action_buffer:
            info_dict = self._prepare_info(info_dict)

            pixels = info_dict["pixels"].to(self.device)  # [N, T, C, H, W]
            goal   = info_dict["goal"].to(self.device)    # [N, T, C, H, W]

            with torch.no_grad():
                z_curr      = self.jepa_model.encode({"pixels": pixels})["emb"][:, -1]  # [N, 192]
                z_final_goal = self.jepa_model.encode({"pixels": goal})["emb"][:, -1]   # [N, 192]

                # Replan if: first step, budget exhausted, or subgoal already reached.
                # The third condition is critical — without it the low-level wastes
                # its remaining budget holding at an already-achieved intermediate state.
                subgoal_reached = (
                    self._z_subgoal is not None and
                    torch.norm(z_curr - self._z_subgoal, p=2, dim=-1).max().item() < 2.0
                )
                if (self._z_subgoal is None
                        or self._wm_steps_to_subgoal >= self.gap
                        or subgoal_reached):
                    self._z_subgoal           = self.high_level.predict(z_curr, z_final_goal)
                    self._wm_steps_to_subgoal = 0

                # Low-level actor targets the subgoal, not the final goal
                _, _, actions = self.actor.sample(z_curr, self._z_subgoal)  # [N, 25]

            self._wm_steps_to_subgoal += 1

            actions_np = actions.cpu().numpy()
            N = actions_np.shape[0]
            actions_physical = self.action_scaler.inverse_transform(
                actions_np.reshape(N * 5, 5)
            ).reshape(N, 5, 5)

            for t in range(5):
                self._action_buffer.append(actions_physical[:, t, :])

        return self._action_buffer.pop(0)


# ==========================================================
# 4. EVALUATION LOOP
# ==========================================================
def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])

def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )

# config_path is relative to this source file, so "../config/eval" resolves
# correctly whether the script is run from the repo root or from this subdirectory.
@hydra.main(version_base=None, config_path="../config/eval", config_name="cube")
def run(cfg: DictConfig):
    # ++checkpoint_dir=<path> can be passed on the command line to override this.
    # Paths are resolved relative to the original working dir (before Hydra changes it).
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints_curriculum")
    actor_ckpt_path = hydra.utils.to_absolute_path(
        os.path.join(checkpoint_dir, "actor_policy.pth")
    )

    if not os.path.exists(actor_ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{actor_ckpt_path}'. "
            f"Pass ++checkpoint_dir=<dir> to specify the experiment folder."
        )

    # Derive a human-readable experiment name for results directory.
    # Incorporate the high-level model type so flat/mlp/diffusion runs
    # don't overwrite each other when using the same low-level checkpoint.
    exp_name = os.path.basename(checkpoint_dir).replace("checkpoints_", "", 1)
    high_level_model_type = cfg.get("high_level_model_type", "none")
    if high_level_model_type != "none":
        exp_name = f"{exp_name}_hl_{high_level_model_type}"
    results_path = Path(hydra.utils.to_absolute_path(f"eval_results_{exp_name}"))
    results_path.mkdir(parents=True, exist_ok=True)

    # Create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(cfg.world.width, cfg.world.height))

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    # Fit the action scaler on expert data (same as training pipeline)
    print("Fitting Action Scaler to dataset...")
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(action_data)

    # --- Detect fixed-gap from checkpoint directory name ---
    # e.g. "checkpoints_her_fixed_gap_24_sparse" → gap=24
    gap_match = re.search(r'fixed_gap_(\d+)', checkpoint_dir)
    fixed_gap = int(gap_match.group(1)) if gap_match else None

    # --- Choose flat or hierarchical policy ---
    high_level_model_type = cfg.get("high_level_model_type", "none")
    high_level_ckpt_dir   = cfg.get("high_level_ckpt_dir",
                                    hydra.utils.to_absolute_path("checkpoints_high_level"))

    if fixed_gap is not None and high_level_model_type != "none":
        print(f"\nFixed-gap={fixed_gap} checkpoint detected.")
        print(f"Building hierarchical policy with {high_level_model_type} high-level...")
        high_level_model = _load_high_level(
            model_type=high_level_model_type,
            gap=fixed_gap,
            ckpt_dir=high_level_ckpt_dir,
            device="cuda",
        )
        policy = HierarchicalLatentActorPolicy(
            cfg, actor_ckpt_path, action_scaler=action_scaler,
            high_level_model=high_level_model, gap=fixed_gap,
        )
    else:
        if fixed_gap is not None:
            print(f"\nFixed-gap={fixed_gap} checkpoint detected but high_level_model_type='none'.")
            print("Running flat policy (no high-level). Pass ++high_level_model_type=mlp to enable.")
        policy = LatentActorPolicy(cfg, actor_ckpt_path, action_scaler=action_scaler)

    # Sample evaluation starting points
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)

    print(f"\nEvaluating experiment: {exp_name}")
    print(f"Checkpoint: {actor_ckpt_path}")
    print(f"Results will be saved to: {results_path}\n")

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
    elapsed = time.time() - start_time

    print("\n==== FINAL METRICS ====")
    print(metrics)

    out_file = results_path / "results.txt"
    with out_file.open("a") as f:
        f.write(f"\n==== EXPERIMENT: {exp_name} ====\n")
        f.write(f"checkpoint: {actor_ckpt_path}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.2f} seconds\n")

    print(f"\nResults and videos saved to: {results_path}")


if __name__ == "__main__":
    run()
