"""WMEnv: a gymnasium.Env that wraps a frozen JEPA world model.

Used in the E (Experiment) run as the online environment, replacing the
real OGBench env. One env.step consumes a 25-D action chunk atomically
and advances the latent by 5 real-env timesteps.

Numpy in, numpy out. Internally calls the PyTorch JEPA model via a thin
device round-trip. Reward is sparse (0/1) based on L2 distance to the
fixed task goal latent.

Actions are StandardScaler-normalized before feeding to action_encoder,
matching the convention in eval.py (WorldModelPolicy) and train.py
(get_column_normalizer transform). The WM was trained on normalized actions.
"""

from pathlib import Path

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box


LATENT_DIM = 192
ACTION_CHUNK_DIM = 25  # 5 real actions x 5 dims = one WM step


class WMEnv(gymnasium.Env):
    """Single-env JEPA world-model env. Numpy I/O.

    Actions are 25-D = 5 real-env actions x 5 dims, concatenated row-major
    (action_t[0..4] occupies dims 5t..5t+5). Each 5-D sub-action is expected
    in the qc-conventional [-1, 1] range. WMEnv internally normalizes the
    sub-actions via the supplied scaler mean/std (per-element) BEFORE feeding
    them to the JEPA action encoder -- the WM was trained on
    StandardScaler-normalized actions (see train.py get_column_normalizer).

    Anchored Restart Rollouts (Variant B), enabled when branch_length > 0:
      Every branch_length model steps, search the offline anchor pool for the
      nearest real latent to the current predicted state. If close enough
      (cos >= anchor_threshold_cos AND L2 <= anchor_threshold_l2), stash the
      anchor and emit truncated=True so the outer loop resets. The next
      reset() will start from the stashed anchor instead of a random pool
      sample. If no neighbour passes the threshold, truncate AND fall back
      to random reset -- the drifted rollout is abandoned. The transition
      tuple appended to replay buffer always uses the model output as
      next_observations (no fake snap edge).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jepa_model,
        z_goal_task: np.ndarray,
        z_init_pool: np.ndarray,
        done_threshold: float,
        scaler_mean: np.ndarray,    # (5,) per-element mean
        scaler_std: np.ndarray,     # (5,) per-element std
        max_episode_steps: int = 40,
        device: str = "cuda",
        # --- Variant B anchoring (off by default) ---
        anchor_pool: np.ndarray = None,        # (N, 192) full offline latents; None = no anchoring
        branch_length: int = 0,                # 0 = anchoring disabled
        anchor_threshold_cos: float = 0.95,
        anchor_threshold_l2: float = 4.0,
        # --- Dense vs sparse reward ---
        reward_shape: str = "sparse",          # "sparse" (1 inside threshold) | "dense" (-d/scale)
        dense_reward_scale: float = 10.0,
        # --- Uncertainty-penalised reward (MOPO/LOMPO style) ---
        # uncertainty_mode selects how the per-step uncertainty proxy `u` is
        # computed; the reward is then r' = r - uncertainty_penalty * u.
        #
        #   "ensemble"    : L2 norm of per-element std across ensemble
        #                   predictions (requires len(jepa_model) > 1).
        #   "nn_distance" : L2 distance from the predicted next-latent to its
        #                   nearest neighbour in the offline anchor pool.
        #                   Strong OOD signal; works with a single WM.
        #                   Requires anchor_pool to be set.
        #   "both"        : sum of "ensemble" and "nn_distance" terms.
        #
        # In all cases setting uncertainty_penalty=0.0 disables the penalty,
        # though the per-step uncertainty is still logged in info[...]
        # for diagnostic purposes.
        uncertainty_mode: str = "ensemble",
        uncertainty_penalty: float = 0.0,
    ):
        super().__init__()
        assert z_goal_task.shape == (LATENT_DIM,), f"got {z_goal_task.shape}"
        assert z_init_pool.ndim == 2 and z_init_pool.shape[1] == LATENT_DIM
        assert scaler_mean.shape == (5,) and scaler_std.shape == (5,)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(LATENT_DIM,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_CHUNK_DIM,), dtype=np.float32
        )
        # Single model or list of models for ensemble uncertainty estimation.
        if isinstance(jepa_model, (list, tuple)):
            self._models = list(jepa_model)
        else:
            self._models = [jepa_model]
        # Keep self.model for backward-compat (some callers reach into it for
        # action_encoder, encode, etc -- those use the first member).
        self.model = self._models[0]
        self.uncertainty_penalty = float(uncertainty_penalty)
        assert uncertainty_mode in ("ensemble", "nn_distance", "both"), \
            f"bad uncertainty_mode: {uncertainty_mode}"
        self.uncertainty_mode = uncertainty_mode
        self.device = device
        self.done_threshold = float(done_threshold)
        self.max_episode_steps = int(max_episode_steps)

        # Torch tensors on device for fast inner loop
        self._z_goal_torch = torch.from_numpy(z_goal_task.astype(np.float32)).to(device)
        self._z_pool_torch = torch.from_numpy(z_init_pool.astype(np.float32)).to(device)
        self._scaler_mean = torch.from_numpy(scaler_mean.astype(np.float32)).to(device)  # (5,)
        self._scaler_std = torch.from_numpy(scaler_std.astype(np.float32)).to(device)    # (5,)

        self._z_state = None  # torch [1, 1, 192]
        self._step_count = 0

        # --- Reward shape ---
        assert reward_shape in ("sparse", "dense"), f"bad reward_shape: {reward_shape}"
        self.reward_shape = reward_shape
        self.dense_reward_scale = float(dense_reward_scale)

        # --- Variant B state ---
        self.branch_length = int(branch_length)
        self.anchor_threshold_cos = float(anchor_threshold_cos)
        self.anchor_threshold_l2 = float(anchor_threshold_l2)
        self._anchor_enabled = self.branch_length > 0 and anchor_pool is not None
        self._branch_step = 0
        self._pending_anchor = None      # np.ndarray (192,) or None
        # Anchor accounting (logged via info dict, aggregated by main loop)
        self._anchor_attempts = 0
        self._anchor_hits = 0
        if self._anchor_enabled:
            anchor_pool = np.asarray(anchor_pool, dtype=np.float32)
            assert anchor_pool.ndim == 2 and anchor_pool.shape[1] == LATENT_DIM
            self._anchor_pool_np = anchor_pool                  # (N, 192) raw
            norms = np.linalg.norm(anchor_pool, axis=1, keepdims=True) + 1e-8
            self._anchor_pool_normed = (anchor_pool / norms).astype(np.float32)
            # ALWAYS load the GPU torch pool — it's used for the per-step NN
            # query in nn_distance uncertainty mode (avoiding CPU sync).
            # Memory cost: ~2 * N * 192 * 4 bytes = ~230 MB for N=150k. OK on H100.
            self._anchor_pool_torch = torch.from_numpy(self._anchor_pool_normed).to(device)
            self._anchor_pool_raw_torch = torch.from_numpy(self._anchor_pool_np).to(device)
            try:
                import faiss
                self._faiss_index = faiss.IndexFlatIP(LATENT_DIM)
                self._faiss_index.add(self._anchor_pool_normed)
                self._anchor_backend = "faiss"
            except Exception:
                self._faiss_index = None
                self._anchor_backend = "torch"
        else:
            self._anchor_pool_np = None
            self._anchor_pool_normed = None
            self._faiss_index = None
            self._anchor_backend = "off"

    def _find_anchor(self, z_pred_np):
        """Return (anchor_z_np, cos_sim, l2_dist) of nearest real latent, or
        None if none meets the threshold."""
        z = z_pred_np.astype(np.float32).reshape(-1)
        z_n = z / (np.linalg.norm(z) + 1e-8)
        if self._anchor_backend == "faiss":
            sims, idxs = self._faiss_index.search(z_n[None].astype(np.float32), 1)
            idx = int(idxs[0, 0])
            cos = float(sims[0, 0])
            anchor = self._anchor_pool_np[idx]
        else:  # torch fallback
            with torch.no_grad():
                zt = torch.from_numpy(z_n).to(self.device)
                s = self._anchor_pool_torch @ zt
                idx = int(torch.argmax(s).item())
                cos = float(s[idx].item())
                anchor = self._anchor_pool_raw_torch[idx].detach().cpu().numpy()
        l2 = float(np.linalg.norm(z - anchor))
        if cos >= self.anchor_threshold_cos and l2 <= self.anchor_threshold_l2:
            return anchor.astype(np.float32), cos, l2
        return None, cos, l2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._pending_anchor is not None:
            # Variant B: restart this branch from the previously stashed anchor.
            z0 = torch.from_numpy(self._pending_anchor).to(self.device)
            self._pending_anchor = None
            anchor_used = True
        else:
            idx = int(np.random.randint(self._z_pool_torch.shape[0]))
            z0 = self._z_pool_torch[idx]
            anchor_used = False
        self._z_state = z0.view(1, 1, LATENT_DIM).contiguous()
        self._step_count = 0
        self._branch_step = 0
        obs = z0.detach().cpu().numpy().astype(np.float32)
        info = {"anchor_used_on_reset": float(anchor_used)} if self._anchor_enabled else {}
        return obs, info

    def step(self, action):
        assert self._z_state is not None, "must call reset() before step()"
        action_np = np.asarray(action, dtype=np.float32).reshape(ACTION_CHUNK_DIM)
        a_raw = torch.from_numpy(action_np).to(self.device).view(5, 5)  # 5 sub-actions x 5 dims
        # StandardScaler-normalize each 5-D sub-action, then row-major flatten to 25-D
        a_scaled = (a_raw - self._scaler_mean) / self._scaler_std       # (5, 5)
        a = a_scaled.reshape(1, 1, ACTION_CHUNK_DIM)

        with torch.no_grad():
            if len(self._models) == 1:
                m = self._models[0]
                act_emb = m.action_encoder(a)                            # [1, 1, 192]
                z_next = m.predict(self._z_state, act_emb)[:, -1:]       # [1, 1, 192]
                ensemble_sigma = 0.0
            else:
                # Each ensemble member runs its own action_encoder + predict.
                preds = []
                for m in self._models:
                    act_emb_k = m.action_encoder(a)                      # [1, 1, 192]
                    z_k = m.predict(self._z_state, act_emb_k)[:, -1:]    # [1, 1, 192]
                    preds.append(z_k)
                stacked = torch.stack(preds, dim=0)                      # [K, 1, 1, 192]
                z_next = stacked.mean(dim=0)                             # [1, 1, 192]
                per_dim_std = stacked.std(dim=0, unbiased=False)         # [1, 1, 192]
                ensemble_sigma = float(torch.linalg.vector_norm(per_dim_std).item())

        self._z_state = z_next
        self._step_count += 1
        self._branch_step += 1

        z_flat = z_next.squeeze()  # [192]
        d = torch.norm(z_flat - self._z_goal_torch, p=2).item()
        success = d < self.done_threshold
        if self.reward_shape == "dense":
            reward = -d / self.dense_reward_scale
        else:
            reward = 1.0 if success else 0.0

        # --- Uncertainty signal: ensemble σ and/or NN-distance ---
        nn_l2 = 0.0
        if self.uncertainty_mode in ("nn_distance", "both") and self._anchor_enabled:
            # GPU-resident NN search: normalize z, cosine-NN via matmul against
            # pre-normalised pool, then compute L2 to the raw anchor. Stays on
            # GPU until the final scalar .item() so JAX <-> torch sync is once
            # per step, not three times.
            with torch.no_grad():
                z_norm = z_flat / (torch.linalg.vector_norm(z_flat) + 1e-8)
                sims = self._anchor_pool_torch @ z_norm                  # [N]
                idx = torch.argmax(sims)                                 # scalar GPU
                anchor = self._anchor_pool_raw_torch[idx]                # [192] GPU
                nn_l2_t = torch.linalg.vector_norm(z_flat - anchor)      # scalar GPU
                nn_l2 = float(nn_l2_t.item())                            # single sync

        if self.uncertainty_mode == "ensemble":
            uncertainty = ensemble_sigma
        elif self.uncertainty_mode == "nn_distance":
            uncertainty = nn_l2
        else:  # "both"
            uncertainty = ensemble_sigma + nn_l2

        if self.uncertainty_penalty > 0.0 and uncertainty > 0.0:
            reward = reward - self.uncertainty_penalty * uncertainty
        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        info = {
            "distance": float(d),
            "success": float(success),
            "wm_uncertainty": float(uncertainty),
            "wm_ensemble_sigma": float(ensemble_sigma),
            "wm_nn_l2": float(nn_l2),
        }

        # --- Variant B: end-of-branch anchor logic ---
        if (self._anchor_enabled and not terminated and not truncated
                and self._branch_step >= self.branch_length):
            z_pred_np = z_flat.detach().cpu().numpy()
            anchor_z, cos, l2 = self._find_anchor(z_pred_np)
            self._anchor_attempts += 1
            info["anchor_cos"] = float(cos)
            info["anchor_l2"] = float(l2)
            if anchor_z is not None:
                self._pending_anchor = anchor_z
                self._anchor_hits += 1
                info["anchor_hit"] = 1.0
            else:
                # No usable anchor; abandon this branch (random reset next).
                info["anchor_hit"] = 0.0
            info["anchor_attempts_total"] = float(self._anchor_attempts)
            info["anchor_hit_rate"] = float(self._anchor_hits / max(1, self._anchor_attempts))
            truncated = True
            self._branch_step = 0

        obs = z_flat.detach().cpu().numpy().astype(np.float32)
        return obs, reward, terminated, truncated, info

    def get_state(self):
        """Compat with main.py's --save_all_online_states flag."""
        z = self._z_state.detach().cpu().numpy().reshape(-1) if self._z_state is not None \
            else np.zeros(LATENT_DIM, dtype=np.float32)
        return {"qpos": z, "qvel": np.zeros(0, dtype=np.float32)}


def make_wm_env_and_dataset(
    wm_ckpt_path,
    latent_cache_path,
    hdf5_dataset_path,
    task_id,
    done_threshold,
    max_episode_steps=40,
    wm_device="cuda",
    img_size=224,
    # --- Variant B anchoring (off by default) ---
    branch_length: int = 0,
    anchor_threshold_cos: float = 0.95,
    anchor_threshold_l2: float = 4.0,
    # --- Reward shape ---
    reward_shape: str = "sparse",
    dense_reward_scale: float = 10.0,
    # --- Ensemble uncertainty penalty ---
    # wm_ckpt_path may be a single path OR a list of paths. When a list is
    # given, each WM is loaded independently and WMEnv averages their
    # predictions / penalises reward by the ensemble disagreement (scaled
    # by uncertainty_penalty). The first WM is used as the encoder for
    # offline-data preprocessing (z_goal, init pool, etc) -- the encoder is
    # frozen and shared across ensemble members in our setup, so this is
    # exact.
    uncertainty_mode: str = "ensemble",
    uncertainty_penalty: float = 0.0,
):
    """Build (train_env, eval_env, train_dataset, val_dataset, jepa, real_env, z_goal) for E.

    The done_threshold is ONLY used by WMEnv (for online rewards). The offline
    dataset uses OGBench's task-relabeled rewards via build_for_E.

    hdf5_dataset_path is needed to fit the action StandardScaler that the WM
    was trained with.
    """
    from envs.jepa_loader import load_jepa, encode_pixels_to_latent
    from envs.wm_dataset_builder import build_for_E
    from envs.env_utils import EpisodeMonitor
    from sklearn import preprocessing
    from pathlib import Path
    import gymnasium as _gym
    import ogbench  # registers envs

    # JEPA — single or ensemble. wm_ckpt_path is either a string (one model)
    # or a list (ensemble).
    if isinstance(wm_ckpt_path, (list, tuple)):
        jepa_list = [load_jepa(p, device=wm_device, img_size=img_size)
                     for p in wm_ckpt_path]
        jepa = jepa_list[0]            # canonical model for preprocessing
        ensemble_models = jepa_list    # passed to WMEnv as a list
        print(f"[wm_env] loaded ensemble of {len(jepa_list)} WMs", flush=True)
    else:
        jepa = load_jepa(wm_ckpt_path, device=wm_device, img_size=img_size)
        ensemble_models = jepa         # single-model fast path

    # Real env: use OGBench's singletask env at 224x224 (same as B1, just upscaled
    # for JEPA). reward_task_id is baked in via env registration, so rewards are
    # consistently [-1, 0] both offline (relabeled) and online (env native).
    real_env = _gym.make(
        f"visual-cube-single-singletask-task{task_id}-v0",
        width=224, height=224,
    )
    real_env = EpisodeMonitor(real_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    # render_goal=True asks the env to render & return the goal image at reset
    obs, info = real_env.reset(seed=0, options=dict(render_goal=True))
    goal_img = info.get("goal", info.get("target", None))
    if goal_img is None:
        raise RuntimeError(
            f"reset info missing 'goal'/'target'; got keys: {list(info.keys())}"
        )
    z_goal_task = encode_pixels_to_latent(jepa, goal_img, wm_device)  # (192,)

    # Initial-state pool from the latent cache
    cache = torch.load(latent_cache_path, map_location="cpu", weights_only=False)
    if isinstance(cache, dict) and "all_latents" in cache:
        all_latents = cache["all_latents"]
    else:
        all_latents = cache
    z_init_pool = np.stack(
        [ep[0].numpy() if torch.is_tensor(ep[0]) else np.asarray(ep[0])
         for ep in all_latents],
        axis=0,
    ).astype(np.float32)  # (N_ep, 192)

    # Variant B anchor pool: ALL latents flattened (stride-1 per env step).
    # Only built when branch_length > 0 to avoid memory if unused.
    if branch_length > 0:
        anchor_pool = np.concatenate(
            [(ep.cpu().numpy() if torch.is_tensor(ep) else np.asarray(ep))
             for ep in all_latents],
            axis=0,
        ).astype(np.float32)  # (N_total, 192)
    else:
        anchor_pool = None

    # Fit action StandardScaler on the HDF5 actions -- matches the WM training
    # convention from train.py (get_column_normalizer).
    import stable_worldmodel as swm
    swm_ds = swm.data.HDF5Dataset(
        hdf5_dataset_path,
        keys_to_cache=["action"],
        cache_dir=str(Path(hdf5_dataset_path).parent),
    )
    action_data = swm_ds.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = preprocessing.StandardScaler()
    scaler.fit(action_data)

    # Build offline chunk-granularity dataset.
    # Note: done_threshold is NOT used here -- offline rewards come from
    # OGBench's task-relabeled signal. Threshold is only for WMEnv (online).
    train_dataset_dict = build_for_E(
        all_latents=all_latents,
        task_id=task_id,
    )

    # Dense-reward relabel: replace OGBench's sparse signal with -||z' - z_goal||/scale.
    # Both online (WMEnv.step) and offline (this relabel) must use the SAME reward
    # function or the critic will fit a mixture and produce garbage. The actor is
    # safe (best-of-n is BC + Q-rerank), but the critic will recalibrate from the
    # offline checkpoint to the new scale over the first ~100k online updates.
    if reward_shape == "dense":
        next_obs = train_dataset_dict["next_observations"]            # (N, 192)
        d = np.linalg.norm(next_obs - z_goal_task[None, :], axis=1)   # (N,)
        train_dataset_dict["rewards"] = (-d / float(dense_reward_scale)).astype(np.float32)
        print(f"[wm_env] relabeled offline rewards densely: "
              f"mean={train_dataset_dict['rewards'].mean():.4f}, "
              f"min={train_dataset_dict['rewards'].min():.4f}, "
              f"max={train_dataset_dict['rewards'].max():.4f}", flush=True)

    # Build WMEnv — anchoring is ON for train_env, OFF for eval_env so the
    # in-loop eval metric stays comparable across runs (real-env eval is the
    # authoritative metric via evaluate_real_ogbench).
    train_env = WMEnv(
        jepa_model=ensemble_models,
        z_goal_task=z_goal_task,
        z_init_pool=z_init_pool,
        done_threshold=done_threshold,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
        max_episode_steps=max_episode_steps,
        device=wm_device,
        anchor_pool=anchor_pool,
        branch_length=branch_length,
        anchor_threshold_cos=anchor_threshold_cos,
        anchor_threshold_l2=anchor_threshold_l2,
        reward_shape=reward_shape,
        dense_reward_scale=dense_reward_scale,
        uncertainty_mode=uncertainty_mode,
        uncertainty_penalty=uncertainty_penalty,
    )
    eval_env = WMEnv(
        jepa_model=jepa,
        z_goal_task=z_goal_task,
        z_init_pool=z_init_pool,
        done_threshold=done_threshold,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
        max_episode_steps=max_episode_steps,
        device=wm_device,
        reward_shape=reward_shape,
        dense_reward_scale=dense_reward_scale,
    )
    return train_env, eval_env, train_dataset_dict, None, jepa, real_env, z_goal_task
