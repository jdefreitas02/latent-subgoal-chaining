"""Create the 224x224 swm/OGBCube-v0 env that the WM was trained on, wrapped
in qc's EpisodeMonitor. Used by B1 (with pixel observations) and as the
reference real env for B2/E eval (pixels are then encoded by JEPA at every step).

The B1 offline dataset combines 224x224 pixels from the swm HDF5 file
(`visual-cube-single-play-v0_224`) with OGBench's task-relabeled rewards
(from `cube-single-play-singletask-task{N}-v0`). This way B1 uses the same
reward signal offline as B2/E -- only the observation space differs.
"""

import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor


class TaskIdResetWrapper(gymnasium.Wrapper):
    """Force a fixed task_id on every reset so the env always shows the correct
    goal marker in its pixels.

    Without this, swm/OGBCube-v0 defaults to reward_task_id=2 (see cube_env.py:773)
    regardless of which task the offline dataset was built for.
    """

    def __init__(self, env, task_id: int):
        super().__init__(env)
        self._task_id = task_id

    def reset(self, seed=None, options=None, **kwargs):
        opts = dict(options or {})
        opts['task_id'] = self._task_id
        return self.env.reset(seed=seed, options=opts, **kwargs)


def make_swm_cube_env(seed=0, task_id=None):
    """Create swm/OGBCube-v0 wrapped in EpisodeMonitor.

    Args:
        seed: RNG seed for the env.
        task_id: If set, wraps the env so every reset uses this task. Without
            this, the env defaults to task 2 (cube_env.py:773), which mismatches
            any dataset built for a different task.
    """
    import ogbench  # registers env families
    import stable_worldmodel  # registers swm/OGBCube-v0

    env = gymnasium.make(
        "swm/OGBCube-v0",
        ob_type="pixels",
        env_type="single",
        visualize_info=False,
    )
    if task_id is not None:
        env = TaskIdResetWrapper(env, task_id=task_id)
    env = EpisodeMonitor(env, filter_regexes=[".*privileged.*", ".*proprio.*"])
    env.reset(seed=seed)
    return env


B1_PIX_SIZE = 64  # offline pixels and online env both resized to 64x64


def _resize_pixels(pix_hwc, size=B1_PIX_SIZE):
    """Resize (T, H, W, 3) uint8 array to (T, size, size, 3) using PIL."""
    from PIL import Image
    out = np.empty((pix_hwc.shape[0], size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(pix_hwc):
        out[i] = np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return out


def build_pixel_dataset_b1(hdf5_dataset_path, task_id, action_clip_eps=1e-5):
    """Build B1's offline dataset: 64x64 pixel observations + OGBench-relabeled
    rewards for task `task_id`.

    Loads pixels from the 224x224 swm HDF5 and resizes to 64x64 on the fly
    (~12GB in RAM vs ~150GB for 224x224). Uses qpos from the same HDF5 for
    task-specific reward relabeling — no downloads needed.
    """
    import ogbench
    import gymnasium
    import h5py
    from pathlib import Path

    hdf5_path = hdf5_dataset_path + ".h5" if not hdf5_dataset_path.endswith(".h5") else hdf5_dataset_path

    with h5py.File(hdf5_path, "r") as f:
        actions_full = f["action"][...].astype(np.float32)   # (N, 5)
        qpos_full    = f["qpos"][...].astype(np.float32)     # (N, 21)
        ep_len       = f["ep_len"][...].astype(np.int64)     # (n_eps,)
        ep_offset    = f["ep_offset"][...].astype(np.int64)  # (n_eps,)

    n_total = actions_full.shape[0]
    terminals_full = np.zeros(n_total, dtype=np.float32)
    for offset, length in zip(ep_offset, ep_len):
        terminals_full[offset + length - 1] = 1.0

    # Task-relabeled rewards via state env (no download)
    env_name = f"cube-single-singletask-task{task_id}-v0"
    state_env = gymnasium.make(env_name)
    ds_tmp = {"actions": actions_full, "qpos": qpos_full, "terminals": terminals_full}
    ogbench.relabel_utils.relabel_dataset(env_name, state_env, ds_tmp)
    state_env.close()
    rewards_full = ds_tmp["rewards"].astype(np.float32)
    masks_full   = ds_tmp["masks"].astype(np.float32)

    import stable_worldmodel as swm
    pix_ds = swm.data.HDF5Dataset(
        hdf5_dataset_path,
        cache_dir=str(Path(hdf5_dataset_path).parent),
    )

    obs_chunks, next_obs_chunks = [], []
    act_chunks, rew_chunks, term_chunks, mask_chunks = [], [], [], []

    for ep_idx in range(len(ep_len)):
        T_pix = int(ep_len[ep_idx])
        offset = int(ep_offset[ep_idx])
        T = T_pix - 1
        if T <= 0:
            continue
        chunks = pix_ds.load_chunk(np.array([ep_idx]), np.array([0]), np.array([T_pix]))
        pix = chunks[0]["pixels"]
        if hasattr(pix, "numpy"):
            pix = pix.numpy()
        pix_hwc = np.transpose(pix, (0, 2, 3, 1))       # [T_pix, H, W, 3]
        pix_small = _resize_pixels(pix_hwc, B1_PIX_SIZE)  # [T_pix, 64, 64, 3]
        obs_chunks.append(pix_small[:T])
        next_obs_chunks.append(pix_small[1:T + 1])

        a = actions_full[offset:offset + T].astype(np.float32)
        r = rewards_full[offset:offset + T].astype(np.float32)
        t = terminals_full[offset:offset + T].copy().astype(np.float32)
        m = masks_full[offset:offset + T].copy().astype(np.float32)
        t[-1] = 1.0
        m[-1] = 0.0
        act_chunks.append(a)
        rew_chunks.append(r)
        term_chunks.append(t)
        mask_chunks.append(m)

    observations      = np.concatenate(obs_chunks,      axis=0)
    next_observations = np.concatenate(next_obs_chunks, axis=0)
    actions           = np.concatenate(act_chunks,      axis=0)
    rewards           = np.concatenate(rew_chunks,      axis=0)
    terminals         = np.concatenate(term_chunks,     axis=0)
    masks             = np.concatenate(mask_chunks,     axis=0)

    if action_clip_eps is not None:
        actions = np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps).astype(np.float32)

    return dict(
        observations=observations, actions=actions, rewards=rewards,
        terminals=terminals, masks=masks, next_observations=next_observations,
    )


class ResizeObsWrapper(gymnasium.ObservationWrapper):
    """Resize pixel observations to (size, size, 3) uint8."""

    def __init__(self, env, size=B1_PIX_SIZE):
        super().__init__(env)
        self._size = size
        h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(size, size, c), dtype=np.uint8
        )

    def observation(self, obs):
        from PIL import Image
        return np.array(
            Image.fromarray(obs).resize((self._size, self._size), Image.BILINEAR),
            dtype=np.uint8,
        )


def make_swm_pixel_env_and_dataset(hdf5_dataset_path, task_id, seed=0):
    """Drop-in for qc's make_env_and_datasets: B1 with 64x64 pixel observations."""
    train_env  = ResizeObsWrapper(make_swm_cube_env(seed=seed))
    eval_env   = ResizeObsWrapper(make_swm_cube_env(seed=seed + 1))
    train_dataset_dict = build_pixel_dataset_b1(hdf5_dataset_path, task_id=task_id)
    return train_env, eval_env, train_dataset_dict, None
