"""Offline dataset builders for B2 and E.

Both pull rewards/actions/terminals/masks directly from OGBench's
`cube-single-play-singletask-task{N}-v0` dataset (i.e., the play dataset
with task-relabeled rewards via OGBench's relabel_dataset). Only the
*observations* are replaced -- pixels are dropped in favour of pre-encoded
192-D JEPA latents from the cache. This keeps the offline phase identical
to qc default (same reward signal) and decouples it from --wm_done_threshold,
which now only affects WMEnv at online time.

B2: per-real-step transitions. action_dim=5, horizon_length=5.
E:  stride-5 chunk-granularity transitions. action_dim=25, horizon_length=1.
    Chunk reward = max of the 5 per-step rewards in the chunk (sparse-friendly).
"""

import numpy as np
import torch


LATENT_DIM = 192


def _episodes_to_numpy(all_latents):
    out = []
    for ep in all_latents:
        if torch.is_tensor(ep):
            ep = ep.cpu().numpy()
        out.append(ep.astype(np.float32))
    return out


def _load_ogbench_singletask_dataset(task_id, hdf5_path=None, env_family="cube-single"):
    """Return (actions, rewards, terminals, masks, episode_lens) from the
    visual 224x224 HDF5 dataset with task-specific reward relabeling.

    Loads actions/qpos/ep_len from the visual HDF5 (no download needed), then
    uses ogbench relabeling with a lightweight state env to compute rewards/masks
    from qpos. This keeps the reward signal consistent with OGBench's task
    definitions while using the same dataset the WM was trained on.
    """
    import ogbench
    import gymnasium
    import os
    import h5py

    if hdf5_path is None:
        hdf5_path = os.path.expanduser(
            "~/stable_wm_data/ogbench/visual-cube-single-play-v0_224.h5"
        )

    with h5py.File(hdf5_path, "r") as f:
        actions  = f["action"][...].astype(np.float32)    # (N, 5)
        qpos     = f["qpos"][...].astype(np.float32)      # (N, 21)
        ep_len   = f["ep_len"][...].astype(np.int64)      # (n_eps,)
        ep_offset = f["ep_offset"][...].astype(np.int64)  # (n_eps,)
        # scene / puzzle relabeling needs button_states (drawer/window/button
        # success terms); preserved in the 224 HDF5 by convert/refilm.
        button_states = (f["button_states"][...].astype(np.int64)
                         if "button_states" in f else None)

    # Build flat terminals array: last frame of each episode = 1
    n_total = actions.shape[0]
    terminals = np.zeros(n_total, dtype=np.float32)
    for offset, length in zip(ep_offset, ep_len):
        terminals[offset + length - 1] = 1.0

    # Use ogbench relabeling with a state env to get task-specific rewards/masks
    env_name = f"{env_family}-singletask-task{task_id}-v0"
    env = gymnasium.make(env_name)
    ds = {"actions": actions, "qpos": qpos, "terminals": terminals}
    if button_states is not None:
        ds["button_states"] = button_states
    ogbench.relabel_utils.relabel_dataset(env_name, env, ds)
    env.close()

    rewards = np.asarray(ds["rewards"], dtype=np.float32)  # (N,)
    masks   = np.asarray(ds["masks"],   dtype=np.float32)  # (N,)

    # Derive episode boundaries (terminals=1 marks episode end).
    term_locs = np.nonzero(terminals > 0)[0]
    if term_locs.size == 0:
        episode_lens = np.array([len(actions)], dtype=np.int64)
    else:
        episode_lens = np.diff(np.concatenate([[-1], term_locs])).astype(np.int64)
    return actions, rewards, terminals, masks, episode_lens


def _split_by_episode(arr, episode_lens):
    """Split a flat array into a list of per-episode arrays."""
    out = []
    start = 0
    for L in episode_lens:
        out.append(arr[start:start + int(L)])
        start += int(L)
    return out


def _align_latent_cache_to_episodes(all_latents, episode_lens):
    """Verify episode lengths match between cache and OGBench dataset, then
    return latents per episode as np.ndarray (T, 192)."""
    latents = _episodes_to_numpy(all_latents)
    if len(latents) != len(episode_lens):
        raise RuntimeError(
            f"Latent cache has {len(latents)} episodes but OGBench dataset has "
            f"{len(episode_lens)} episodes."
        )
    for i, (z_ep, L) in enumerate(zip(latents, episode_lens)):
        if abs(z_ep.shape[0] - int(L)) > 1:
            raise RuntimeError(
                f"Episode {i}: latent length {z_ep.shape[0]} vs OGBench length {L}"
            )
    return latents


def build_for_B2(all_latents, task_id, action_clip_eps=1e-5,
                 env_family="cube-single"):
    """Per-real-step transitions: 192-D JEPA latent obs + 5-D action +
    OGBench's task-relabeled reward signal. The done threshold is NOT used here.
    """
    actions, rewards, terminals, masks, episode_lens = \
        _load_ogbench_singletask_dataset(task_id, env_family=env_family)

    latents = _align_latent_cache_to_episodes(all_latents, episode_lens)

    actions_per_ep = _split_by_episode(actions, episode_lens)
    rewards_per_ep = _split_by_episode(rewards, episode_lens)
    terminals_per_ep = _split_by_episode(terminals, episode_lens)
    masks_per_ep = _split_by_episode(masks, episode_lens)

    obs_list, next_obs_list, act_list = [], [], []
    rew_list, term_list, mask_list = [], [], []

    for z_ep, a_ep, r_ep, t_ep, m_ep in zip(latents, actions_per_ep, rewards_per_ep,
                                              terminals_per_ep, masks_per_ep):
        T_z = z_ep.shape[0]
        T_a = a_ep.shape[0]
        T = min(T_z - 1, T_a)
        if T <= 0:
            continue
        obs_list.append(z_ep[:T])
        next_obs_list.append(z_ep[1:T + 1])
        act_list.append(a_ep[:T])
        rew_list.append(r_ep[:T])
        # Enforce terminal at last transition of each episode for downstream
        # sample_sequence boundary handling.
        term = t_ep[:T].copy()
        term[-1] = 1.0
        mask = m_ep[:T].copy()
        mask[-1] = 0.0
        term_list.append(term)
        mask_list.append(mask)

    observations = np.concatenate(obs_list, axis=0).astype(np.float32)
    next_observations = np.concatenate(next_obs_list, axis=0).astype(np.float32)
    actions_flat = np.concatenate(act_list, axis=0).astype(np.float32)
    rewards_flat = np.concatenate(rew_list, axis=0).astype(np.float32)
    terminals_flat = np.concatenate(term_list, axis=0).astype(np.float32)
    masks_flat = np.concatenate(mask_list, axis=0).astype(np.float32)

    if action_clip_eps is not None:
        actions_flat = np.clip(actions_flat, -1 + action_clip_eps, 1 - action_clip_eps).astype(np.float32)

    return dict(
        observations=observations,
        actions=actions_flat,
        rewards=rewards_flat,
        terminals=terminals_flat,
        masks=masks_flat,
        next_observations=next_observations,
    )


def build_for_E(all_latents, task_id, stride=5, action_clip_eps=1e-5,
                env_family="cube-single", hdf5_path=None):
    """Stride-5 chunk-granularity transitions for E.

    Each transition: (z[5k], chunk_25, z[5(k+1)], r_chunk, term).
    chunk_25 = concatenation of the 5 per-step 5-D actions (row-major).
    r_chunk  = max of the 5 per-step OGBench-relabeled rewards in the window
               (sparse-friendly: chunk fires if any sub-step crosses the goal).
    Done threshold is NOT used here.
    """
    actions, rewards, terminals, masks, episode_lens = \
        _load_ogbench_singletask_dataset(task_id, hdf5_path=hdf5_path,
                                         env_family=env_family)

    latents = _align_latent_cache_to_episodes(all_latents, episode_lens)
    actions_per_ep = _split_by_episode(actions, episode_lens)
    rewards_per_ep = _split_by_episode(rewards, episode_lens)

    obs_list, next_obs_list, chunk_list = [], [], []
    rew_list, term_list, mask_list = [], [], []

    for z_ep, a_ep, r_ep in zip(latents, actions_per_ep, rewards_per_ep):
        T_z = z_ep.shape[0]
        T_a = a_ep.shape[0]
        T_pair = min(T_z, T_a + 1)
        num_chunks = (T_pair - 1) // stride
        if num_chunks <= 0:
            continue
        ep_obs, ep_next, ep_chunks, ep_rew = [], [], [], []
        for k in range(num_chunks):
            t = stride * k
            t_next = stride * (k + 1)
            chunk = a_ep[t:t_next].reshape(-1).astype(np.float32)
            if chunk.shape[0] != stride * a_ep.shape[-1]:
                continue
            ep_obs.append(z_ep[t])
            ep_next.append(z_ep[t_next])
            ep_chunks.append(chunk)
            ep_rew.append(float(np.max(r_ep[t:t_next])))   # sparse-friendly aggregation
        if not ep_obs:
            continue
        L = len(ep_obs)
        term_arr = np.zeros(L, dtype=np.float32)
        term_arr[-1] = 1.0
        mask_arr = np.ones(L, dtype=np.float32)
        mask_arr[-1] = 0.0
        obs_list.extend(ep_obs)
        next_obs_list.extend(ep_next)
        chunk_list.extend(ep_chunks)
        rew_list.extend(ep_rew)
        term_list.append(term_arr)
        mask_list.append(mask_arr)

    observations = np.stack(obs_list, axis=0).astype(np.float32)
    next_observations = np.stack(next_obs_list, axis=0).astype(np.float32)
    actions_flat = np.stack(chunk_list, axis=0).astype(np.float32)
    rewards_flat = np.array(rew_list, dtype=np.float32)
    terminals_flat = np.concatenate(term_list, axis=0).astype(np.float32)
    masks_flat = np.concatenate(mask_list, axis=0).astype(np.float32)

    if action_clip_eps is not None:
        actions_flat = np.clip(actions_flat, -1 + action_clip_eps, 1 - action_clip_eps).astype(np.float32)

    return dict(
        observations=observations,
        actions=actions_flat,
        rewards=rewards_flat,
        terminals=terminals_flat,
        masks=masks_flat,
        next_observations=next_observations,
    )
