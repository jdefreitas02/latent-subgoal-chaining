"""Goal-conditioned HER batch sampler — HIQL-style indicator reward.

Mirrors the HIQL sampling pattern from wgsp/train_hiql_baseline.py:
for each sampled index i, the goal g is drawn from one of:
  - "cur":    g = observations[i]                          (P = p_curgoal)
  - "traj":   g = observations[idx_future], same episode   (P = p_trajgoal)
              future offset ~ Geometric(p_geom), clamped to episode end
  - "random": g = observations[rand_idx], anywhere         (P = p_randomgoal)
  - "task":   g = task_goals[k] (uniform K), optional      (P = p_taskgoal)

Reward is an *index-based indicator* (no latent distance):
  r = 1  iff  the current transition's flat idx equals the goal's flat idx,
              i.e., the trajectory is AT the goal step
  r = 0  otherwise

Concretely:
  - cur:    g comes from idx i itself -> r = 1, mask = 0 (terminal)
              V(s, s) = 1 -- "at the goal" semantics
  - traj:   g comes from idx > i (offset >= 1) -> r = 0, mask = 1
              Value propagates back via TD over many gradient steps
  - random: g comes from arbitrary idx -> r = 0, mask = 1
  - task:   no idx-based match available -> r = 0 (no signal). The optional
              `task_rewards` array (N_offline, K) overrides this with
              precomputed OGBench task rewards for offline transitions.

This avoids any dependence on geometry of the WM latent space — we never
compute ||z - g|| anywhere.

Mask = 1 - r (cut bootstrap on success).
Terminal = original terminal | (r > 0).

`obs_aug = concat([z, g], -1)`  → (B, 2D)
`next_obs_aug = concat([z_next, g], -1)` → (B, 2D)

The output dict matches Dataset.sample_sequence() with sequence_length=1
so ACFQLAgent.update() works unchanged. Specifically:
  observations:       (B, 2D)
  full_observations:  (B, 1, 2D)
  next_observations:  (B, 1, 2D)
  actions:            (B, 1, A)
  next_actions:       (B, 1, A)
  rewards:            (B, 1)
  masks:              (B, 1)
  terminals:          (B, 1)
  valid:              (B, 1)
"""
import numpy as np


def build_episode_metadata(terminals):
    """Derive (ep_starts, ep_lens, ep_ids, t_within) from a flat terminals array.

    Assumes terminals[i] == 1.0 marks the end of each episode (as built by
    wm_dataset_builder.build_for_E / build_for_B2).

    Returns:
        ep_starts: (n_eps,) int64 — flat index of first transition per episode
        ep_lens:   (n_eps,) int64 — number of transitions per episode
        ep_ids:    (N,)     int32 — ep_id per transition
        t_within:  (N,)     int32 — index within episode per transition
    """
    terminals = np.asarray(terminals)
    n_total = terminals.shape[0]
    term_locs = np.nonzero(terminals > 0)[0]
    if term_locs.size == 0 or term_locs[-1] != n_total - 1:
        # Append a synthetic terminal at the last index so we don't lose tail data
        term_locs = np.concatenate([term_locs, [n_total - 1]])
    ep_starts = np.concatenate([[0], term_locs[:-1] + 1]).astype(np.int64)
    ep_lens = (term_locs - ep_starts + 1).astype(np.int64)

    ep_ids = np.empty(n_total, dtype=np.int32)
    t_within = np.empty(n_total, dtype=np.int32)
    for e, (s, L) in enumerate(zip(ep_starts, ep_lens)):
        ep_ids[s:s + L] = e
        t_within[s:s + L] = np.arange(L, dtype=np.int32)
    return ep_starts, ep_lens, ep_ids, t_within


class GCHERSampler:
    """Stateless HER sampler with HIQL-style indicator reward.

    Holds the task goals and mix probabilities. At each sample() call, reads
    from buffer arrays + externally maintained ep_starts/ep_lens.

    Args:
        task_goals: (K, D) numpy array of task-goal latents. Used only if
            p_taskgoal > 0.
        p_curgoal:    P(goal = current obs[i])   -- gives r=1, terminal.
        p_trajgoal:   P(goal = future obs in same trajectory) -- r=0, TD prop.
        p_randomgoal: P(goal = uniformly random obs) -- r=0, no signal.
        p_taskgoal:   P(goal = uniformly random task goal). Default 0.
                      Without precomputed `task_rewards`, all task-goal samples
                      get r=0 (signal arrives only via the eval distribution).
        p_geom:       Geometric distribution parameter for future-offset
                      sampling (mean offset = 1 / p_geom).
        task_rewards: Optional (N_offline, K) numpy array of OGBench-relabeled
                      task rewards. If provided AND idx < N_offline, the
                      task-goal sample uses this ground-truth reward. For
                      online idx >= N_offline, falls back to r=0.
        done_threshold: kept for API back-compat; unused under indicator mode.
    """
    def __init__(self, task_goals, done_threshold=2.0,
                 p_curgoal=0.2, p_trajgoal=0.5, p_randomgoal=0.3,
                 p_taskgoal=0.0, p_geom=0.1, task_rewards=None):
        self.task_goals = np.asarray(task_goals, dtype=np.float32)  # (K, D)
        assert self.task_goals.ndim == 2
        self.K, self.goal_dim = self.task_goals.shape
        self.done_threshold = float(done_threshold)
        self.p_taskgoal = float(p_taskgoal)
        total = p_curgoal + p_trajgoal + p_randomgoal
        if total <= 0:
            raise ValueError("p_curgoal + p_trajgoal + p_randomgoal must be > 0")
        self.p_curgoal = float(p_curgoal) / total
        self.p_trajgoal = float(p_trajgoal) / total
        self.p_randomgoal = float(p_randomgoal) / total
        self.p_geom = float(p_geom)
        self.task_rewards = (np.asarray(task_rewards, dtype=np.float32)
                             if task_rewards is not None else None)
        if self.task_rewards is not None:
            assert self.task_rewards.shape[1] == self.K, \
                f"task_rewards has K={self.task_rewards.shape[1]} but K={self.K}"

    def sample(self, buffer_dict, current_size, ep_starts, ep_lens,
               batch_size, np_rng, idx_low=0, idx_high=None):
        """Sample a batch with HIQL-style HER-relabeled goals + indicator reward.

        Returns batch dict matching Dataset.sample_sequence(..., sequence_length=1).
        Observations augmented to (B, 2D) by concatenating goals.

        idx_low / idx_high restrict the sampled transitions (and random goals)
        to a contiguous slice of the buffer, e.g. [0, n_offline) to sample only
        the offline portion. Default covers the whole valid buffer.
        """
        hi = current_size if idx_high is None else int(idx_high)
        idxs = np_rng.integers(idx_low, hi, size=batch_size)
        obs       = buffer_dict['observations'][idxs]       # (B, D)
        next_obs  = buffer_dict['next_observations'][idxs]  # (B, D)
        actions   = buffer_dict['actions'][idxs]            # (B, A)
        terms_orig = buffer_dict['terminals'][idxs]         # (B,)
        ep_ids_b  = buffer_dict['ep_id'][idxs]              # (B,)
        t_within_b = buffer_dict['t_within'][idxs]          # (B,)

        # ---- Decide goal source per index ----
        roll = np_rng.uniform(0.0, 1.0, size=batch_size)
        is_task = roll < self.p_taskgoal
        roll2 = np_rng.uniform(0.0, 1.0, size=batch_size)
        is_cur    = (~is_task) & (roll2 < self.p_curgoal)
        is_traj   = (~is_task) & (roll2 >= self.p_curgoal) & \
                    (roll2 <  self.p_curgoal + self.p_trajgoal)
        is_random = (~is_task) & (roll2 >= self.p_curgoal + self.p_trajgoal)

        # ---- Build goal latents (B, D) AND indicator rewards (B,) ----
        goals = np.empty_like(next_obs)
        rewards = np.zeros(batch_size, dtype=np.float32)

        # HIQL convention: success = (idxs == value_goal_idxs). Reward is 1
        # whenever the goal's flat idx equals the current transition's flat idx.

        # cur: goal_idx = idxs (trivially equal) -> r = 1
        if is_cur.any():
            goals[is_cur] = obs[is_cur]
            rewards[is_cur] = 1.0

        # traj: geometric future within same episode, clamped to ep end.
        # If the clamp makes idx_future == idxs (already at end-of-episode),
        # that still counts as "at the goal" -> r = 1.
        if is_traj.any():
            traj_pos = np.nonzero(is_traj)[0]
            ep_b = ep_ids_b[traj_pos]
            t_b  = t_within_b[traj_pos]
            ep_len_b = ep_lens[ep_b]
            ep_start_b = ep_starts[ep_b]
            offset = np_rng.geometric(self.p_geom, size=traj_pos.size).astype(np.int64)
            t_future = np.minimum(t_b + offset, ep_len_b - 1).astype(np.int64)
            idx_future = np.minimum((ep_start_b + t_future).astype(np.int64),
                                    hi - 1)
            goals[traj_pos] = buffer_dict['observations'][idx_future]
            rewards[traj_pos] = (idx_future == idxs[traj_pos]).astype(np.float32)

        # random: g = obs[rand_idx]; r = 1 only on the rare idx coincidence
        if is_random.any():
            rand_pos = np.nonzero(is_random)[0]
            rand_idx = np_rng.integers(idx_low, hi, size=rand_pos.size)
            goals[rand_pos] = buffer_dict['observations'][rand_idx]
            rewards[rand_pos] = (rand_idx == idxs[rand_pos]).astype(np.float32)

        # task: g = task_goals[k]; r from precomputed OGBench rewards if available
        if is_task.any():
            task_pos = np.nonzero(is_task)[0]
            task_k = np_rng.integers(0, self.K, size=task_pos.size)
            goals[task_pos] = self.task_goals[task_k]
            if self.task_rewards is not None:
                N_off = self.task_rewards.shape[0]
                offline_mask = idxs[task_pos] < N_off
                if offline_mask.any():
                    off_pos = task_pos[offline_mask]
                    off_idx = idxs[off_pos]
                    off_k   = task_k[offline_mask]
                    rewards[off_pos] = self.task_rewards[off_idx, off_k]
                # online indices: leave r=0 (no ground truth)

        masks = 1.0 - rewards
        terminals = np.maximum(terms_orig.astype(np.float32), rewards)
        valid = np.ones(batch_size, dtype=np.float32)

        # ---- Augment observations: concat([z, g], -1) ----
        obs_aug      = np.concatenate([obs, goals], axis=-1).astype(np.float32)
        next_obs_aug = np.concatenate([next_obs, goals], axis=-1).astype(np.float32)

        # ---- Package in sample_sequence(..., sequence_length=1) format ----
        # ACFQLAgent.critic_loss reads:
        #   batch['observations']               -> (B, 2D)
        #   batch['next_observations'][..., -1, :] -> needs seq dim
        #   batch['actions'][..., 0, :]         -> needs seq dim
        #   batch['rewards'][..., -1]           -> needs seq dim
        #   batch['masks'][..., -1]             -> needs seq dim
        #   batch['valid'][..., -1]             -> needs seq dim
        out = dict(
            observations=obs_aug,                          # (B, 2D)
            full_observations=obs_aug[:, None, :],         # (B, 1, 2D)
            actions=actions[:, None, :],                   # (B, 1, A)
            next_actions=actions[:, None, :],              # (B, 1, A) -- unused with action_chunking=False
            rewards=rewards[:, None],                      # (B, 1)
            masks=masks[:, None],                          # (B, 1)
            terminals=terminals[:, None],                  # (B, 1)
            valid=valid[:, None],                          # (B, 1)
            next_observations=next_obs_aug[:, None, :],    # (B, 1, 2D)
        )
        return out

    def sample_stored_goal(self, buffer_dict, batch_size, np_rng,
                           idx_low, idx_high):
        """Sample transitions conditioned on the STORED goal — no HER relabel.

        For online MPC transitions, buffer_dict['goal'] holds g_active, the goal
        the MPC action was actually computed for. Pairing (z, g_active) -> a_mpc
        is the correct goal->action mapping; relabeling with an unrelated HER
        goal would clone a goal-specific action under the wrong goal (see E3b
        analysis). Reward/mask/terminal come from stored values and are only
        used when the critic is unfrozen.
        """
        idxs = np_rng.integers(idx_low, idx_high, size=batch_size)
        obs       = buffer_dict['observations'][idxs]
        next_obs  = buffer_dict['next_observations'][idxs]
        actions   = buffer_dict['actions'][idxs]
        goals     = buffer_dict['goal'][idxs]
        rewards   = buffer_dict['rewards'][idxs].astype(np.float32)
        masks     = (1.0 - rewards).astype(np.float32)
        terminals = np.maximum(buffer_dict['terminals'][idxs].astype(np.float32),
                               rewards)
        valid     = np.ones(batch_size, dtype=np.float32)

        obs_aug      = np.concatenate([obs, goals], axis=-1).astype(np.float32)
        next_obs_aug = np.concatenate([next_obs, goals], axis=-1).astype(np.float32)
        return dict(
            observations=obs_aug,
            full_observations=obs_aug[:, None, :],
            actions=actions[:, None, :],
            next_actions=actions[:, None, :],
            rewards=rewards[:, None],
            masks=masks[:, None],
            terminals=terminals[:, None],
            valid=valid[:, None],
            next_observations=next_obs_aug[:, None, :],
        )
