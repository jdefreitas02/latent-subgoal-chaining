import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        # For multi-cube families (cube-double, cube-triple, cube-quadruple),
        # ogbench.make_env_and_datasets strips the cube-count word ('double',
        # 'triple', etc.) thinking it's a dataset-type qualifier, producing the
        # wrong env name (e.g. 'visual-cube-singletask-task1-v0' instead of
        # 'visual-cube-double-singletask-task1-v0'). Fix by calling lower-level
        # ogbench functions directly with the correct names.
        _MULTI_CUBE_FAMILIES = ('cube-double', 'cube-triple', 'cube-quadruple')
        if any(mc in env_name for mc in _MULTI_CUBE_FAMILIES):
            import os as _os, numpy as _np, gymnasium as _gym
            _splits = env_name.split('-')
            _pos = _splits.index('singletask')
            # Multi-cube play datasets include 'play' in their name
            # (e.g. visual-cube-double-play-v0), unlike cube-single which
            # omits it (visual-cube-single-v0). Insert 'play' before 'v0'.
            # e.g. visual-cube-double-singletask-task1-v0
            #   -> visual-cube-double-play-v0
            _dataset_name = '-'.join(_splits[:_pos] + ['play'] + _splits[-1:])
            # Use already-downloaded copy from STABLEWM_HOME if available.
            _stablewm = _os.path.expanduser('~/stable_wm_data/ogbench')
            _local_path = _os.path.join(_stablewm, f'{_dataset_name}.npz')
            if _os.path.exists(_local_path):
                _dataset_dir = _stablewm
            else:
                _dataset_dir = _os.path.expanduser('~/.ogbench/data')
                ogbench.download_datasets([_dataset_name], _dataset_dir)
            _train_path = _os.path.join(_dataset_dir, f'{_dataset_name}.npz')
            _val_path   = _os.path.join(_dataset_dir, f'{_dataset_name}-val.npz')
            env      = _gym.make(env_name)
            eval_env = _gym.make(env_name)
            _train_raw = ogbench.load_dataset(_train_path, ob_dtype=_np.uint8, add_info=True)
            _val_raw   = ogbench.load_dataset(_val_path,   ob_dtype=_np.uint8, add_info=True)
            ogbench.relabel_utils.relabel_dataset(env_name, env, _train_raw)
            ogbench.relabel_utils.relabel_dataset(env_name, env, _val_raw)
            for _k in ('qpos', 'qvel', 'button_states'):
                _train_raw.pop(_k, None)
                _val_raw.pop(_k, None)
            train_dataset = Dataset.create(**_train_raw)
            val_dataset   = Dataset.create(**_val_raw)
        else:
            env, _train_raw, _val_raw = ogbench.make_env_and_datasets(env_name)
            eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
            train_dataset = Dataset.create(**_train_raw)
            val_dataset   = Dataset.create(**_val_raw)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("lift") or env_name.startswith("can") or env_name.startswith("square") or \
        env_name.startswith("transport") or env_name.startswith("tool_hang"):
        # RoboMimic.
        from envs import robomimic_utils

        env = robomimic_utils.make_env(env_name, seed=0)
        eval_env = robomimic_utils.make_env(env_name, seed=42)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = robomimic_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
