import glob, tqdm, wandb, os, json, random, time, jax
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

# ----- B2 / E flags (qc + JEPA encoder / qc + JEPA + WM env) -----
flags.DEFINE_bool('use_jepa_obs', False,
                  "B2: wrap the real swm/OGBCube-v0 env with a JEPA obs encoder. "
                  "Offline dataset is per-real-step 192-D latents.")
flags.DEFINE_bool('use_world_model', False,
                  "E: run online phase inside the JEPA world model (WMEnv). "
                  "Offline dataset is stride-5 chunk-granularity.")
flags.DEFINE_string('wm_ckpt_path',
                    os.path.expanduser("~/stable_wm_data/cube/lejepa"),
                    "Path to the JEPA checkpoint used for B2/E. AutoCostModel appends "
                    "_weights.ckpt / _object.ckpt internally, so pass the base path.")
flags.DEFINE_string('wm_latent_cache',
                    os.path.expanduser("~/stable_wm_data/ogbench/lewm_224_latents_cache.pt"),
                    "Path to the pre-encoded latent cache (per-frame 192-D).")
flags.DEFINE_string('wm_hdf5_dataset_path',
                    os.path.expanduser("~/stable_wm_data/ogbench/visual-cube-single-play-v0_224"),
                    "Path (no .h5) to the swm HDF5 dataset that matches the latent cache.")
flags.DEFINE_integer('wm_task_id', 1, "OGBench cube-single task id for the goal (1..5).")
flags.DEFINE_float('wm_done_threshold', 9.36,
                   "L2-latent distance threshold for the sparse 0/1 reward in E (offline and online).")
flags.DEFINE_integer('wm_max_episode_steps', 40,
                     "WMEnv truncation length in WM steps (each step = 5 real env steps).")
# --- Variant B anchored restart rollouts (off by default) ---
flags.DEFINE_integer('wm_branch_length', 0,
                     "If > 0, enable anchored restart rollouts: every N WMEnv steps, "
                     "snap the rollout's *next* reset to the nearest offline latent. "
                     "0 = standard WMEnv rollouts (no anchoring).")
flags.DEFINE_float('wm_anchor_threshold_cos', 0.95,
                   "Min cosine similarity to accept a nearest-neighbor anchor.")
flags.DEFINE_float('wm_anchor_threshold_l2', 4.0,
                   "Max L2 distance to accept a nearest-neighbor anchor.")
flags.DEFINE_string('wm_reward_shape', 'sparse',
                    "WMEnv reward shape: 'sparse' (1 inside threshold, else 0) or "
                    "'dense' (-||z - z_goal||_2 / scale). Dense mode ALSO relabels the "
                    "offline replay rewards so the critic sees one consistent reward "
                    "distribution across offline+online.")
flags.DEFINE_float('wm_dense_reward_scale', 10.0,
                   "Divisor for dense reward (i.e., r = -d/scale). 10.0 gives ~ -1.5 per "
                   "step at the median offline latent-to-goal distance.")
flags.DEFINE_list('wm_ensemble_paths', [],
                  "Optional list of additional WM checkpoint paths (besides wm_ckpt_path) "
                  "to form an ensemble for uncertainty-penalised WMEnv rewards. Empty list "
                  "(default) means no ensemble. When set, WMEnv averages predictions across "
                  "members and subtracts uncertainty_penalty * std-norm from the reward.")
flags.DEFINE_float('wm_uncertainty_penalty', 0.0,
                   "Coefficient for the uncertainty reward penalty (MOPO/LOMPO style). "
                   "Combined with --wm_uncertainty_mode to pick the uncertainty signal.")
flags.DEFINE_string('wm_uncertainty_mode', 'ensemble',
                    "Uncertainty signal for the reward penalty: 'ensemble' (std across "
                    "ensemble members; requires --wm_ensemble_paths), 'nn_distance' (L2 "
                    "from predicted latent to its nearest offline neighbour; requires "
                    "wm_branch_length>0 since it reuses the anchor pool), or 'both'.")
# --- Phase 1: differentiable rollouts (analytic policy gradient through frozen WM) ---
flags.DEFINE_float('rollout_loss_weight', 0.0,
                   "If > 0, augment the actor loss with -E[sum gamma^t r_t] computed "
                   "by rolling out the BC-flow actions through a frozen JAX WM. "
                   "0 disables this term entirely (default).")
flags.DEFINE_integer('rollout_horizon', 3,
                     "Number of WM steps in the differentiable rollout. Matches the "
                     "anchored branch length we already validated.")
flags.DEFINE_float('rollout_dense_scale', 10.0,
                   "Scale for dense reward inside the rollout (same convention as "
                   "wm_dense_reward_scale).")
# --- Joint WM + policy training (offline phase) ---
flags.DEFINE_bool('use_joint_wm_training', False,
                  "If true, wrap the WM in a JAX TrainState and update it during "
                  "the offline phase using L_pred + beta * L_value, alongside the "
                  "actor and critic. The WM is initialised from --wm_ckpt_path. "
                  "Only valid in the E (WM) pipeline.")
flags.DEFINE_float('joint_alpha', 1.0,
                   "Weight on L_pred (one-step prediction MSE) in joint WM loss.")
flags.DEFINE_float('joint_beta', 0.1,
                   "Final weight on L_value (Bellman residual via WM) in joint "
                   "WM loss; ramped from 0 over joint_beta_ramp_steps.")
flags.DEFINE_integer('joint_beta_ramp_steps', 50000,
                     "Linear ramp length for joint_beta. Gives critic time to "
                     "settle before WM starts following value gradient.")
flags.DEFINE_float('joint_wm_lr', 1e-5,
                   "Learning rate for the WM optimizer during joint training.")
flags.DEFINE_integer('eval_max_episode_steps', 500,
                     "Hard step cap per episode in evaluate(). Prevents infinite loops when env lacks TimeLimit.")
flags.DEFINE_string('wm_device', 'cuda',
                    "Device for the PyTorch JEPA model. Use 'cuda:1' if JAX takes cuda:0.")
flags.DEFINE_integer('wm_img_size', 224, "JEPA image size (224 for lejepa).")
flags.DEFINE_integer('wm_patch_size', 14, "JEPA ViT patch size (14 for 224x224).")
flags.DEFINE_bool('real_env_eval', True,
                  "When using --use_jepa_obs or --use_world_model, also eval against the "
                  "real swm/OGBCube-v0 env with JEPA encoding. Default ON for B2/E.")
flags.DEFINE_integer('offline_final_eval_episodes', 250,
                     "Total episodes for the dedicated end-of-offline real-env eval "
                     "(distributed evenly across the 5 tasks).")
flags.DEFINE_string('offline_checkpoint_subdir', 'offline_final',
                    "Subdir under save_dir where the offline-only checkpoint is saved.")
flags.DEFINE_bool('use_b1_swm_env', False,
                  "B1: train pure qc on swm/OGBCube-v0 (224x224) with the swm HDF5 play "
                  "dataset. Mutually exclusive with --use_jepa_obs / --use_world_model.")
flags.DEFINE_string('load_offline_ckpt', None,
                    "If set, load the agent from this .pkl checkpoint AFTER building the "
                    "dataset and skip the offline training loop. Used to reuse one offline-"
                    "trained agent across an online threshold sweep without paying the "
                    "offline compute cost each time.")
flags.DEFINE_bool('continue_offline', False,
                  "If True together with --load_offline_ckpt, DO NOT skip the offline loop. "
                  "Instead continue offline training from the loaded checkpoint. Used for "
                  "alternating policy/JEPA training where each phase resumes from the prior.")


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    # ---------- env + dataset dispatch ----------
    # Three mutually-exclusive paths:
    #   E:  use_world_model -> WMEnv online + stride-5 latent offline dataset
    #   B2: use_jepa_obs    -> real swm/OGBCube-v0 wrapped + per-step latent offline dataset
    #   B1: use_b1_swm_env  -> real swm/OGBCube-v0 raw pixels + HDF5 pixel dataset
    #   default:            -> original qc behavior (OGBench / D4RL / robomimic)
    jepa_model = None
    real_env_for_eval = None   # only used for B2/E real-env-with-JEPA eval

    n_flags_set = sum([FLAGS.use_world_model, FLAGS.use_jepa_obs, FLAGS.use_b1_swm_env])
    if n_flags_set > 1:
        raise ValueError("--use_world_model, --use_jepa_obs, --use_b1_swm_env are mutually exclusive.")

    if FLAGS.use_world_model:
        from envs.wm_env import make_wm_env_and_dataset
        # If extra ensemble paths are provided, build a list with wm_ckpt_path
        # first (the canonical encoder used for preprocessing).
        if FLAGS.wm_ensemble_paths:
            wm_paths = [FLAGS.wm_ckpt_path] + list(FLAGS.wm_ensemble_paths)
            print(f"[main] using ensemble of {len(wm_paths)} WMs: {wm_paths}", flush=True)
        else:
            wm_paths = FLAGS.wm_ckpt_path
        (env, eval_env, train_dataset_dict, val_dataset,
         jepa_model, real_env_for_eval, _z_goal_task) = make_wm_env_and_dataset(
            wm_ckpt_path=wm_paths,
            latent_cache_path=FLAGS.wm_latent_cache,
            hdf5_dataset_path=FLAGS.wm_hdf5_dataset_path,
            task_id=FLAGS.wm_task_id,
            done_threshold=FLAGS.wm_done_threshold,
            max_episode_steps=FLAGS.wm_max_episode_steps,
            wm_device=FLAGS.wm_device,
            img_size=FLAGS.wm_img_size,
            branch_length=FLAGS.wm_branch_length,
            anchor_threshold_cos=FLAGS.wm_anchor_threshold_cos,
            anchor_threshold_l2=FLAGS.wm_anchor_threshold_l2,
            reward_shape=FLAGS.wm_reward_shape,
            dense_reward_scale=FLAGS.wm_dense_reward_scale,
            uncertainty_mode=FLAGS.wm_uncertainty_mode,
            uncertainty_penalty=FLAGS.wm_uncertainty_penalty,
        )
        train_dataset = train_dataset_dict
    elif FLAGS.use_jepa_obs:
        # B2 uses the SAME env as B1 (visual-cube-single-singletask-task{N}-v0)
        # rendered at 224x224 to match JEPA's pretraining resolution. The only
        # difference vs B1 is observations: JEPA-encoded 192-D latents instead of
        # raw 64x64 pixels through IMPALA.
        from envs.jepa_obs_wrapper import JEPAObsWrapper
        from envs.jepa_loader import load_jepa
        from envs.wm_dataset_builder import build_for_B2
        from envs.env_utils import EpisodeMonitor
        import gymnasium as _gym
        import ogbench  # registers env families
        import torch as _torch

        jepa_model = load_jepa(FLAGS.wm_ckpt_path, device=FLAGS.wm_device,
                               img_size=FLAGS.wm_img_size, patch_size=FLAGS.wm_patch_size)

        def _make_b1_env_224(seed):
            e = _gym.make(FLAGS.env_name, width=224, height=224)
            e = EpisodeMonitor(e, filter_regexes=['.*privileged.*', '.*proprio.*'])
            e.reset(seed=seed)
            return e

        real_env_for_eval = _make_b1_env_224(seed=FLAGS.seed + 1)
        env = JEPAObsWrapper(_make_b1_env_224(seed=FLAGS.seed), jepa_model, device=FLAGS.wm_device)
        eval_env = JEPAObsWrapper(_make_b1_env_224(seed=FLAGS.seed + 2), jepa_model, device=FLAGS.wm_device)

        cache = _torch.load(FLAGS.wm_latent_cache, map_location="cpu", weights_only=False)
        all_latents = cache["all_latents"] if isinstance(cache, dict) and "all_latents" in cache else cache
        train_dataset = build_for_B2(all_latents, task_id=FLAGS.wm_task_id)
        val_dataset = None
    elif FLAGS.use_b1_swm_env:
        from envs.swm_env_register import make_swm_pixel_env_and_dataset
        env, eval_env, train_dataset, val_dataset = make_swm_pixel_env_and_dataset(
            FLAGS.wm_hdf5_dataset_path, task_id=FLAGS.wm_task_id, seed=FLAGS.seed
        )
    else:
        # data loading -- original qc behavior
        if FLAGS.ogbench_dataset_dir is not None:
            assert FLAGS.dataset_replace_interval != 0
            assert FLAGS.dataset_proportion == 1.0
            dataset_idx = 0
            dataset_paths = [
                file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
            ]
            env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
            )
        else:
            env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0

    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)

        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds

    # process_train_dataset always runs: it normalises both raw dicts (our new
    # B1/B2/E paths) and existing Datasets (qc default path -- Dataset.create(**ds)
    # is idempotent because FrozenDict supports ** unpacking).
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    # Optional: load a JAX WM for differentiable-rollout actor loss.
    # Only relevant when --rollout_loss_weight > 0 and we are using the E (WM)
    # pipeline (so a frozen WM is already loaded for the env). We reuse the
    # ckpt path the user already provided via --wm_ckpt_path; the JAX WM has
    # been parity-tested against the Torch WM.
    wm_model_jax = None
    wm_params_jax = None
    z_goal_jax = None
    if FLAGS.rollout_loss_weight > 0.0 and FLAGS.use_world_model:
        from wm_jax import load_wm_jax
        print(f"[diff-rollout] loading JAX WM from {FLAGS.wm_ckpt_path}", flush=True)
        wm_model_jax, wm_params_jax = load_wm_jax(FLAGS.wm_ckpt_path)
        z_goal_jax = jnp.asarray(_z_goal_task)
        # Pass these into the config so they're plumbed through the agent.
        config['rollout_loss_weight'] = FLAGS.rollout_loss_weight
        config['rollout_horizon'] = FLAGS.rollout_horizon
        config['rollout_dense_scale'] = FLAGS.rollout_dense_scale
        print(f"[diff-rollout] enabled: weight={FLAGS.rollout_loss_weight} "
              f"horizon={FLAGS.rollout_horizon} scale={FLAGS.rollout_dense_scale}",
              flush=True)

    # Optional: joint WM+policy training (offline phase). The WM is wrapped in
    # a JAX TrainState with its own optax optimizer and updates alongside the
    # actor/critic via L_pred + beta * L_value (see thesis/joint_wm_policy_offline_design.md).
    wm_train_state = None
    if FLAGS.use_joint_wm_training and FLAGS.use_world_model:
        from wm_jax import make_wm_trainstate
        print(f"[joint-wm] wrapping {FLAGS.wm_ckpt_path} in JAX TrainState "
              f"(lr={FLAGS.joint_wm_lr})", flush=True)
        wm_train_state = make_wm_trainstate(FLAGS.wm_ckpt_path, lr=FLAGS.joint_wm_lr)
        config['joint_alpha'] = FLAGS.joint_alpha
        config['joint_beta'] = FLAGS.joint_beta
        config['joint_beta_ramp_steps'] = FLAGS.joint_beta_ramp_steps
        print(f"[joint-wm] enabled: alpha={FLAGS.joint_alpha} "
              f"beta_final={FLAGS.joint_beta} ramp={FLAGS.joint_beta_ramp_steps}",
              flush=True)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        wm_model=wm_model_jax,
        wm_params=wm_params_jax,
        z_goal=z_goal_jax,
        wm_train_state=wm_train_state,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
        prefixes.append("offline_only")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
        prefixes.append("online_only")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # ---------- optionally short-circuit the offline phase ----------
    # When --load_offline_ckpt is set we skip the offline gradient loop entirely
    # and load a previously-trained agent. The train_dataset is still rebuilt
    # (with rewards computed at the CURRENT --wm_done_threshold) so the replay
    # buffer's offline transitions are consistent with the online phase's
    # threshold during the threshold sweep.
    if FLAGS.load_offline_ckpt is not None:
        print(f"[offline-load] loading agent from {FLAGS.load_offline_ckpt}", flush=True)
        agent = restore_agent_with_file(agent, FLAGS.load_offline_ckpt)
        if FLAGS.continue_offline:
            print(f"[offline-load] --continue_offline=True: running {FLAGS.offline_steps} "
                  f"more offline steps from the loaded checkpoint.", flush=True)
        else:
            log_step = FLAGS.offline_steps  # bump log step so online metrics align
            if FLAGS.offline_steps > 0:
                print(f"[offline-load] --offline_steps={FLAGS.offline_steps} ignored; "
                      f"skipping offline loop (pass --continue_offline=True to continue).",
                      flush=True)

    # Offline RL
    dataset_idx = 0
    # Skip offline ONLY when a checkpoint was loaded AND continue_offline was not requested.
    skip_offline = (FLAGS.load_offline_ckpt is not None) and (not FLAGS.continue_offline)
    offline_steps_to_run = 0 if skip_offline else FLAGS.offline_steps
    for i in tqdm.tqdm(range(1, offline_steps_to_run + 1)):
        log_step += 1

        if (FLAGS.ogbench_dataset_dir is not None
                and FLAGS.dataset_replace_interval != 0
                and i % FLAGS.dataset_replace_interval == 0
                and not (FLAGS.use_world_model or FLAGS.use_jepa_obs or FLAGS.use_b1_swm_env)):
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                max_episode_steps=FLAGS.eval_max_episode_steps,
            )
            logger.log(eval_info, "eval", step=log_step)

    # =================================================================
    # End-of-offline: save dedicated checkpoint + run authoritative eval
    # Skipped when --load_offline_ckpt is set (the checkpoint already exists
    # and the eval was already produced by the run that trained it).
    # =================================================================
    # Save the offline-final checkpoint+eval if we actually ran offline training,
    # either fresh or via --continue_offline.
    if offline_steps_to_run > 0:
        offline_ckpt_dir = os.path.join(FLAGS.save_dir, FLAGS.offline_checkpoint_subdir)
        os.makedirs(offline_ckpt_dir, exist_ok=True)
        save_agent(agent, offline_ckpt_dir, log_step)
        print(f"[offline-final] saved checkpoint to {offline_ckpt_dir}", flush=True)

        # Authoritative real-env eval (used by all three runs for direct comparability)
        try:
            if FLAGS.use_jepa_obs:
                # B2: eval_env is already a JEPA-wrapped OGBench singletask env
                # (same env as B1, just rendered at 224x224 with JEPA encoding).
                # Use qc's standard evaluate() for direct comparability with B1.
                offline_metrics, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.offline_final_eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=1,
                    max_episode_steps=FLAGS.eval_max_episode_steps,
                )
            elif FLAGS.use_world_model:
                # E: WMEnv for training, real-OGBench-singletask env for offline-final eval.
                # real_env_for_eval is the OGBench singletask env at 224x224 (task_id
                # baked in via env registration). We evaluate only the configured task,
                # using all offline_final_eval_episodes against it.
                from envs.real_ogbench_eval import evaluate_real_ogbench
                if real_env_for_eval is None:
                    raise RuntimeError("E run did not produce real_env_for_eval; "
                                       "check make_wm_env_and_dataset.")
                offline_metrics = evaluate_real_ogbench(
                    agent=agent,
                    real_env=real_env_for_eval,
                    jepa_model=jepa_model,
                    device=FLAGS.wm_device,
                    task_ids=(FLAGS.wm_task_id,),  # OGBench env has task_id baked in
                    num_episodes_per_task=FLAGS.offline_final_eval_episodes,
                    action_dispatch="chunk25",
                    pass_task_id_on_reset=False,
                )
            elif FLAGS.use_b1_swm_env:
                # B1 raw-pixels eval on the same swm/OGBCube-v0 env via qc's evaluate()
                offline_metrics, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.offline_final_eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=1,
                    max_episode_steps=FLAGS.eval_max_episode_steps,
                )
            else:
                # qc default path: just run the standard eval at offline-final
                offline_metrics, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.offline_final_eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=1,
                    max_episode_steps=FLAGS.eval_max_episode_steps,
                )

            logger.log(offline_metrics, "offline_only", step=log_step)
            with open(os.path.join(offline_ckpt_dir, "offline_only_eval.json"), "w") as f:
                json.dump({k: float(v) for k, v in offline_metrics.items()}, f, indent=2)
            print(f"[offline-final] eval written to {offline_ckpt_dir}/offline_only_eval.json",
                  flush=True)
        except Exception as e:
            print(f"[offline-final] eval failed (continuing to online phase): {e}", flush=True)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )

    ob, _ = env.reset()

    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        if FLAGS.use_world_model:
            # E: agent emits a 25-D action that IS the chunk. WMEnv consumes
            # the full 25-D atomically (one WM step = 5 real env steps).
            # Bypass qc's per-action queue.
            action = agent.sample_actions(observations=ob, rng=key)
            action = np.array(action).reshape(-1).astype(np.float32)  # (25,)
            next_ob, int_reward, terminated, truncated, info = env.step(action)
            applied_action = action
        else:
            # B1, B2, default: qc's normal queued per-step dispatch.
            if len(action_queue) == 0:
                action = agent.sample_actions(observations=ob, rng=key)
                action_chunk = np.array(action).reshape(-1, action_dim)
                for action in action_chunk:
                    action_queue.append(action)
            action = action_queue.pop(0)
            next_ob, int_reward, terminated, truncated, info = env.step(action)
            applied_action = action
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            if hasattr(env, 'get_state'):
                state = env.get_state()
                data["steps"].append(i)
                data["obs"].append(np.copy(next_ob))
                data["qpos"].append(np.copy(state.get("qpos", np.zeros(0))))
                data["qvel"].append(np.copy(state.get("qvel", np.zeros(0))))
                if "button_states" in state:
                    data["button_states"].append(np.copy(state["button_states"]))

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=applied_action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)

        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            # In E (action_chunking=False, horizon_length=1) sample_sequence with
            # sequence_length=1 returns single-step batches. Otherwise the standard
            # horizon-length chunks.
            sample_seq_len = 1 if FLAGS.use_world_model else FLAGS.horizon_length
            batch = replay_buffer.sample_sequence(
                config['batch_size'] * FLAGS.utd_ratio,
                sequence_length=sample_seq_len,
                discount=discount,
            )
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)

        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                max_episode_steps=FLAGS.eval_max_episode_steps,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    # =================================================================
    # Online-final checkpoint + authoritative real-env eval.
    # Mirrors the offline_final block so threshold sweeps can be compared
    # against the same real-env metric. Without this, the only eval inside
    # the online loop is against eval_env (WMEnv for E), whose latent-distance
    # success criterion is rarely triggered and tells us nothing about real
    # task performance.
    # =================================================================
    if FLAGS.online_steps > 0:
        online_ckpt_dir = os.path.join(FLAGS.save_dir, "online_final")
        os.makedirs(online_ckpt_dir, exist_ok=True)
        save_agent(agent, online_ckpt_dir, log_step)
        print(f"[online-final] saved checkpoint to {online_ckpt_dir}", flush=True)

        try:
            if FLAGS.use_world_model:
                from envs.real_ogbench_eval import evaluate_real_ogbench
                if real_env_for_eval is None:
                    raise RuntimeError("E run did not produce real_env_for_eval.")
                online_metrics = evaluate_real_ogbench(
                    agent=agent,
                    real_env=real_env_for_eval,
                    jepa_model=jepa_model,
                    device=FLAGS.wm_device,
                    task_ids=(FLAGS.wm_task_id,),
                    num_episodes_per_task=FLAGS.offline_final_eval_episodes,
                    action_dispatch="chunk25",
                    pass_task_id_on_reset=False,
                )
            else:
                online_metrics, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.offline_final_eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=1,
                    max_episode_steps=FLAGS.eval_max_episode_steps,
                )

            logger.log(online_metrics, "online_only", step=log_step)
            with open(os.path.join(online_ckpt_dir, "online_only_eval.json"), "w") as f:
                json.dump({k: float(v) for k, v in online_metrics.items()}, f, indent=2)
            print(f"[online-final] eval written to {online_ckpt_dir}/online_only_eval.json",
                  flush=True)
        except Exception as e:
            print(f"[online-final] eval failed: {e}", flush=True)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0) if data["qpos"] else np.zeros(0),
                 "qvel": np.stack(data["qvel"], axis=0) if data["qvel"] else np.zeros(0),
                 "obs": np.stack(data["obs"], axis=0) if data["obs"] else np.zeros(0),
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
