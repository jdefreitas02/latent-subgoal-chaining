#!/usr/bin/env bash
# E_anchored -- qc + JEPA encoder + WM env online, with anchored restart rollouts.
#
# Same as run_experiment.sh, with anchored rollouts enabled for the ONLINE phase:
# every WM_BRANCH_LENGTH WM-steps the rollout truncates and the next reset
# snaps to the nearest offline latent (if within thresholds). Bounds compounding
# error to ~branch_length steps and keeps the online replay buffer concentrated
# on the data manifold.
#
# Defaults chosen from the anchor-viability diagnostic (logs/diag_anchor_55288.out):
#   - At h=3 chunks (15 env steps), pred-to-real cos_med=0.912, rel_l2=0.417 — safe.
#   - At h=5+ the JEPA predictor's positional encoding falls off — drift becomes large.
#   - Anchor pool cos to a random latent is 0.998 vs random pair ~0 → strong signal.
#   - 64% of NNs come from a different trajectory → real cross-trajectory stitching.
#
# Stage 1 / Stage 2 split mirrors run_experiment.sh.

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

WM_CKPT="${WM_CKPT:-${STABLEWM_HOME}/cube/lejepa}"
WM_CACHE="${WM_CACHE:-${STABLEWM_HOME}/ogbench/lewm_224_latents_cache.pt}"
RUN_GROUP="${RUN_GROUP:-e_qc_jepa_wm_anchored_${WM_REWARD_SHAPE:-sparse}}"
WM_HDF5="${STABLEWM_HOME}/ogbench/visual-cube-single-play-v0_224"

OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
START_TRAINING="${START_TRAINING:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"
WM_DONE_THRESHOLDS="${WM_DONE_THRESHOLDS:-12.0}"
WM_MAX_EPISODE_STEPS="${WM_MAX_EPISODE_STEPS:-40}"
WM_DEVICE="${WM_DEVICE:-cuda}"
LOAD_OFFLINE_CKPT="${LOAD_OFFLINE_CKPT:-}"
# --- Variant B anchoring params ---
WM_BRANCH_LENGTH="${WM_BRANCH_LENGTH:-3}"
WM_ANCHOR_COS="${WM_ANCHOR_COS:-0.95}"
WM_ANCHOR_L2="${WM_ANCHOR_L2:-4.0}"
# --- Reward shape (sparse | dense) ---
WM_REWARD_SHAPE="${WM_REWARD_SHAPE:-sparse}"
WM_DENSE_SCALE="${WM_DENSE_SCALE:-10.0}"
# Ensemble paths (comma-separated, in addition to WM_CKPT) + uncertainty penalty.
# Empty = no ensemble.
WM_ENSEMBLE_PATHS="${WM_ENSEMBLE_PATHS:-}"
WM_UNCERTAINTY_PENALTY="${WM_UNCERTAINTY_PENALTY:-0.0}"
WM_UNCERTAINTY_MODE="${WM_UNCERTAINTY_MODE:-ensemble}"
# --- Phase 1: differentiable rollouts (analytic policy gradient through frozen WM) ---
# 0.0 disables; small (~0.1) treats it as a BC regularizer; large (>=1.0) lets
# the rollout gradient unfreeze the actor from BC.
ROLLOUT_LOSS_WEIGHT="${ROLLOUT_LOSS_WEIGHT:-0.0}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-3}"
ROLLOUT_DENSE_SCALE="${ROLLOUT_DENSE_SCALE:-10.0}"

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
    for task_id in $WM_TASK_IDS; do
        for thr in $WM_DONE_THRESHOLDS; do
            if [ -n "$LOAD_OFFLINE_CKPT" ]; then
                load_arg="--load_offline_ckpt ${LOAD_OFFLINE_CKPT}"
                job_name="qcA_e_t${task_id}_thr${thr}_h${WM_BRANCH_LENGTH}_${WM_REWARD_SHAPE}_s${seed}_online"
                time_limit="${ONLINE_TIME_LIMIT:-08:00:00}"
            else
                load_arg=""
                job_name="qcA_e_t${task_id}_thr${thr}_h${WM_BRANCH_LENGTH}_${WM_REWARD_SHAPE}_s${seed}"
                time_limit="24:00:00"
            fi

            sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=${time_limit}
#SBATCH --output=${WORKDIR}/logs/${job_name}_%j.out

source ${VENV}
cd ${QCDIR}
export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

echo "=== qc E (ANCHORED, h=${WM_BRANCH_LENGTH}) task=${task_id} thr=${thr} seed=${seed} ==="

python main.py \\
    --run_group "${RUN_GROUP}" \\
    --seed ${seed} \\
    --env_name "visual-cube-single-singletask-task${task_id}-v0" \\
    --use_world_model=true \\
    --wm_ckpt_path "${WM_CKPT}" \\
    --wm_latent_cache "${WM_CACHE}" \\
    --wm_hdf5_dataset_path "${WM_HDF5}" \\
    --wm_task_id ${task_id} \\
    --wm_done_threshold ${thr} \\
    --wm_max_episode_steps ${WM_MAX_EPISODE_STEPS} \\
    --wm_device "${WM_DEVICE}" \\
    --wm_branch_length ${WM_BRANCH_LENGTH} \\
    --wm_anchor_threshold_cos ${WM_ANCHOR_COS} \\
    --wm_anchor_threshold_l2 ${WM_ANCHOR_L2} \\
    --wm_reward_shape ${WM_REWARD_SHAPE} \\
    --wm_dense_reward_scale ${WM_DENSE_SCALE} \\
    --wm_ensemble_paths="${WM_ENSEMBLE_PATHS}" \\
    --wm_uncertainty_penalty ${WM_UNCERTAINTY_PENALTY} \\
    --wm_uncertainty_mode ${WM_UNCERTAINTY_MODE} \\
    --rollout_loss_weight ${ROLLOUT_LOSS_WEIGHT} \\
    --rollout_horizon ${ROLLOUT_HORIZON} \\
    --rollout_dense_scale ${ROLLOUT_DENSE_SCALE} \\
    --offline_steps ${OFFLINE_STEPS} \\
    --online_steps  ${ONLINE_STEPS} \\
    --start_training ${START_TRAINING} \\
    --eval_interval ${EVAL_INTERVAL} \\
    --eval_episodes ${EVAL_EPISODES} \\
    --horizon_length 1 \\
    --agent agents/acfql.py \\
    --agent.encoder=jepa_head \\
    --agent.actor_type=best-of-n \\
    --agent.actor_num_samples=4 \\
    --agent.action_chunking=False \\
    ${load_arg}

echo "=== ${job_name} done ==="
EOF
            echo "Submitted ${job_name}"
        done
    done
done
