#!/usr/bin/env bash
# E -- qc + JEPA encoder + WM env online.
#
# Two phases supported:
#   1. Stage 1 (default): train the offline phase ONCE and run the online phase
#      with WM_DONE_THRESHOLDS[0] as the baseline threshold. Saves the offline
#      checkpoint to {save_dir}/offline_final/params_{offline_steps}.pkl.
#   2. Stage 2 (when LOAD_OFFLINE_CKPT is set): SKIP offline training, load the
#      previously-saved offline checkpoint, and run only the online phase. Use
#      this for the threshold sweep over WM_DONE_THRESHOLDS[1..N].
#
# Recommended workflow for a threshold sweep over {1.5, 2.0, 2.5, 3.0}:
#
#   # Stage 1 -- one full run at the baseline threshold:
#   WM_DONE_THRESHOLDS="2.0" bash run_experiment.sh
#   # ... wait for the SLURM job to complete and note the save_dir from logs ...
#
#   # Stage 2 -- launch the remaining thresholds in parallel, all loading the
#   # same offline checkpoint:
#   LOAD_OFFLINE_CKPT=/path/to/exp/qc/e_qc_jepa_wm/.../offline_final/params_500000.pkl \
#   WM_DONE_THRESHOLDS="1.5 2.5 3.0" bash run_experiment.sh

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

WM_CKPT="${WM_CKPT:-${STABLEWM_HOME}/cube/lejepa}"
WM_CACHE="${WM_CACHE:-${STABLEWM_HOME}/ogbench/lewm_224_latents_cache.pt}"
WM_HDF5="${WM_HDF5:-${STABLEWM_HOME}/ogbench/visual-cube-single-play-v0_224}"
RUN_GROUP="${RUN_GROUP:-e_qc_jepa_wm}"

OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
START_TRAINING="${START_TRAINING:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"               # default task 1; pass "1 2 3 4 5" for full sweep
WM_DONE_THRESHOLDS="${WM_DONE_THRESHOLDS:-2.0}"  # space-separated, e.g. "1.5 2.0 2.5 3.0"
WM_MAX_EPISODE_STEPS="${WM_MAX_EPISODE_STEPS:-40}"
WM_DEVICE="${WM_DEVICE:-cuda}"
LOAD_OFFLINE_CKPT="${LOAD_OFFLINE_CKPT:-}"    # empty = train offline; set = reuse ckpt
# Joint WM+policy training (offline phase). Set USE_JOINT_WM=true to enable.
USE_JOINT_WM="${USE_JOINT_WM:-false}"
JOINT_ALPHA="${JOINT_ALPHA:-1.0}"
JOINT_BETA="${JOINT_BETA:-0.1}"
JOINT_BETA_RAMP_STEPS="${JOINT_BETA_RAMP_STEPS:-50000}"
JOINT_WM_LR="${JOINT_WM_LR:-1e-5}"

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
    for task_id in $WM_TASK_IDS; do
        for thr in $WM_DONE_THRESHOLDS; do
            # Stage 2 (loading): SLURM job is online-only, smaller resource footprint
            #                   and the offline_steps flag is ignored by main.py.
            # Stage 1 (training): full offline+online job.
            if [ -n "$LOAD_OFFLINE_CKPT" ]; then
                load_arg="--load_offline_ckpt ${LOAD_OFFLINE_CKPT}"
                job_name="qc_e_t${task_id}_thr${thr}_s${seed}_online"
                time_limit="08:00:00"
            else
                load_arg=""
                job_name="qc_e_t${task_id}_thr${thr}_s${seed}"
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

echo "=== qc E task=${task_id} thr=${thr} seed=${seed} ==="

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
    --use_joint_wm_training=${USE_JOINT_WM} \\
    --joint_alpha ${JOINT_ALPHA} \\
    --joint_beta ${JOINT_BETA} \\
    --joint_beta_ramp_steps ${JOINT_BETA_RAMP_STEPS} \\
    --joint_wm_lr ${JOINT_WM_LR} \\
    ${load_arg}

echo "=== ${job_name} done ==="
EOF
            echo "Submitted ${job_name}"
        done
    done
done
