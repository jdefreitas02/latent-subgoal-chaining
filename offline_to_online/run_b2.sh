#!/usr/bin/env bash
# B2 -- qc + JEPA encoder on real swm/OGBCube-v0.
# Agent observes 192-D JEPA latents; no IMPALA. Real env interaction online.

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

WM_CKPT="${STABLEWM_HOME}/cube/lejepa"
WM_CACHE="${STABLEWM_HOME}/ogbench/lewm_224_latents_cache.pt"
WM_HDF5="${STABLEWM_HOME}/ogbench/visual-cube-single-play-v0_224"

OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
START_TRAINING="${START_TRAINING:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"   # default task 1; pass "1 2 3 4 5" for full sweep
WM_DEVICE="${WM_DEVICE:-cuda}"

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
  for task_id in $WM_TASK_IDS; do
    job_name="qc_b2_t${task_id}_s${seed}"
    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${WORKDIR}/logs/${job_name}_%j.out

source ${VENV}
cd ${QCDIR}
export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

echo "=== qc B2 (qc + JEPA encoder, real env online) seed ${seed} ==="

python main.py \\
    --run_group "b2_qc_jepa_encoder" \\
    --seed ${seed} \\
    --env_name "visual-cube-single-singletask-task${task_id}-v0" \\
    --use_jepa_obs=true \\
    --wm_ckpt_path "${WM_CKPT}" \\
    --wm_latent_cache "${WM_CACHE}" \\
    --wm_task_id ${task_id} \\
    --wm_device "${WM_DEVICE}" \\
    --offline_steps ${OFFLINE_STEPS} \\
    --online_steps  ${ONLINE_STEPS} \\
    --start_training ${START_TRAINING} \\
    --eval_interval ${EVAL_INTERVAL} \\
    --eval_episodes ${EVAL_EPISODES} \\
    --horizon_length 5 \\
    --agent agents/acfql.py \\
    --agent.encoder=jepa_head \\
    --agent.actor_type=distill-ddpg \\
    --agent.action_chunking=True

echo "=== ${job_name} done ==="
EOF
    echo "Submitted ${job_name}"
  done
done
