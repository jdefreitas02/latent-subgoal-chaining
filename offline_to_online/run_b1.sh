#!/usr/bin/env bash
# B1 -- pure qc on OGBench visual-cube-single at 64x64. Standard qc pipeline.
#
# Uses qc's built-in IMPALA encoder on the standard OGBench visual dataset
# (~/.ogbench/data/visual-cube-single-play-v0.npz). No JEPA, no WM.
#
# Usage:
#   bash run_b1.sh                                    # full run, seed 0
#   SEEDS="0 1 2" bash run_b1.sh                      # multiple seeds
#   OFFLINE_STEPS=1000 ONLINE_STEPS=0 bash run_b1.sh  # smoke test

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
START_TRAINING="${START_TRAINING:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
  for task_id in $WM_TASK_IDS; do
    job_name="qc_b1_t${task_id}_s${seed}"
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
export MUJOCO_GL="${MUJOCO_GL}"

echo "=== qc B1 (pure qc + IMPALA on 64x64 pixels) task ${task_id} seed ${seed} ==="

python main.py \\
    --run_group "b1_pure_qc" \\
    --seed ${seed} \\
    --env_name "visual-cube-single-play-singletask-task${task_id}-v0" \\
    --offline_steps ${OFFLINE_STEPS} \\
    --online_steps  ${ONLINE_STEPS} \\
    --start_training ${START_TRAINING} \\
    --eval_interval ${EVAL_INTERVAL} \\
    --eval_episodes ${EVAL_EPISODES} \\
    --horizon_length 5 \\
    --agent agents/acfql.py \\
    --agent.encoder=impala_small \\
    --agent.actor_type=distill-ddpg \\
    --agent.action_chunking=True

echo "=== ${job_name} done ==="
EOF
    echo "Submitted ${job_name}"
  done
done
