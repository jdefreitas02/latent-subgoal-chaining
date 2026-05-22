#!/usr/bin/env bash
# B2 -- qc + JEPA encoder (frozen) on real swm/OGBCube-v0, OFFLINE ONLY.
# Same as run_b2.sh but:
#   - actor_type=best-of-n (no distill_loss, no q_loss on actor)
#   - actor_num_samples=4 during training for speed and TD-target conservatism
#   - online_steps=0  (offline-only run; eval at end via standalone script)
#
# Motivation: we showed that the existing B2 distill-ddpg checkpoint gives
# 3% offline-final, but the *same weights* re-evaluated with best-of-n N=32
# gives 32%. Diagnosis: the one-step actor exploits OOD Q-peaks because the
# JEPA encoder is frozen and can't reshape the value surface.
# This run removes the broken one-step head from training entirely.

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

WM_CKPT="${STABLEWM_HOME}/cube/lejepa"
WM_CACHE="${STABLEWM_HOME}/ogbench/lewm_224_latents_cache.pt"

OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
ACTOR_NUM_SAMPLES="${ACTOR_NUM_SAMPLES:-4}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"
WM_DEVICE="${WM_DEVICE:-cuda}"

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
  for task_id in $WM_TASK_IDS; do
    job_name="qc_b2bon_t${task_id}_s${seed}"
    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=${WORKDIR}/logs/${job_name}_%j.out

set -e
source ${VENV}
cd ${QCDIR}
export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

echo "=== qc B2-best-of-n (offline only, actor_type=best-of-n, N=${ACTOR_NUM_SAMPLES}) seed ${seed} task ${task_id} ==="

python main.py \\
    --run_group "b2bon_qc_jepa_offline" \\
    --seed ${seed} \\
    --env_name "visual-cube-single-singletask-task${task_id}-v0" \\
    --use_jepa_obs=true \\
    --wm_ckpt_path "${WM_CKPT}" \\
    --wm_latent_cache "${WM_CACHE}" \\
    --wm_task_id ${task_id} \\
    --wm_device "${WM_DEVICE}" \\
    --offline_steps ${OFFLINE_STEPS} \\
    --online_steps  0 \\
    --eval_interval ${EVAL_INTERVAL} \\
    --eval_episodes ${EVAL_EPISODES} \\
    --horizon_length 5 \\
    --agent agents/acfql.py \\
    --agent.encoder=jepa_head \\
    --agent.actor_type=best-of-n \\
    --agent.actor_num_samples=${ACTOR_NUM_SAMPLES} \\
    --agent.action_chunking=True

echo "=== ${job_name} done ==="
EOF
    echo "Submitted ${job_name}"
  done
done
