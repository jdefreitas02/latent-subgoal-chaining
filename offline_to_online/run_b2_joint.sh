#!/usr/bin/env bash
# B2-joint -- alternating qc policy training + JEPA fine-tuning with a
# task-aware aux loss. Offline-only.
#
# Per iteration:
#   1. Train policy for OFFLINE_STEPS_PER_ITER steps using current cache
#   2. Fine-tune JEPA with pred_loss + sigreg + lambda_aux * task_distance_loss
#   3. Re-encode the cache with the updated JEPA
#   4. Verify predictor accuracy
# After all iterations, save the final policy and updated JEPA.

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_API_KEY=$(cat ~/.wandb)

WM_CKPT_INIT_FILE="${STABLEWM_HOME}/cube/lejepa_object.ckpt"
WM_CKPT_INIT_STEM="${STABLEWM_HOME}/cube/lejepa"
WM_HDF5="${STABLEWM_HOME}/ogbench/visual-cube-single-play-v0_224"
CACHE_INIT="${STABLEWM_HOME}/ogbench/lewm_224_latents_cache.pt"

# Run dir for this experiment's WM checkpoints and caches
EXP_TAG="${EXP_TAG:-b2_joint_$(date +%Y%m%d_%H%M%S)}"
EXP_DIR="${STABLEWM_HOME}/runs/${EXP_TAG}"

OFFLINE_TOTAL_STEPS="${OFFLINE_TOTAL_STEPS:-500000}"
N_ITERATIONS="${N_ITERATIONS:-5}"
JEPA_FINETUNE_STEPS="${JEPA_FINETUNE_STEPS:-2000}"
JEPA_FINETUNE_BS="${JEPA_FINETUNE_BS:-16}"
LAMBDA_AUX="${LAMBDA_AUX:-1.0}"
LR="${LR:-2e-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
SEEDS="${SEEDS:-0}"
WM_TASK_IDS="${WM_TASK_IDS:-1}"
WM_DEVICE="${WM_DEVICE:-cuda}"

# Steps per iteration
STEPS_PER_ITER=$((OFFLINE_TOTAL_STEPS / N_ITERATIONS))

mkdir -p "${WORKDIR}/logs"

for seed in $SEEDS; do
  for task_id in $WM_TASK_IDS; do
    job_name="qc_b2joint_t${task_id}_s${seed}"
    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=${WORKDIR}/logs/${job_name}_%j.out

set -e
source ${VENV}
export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

EXP_DIR="${EXP_DIR}_t${task_id}_s${seed}"
mkdir -p "\${EXP_DIR}"

echo "=== qc B2-joint task=${task_id} seed=${seed} ==="
echo "EXP_DIR=\${EXP_DIR}"
echo "STEPS_PER_ITER=${STEPS_PER_ITER}  N_ITERATIONS=${N_ITERATIONS}"

CUR_WM_FILE="${WM_CKPT_INIT_FILE}"   # full path to _object.ckpt
CUR_WM_STEM="${WM_CKPT_INIT_STEM}"   # for load_jepa (without _object.ckpt suffix)
CUR_CACHE="${CACHE_INIT}"
CUR_AGENT_CKPT=""

cd ${QCDIR}

for ITER in \$(seq 1 ${N_ITERATIONS}); do
    echo ""
    echo "=== ITERATION \${ITER} of ${N_ITERATIONS} ==="

    # --- 1. Train policy ---
    if [ -z "\${CUR_AGENT_CKPT}" ]; then
        echo "Iter \${ITER}: starting policy training from scratch"
        CONTINUE_ARG=""
    else
        echo "Iter \${ITER}: continuing policy training from \${CUR_AGENT_CKPT}"
        CONTINUE_ARG="--load_offline_ckpt=\${CUR_AGENT_CKPT} --continue_offline=True"
    fi

    python main.py \\
        --run_group "b2joint_qc_jepa_iter\${ITER}" \\
        --seed ${seed} \\
        --env_name "visual-cube-single-singletask-task${task_id}-v0" \\
        --use_jepa_obs=true \\
        --wm_ckpt_path "\${CUR_WM_STEM}" \\
        --wm_latent_cache "\${CUR_CACHE}" \\
        --wm_task_id ${task_id} \\
        --wm_device "${WM_DEVICE}" \\
        --offline_steps ${STEPS_PER_ITER} \\
        --online_steps  0 \\
        --eval_interval ${EVAL_INTERVAL} \\
        --eval_episodes ${EVAL_EPISODES} \\
        --save_interval ${STEPS_PER_ITER} \\
        --offline_checkpoint_subdir "iter\${ITER}_offline_final" \\
        --horizon_length 5 \\
        --agent agents/acfql.py \\
        --agent.encoder=jepa_head_deep \\
        --agent.actor_type=distill-ddpg \\
        --agent.action_chunking=True \\
        \${CONTINUE_ARG}

    # Find the just-saved offline_final params file
    LAST_RUN_DIR=\$(ls -td ${QCDIR}/exp/qc/b2joint_qc_jepa_iter\${ITER}/visual-cube-single-singletask-task${task_id}-v0/sd*/iter\${ITER}_offline_final 2>/dev/null | head -1)
    CUR_AGENT_CKPT=\$(ls \${LAST_RUN_DIR}/params_*.pkl | head -1)
    echo "Iter \${ITER} policy ckpt: \${CUR_AGENT_CKPT}"

    # If this is the LAST iteration, skip JEPA fine-tune and cache rebuild
    if [ \${ITER} -eq ${N_ITERATIONS} ]; then
        echo "Iter \${ITER}: last iteration; skipping JEPA finetune."
        break
    fi

    # --- 2a. Compute policy distillation targets from the just-trained policy ---
    DISTILL_TARGETS="\${EXP_DIR}/iter\${ITER}_policy_targets.pt"
    echo "Iter \${ITER}: computing policy distillation targets -> \${DISTILL_TARGETS}"
    python ${QCDIR}/compute_policy_targets.py \\
        --ckpt "\${CUR_AGENT_CKPT}" \\
        --cache "\${CUR_CACHE}" \\
        --task_id ${task_id} \\
        --encoder jepa_head_deep \\
        --out "\${DISTILL_TARGETS}"

    # --- 2b. Fine-tune JEPA with policy distillation ---
    NEW_WM_FILE="\${EXP_DIR}/iter\${ITER}_lejepa_object.ckpt"
    NEW_WM_STEM="\${EXP_DIR}/iter\${ITER}_lejepa"
    echo "Iter \${ITER}: fine-tuning JEPA -> \${NEW_WM_FILE}"
    python ${WORKDIR}/finetune_jepa.py \\
        --jepa_in "\${CUR_WM_FILE}" \\
        --jepa_out "\${NEW_WM_FILE}" \\
        --hdf5 "${WM_HDF5}" \\
        --task_id ${task_id} \\
        --steps ${JEPA_FINETUNE_STEPS} \\
        --batch_size ${JEPA_FINETUNE_BS} \\
        --distill_targets "\${DISTILL_TARGETS}" \\
        --lambda_q_distill 1.0 \\
        --lambda_a_distill 1.0 \\
        --lambda_aux 0 \\
        --lr ${LR} \\
        --device ${WM_DEVICE}

    # --- 3. Re-encode cache ---
    NEW_CACHE="\${EXP_DIR}/iter\${ITER}_cache.pt"
    echo "Iter \${ITER}: re-encoding cache -> \${NEW_CACHE}"
    python ${WORKDIR}/encode_cache_b2.py \\
        --jepa "\${NEW_WM_FILE}" \\
        --hdf5 "${WM_HDF5}" \\
        --out "\${NEW_CACHE}" \\
        --batch_size 32 \\
        --device ${WM_DEVICE}

    # --- 4. Verify predictor ---
    echo "Iter \${ITER}: verifying predictor accuracy"
    python ${WORKDIR}/verify_predictor.py \\
        --jepa "\${NEW_WM_FILE}" \\
        --hdf5 "${WM_HDF5}" \\
        --n_episodes 20 \\
        --device ${WM_DEVICE}

    CUR_WM_FILE="\${NEW_WM_FILE}"
    CUR_WM_STEM="\${NEW_WM_STEM}"
    CUR_CACHE="\${NEW_CACHE}"
done

echo "=== ${job_name} done ==="
echo "Final policy: \${CUR_AGENT_CKPT}"
echo "Final WM (if updated): \${CUR_WM}"
echo "Final cache: \${CUR_CACHE}"
EOF
    echo "Submitted ${job_name}"
  done
done
