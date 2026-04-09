#!/bin/bash
# ------------------------------------------------------------------
# run_bc_experiments.sh
#
# Tests HER + sparse reward + behavioural cloning regularisation.
# Submits jobs for two task modes (pure_distance, fixed gap=5) and
# four BC alpha values (0.0, 0.1, 0.5, 1.0) — 8 RL jobs in total.
#
# If the BC model checkpoint does not yet exist, a BC training job
# is submitted first and all RL jobs are held until it completes.
#
# Run directly from the submission node (NOT via qsub):
#
#   cd /path/to/latent_hindsight_rl
#   bash run_bc_experiments.sh
#
# Monitor with: qstat -u $USER
# ------------------------------------------------------------------

WORKDIR="$(pwd)"
VENV="$HOME/leworldmodel/.venv/bin/activate"
BC_MODEL="$WORKDIR/checkpoints_bc/bc_policy.pth"

PBS_RESOURCES="walltime=12:00:00,select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S"
BC_RESOURCES="walltime=4:00:00,select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S"

# ------------------------------------------------------------------
# Helper: submit a single RL experiment job.
#   $1 job_name   $2 mode   $3 extra_args   $4 dependency (optional)
# ------------------------------------------------------------------
submit_rl_job() {
    local job_name="$1"
    local mode="$2"
    local extra_args="$3"
    local depend_flag="${4:-}"

    local job_id
    job_id=$(qsub ${depend_flag} <<EOF
#!/bin/bash
#PBS -N ${job_name}
#PBS -l ${PBS_RESOURCES}
#PBS -j oe

cd ${WORKDIR}
echo "=== Job: ${job_name} | Mode: ${mode} ${extra_args} ==="
echo "Working dir: \$(pwd)"

module load Python/3.10

export STABLEWM_HOME="\$EPHEMERAL/stable_wm_data"
export EPHEMERAL=\$EPHEMERAL

source ${VENV}

python train.py --mode ${mode} --reward_mode sparse ${extra_args}

echo "=== ${job_name} finished ==="
EOF
    )
    echo "Submitted ${job_name} -> ${job_id}"
    echo "$job_id"
}

# ------------------------------------------------------------------
# Step 1: Train BC model if checkpoint is missing
# ------------------------------------------------------------------
bc_depend=""

if [ ! -f "$BC_MODEL" ]; then
    echo "BC checkpoint not found — submitting BC training job first."

    bc_job_id=$(qsub <<EOF
#!/bin/bash
#PBS -N bc_train
#PBS -l ${BC_RESOURCES}
#PBS -j oe

cd ${WORKDIR}
echo "=== BC Training Job ==="
echo "Working dir: \$(pwd)"

module load Python/3.10

export STABLEWM_HOME="\$EPHEMERAL/stable_wm_data"
export EPHEMERAL=\$EPHEMERAL

source ${VENV}

python train_bc.py \
    --save_dir ./checkpoints_bc \
    --epochs 50 \
    --batch_size 1024 \
    --lr 3e-4 \
    --goals_per_step 4

echo "=== bc_train finished ==="
EOF
    )
    echo "Submitted bc_train -> ${bc_job_id}"
    bc_depend="-W depend=afterok:${bc_job_id}"
else
    echo "Found existing BC checkpoint: $BC_MODEL"
fi

# ------------------------------------------------------------------
# Step 2: Submit RL experiments, optionally gated on BC training
# ------------------------------------------------------------------
#ALPHAS=("0.0" "0.1" "0.5" "1.0")
ALPHAS=("0.1" "0.5" "1.0")

for ALPHA in "${ALPHAS[@]}"; do
    # Format alpha for job name (replace dot with underscore)
    alpha_tag="${ALPHA//./_}"

    if [ "$ALPHA" = "0.0" ]; then
        bc_args=""
    else
        bc_args="--bc_model_path $BC_MODEL --bc_alpha $ALPHA"
    fi

    # pure_distance
    submit_rl_job \
        "her_pure_bc${alpha_tag}" \
        "pure_distance" \
        "--tmax 40 ${bc_args}" \
        "$bc_depend" > /dev/null

    # fixed gap=5
    submit_rl_job \
        "her_gap5_bc${alpha_tag}" \
        "fixed" \
        "--gap 5 ${bc_args}" \
        "$bc_depend" > /dev/null
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "Checkpoint directories (created at runtime):"
echo "  checkpoints_her_pure_distance_sparse/"
echo "  checkpoints_her_pure_distance_sparse_bc0.1/"
echo "  checkpoints_her_pure_distance_sparse_bc0.5/"
echo "  checkpoints_her_pure_distance_sparse_bc1.0/"
echo "  checkpoints_her_fixed_gap_5_sparse/"
echo "  checkpoints_her_fixed_gap_5_sparse_bc0.1/"
echo "  checkpoints_her_fixed_gap_5_sparse_bc0.5/"
echo "  checkpoints_her_fixed_gap_5_sparse_bc1.0/"
