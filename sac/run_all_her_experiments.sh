#!/bin/bash
# ------------------------------------------------------------------
# run_all_her_experiments.sh
#
# Submits all 5 HER training experiments as independent PBS jobs.
# Uses train.py (VectorizedEpisodicHERBuffer with future-goal relabelling)
# across pure_distance, fixed-gap, and curriculum modes.
#
# Run this script directly (NOT via qsub) from the job-submission node:
#
#   cd /path/to/latent_hindsight_rl
#   bash run_all_her_experiments.sh
#
# Each experiment gets its own job name, output log, and checkpoint
# directory. Use `qstat` to monitor progress.
# ------------------------------------------------------------------

WORKDIR="$(pwd)"
VENV="$HOME/leworldmodel/.venv/bin/activate"

# Common PBS resource request
PBS_RESOURCES="walltime=10:00:00,select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S"

submit_job() {
    local job_name="$1"
    local mode="$2"
    local extra_args="$3"

    local job_id
    job_id=$(qsub <<EOF
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

python sac_train.py --mode ${mode} ${extra_args}

echo "=== ${job_name} finished ==="
EOF
    )
    echo "Submitted ${job_name} -> ${job_id}"
}

# ------------------------------------------------------------------
# 1. Pure distance + HER  — start=ep[0], goal=ep[-1]; HER relabels
#    goals with future visited states, providing implicit sub-goals.
# ------------------------------------------------------------------
submit_job "her_pure_distance"    "pure_distance"  "--tmax 40"

# ------------------------------------------------------------------
# 2. Fixed gap (small) + HER  — gap=2  WM steps (~10 video frames)
# ------------------------------------------------------------------
submit_job "her_fixed_gap_2"      "fixed"          "--gap 2"

# ------------------------------------------------------------------
# 3. Fixed gap (medium) + HER — gap=8  WM steps (~40 video frames)
# ------------------------------------------------------------------
submit_job "her_fixed_gap_8"      "fixed"          "--gap 8"

# ------------------------------------------------------------------
# 4. Fixed gap (large) + HER  — gap=24 WM steps (~120 video frames)
# ------------------------------------------------------------------
submit_job "her_fixed_gap_24"     "fixed"          "--gap 24"

# ------------------------------------------------------------------
# 5. Adaptive curriculum + HER — gap grows as agent succeeds
# ------------------------------------------------------------------
submit_job "her_curriculum"       "curriculum"     ""

echo ""
echo "All 5 HER experiments submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "Checkpoint directories (created at runtime):"
echo "  checkpoints_her_pure_distance/"
echo "  checkpoints_her_fixed_gap_2/"
echo "  checkpoints_her_fixed_gap_8/"
echo "  checkpoints_her_fixed_gap_24/"
echo "  checkpoints_her_curriculum/"
