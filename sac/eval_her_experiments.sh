#!/bin/bash
# ------------------------------------------------------------------
# eval_her_experiments.sh
#
# Submits evaluation jobs for all 5 HER-trained experiments.
# Run directly (NOT via qsub) from the latent_hindsight_rl directory:
#
#   cd /path/to/latent_hindsight_rl
#   bash eval_her_experiments.sh
#
# Each experiment is evaluated in its own PBS job. Results are saved
# to eval_results_<experiment>/ alongside the checkpoint directories.
# Use `qstat` to monitor progress.
# ------------------------------------------------------------------

WORKDIR="$(pwd)"
VENV="$HOME/leworldmodel/.venv/bin/activate"
PBS_RESOURCES="walltime=00:40:00,select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S"

submit_eval() {
    local job_name="$1"
    local checkpoint_dir="$2"   # relative to WORKDIR (latent_hindsight_rl/)

    local job_id
    job_id=$(qsub <<EOF
#!/bin/bash
#PBS -N ${job_name}
#PBS -l ${PBS_RESOURCES}
#PBS -j oe

cd ${WORKDIR}
echo "=== Eval: ${job_name} | Checkpoint: ${checkpoint_dir} ==="

module load Python/3.10
export STABLEWM_HOME="\$EPHEMERAL/stable_wm_data"
export EPHEMERAL=\$EPHEMERAL

source ${VENV}

export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_GL=glfw

xvfb-run -a --server-args="-screen 0 1024x768x24 +extension RANDR" \
    python eval_actor.py \
    --config-name=cube.yaml \
    policy=cube/lejepa \
    ++checkpoint_dir=${checkpoint_dir}

echo "=== ${job_name} finished ==="
ls -lh ${WORKDIR}/eval_results_${checkpoint_dir#checkpoints_}/
EOF
    )
    echo "Submitted ${job_name} -> ${job_id}"
}

# Submit one eval job per HER experiment
submit_eval "eval_her_pure_distance"  "checkpoints_her_pure_distance"
submit_eval "eval_her_fixed_gap_2"    "checkpoints_her_fixed_gap_2"
submit_eval "eval_her_fixed_gap_8"    "checkpoints_her_fixed_gap_8"
submit_eval "eval_her_fixed_gap_24"   "checkpoints_her_fixed_gap_24"
submit_eval "eval_her_curriculum"     "checkpoints_her_curriculum"

echo ""
echo "All 5 HER evaluation jobs submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  eval_results_her_pure_distance/results.txt"
echo "  eval_results_her_fixed_gap_2/results.txt"
echo "  eval_results_her_fixed_gap_8/results.txt"
echo "  eval_results_her_fixed_gap_24/results.txt"
echo "  eval_results_her_curriculum/results.txt"
