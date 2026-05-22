#!/usr/bin/env bash
# Submit preflight.py to a GPU node to verify (3) chunk-ordering and (4)
# threshold calibration. Checks (1) and (2) already pass on CPU.

set -euo pipefail

WORKDIR="$HOME/leworldmodel/latent_hindsight_rl"
QCDIR="${WORKDIR}/offline_to_online"
VENV="${WORKDIR}/.venv/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable_wm_data}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${WORKDIR}/logs"

sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=qc_preflight
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=${WORKDIR}/logs/qc_preflight_%j.out

source ${VENV}
cd ${QCDIR}
export STABLEWM_HOME="${STABLEWM_HOME}"
export MUJOCO_GL="${MUJOCO_GL}"

echo "=== qc preflight (chunk ordering + threshold calibration) ==="

python preflight.py

echo "=== qc preflight done ==="
EOF

echo "Submitted qc_preflight"
