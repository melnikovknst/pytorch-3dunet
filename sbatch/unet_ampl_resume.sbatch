#!/bin/bash
#SBATCH -J unet_ampl_resume
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#
# Add a GPU directive here if your Slurm cluster requires it, for example:
##SBATCH --gres=gpu:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ] && [ -d "${SUBMIT_DIR}/pytorch3dunet" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../pyproject.toml" ] && [ -d "${SUBMIT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [ -f "${SCRIPT_DIR}/../pyproject.toml" ] && [ -d "${SCRIPT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repository root from submit directory: ${SUBMIT_DIR}"
    echo "Submit this job from the repository root or from the sbatch directory."
    exit 1
fi

export JOB_LOG_NAME="${JOB_LOG_NAME:-unet_ampl_resume}"
export RUN_INFERENCE_AFTER_TRAIN="${RUN_INFERENCE_AFTER_TRAIN:-1}"
exec "${SCRIPT_DIR}/unet_ampl_train.sbatch"
