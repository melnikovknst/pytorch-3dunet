#!/bin/bash
#SBATCH -J parihaka_predict
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ] && [ -d "${SUBMIT_DIR}/pytorch3dunet" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SCRIPT_DIR}/../pyproject.toml" ] && [ -d "${SCRIPT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repository root. Submit from the repository root or from the sbatch directory."
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/outputs"
SLURM_LOG_DIR="${OUTPUT_DIR}/slurm_logs"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-parihaka_predict}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"

cd "${REPO_ROOT}"
mkdir -p \
    outputs/h5_parihaka \
    outputs/diagnostics \
    outputs/checkpoints_parihaka \
    outputs/logs/tensorboard_parihaka \
    outputs/predictions_parihaka \
    outputs/visualizations_parihaka \
    "${SCRIPT_LOG_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

if command -v module >/dev/null 2>&1; then
    module load nvidia/cuda || true
fi

if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/checkpoints_parihaka/best_checkpoint.pytorch}"

echo "hostname=$(hostname)"
echo "start_date=$(date)"
echo "repo=${REPO_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "python=${PYTHON_BIN}"
echo "checkpoint=${CHECKPOINT_PATH}"
"${PYTHON_BIN}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
"${PYTHON_BIN}" -c "import h5py, numpy, torch, yaml"
if [ "${REQUIRE_CUDA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is not available to PyTorch.')"
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: missing checkpoint: ${CHECKPOINT_PATH}"
    exit 1
fi

CHECK_CMD=(
    "${PYTHON_BIN}" scripts/check_parihaka_pipeline.py
    --h5-dir outputs/h5_parihaka
    --train-config configs/parihaka_train.yaml
    --predict-config configs/parihaka_predict.yaml
)
printf 'command:'
printf ' %q' "${CHECK_CMD[@]}"
printf '\n'
"${CHECK_CMD[@]}"

if command -v predict3dunet >/dev/null 2>&1; then
    CMD=(predict3dunet --config configs/parihaka_predict.yaml --model_path "${CHECKPOINT_PATH}")
else
    CMD=("${PYTHON_BIN}" -m pytorch3dunet.predict --config configs/parihaka_predict.yaml --model_path "${CHECKPOINT_PATH}")
fi
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "finish_date=$(date)"
