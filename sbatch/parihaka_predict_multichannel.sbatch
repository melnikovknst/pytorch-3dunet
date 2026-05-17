#!/bin/bash
#SBATCH -J parihaka_predict_mc
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o outputs/slurm_logs/%x-%j.out
#SBATCH -e outputs/slurm_logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ] && [ -d "${SUBMIT_DIR}/pytorch3dunet" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SCRIPT_DIR}/../pyproject.toml" ] && [ -d "${SCRIPT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repository root. Submit from the repository root or pass the sbatch file path from there."
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p \
    outputs/h5_axis1_multichannel \
    outputs/diagnostics \
    outputs/checkpoints_multichannel \
    outputs/logs/tensorboard_multichannel \
    outputs/predictions_multichannel \
    outputs/slurm_logs

if command -v module >/dev/null 2>&1; then
    module load nvidia/cuda || true
fi

if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/checkpoints_multichannel/best_checkpoint.pytorch}"

echo "hostname=$(hostname)"
echo "start_date=$(date)"
echo "repo=${REPO_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
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
    --h5-dir outputs/h5_axis1_multichannel
    --train-config configs/parihaka_train_multichannel.yaml
    --predict-config configs/parihaka_predict_multichannel.yaml
)
printf 'command:'
printf ' %q' "${CHECK_CMD[@]}"
printf '\n'
"${CHECK_CMD[@]}"

if command -v predict3dunet >/dev/null 2>&1; then
    CMD=(predict3dunet --config configs/parihaka_predict_multichannel.yaml --model_path "${CHECKPOINT_PATH}")
else
    CMD=("${PYTHON_BIN}" -m pytorch3dunet.predict --config configs/parihaka_predict_multichannel.yaml --model_path "${CHECKPOINT_PATH}")
fi
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "finish_date=$(date)"
