#!/bin/bash
#SBATCH -J parihaka_pipeline
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ] && [ -d "${SUBMIT_DIR}/pytorch3dunet" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../pyproject.toml" ] && [ -d "${SUBMIT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [ -f "${SCRIPT_DIR}/../pyproject.toml" ] && [ -d "${SCRIPT_DIR}/../pytorch3dunet" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repository root. Submit from the repository root or from the sbatch directory."
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/outputs"
SLURM_LOG_DIR="${OUTPUT_DIR}/slurm_logs"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-parihaka_pipeline}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
H5_DIR="${OUTPUT_DIR}/h5_parihaka"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints_parihaka"

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

PATCH_SHAPE_VALUES="${PATCH_SHAPE:-64 128 128}"
STRIDE_SHAPE_VALUES="${STRIDE_SHAPE:-48 96 96}"
read -r -a PATCH_SHAPE_ARGS <<< "${PATCH_SHAPE_VALUES}"
read -r -a STRIDE_SHAPE_ARGS <<< "${STRIDE_SHAPE_VALUES}"
if [ "${#PATCH_SHAPE_ARGS[@]}" -ne 3 ]; then
    echo "ERROR: PATCH_SHAPE must contain exactly 3 integers. Got: ${PATCH_SHAPE_VALUES}"
    exit 1
fi
if [ "${#STRIDE_SHAPE_ARGS[@]}" -ne 3 ]; then
    echo "ERROR: STRIDE_SHAPE must contain exactly 3 integers. Got: ${STRIDE_SHAPE_VALUES}"
    exit 1
fi

echo "hostname=$(hostname)"
echo "start_date=$(date)"
echo "repo=${REPO_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "python=${PYTHON_BIN}"
echo "h5_dir=${H5_DIR}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "force_prepare=${FORCE_PREPARE:-0}"
"${PYTHON_BIN}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
"${PYTHON_BIN}" -c "import h5py, matplotlib, numpy, scipy, torch, yaml"
if [ "${REQUIRE_CUDA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is not available to PyTorch.')"
fi

if [ "${FORCE_PREPARE:-0}" = "1" ] \
    || [ ! -f "${H5_DIR}/parihaka_train.h5" ] \
    || [ ! -f "${H5_DIR}/parihaka_val.h5" ] \
    || [ ! -f "${H5_DIR}/parihaka_test.h5" ]; then
    PREPARE_CMD=(
        "${PYTHON_BIN}" scripts/prepare_parihaka_multichannel_h5.py
        --split-axis 1
        --out-dir outputs/h5_parihaka
        --patch-shape "${PATCH_SHAPE_ARGS[@]}"
        --local-std-window 5 5 5
        --num-classes 6
    )
    printf 'command:'
    printf ' %q' "${PREPARE_CMD[@]}"
    printf '\n'
    "${PREPARE_CMD[@]}"
else
    echo "Found existing Parihaka H5 splits in ${H5_DIR}; skipping prepare. Set FORCE_PREPARE=1 to regenerate."
fi

ANALYZE_CMD=(
    "${PYTHON_BIN}" scripts/analyze_parihaka_h5.py
    --h5-dir outputs/h5_parihaka
    --patch-shape "${PATCH_SHAPE_ARGS[@]}"
    --stride-shape "${STRIDE_SHAPE_ARGS[@]}"
    --out-json outputs/diagnostics/parihaka_h5_analysis.json
)
printf 'command:'
printf ' %q' "${ANALYZE_CMD[@]}"
printf '\n'
"${ANALYZE_CMD[@]}"

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

if command -v train3dunet >/dev/null 2>&1; then
    TRAIN_CMD=(train3dunet --config configs/parihaka_train.yaml)
else
    TRAIN_CMD=("${PYTHON_BIN}" -m pytorch3dunet.train --config configs/parihaka_train.yaml)
fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    TRAIN_CMD+=(--trainer.resume "${RESUME_CHECKPOINT}")
fi
printf 'command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/best_checkpoint.pytorch}"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: missing prediction checkpoint: ${CHECKPOINT_PATH}"
    exit 1
fi

if command -v predict3dunet >/dev/null 2>&1; then
    PREDICT_CMD=(predict3dunet --config configs/parihaka_predict.yaml --model_path "${CHECKPOINT_PATH}")
else
    PREDICT_CMD=("${PYTHON_BIN}" -m pytorch3dunet.predict --config configs/parihaka_predict.yaml --model_path "${CHECKPOINT_PATH}")
fi
printf 'command:'
printf ' %q' "${PREDICT_CMD[@]}"
printf '\n'
"${PREDICT_CMD[@]}"

if [ "${VISUALIZE_RESULTS:-1}" = "1" ]; then
    PREDICTION_PATH="${PREDICTION_PATH:-${REPO_ROOT}/outputs/predictions_parihaka/parihaka_test_predictions.h5}"
    VISUALIZE_CMD=(
        "${PYTHON_BIN}" scripts/visualize_parihaka_predictions.py
        --h5 outputs/h5_parihaka/parihaka_test.h5
        --pred "${PREDICTION_PATH}"
        --out-dir outputs/visualizations_parihaka
        --axis "${VISUALIZE_AXIS:-1}"
        --num-slices "${VISUALIZE_NUM_SLICES:-10}"
        --raw-channel "${VISUALIZE_RAW_CHANNEL:-0}"
    )
    printf 'command:'
    printf ' %q' "${VISUALIZE_CMD[@]}"
    printf '\n'
    "${VISUALIZE_CMD[@]}"
else
    echo "Skipping visualization. Set VISUALIZE_RESULTS=1 to enable it."
fi

echo "finish_date=$(date)"
