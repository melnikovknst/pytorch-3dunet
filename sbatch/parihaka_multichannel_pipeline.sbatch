#!/bin/bash
#SBATCH -J parihaka_mc
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
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
    echo "ERROR: cannot locate repository root from submit directory: ${SUBMIT_DIR}"
    echo "Submit this job from the repository root or from the sbatch directory."
    exit 1
fi

OUTPUT_DIR="${REPO_ROOT}/outputs"
SLURM_LOG_DIR="${OUTPUT_DIR}/slurm_logs"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-parihaka_multichannel_pipeline}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
H5_DIR="${OUTPUT_DIR}/h5_axis1_multichannel"
DIAGNOSTICS_DIR="${OUTPUT_DIR}/diagnostics"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints_multichannel"
TENSORBOARD_DIR="${OUTPUT_DIR}/logs/tensorboard_multichannel"
PREDICTION_DIR="${OUTPUT_DIR}/predictions_multichannel"
VISUALIZATION_DIR="${OUTPUT_DIR}/visualizations_multichannel"

mkdir -p \
    "${SCRIPT_LOG_DIR}" \
    "${H5_DIR}" \
    "${DIAGNOSTICS_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${PREDICTION_DIR}" \
    "${VISUALIZATION_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then
    module load nvidia/cuda || true
fi

BOOTSTRAP_PYTHON=""
for PY_CANDIDATE in python3.12 python3.11 python3 python; do
    if command -v "${PY_CANDIDATE}" >/dev/null 2>&1; then
        PY_CANDIDATE_PATH="$(command -v "${PY_CANDIDATE}")"
        if "${PY_CANDIDATE_PATH}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then
            BOOTSTRAP_PYTHON="${PY_CANDIDATE_PATH}"
            break
        fi
    fi
done

if [ -z "${BOOTSTRAP_PYTHON}" ]; then
    echo "ERROR: compatible Python interpreter not found."
    echo "This project requires Python 3.11 or newer."
    echo "Load a compatible Python module before submitting the job."
    exit 1
fi

VENV_DIR="${REPO_ROOT}/.venv"
if [ -x "${VENV_DIR}/bin/python" ]; then
    if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then
        echo "ERROR: existing ${VENV_DIR} uses an incompatible Python."
        echo "Remove it and rerun: rm -rf ${VENV_DIR}"
        echo "This project requires Python 3.11 or newer."
        exit 1
    fi
else
    echo "Creating virtual environment: ${VENV_DIR}"
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
PYTHON_BIN="${VENV_DIR}/bin/python"

PROJECT_HASH="$(cksum "${REPO_ROOT}/pyproject.toml" | awk '{print $1}')"
PROJECT_MARKER="${VENV_DIR}/.project-${PROJECT_HASH}.installed"

if [ ! -f "${PROJECT_MARKER}" ]; then
    echo "Installing Python dependencies into ${VENV_DIR}"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    if [ "${INSTALL_PYTORCH_CUDA:-1}" = "1" ]; then
        TORCH_CUDA_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)"
        if [ "${TORCH_CUDA_VERSION}" != "12.6" ]; then
            echo "Installing PyTorch build for CUDA 12.6"
            "${PYTHON_BIN}" -m pip install --upgrade torch --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        fi
    fi
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
    if [ "${INSTALL_PYTORCH_CUDA:-1}" = "1" ]; then
        TORCH_CUDA_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)"
        if [ "${TORCH_CUDA_VERSION}" != "12.6" ]; then
            echo "Reinstalling PyTorch build for CUDA 12.6 after project dependency resolution"
            "${PYTHON_BIN}" -m pip install --upgrade torch --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        fi
    fi
    rm -f "${VENV_DIR}"/.project-*.installed
    touch "${PROJECT_MARKER}"
else
    echo "Python dependencies already installed for current pyproject.toml"
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/parihaka_train_multichannel.yaml}"
PREDICT_CONFIG="${PREDICT_CONFIG:-configs/parihaka_predict_multichannel.yaml}"
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

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "start_date=$(date)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "h5_dir=${H5_DIR}"
echo "diagnostics=${DIAGNOSTICS_DIR}"
echo "checkpoints=${CHECKPOINT_DIR}"
echo "tensorboard=${TENSORBOARD_DIR}"
echo "predictions=${PREDICTION_DIR}"
echo "visualizations=${VISUALIZATION_DIR}"
echo "train_config=${TRAIN_CONFIG}"
echo "predict_config=${PREDICT_CONFIG}"
echo "force_prepare=${FORCE_PREPARE:-0}"
echo "run_predict_after_train=${RUN_PREDICT_AFTER_TRAIN:-1}"
echo "run_visualization_after_predict=${RUN_VISUALIZATION_AFTER_PREDICT:-1}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"${PYTHON_BIN}" -c "import h5py, matplotlib, numpy, scipy, torch, yaml"
if [ "${REQUIRE_CUDA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is not available to PyTorch. Check GPU allocation, NVIDIA driver, or PyTorch CUDA wheel.')"
fi

if [ "${FORCE_PREPARE:-0}" = "1" ] \
    || [ ! -f "${H5_DIR}/parihaka_train.h5" ] \
    || [ ! -f "${H5_DIR}/parihaka_val.h5" ] \
    || [ ! -f "${H5_DIR}/parihaka_test.h5" ]; then
    PREPARE_CMD=(
        "${PYTHON_BIN}" scripts/prepare_parihaka_multichannel_h5.py
        --split-axis 1
        --out-dir outputs/h5_axis1_multichannel
        --patch-shape "${PATCH_SHAPE_ARGS[@]}"
    )
    printf 'command:'
    printf ' %q' "${PREPARE_CMD[@]}"
    printf '\n'
    "${PREPARE_CMD[@]}"
else
    echo "Found existing multichannel H5 splits in ${H5_DIR}; skipping prepare. Set FORCE_PREPARE=1 to regenerate."
fi

ANALYZE_CMD=(
    "${PYTHON_BIN}" scripts/analyze_parihaka_h5.py
    --h5-dir outputs/h5_axis1_multichannel
    --patch-shape "${PATCH_SHAPE_ARGS[@]}"
    --stride-shape "${STRIDE_SHAPE_ARGS[@]}"
    --out-json outputs/diagnostics/parihaka_multichannel_h5_analysis.json
)
printf 'command:'
printf ' %q' "${ANALYZE_CMD[@]}"
printf '\n'
"${ANALYZE_CMD[@]}"

CHECK_CMD=(
    "${PYTHON_BIN}" scripts/check_parihaka_pipeline.py
    --h5-dir outputs/h5_axis1_multichannel
    --train-config "${TRAIN_CONFIG}"
    --predict-config "${PREDICT_CONFIG}"
)
printf 'command:'
printf ' %q' "${CHECK_CMD[@]}"
printf '\n'
"${CHECK_CMD[@]}"

if command -v train3dunet >/dev/null 2>&1; then
    TRAIN_CMD=(train3dunet --config "${TRAIN_CONFIG}")
else
    TRAIN_CMD=("${PYTHON_BIN}" -m pytorch3dunet.train --config "${TRAIN_CONFIG}")
fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    TRAIN_CMD+=(--trainer.resume "${RESUME_CHECKPOINT}")
fi
printf 'command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [ "${RUN_PREDICT_AFTER_TRAIN:-1}" = "1" ]; then
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/best_checkpoint.pytorch}"
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "ERROR: missing prediction checkpoint: ${CHECKPOINT_PATH}"
        exit 1
    fi

    if command -v predict3dunet >/dev/null 2>&1; then
        PREDICT_CMD=(predict3dunet --config "${PREDICT_CONFIG}" --model_path "${CHECKPOINT_PATH}")
    else
        PREDICT_CMD=("${PYTHON_BIN}" -m pytorch3dunet.predict --config "${PREDICT_CONFIG}" --model_path "${CHECKPOINT_PATH}")
    fi
    printf 'command:'
    printf ' %q' "${PREDICT_CMD[@]}"
    printf '\n'
    "${PREDICT_CMD[@]}"

    if [ "${RUN_VISUALIZATION_AFTER_PREDICT:-1}" = "1" ]; then
        for AXIS in 0 1 2; do
            VIS_CMD=(
                "${PYTHON_BIN}" scripts/visualize_parihaka_predictions.py
                --h5 outputs/h5_axis1_multichannel/parihaka_test.h5
                --prediction outputs/predictions_multichannel/parihaka_test_predictions.h5
                --raw-channel 0
                --axis "${AXIS}"
                --out-dir "outputs/visualizations_multichannel/axis${AXIS}"
            )
            printf 'command:'
            printf ' %q' "${VIS_CMD[@]}"
            printf '\n'
            "${VIS_CMD[@]}"
        done
    fi
fi

echo "finish_date=$(date)"
