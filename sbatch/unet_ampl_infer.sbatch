#!/bin/bash
#SBATCH -J 3Dunet_ampl_infer
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#
# Add a GPU directive here if your Slurm cluster requires it, for example:
##SBATCH --gres=gpu:1

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
SLURM_LOG_DIR="${OUTPUT_DIR}/logs/slurm"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-unet_ampl_infer}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
PREDICTION_DIR="${OUTPUT_DIR}/predictions"
VISUALIZATION_DIR="${OUTPUT_DIR}/visualizations"
H5_DIR="${OUTPUT_DIR}/h5"

mkdir -p "${SCRIPT_LOG_DIR}" "${CHECKPOINT_DIR}" "${PREDICTION_DIR}" "${VISUALIZATION_DIR}" "${H5_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

module load nvidia/cuda

cd "${REPO_ROOT}"

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
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="${VENV_DIR}/bin/python"
else
    echo "Creating virtual environment: ${VENV_DIR}"
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="${VENV_DIR}/bin/python"
fi

PROJECT_HASH="$(cksum "${REPO_ROOT}/pyproject.toml" | awk '{print $1}')"
PROJECT_MARKER="${VENV_DIR}/.project-${PROJECT_HASH}.installed"

if [ ! -f "${PROJECT_MARKER}" ]; then
    echo "Installing Python dependencies into ${VENV_DIR}"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    if [ "${INSTALL_PYTORCH_CUDA:-0}" = "1" ]; then
        TORCH_CUDA_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)"
        if [ "${TORCH_CUDA_VERSION}" != "12.6" ]; then
            echo "Installing PyTorch build for CUDA 12.6"
            "${PYTHON_BIN}" -m pip install --upgrade torch --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        fi
    fi
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
    if [ "${INSTALL_PYTORCH_CUDA:-0}" = "1" ]; then
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
    if [ "${INSTALL_PYTORCH_CUDA:-0}" = "1" ]; then
        TORCH_CUDA_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)"
        if [ "${TORCH_CUDA_VERSION}" != "12.6" ]; then
            echo "Installing PyTorch build for CUDA 12.6"
            "${PYTHON_BIN}" -m pip install --upgrade torch --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
        fi
    else
        echo "Skipping PyTorch CUDA auto-install; set INSTALL_PYTORCH_CUDA=1 to reinstall it."
    fi
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/best_checkpoint.pytorch}"
INFER_SPLIT="${INFER_SPLIT:-val}"
case "${INFER_SPLIT}" in
    train|val|test) ;;
    *)
        echo "ERROR: INFER_SPLIT must be one of: train, val, test. Got: ${INFER_SPLIT}"
        exit 1
        ;;
esac
INFER_H5_PATH="${H5_DIR}/parihaka_${INFER_SPLIT}.h5"
PREDICTION_PATH="${PREDICTION_DIR}/parihaka_${INFER_SPLIT}_predictions.h5"
VISUALIZATION_SPLIT_DIR="${VISUALIZATION_DIR}/${INFER_SPLIT}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "infer_split=${INFER_SPLIT}"
echo "infer_h5=${INFER_H5_PATH}"
echo "h5=${H5_DIR}"
echo "predictions=${PREDICTION_DIR}"
echo "prediction_file=${PREDICTION_PATH}"
echo "visualizations=${VISUALIZATION_SPLIT_DIR}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi

"${PYTHON_BIN}" -c "import h5py, matplotlib, numpy, torch, yaml"
if [ "${REQUIRE_CUDA:-1}" = "1" ]; then
    "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is not available to PyTorch. Check GPU allocation, NVIDIA driver, or PyTorch CUDA wheel.')"
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: missing checkpoint: ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -f "${INFER_H5_PATH}" ]; then
    echo "ERROR: missing H5 split: ${INFER_H5_PATH}"
    echo "Run sbatch/unet_ampl_train.sbatch first, or create the H5 files before inference."
    exit 1
fi

read -r PATCH_D PATCH_H PATCH_W < <("${PYTHON_BIN}" -c 'import yaml; cfg = yaml.safe_load(open("configs/parihaka_predict.yaml")); print(*cfg["loaders"]["test"]["slice_builder"]["patch_shape"])')
PATCH_SHAPE=("${PATCH_D}" "${PATCH_H}" "${PATCH_W}")
echo "predict_patch_shape=${PATCH_SHAPE[*]}"
"${PYTHON_BIN}" scripts/validate_parihaka_h5.py --patch-shape "${PATCH_SHAPE[@]}" --splits "${INFER_SPLIT}"

PREDICT_ARGS=(
    --config configs/parihaka_predict.yaml
    --model_path "${CHECKPOINT_PATH}"
    --loaders.test.file_paths "${INFER_H5_PATH}"
)
echo "Starting UNet inference with args: ${PREDICT_ARGS[*]}"
if command -v predict3dunet >/dev/null 2>&1; then
    predict3dunet "${PREDICT_ARGS[@]}"
else
    "${PYTHON_BIN}" -m pytorch3dunet.predict "${PREDICT_ARGS[@]}"
fi
echo "UNet inference finished."

if [ "${RUN_VISUALIZATION_AFTER_INFERENCE:-1}" = "1" ]; then
    echo "Starting Parihaka prediction visualization."
    "${PYTHON_BIN}" scripts/visualize_parihaka_predictions.py \
        --h5 "${INFER_H5_PATH}" \
        --pred "${PREDICTION_PATH}" \
        --out-dir "${VISUALIZATION_SPLIT_DIR}"
    echo "Visualization finished."
fi
