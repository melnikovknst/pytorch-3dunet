#!/bin/bash
#SBATCH -J 3Dunet_ampl_train
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
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/unet_ampl_train"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
TENSORBOARD_DIR="${OUTPUT_DIR}/logs/tensorboard"
PREDICTION_DIR="${OUTPUT_DIR}/predictions"
VISUALIZATION_DIR="${OUTPUT_DIR}/visualizations"
H5_DIR="${OUTPUT_DIR}/h5"

mkdir -p "${SCRIPT_LOG_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${PREDICTION_DIR}" "${VISUALIZATION_DIR}" "${H5_DIR}"
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
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
    rm -f "${VENV_DIR}"/.project-*.installed
    touch "${PROJECT_MARKER}"
else
    echo "Python dependencies already installed for current pyproject.toml"
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "checkpoints=${CHECKPOINT_DIR}"
echo "tensorboard=${TENSORBOARD_DIR}"
echo "h5=${H5_DIR}"
echo "predictions=${PREDICTION_DIR}"
echo "visualizations=${VISUALIZATION_DIR}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi

"${PYTHON_BIN}" -c "import h5py, matplotlib, numpy, torch, yaml"

if [ ! -f "${H5_DIR}/parihaka_train.h5" ] || [ ! -f "${H5_DIR}/parihaka_val.h5" ] || [ ! -f "${H5_DIR}/parihaka_test.h5" ]; then
    "${PYTHON_BIN}" scripts/prepare_parihaka_h5.py
fi

if command -v train3dunet >/dev/null 2>&1; then
    train3dunet --config configs/parihaka_train.yaml
else
    "${PYTHON_BIN}" -m pytorch3dunet.train --config configs/parihaka_train.yaml
fi
