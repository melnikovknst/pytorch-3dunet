#!/bin/bash
#SBATCH -J parihaka_visualize
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
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
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-parihaka_visualize}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"

cd "${REPO_ROOT}"
mkdir -p \
    outputs/visualizations_parihaka \
    "${SCRIPT_LOG_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

H5_PATH="${H5_PATH:-outputs/h5_parihaka/parihaka_test.h5}"
PREDICTION_PATH="${PREDICTION_PATH:-outputs/predictions_parihaka/parihaka_test_predictions.h5}"
VISUALIZATION_DIR="${VISUALIZATION_DIR:-outputs/visualizations_parihaka}"

echo "hostname=$(hostname)"
echo "start_date=$(date)"
echo "repo=${REPO_ROOT}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "python=${PYTHON_BIN}"
echo "h5=${H5_PATH}"
echo "prediction=${PREDICTION_PATH}"
echo "visualization_dir=${VISUALIZATION_DIR}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c "import h5py, matplotlib, numpy"

if [ ! -f "${H5_PATH}" ]; then
    echo "ERROR: missing HDF5 test file: ${H5_PATH}"
    exit 1
fi
if [ ! -f "${PREDICTION_PATH}" ]; then
    echo "ERROR: missing prediction file: ${PREDICTION_PATH}"
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" scripts/visualize_parihaka_predictions.py
    --h5 "${H5_PATH}"
    --pred "${PREDICTION_PATH}"
    --out-dir "${VISUALIZATION_DIR}"
    --axis "${VISUALIZE_AXIS:-0}"
    --num-slices "${VISUALIZE_NUM_SLICES:-10}"
    --raw-channel "${VISUALIZE_RAW_CHANNEL:-0}"
)
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "finish_date=$(date)"
