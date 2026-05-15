#!/bin/bash
#SBATCH -J parihaka_h5_diag
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
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
SLURM_LOG_DIR="${OUTPUT_DIR}/logs/slurm"
SCRIPT_LOG_DIR="${SLURM_LOG_DIR}/${JOB_LOG_NAME:-analyze_parihaka_h5}"
JOB_OUT="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.out"
JOB_ERR="${SCRIPT_LOG_DIR}/${SLURM_JOB_ID:-manual}.err"
H5_DIR="${H5_DIR:-${OUTPUT_DIR}/h5}"
DIAGNOSTICS_DIR="${OUTPUT_DIR}/diagnostics"
OUT_JSON="${OUT_JSON:-${DIAGNOSTICS_DIR}/parihaka_h5_analysis.json}"

mkdir -p "${SCRIPT_LOG_DIR}" "${DIAGNOSTICS_DIR}"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

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

PATCH_SHAPE_VALUES="${PATCH_SHAPE:-64 128 128}"
STRIDE_SHAPE_VALUES="${STRIDE_SHAPE:-48 96 96}"
SPLIT_VALUES="${SPLITS:-train val test}"
read -r -a PATCH_SHAPE_ARGS <<< "${PATCH_SHAPE_VALUES}"
read -r -a STRIDE_SHAPE_ARGS <<< "${STRIDE_SHAPE_VALUES}"
read -r -a SPLIT_ARGS <<< "${SPLIT_VALUES}"

if [ "${#PATCH_SHAPE_ARGS[@]}" -ne 3 ]; then
    echo "ERROR: PATCH_SHAPE must contain exactly 3 integers. Got: ${PATCH_SHAPE_VALUES}"
    exit 1
fi
if [ "${#STRIDE_SHAPE_ARGS[@]}" -ne 3 ]; then
    echo "ERROR: STRIDE_SHAPE must contain exactly 3 integers. Got: ${STRIDE_SHAPE_VALUES}"
    exit 1
fi

for split in "${SPLIT_ARGS[@]}"; do
    if [ ! -f "${H5_DIR}/parihaka_${split}.h5" ]; then
        echo "ERROR: missing H5 split: ${H5_DIR}/parihaka_${split}.h5"
        echo "This diagnostic job does not create or modify HDF5 files."
        exit 1
    fi
done

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "slurm_log_dir=${SCRIPT_LOG_DIR}"
echo "job_stdout=${JOB_OUT}"
echo "job_stderr=${JOB_ERR}"
echo "h5_dir=${H5_DIR}"
echo "splits=${SPLIT_ARGS[*]}"
echo "patch_shape=${PATCH_SHAPE_ARGS[*]}"
echo "stride_shape=${STRIDE_SHAPE_ARGS[*]}"
echo "min_class_voxels=${MIN_CLASS_VOXELS:-256}"
echo "num_classes=${NUM_CLASSES:-6}"
echo "raw_key=${RAW_KEY:-raw}"
echo "label_key=${LABEL_KEY:-label}"
echo "out_json=${OUT_JSON}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c "import h5py, numpy"

"${PYTHON_BIN}" scripts/analyze_parihaka_h5.py \
    --h5-dir "${H5_DIR}" \
    --splits "${SPLIT_ARGS[@]}" \
    --raw-key "${RAW_KEY:-raw}" \
    --label-key "${LABEL_KEY:-label}" \
    --num-classes "${NUM_CLASSES:-6}" \
    --patch-shape "${PATCH_SHAPE_ARGS[@]}" \
    --stride-shape "${STRIDE_SHAPE_ARGS[@]}" \
    --min-class-voxels "${MIN_CLASS_VOXELS:-256}" \
    --out-json "${OUT_JSON}"

echo "Parihaka H5 diagnostics finished."
