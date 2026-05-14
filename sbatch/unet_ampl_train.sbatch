#!/bin/bash
#SBATCH -J unet_ampl_train
#SBATCH -p a100serv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o outputs/logs/slurm/%x.%j.out
#SBATCH -e outputs/logs/slurm/%x.%j.err
#
# Add a GPU directive here if your Slurm cluster requires it, for example:
##SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/logs/slurm outputs/checkpoints outputs/logs/tensorboard outputs/predictions outputs/visualizations outputs/h5

module load nvidia/cuda

# TODO: activate the correct environment if .venv is not used, for example:
# source ~/.bashrc
# conda activate 3dunet
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi || true

python -c "import h5py, matplotlib, numpy, torch, yaml"

if [[ ! -f outputs/h5/parihaka_train.h5 || ! -f outputs/h5/parihaka_val.h5 || ! -f outputs/h5/parihaka_test.h5 ]]; then
  python scripts/prepare_parihaka_h5.py
fi

if command -v train3dunet >/dev/null 2>&1; then
  train3dunet --config configs/parihaka_train.yaml
else
  python -m pytorch3dunet.train --config configs/parihaka_train.yaml
fi
