#!/bin/bash
#SBATCH -J unet_ampl_train
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/logs/slurm outputs/checkpoints outputs/logs/tensorboard outputs/predictions outputs/visualizations outputs/h5

JOB_OUT="outputs/logs/slurm/${SLURM_JOB_ID:-manual}_unet_ampl_train.out"
JOB_ERR="outputs/logs/slurm/${SLURM_JOB_ID:-manual}_unet_ampl_train.err"
exec > "${JOB_OUT}" 2> "${JOB_ERR}"

# Keep the CUDA module style from the legacy unet_ampl.sbatch if modules are available on the cluster.
if command -v module >/dev/null 2>&1; then
  module load nvidia/cuda || true
fi

# TODO: activate the correct environment, for example:
# source ~/.bashrc
# conda activate 3dunet

if [[ ! -f outputs/h5/parihaka_train.h5 || ! -f outputs/h5/parihaka_val.h5 || ! -f outputs/h5/parihaka_test.h5 ]]; then
  python scripts/prepare_parihaka_h5.py
fi

if command -v train3dunet >/dev/null 2>&1; then
  train3dunet --config configs/parihaka_train.yaml
else
  python -m pytorch3dunet.train --config configs/parihaka_train.yaml
fi
