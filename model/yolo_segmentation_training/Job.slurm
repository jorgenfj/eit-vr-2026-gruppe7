#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=studiegrupper-vortex
#SBATCH --job-name=vortex-instance-seg-train
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu40g|gpu80g|gpu32g"
#SBATCH --mem=48G
#SBATCH --output=%x_%j.out

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Helps reduce fragmentation for some training runs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

module purge
module --ignore_cache load foss/2022a
module --ignore_cache load Python/3.10.4-GCCcore-11.3.0

VENV_BASE="${SLURM_TMPDIR:-/tmp}"
if ! VENV_DIR="$(mktemp -d -p "$VENV_BASE" venv-objdetect-XXXXXXXX)"; then
    echo "Error: Failed to create temporary virtualenv directory under '$VENV_BASE'." >&2
    exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

srun python train.py

deactivate
rm -rf "$VENV_DIR"
