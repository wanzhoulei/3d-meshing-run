#!/bin/bash
#SBATCH --job-name=3dmesh-train
#SBATCH --account=fc_dgflow
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:40:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e
set -o pipefail

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "PWD0:  $(pwd)"

cd /global/scratch/users/$USER/3d-meshing-run
echo "A) after cd: $(date)  PWD=$(pwd)"

# ---- Micromamba on scratch (avoids /global/software hangs) ----
export MAMBA_ROOT_PREFIX=/global/scratch/users/$USER/micromamba
export PATH=/global/scratch/users/$USER/bin:$PATH
MM=/global/scratch/users/$USER/bin/micromamba

echo "B) micromamba version: $($MM --version)  at $(date)"

echo "C) python -V (env): $(date)"
$MM run -n mesh_torch python -V

echo "D) nvidia-smi: $(date)"
nvidia-smi

echo "E) torch cuda check (env): $(date)"
$MM run -n mesh_torch python -u -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "F) launching train.py: $(date)"
PYTHONUNBUFFERED=1 $MM run -n mesh_torch python -u train.py
echo "G) train.py finished: $(date)"
