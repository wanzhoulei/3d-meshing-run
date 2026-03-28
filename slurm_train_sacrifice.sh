#!/bin/bash
# ============================================================
#  SLURM job: train sacrifice-moves RL agent (1 GPU, V100)
#  Estimated wall time: ~8–12 hours for 180 rounds on V100
# ============================================================
#SBATCH --job-name=train_sacrifice
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_dgflow
#SBATCH --qos=v100_gpu3_normal
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ---- environment ------------------------------------------------
set -e
set -o pipefail

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"

cd /global/scratch/users/$USER/3d-meshing-run

export MAMBA_ROOT_PREFIX=/global/scratch/users/$USER/micromamba
export PATH=/global/scratch/users/$USER/bin:$PATH
MM=/global/scratch/users/$USER/bin/micromamba

echo "micromamba: $($MM --version)"
$MM run -n mesh_torch python -V

nvidia-smi

PYTHONUNBUFFERED=1 $MM run -n mesh_torch python -u -c \
    "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "Launching train_sacrifice.py: $(date)"
PYTHONUNBUFFERED=1 $MM run -n mesh_torch python -u train_sacrifice.py
echo "Done: $(date)"

