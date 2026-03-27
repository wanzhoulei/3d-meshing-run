#!/bin/bash
# ============================================================
#  SLURM job: train sacrifice-moves RL agent (1 GPU)
#  Estimated wall time: ~6–10 hours for 180 rounds on A40
# ============================================================
#SBATCH --job-name=train_sacrifice
#SBATCH --partition=savio4_gpu      # A40 GPUs (40 GB); or savio3_gpu for V100
#SBATCH --account=YOUR_ACCOUNT      # e.g. fc_yourlab  ← CHANGE THIS
#SBATCH --qos=savio_normal          # matches partition; use savio_lowprio for preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # workers for CPU-side env stepping
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # 12-hour wall limit
#SBATCH --output=logs/train_sacrifice_%j.out
#SBATCH --error=logs/train_sacrifice_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL      # ← CHANGE THIS

# ---- environment ------------------------------------------------
mkdir -p logs

module purge
module load python/3.10.12          # adjust to available version on Savio
module load cuda/12.1               # adjust to available CUDA version

# Activate your conda environment
# Option A (conda):
source activate 3d-mesh             # ← CHANGE to your env name
# Option B (venv):
# source ~/envs/3d-mesh/bin/activate

cd $SLURM_SUBMIT_DIR

echo "=============================="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Dir:    $(pwd)"
echo "Start:  $(date)"
echo "=============================="

# Verify GPU is visible
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ---- run --------------------------------------------------------
python train_sacrifice.py

echo "=============================="
echo "Done:   $(date)"
echo "=============================="
