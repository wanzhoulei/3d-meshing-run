#!/bin/bash
# ============================================================
#  SLURM job: generate sacrifice-moves dataset (CPU only)
#  Estimated wall time: ~2–4 hours for 2000 hard samples
# ============================================================
#SBATCH --job-name=gen_sacrifice
#SBATCH --partition=savio2          # CPU partition; change to savio3 if needed
#SBATCH --account=YOUR_ACCOUNT      # e.g. fc_yourlab  ← CHANGE THIS
#SBATCH --qos=savio_normal          # matches savio2 partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00             # 6-hour wall limit
#SBATCH --output=logs/gen_sacrifice_%j.out
#SBATCH --error=logs/gen_sacrifice_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL      # ← CHANGE THIS

# ---- environment ------------------------------------------------
mkdir -p logs

module purge
module load python/3.10.12          # adjust to available version on Savio

# Activate your conda environment
# Option A (conda):
source activate 3d-mesh             # ← CHANGE to your env name
# Option B (venv):
# source ~/envs/3d-mesh/bin/activate

cd $SLURM_SUBMIT_DIR

echo "=============================="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Dir:    $(pwd)"
echo "Start:  $(date)"
echo "=============================="

# ---- run --------------------------------------------------------
python gen_sacrifice_dataset.py \
    --source  tet_dataset_grid125_sigma1e-02_N2000.mat \
    --output  tet_dataset_sacrifice_k20_N2000.npz \
    --k 20 \
    --target-n 2000 \
    --threshold 0.05 \
    --greedy-patience 30 \
    --greedy-max-steps 150 \
    --seed 42

echo "=============================="
echo "Done:   $(date)"
echo "=============================="
