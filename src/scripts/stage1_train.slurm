#!/bin/bash
#SBATCH -J stage1_train
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/stage1_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "SLURM job ${SLURM_JOB_ID} starting in ${REPO_DIR}"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

python -u src/train/train.py --data_dir data/dataset/semantickitti/dataset/sequences --train_seq 00 --epochs 1 \
  2>&1 | tee logs/stage1_train.out
