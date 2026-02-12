#!/bin/bash
#SBATCH -J stage1_baseline
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/stage1_baseline_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "SLURM job ${SLURM_JOB_ID} starting in ${REPO_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

# Dynamic naming: pass explicit run_id so checkpoint/logs are traceable without guessing.
RUN_ID="s1_ae_j${SLURM_JOB_ID}_q8_ns0.10_lr0.001_bs4_e20_sd42_bc64_lc64_st4_bp1_nmbatch_acrelu_do0"

# Common SemanticKITTI split: train=00-07,09,10 / val=08 / test=11-21
python -u src/train/train.py \
  --data_dir data/dataset/semantickitti/dataset/sequences \
  --train_seqs 00 01 02 03 04 05 06 07 09 10 \
  --epochs 20 \
  --batch_size 4 \
  --num_workers 4 \
  --noise_std 0.10 \
  --quant_bits 8 \
  --run_id "${RUN_ID}" \
  --save_name "${RUN_ID}.pth" \
  --alias_name stage1_baseline.pth \
  2>&1 | tee "logs/${RUN_ID}_train.out"

python -u src/train/evaluate.py \
  --model "data/results/checkpoints/${RUN_ID}.pth" \
  --quant_bits 8 \
  --val_seq 08 \
  --max_frames 999999 \
  --log_path logs/stage1_eval_baseline.out \
  2>&1 | tee -a "logs/${RUN_ID}_train.out"

python -u src/train/evaluate.py \
  --model "data/results/checkpoints/${RUN_ID}.pth" \
  --quant_bits 4 \
  --val_seq 08 \
  --max_frames 999999 \
  --log_path logs/stage1_eval_baseline.out \
  2>&1 | tee -a "logs/${RUN_ID}_train.out"

echo "Done time: $(date '+%Y-%m-%d %H:%M:%S')"
