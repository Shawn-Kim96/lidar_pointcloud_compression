#!/bin/bash
#SBATCH -J stage2_overnight
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o logs/stage2_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "SLURM job ${SLURM_JOB_ID} starting in ${REPO_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

RUN_ID="s2_adapt_j${SLURM_JOB_ID}_q8_ns0.02_lr0.0002_bs4_e12_sd42_bc64_lc64_st4_bp1_nmbatch_acrelu_do0_rl256_bl16_lt2.0_be0.005_rw10.0"

python -u src/train/train_stage2.py \
  --data_dir data/dataset/semantickitti/dataset/sequences \
  --train_seqs 00 01 02 03 04 05 06 07 09 10 \
  --val_seq 08 \
  --epochs 12 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 2e-4 \
  --noise_std 0.02 \
  --lambda_task 2.0 \
  --beta_entropy 0.005 \
  --roi_recon_weight 10.0 \
  --xyz_weight 0.5 \
  --val_eval_batches 64 \
  --init_stage1 data/results/checkpoints/stage1_baseline.pth \
  --run_id "${RUN_ID}" \
  --save_name "${RUN_ID}.pth" \
  --alias_name stage2_adaptive.pth \
  2>&1 | tee "logs/${RUN_ID}_train.out"

python -u src/train/evaluate_stage2.py \
  --stage1_model data/results/checkpoints/stage1_baseline.pth \
  --stage2_ckpt "data/results/checkpoints/${RUN_ID}.pth" \
  --val_seq 08 \
  --max_frames 2048 \
  --batch_size 4 \
  --num_workers 2 \
  --output_plot data/results/visualization/comparison_stage1_stage2.png \
  --output_log logs/stage2_eval_comparison.out \
  2>&1 | tee -a "logs/${RUN_ID}_train.out"

echo "Done time: $(date '+%Y-%m-%d %H:%M:%S')"
