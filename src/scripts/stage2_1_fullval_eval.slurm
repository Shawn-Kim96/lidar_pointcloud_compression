#!/bin/bash
#SBATCH -J stage2_1_fullval_eval
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/stage2_1_fullval_eval_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/visualization
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

STAGE2_1_CKPT="${1:-data/results/checkpoints/s2_1_s2_1_smoke_ns0.02_lr0.0002_bs8_e1_sd42_bc64_lc64_st4_bp1_nmbatch_acrelu_do0.0_rl256_bl16_20260210T063614Z_ec198f48.pth}"
TAG="$(basename "${STAGE2_1_CKPT}" .pth)"

echo "Eval checkpoint: ${STAGE2_1_CKPT}"
echo "Tag: ${TAG}"

python -u src/train/evaluate_stage2_1.py \
  --stage1_model data/results/checkpoints/stage1_baseline.pth \
  --stage2_1_ckpt "${STAGE2_1_CKPT}" \
  --val_seq 08 \
  --max_frames 4071 \
  --batch_size 4 \
  --num_workers 2 \
  --teacher_backend proxy \
  --teacher_ckpt "" \
  --output_plot "data/results/visualization/stage2_1_rd_curve_${TAG}_fullval.png" \
  --output_log "logs/stage2_1_eval_${TAG}_fullval.out"

python -u src/train/evaluate_teacher_scores.py \
  --model_stage stage2_1 \
  --model_ckpt "${STAGE2_1_CKPT}" \
  --val_seq 08 \
  --max_frames 4071 \
  --batch_size 4 \
  --num_workers 2 \
  --teacher_backend proxy \
  --teacher_ckpt "" \
  --no_labels \
  --output_csv "logs/stage2_1_teacher_scores_${TAG}_fullval.csv"

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
