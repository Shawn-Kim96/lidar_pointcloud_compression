#!/bin/bash
#SBATCH --job-name=stage2_distill
#SBATCH --output=logs/stage2_%A_%a.out
#SBATCH --error=logs/stage2_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2

# Create logs directory
mkdir -p logs

# Define Experiment Arrays
# We sweep over Distillation Weights
# Run 0: Balanced (0.5)
# Run 1: High Distill (1.0)
# Run 2: Low Distill (0.1)
LAMBDA_DISTILLS=(0.5 1.0 0.1)

# Get current configuration
L_DISTILL=${LAMBDA_DISTILLS[$SLURM_ARRAY_TASK_ID]}

echo "Running Stage 2 Experiment: Distill Weight=$L_DISTILL"

# Experiment Run ID
RUN_ID="s2_run_${SLURM_ARRAY_TASK_ID}"

# Run Training
# Uses Darknet by default (best form Stage 1)
# Teacher Enabled (default)
python src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "darknet" \
    --lr 1e-4 \
    --epochs 50 \
    --batch_size 4 \
    --teacher_backend "proxy" \
    --run_id $RUN_ID \
    --roi_levels 256 \
    --bg_levels 16 \
    --lambda_recon 1.0 \
    --lambda_rate 0.1 \
    --lambda_distill $L_DISTILL \
    --lambda_importance 0.5

echo "Done Stage 2 Experiment $SLURM_ARRAY_TASK_ID"
