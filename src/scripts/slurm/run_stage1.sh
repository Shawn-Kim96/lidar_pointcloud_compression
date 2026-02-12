#!/bin/bash
#SBATCH --job-name=stage1_backbone
#SBATCH --output=logs/stage1_%A_%a.out
#SBATCH --error=logs/stage1_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# Create logs directory
mkdir -p logs

# Define Experiment Arrays
# We sweep over Backbone + Learning Rate
BACKBONES=("darknet" "resnet" "darknet" "resnet")
LRS=(1e-4 1e-4 5e-5 5e-5)

# Get current configuration based on Array Task ID
BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Running Stage 1 Experiment: Backbone=$BACKBONE, LR=$LR"

# Experiment Run ID
RUN_ID="s1_run_${SLURM_ARRAY_TASK_ID}"

# Run Training
# --no_teacher ensures this is Stage 1 (Backbone Only)
# --roi_levels / --bg_levels set the quantization granularity
python src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone $BACKBONE \
    --lr $LR \
    --epochs 50 \
    --batch_size 4 \
    --no_teacher \
    --run_id $RUN_ID \
    --roi_levels 256 \
    --bg_levels 16 \
    --lambda_recon 1.0 \
    --lambda_rate 0.1 \
    --lambda_distill 0.0 \
    --lambda_importance 0.5

echo "Done Stage 1 Experiment $SLURM_ARRAY_TASK_ID"
