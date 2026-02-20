#!/bin/bash
#SBATCH --job-name=lidar_distill
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
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

# Shared training params
BACKBONE="darknet"
TEACHER_BACKEND="proxy"
EPOCHS=50
BATCH_SIZE=4
NUM_WORKERS=4
LR=1e-4
ROI_LEVELS=256
BG_LEVELS=16
L_RECON=1.0
L_RATE=0.1
L_IMPORTANCE=0.5

# Get current configuration
L_DISTILL=${LAMBDA_DISTILLS[$SLURM_ARRAY_TASK_ID]}

# Write model-centric logs alongside Slurm default logs.
DATE_TAG=$(date +%y%m%d)
LOG_PREFIX="${DATE_TAG}_darknet_ld${L_DISTILL}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

# Experiment Run ID
RUN_ID="r${SLURM_ARRAY_TASK_ID}"
MODE_TAG="distill"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

echo "============================================================"
echo "[Experiment Metadata]"
echo "stage: 2"
echo "training_mode: distillation (teacher enabled)"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: ${TEACHER_BACKEND}"
echo "run_id: ${RUN_ID}"
echo "save_dir: ${SAVE_DIR}"
echo "dataset_root: data/dataset/semantickitti/dataset/sequences"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "lr: ${LR}"
echo "roi_levels: ${ROI_LEVELS}"
echo "bg_levels: ${BG_LEVELS}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

# Run Training
# Uses Darknet by default (best form Stage 1)
# Teacher Enabled (default)
python src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "$BACKBONE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --teacher_backend "$TEACHER_BACKEND" \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --roi_levels "$ROI_LEVELS" \
    --bg_levels "$BG_LEVELS" \
    --lambda_recon "$L_RECON" \
    --lambda_rate "$L_RATE" \
    --lambda_distill "$L_DISTILL" \
    --lambda_importance "$L_IMPORTANCE"

echo "Done Experiment $SLURM_ARRAY_TASK_ID"
