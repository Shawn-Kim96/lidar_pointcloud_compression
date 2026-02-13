#!/bin/bash
#SBATCH --job-name=lidar_train
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
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

# Shared training params
EPOCHS=50
BATCH_SIZE=4
NUM_WORKERS=4
ROI_LEVELS=256
BG_LEVELS=16
L_RECON=1.0
L_RATE=0.1
L_DISTILL=0.0
L_IMPORTANCE=0.5

# Get current configuration based on Array Task ID
BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

# Write model-centric logs alongside Slurm default logs.
DATE_TAG=$(date +%y%m%d)
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_lr${LR}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

# Experiment Run ID
RUN_ID="r${SLURM_ARRAY_TASK_ID}"
MODE_TAG="solo"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

echo "============================================================"
echo "[Experiment Metadata]"
echo "stage: 1"
echo "training_mode: baseline (teacher disabled)"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: none"
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
# --no_teacher ensures this is Stage 1 (Backbone Only)
# --roi_levels / --bg_levels set the quantization granularity
python src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "$BACKBONE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --no_teacher \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --roi_levels "$ROI_LEVELS" \
    --bg_levels "$BG_LEVELS" \
    --lambda_recon "$L_RECON" \
    --lambda_rate "$L_RATE" \
    --lambda_distill "$L_DISTILL" \
    --lambda_importance "$L_IMPORTANCE"

echo "Done Experiment $SLURM_ARRAY_TASK_ID"
