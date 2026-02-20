#!/bin/bash
# Uniform quantization baseline sweep.
# Purpose:
# - Provide ROI-unaware compression baseline using uniform quantization.
# - Compare against adaptive (importance-aware) quantization runs.
#
# Usage:
#   sbatch src/scripts/run_uniform_baseline.sh
#
# Array layout (backbone x quant_bits):
#   0: darknet, q4
#   1: darknet, q6
#   2: darknet, q8
#   3: resnet,  q4
#   4: resnet,  q6
#   5: resnet,  q8

#SBATCH --job-name=lidar_uniform
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5

set -euo pipefail

mkdir -p logs

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  PYTHON_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  PYTHON_ENV_DESC="python:$(command -v python)"
fi

BACKBONES=("darknet" "darknet" "darknet" "resnet" "resnet" "resnet")
QBITS=(4 6 8 4 6 8)

BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
QUANT_BITS=${QBITS[$SLURM_ARRAY_TASK_ID]}

QUANTIZER_MODE="uniform"
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.0}
L_DISTILL=${L_DISTILL:-0.0}
L_IMPORTANCE=${L_IMPORTANCE:-0.0}

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_uniform_q${QUANT_BITS}_j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

RUN_ID="j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
MODE_TAG="uniform"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_q${QUANT_BITS}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

echo "============================================================"
echo "[Experiment Metadata]"
echo "stage: 0"
echo "training_mode: uniform quantization baseline"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: none"
echo "run_id: ${RUN_ID}"
echo "save_dir: ${SAVE_DIR}"
echo "dataset_root: data/dataset/semantickitti/dataset/sequences"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "lr: ${LR}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "quantizer_mode: ${QUANTIZER_MODE}"
echo "quant_bits: ${QUANT_BITS}"
echo "roi_levels: n/a"
echo "bg_levels: n/a"
echo "roi_target_mode: n/a"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

TRAIN_CMD=(
    "${PYTHON_RUNNER[@]}" src/main_train.py
    --data_root "data/dataset/semantickitti/dataset/sequences"
    --backbone "$BACKBONE"
    --lr "$LR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --no_teacher
    --run_id "$RUN_ID"
    --save_dir "$SAVE_DIR"
    --quantizer_mode "$QUANTIZER_MODE"
    --quant_bits "$QUANT_BITS"
    --lambda_recon "$L_RECON"
    --lambda_rate "$L_RATE"
    --lambda_distill "$L_DISTILL"
    --lambda_importance "$L_IMPORTANCE"
)

if [[ "${MAX_TRAIN_FRAMES}" -gt 0 ]]; then
    TRAIN_CMD+=(--max_train_frames "$MAX_TRAIN_FRAMES")
fi

"${TRAIN_CMD[@]}"

echo "Done uniform baseline task ${SLURM_ARRAY_TASK_ID} (${BACKBONE}, q${QUANT_BITS})"
