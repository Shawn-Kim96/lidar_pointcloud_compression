#!/bin/bash
# Parallel Stage2 sweep over ROI target downsampling mode.
# Usage:
#   sbatch src/scripts/run_stage2_roi_target_sweep.sh
#
# Array layout:
#   0 -> nearest
#   1 -> maxpool
#   2 -> area

#SBATCH --job-name=lidar_rt_sweep
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2

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

ROI_TARGET_MODES=("nearest" "maxpool" "area")
ROI_TARGET_MODE=${ROI_TARGET_MODES[$SLURM_ARRAY_TASK_ID]}

# Fixed config for controlled ablation against existing Stage2 setup.
BACKBONE="darknet"
TEACHER_BACKEND="proxy"
EPOCHS=50
BATCH_SIZE=4
NUM_WORKERS=4
LR=1e-4
ROI_LEVELS=256
BG_LEVELS=16
QUANTIZER_MODE="adaptive"
QUANT_BITS=8
L_RECON=1.0
L_RATE=0.1
L_DISTILL=0.1
L_IMPORTANCE=0.5

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_ld${L_DISTILL}_rt${ROI_TARGET_MODE}_j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

RUN_ID="j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
MODE_TAG="distill_rtsweep"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_rt${ROI_TARGET_MODE}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

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
echo "quantizer_mode: ${QUANTIZER_MODE}"
echo "quant_bits: ${QUANT_BITS}"
echo "roi_levels: ${ROI_LEVELS}"
echo "bg_levels: ${BG_LEVELS}"
echo "roi_target_mode: ${ROI_TARGET_MODE}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

"${PYTHON_RUNNER[@]}" src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "$BACKBONE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --teacher_backend "$TEACHER_BACKEND" \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --quantizer_mode "$QUANTIZER_MODE" \
    --quant_bits "$QUANT_BITS" \
    --roi_levels "$ROI_LEVELS" \
    --bg_levels "$BG_LEVELS" \
    --roi_target_mode "$ROI_TARGET_MODE" \
    --lambda_recon "$L_RECON" \
    --lambda_rate "$L_RATE" \
    --lambda_distill "$L_DISTILL" \
    --lambda_importance "$L_IMPORTANCE"

echo "Done ROI target mode sweep task ${SLURM_ARRAY_TASK_ID} (${ROI_TARGET_MODE})"
