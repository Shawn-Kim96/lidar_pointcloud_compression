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

set -euo pipefail

# Create logs directory
mkdir -p logs

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  PYTHON_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  PYTHON_ENV_DESC="python:$(command -v python)"
fi

# Ensure each Slurm task binds to its assigned GPU, avoiding shared cuda:0 collisions.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  RAW_JOB_GPUS="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  if [[ -n "${RAW_JOB_GPUS}" ]]; then
    FIRST_GPU="${RAW_JOB_GPUS%%,*}"
    if [[ "${FIRST_GPU}" == *"["* ]]; then
      FIRST_GPU="${FIRST_GPU#*[}"
      FIRST_GPU="${FIRST_GPU%%]*}"
    fi
    FIRST_GPU="${FIRST_GPU//[^0-9]/}"
    if [[ -n "${FIRST_GPU}" ]]; then
      export CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
    fi
  fi
fi
echo "[slurm-gpu] job_gpus=${SLURM_JOB_GPUS:-unset} step_gpus=${SLURM_STEP_GPUS:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

# Define Experiment Arrays
# We sweep over Backbone + Learning Rate
BACKBONES=("darknet" "resnet" "darknet" "resnet")
LRS=(1e-4 1e-4 5e-5 5e-5)

# Shared training params
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
DATASET_TYPE=${DATASET_TYPE:-kitti3dobject}
if [[ "${DATASET_TYPE}" == "kitti3dobject" ]]; then
  DATA_ROOT=${DATA_ROOT:-data/dataset/kitti3dobject}
  KITTI_SPLIT=${KITTI_SPLIT:-train}
  KITTI_IMAGESET_FILE=${KITTI_IMAGESET_FILE:-}
  KITTI_ROI_CLASSES=${KITTI_ROI_CLASSES:-Car,Pedestrian,Cyclist}
  WAIT_FOR_KITTI_SEC=${WAIT_FOR_KITTI_SEC:-0}
  WAIT_POLL_SEC=${WAIT_POLL_SEC:-60}
else
  DATA_ROOT=${DATA_ROOT:-data/dataset/semantickitti/dataset/sequences}
  KITTI_SPLIT=${KITTI_SPLIT:-train}
  KITTI_IMAGESET_FILE=${KITTI_IMAGESET_FILE:-}
  KITTI_ROI_CLASSES=${KITTI_ROI_CLASSES:-Car,Pedestrian,Cyclist}
fi
ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.02}
L_DISTILL=${L_DISTILL:-0.0}
L_IMPORTANCE=${L_IMPORTANCE:-1.0}
L_IMP_SEPARATION=${L_IMP_SEPARATION:-0.0}
IMP_SEPARATION_MARGIN=${IMP_SEPARATION_MARGIN:-0.05}
LOSS_RECIPE=${LOSS_RECIPE:-balanced_v1}
RATE_LOSS_MODE=${RATE_LOSS_MODE:-normalized_global}
IMPORTANCE_LOSS_MODE=${IMPORTANCE_LOSS_MODE:-weighted_bce}
IMPORTANCE_POS_WEIGHT_MODE=${IMPORTANCE_POS_WEIGHT_MODE:-auto}
IMPORTANCE_POS_WEIGHT=${IMPORTANCE_POS_WEIGHT:-1.0}
IMPORTANCE_POS_WEIGHT_MAX=${IMPORTANCE_POS_WEIGHT_MAX:-50.0}
DISTILL_LOGIT_LOSS=${DISTILL_LOGIT_LOSS:-auto}
DISTILL_TEMPERATURE=${DISTILL_TEMPERATURE:-1.0}
DISTILL_FEATURE_WEIGHT=${DISTILL_FEATURE_WEIGHT:-1.0}
DISTILL_LOGIT_WEIGHT=${DISTILL_LOGIT_WEIGHT:-1.0}
IMPORTANCE_HEAD_TYPE=${IMPORTANCE_HEAD_TYPE:-pp_lite}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

# Get current configuration based on Array Task ID
BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

# Write model-centric logs alongside Slurm default logs.
DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_lr${LR}_j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

# Experiment Run ID
RUN_ID="j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
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
echo "dataset_root: ${DATA_ROOT}"
echo "dataset_type: ${DATASET_TYPE}"
if [[ "${DATASET_TYPE}" == "kitti3dobject" ]]; then
echo "kitti_split: ${KITTI_SPLIT}"
echo "kitti_imageset_file: ${KITTI_IMAGESET_FILE:-auto(ImageSets/<split>.txt)}"
echo "kitti_roi_classes: ${KITTI_ROI_CLASSES}"
fi
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "lr: ${LR}"
echo "quantizer_mode: ${QUANTIZER_MODE}"
echo "quant_bits: ${QUANT_BITS}"
echo "roi_levels: ${ROI_LEVELS}"
echo "bg_levels: ${BG_LEVELS}"
echo "roi_target_mode: ${ROI_TARGET_MODE}"
echo "loss_recipe: ${LOSS_RECIPE}"
echo "rate_loss_mode: ${RATE_LOSS_MODE}"
echo "importance_loss_mode: ${IMPORTANCE_LOSS_MODE}"
echo "importance_pos_weight_mode: ${IMPORTANCE_POS_WEIGHT_MODE}"
echo "importance_pos_weight: ${IMPORTANCE_POS_WEIGHT}"
echo "importance_pos_weight_max: ${IMPORTANCE_POS_WEIGHT_MAX}"
echo "lambda_imp_separation: ${L_IMP_SEPARATION}"
echo "imp_separation_margin: ${IMP_SEPARATION_MARGIN}"
echo "distill_logit_loss: ${DISTILL_LOGIT_LOSS}"
echo "distill_temperature: ${DISTILL_TEMPERATURE}"
echo "distill_feature_weight: ${DISTILL_FEATURE_WEIGHT}"
echo "distill_logit_weight: ${DISTILL_LOGIT_WEIGHT}"
echo "importance_head_type: ${IMPORTANCE_HEAD_TYPE}"
echo "importance_hidden_channels: ${IMPORTANCE_HIDDEN_CHANNELS}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}, imp_separation=${L_IMP_SEPARATION}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

if [[ "${DATASET_TYPE}" == "kitti3dobject" && "${WAIT_FOR_KITTI_SEC}" -gt 0 ]]; then
    KITTI_REQUIRED=("ImageSets" "training/velodyne" "training/label_2" "training/calib")
    deadline=$(( $(date +%s) + WAIT_FOR_KITTI_SEC ))
    while true; do
        ready=1
        for rel in "${KITTI_REQUIRED[@]}"; do
            if [[ ! -d "${DATA_ROOT}/${rel}" ]]; then
                ready=0
                break
            fi
        done
        if [[ "${ready}" == "1" && -f "${DATA_ROOT}/ImageSets/train.txt" ]]; then
            bin_count=$(find "${DATA_ROOT}/training/velodyne" -maxdepth 1 -name '*.bin' | wc -l | tr -d '[:space:]')
            if [[ "${bin_count}" -ge 1000 ]]; then
                break
            fi
        fi
        now=$(date +%s)
        if [[ "${now}" -ge "${deadline}" ]]; then
            echo "Error: KITTI dataset not ready within WAIT_FOR_KITTI_SEC=${WAIT_FOR_KITTI_SEC}" >&2
            exit 1
        fi
        echo "[stage1] waiting for KITTI extraction ... (bin_count=${bin_count:-0})"
        sleep "${WAIT_POLL_SEC}"
    done
fi

# Run Training
# --no_teacher ensures this is Stage 1 (Backbone Only)
# --roi_levels / --bg_levels set the quantization granularity
TRAIN_CMD=(
    "${PYTHON_RUNNER[@]}" src/main_train.py
    --data_root "${DATA_ROOT}"
    --dataset_type "${DATASET_TYPE}"
    --kitti_split "${KITTI_SPLIT}"
    --kitti_roi_classes "${KITTI_ROI_CLASSES}"
    --backbone "$BACKBONE"
    --lr "$LR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --max_train_frames "$MAX_TRAIN_FRAMES"
    --no_teacher
    --run_id "$RUN_ID"
    --save_dir "$SAVE_DIR"
    --quantizer_mode "$QUANTIZER_MODE"
    --quant_bits "$QUANT_BITS"
    --roi_levels "$ROI_LEVELS"
    --bg_levels "$BG_LEVELS"
    --roi_target_mode "$ROI_TARGET_MODE"
    --loss_recipe "$LOSS_RECIPE"
    --rate_loss_mode "$RATE_LOSS_MODE"
    --importance_loss_mode "$IMPORTANCE_LOSS_MODE"
    --importance_pos_weight_mode "$IMPORTANCE_POS_WEIGHT_MODE"
    --importance_pos_weight "$IMPORTANCE_POS_WEIGHT"
    --importance_pos_weight_max "$IMPORTANCE_POS_WEIGHT_MAX"
    --lambda_recon "$L_RECON"
    --lambda_rate "$L_RATE"
    --lambda_distill "$L_DISTILL"
    --lambda_importance "$L_IMPORTANCE"
    --lambda_imp_separation "$L_IMP_SEPARATION"
    --imp_separation_margin "$IMP_SEPARATION_MARGIN"
    --distill_logit_loss "$DISTILL_LOGIT_LOSS"
    --distill_temperature "$DISTILL_TEMPERATURE"
    --distill_feature_weight "$DISTILL_FEATURE_WEIGHT"
    --distill_logit_weight "$DISTILL_LOGIT_WEIGHT"
    --importance_head_type "$IMPORTANCE_HEAD_TYPE"
    --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS"
)
if [[ -n "${KITTI_IMAGESET_FILE}" ]]; then
    TRAIN_CMD+=(--kitti_imageset_file "${KITTI_IMAGESET_FILE}")
fi
"${TRAIN_CMD[@]}"

echo "Done Experiment $SLURM_ARRAY_TASK_ID"
