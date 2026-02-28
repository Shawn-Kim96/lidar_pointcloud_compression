#!/bin/bash
#SBATCH --job-name=stage2_distill_fix
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#
# Stage2 distillation alignment ablation:
# 0) legacy_mean_resize
# 1) energy_pool_16x32
# 2) energy_pool_16x32_scoregate015

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

IDX=${SLURM_ARRAY_TASK_ID:-0}

BACKBONE=${BACKBONE:-resnet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
DATASET_TYPE=${DATASET_TYPE:-kitti3dobject}
if [[ "${DATASET_TYPE}" == "kitti3dobject" ]]; then
  DATA_ROOT=${DATA_ROOT:-data/dataset/kitti3dobject}
  KITTI_SPLIT=${KITTI_SPLIT:-train}
  KITTI_IMAGESET_FILE=${KITTI_IMAGESET_FILE:-}
  KITTI_ROI_CLASSES=${KITTI_ROI_CLASSES:-Car,Pedestrian,Cyclist}
else
  DATA_ROOT=${DATA_ROOT:-data/dataset/semantickitti/dataset/sequences}
  KITTI_SPLIT=${KITTI_SPLIT:-train}
  KITTI_IMAGESET_FILE=${KITTI_IMAGESET_FILE:-}
  KITTI_ROI_CLASSES=${KITTI_ROI_CLASSES:-Car,Pedestrian,Cyclist}
fi
LR=${LR:-1e-4}

QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
LOSS_RECIPE=${LOSS_RECIPE:-balanced_v2}
RATE_LOSS_MODE=${RATE_LOSS_MODE:-normalized_bg}
IMPORTANCE_LOSS_MODE=${IMPORTANCE_LOSS_MODE:-weighted_bce}
IMPORTANCE_POS_WEIGHT_MODE=${IMPORTANCE_POS_WEIGHT_MODE:-auto}
IMPORTANCE_POS_WEIGHT=${IMPORTANCE_POS_WEIGHT:-1.0}
IMPORTANCE_POS_WEIGHT_MAX=${IMPORTANCE_POS_WEIGHT_MAX:-50.0}

L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.02}
L_DISTILL=${L_DISTILL:-0.1}
L_IMPORTANCE=${L_IMPORTANCE:-1.0}
L_IMP_SEP=${L_IMP_SEP:-0.2}
IMP_MARGIN=${IMP_MARGIN:-0.05}

DISTILL_LOGIT_LOSS=${DISTILL_LOGIT_LOSS:-auto}
DISTILL_TEMP=${DISTILL_TEMP:-1.0}
DISTILL_FEATURE_WEIGHT=${DISTILL_FEATURE_WEIGHT:-1.0}
DISTILL_LOGIT_WEIGHT=${DISTILL_LOGIT_WEIGHT:-1.0}

IMPORTANCE_HEAD_TYPE=${IMPORTANCE_HEAD_TYPE:-pp_lite}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

DISTILL_FEATURE_SOURCES=("channel_mean" "energy_map" "energy_map")
DISTILL_ALIGN_MODES=("resize" "adaptive_pool" "adaptive_pool")
DISTILL_ALIGN_HWS=("0,0" "16,32" "16,32")
DISTILL_TEACHER_SCORE_MINS=("0.0" "0.0" "0.15")
DISTILL_TEACHER_SCORE_WEIGHTS=("0" "1" "1")
CASE_TAGS=("legacy_mean_resize" "energy_pool_16x32" "energy_pool_16x32_scoregate015")

DISTILL_FEATURE_SOURCE=${DISTILL_FEATURE_SOURCES[$IDX]}
DISTILL_ALIGN_MODE=${DISTILL_ALIGN_MODES[$IDX]}
DISTILL_ALIGN_HW=${DISTILL_ALIGN_HWS[$IDX]}
DISTILL_TEACHER_SCORE_MIN=${DISTILL_TEACHER_SCORE_MINS[$IDX]}
DISTILL_TEACHER_SCORE_WEIGHT=${DISTILL_TEACHER_SCORE_WEIGHTS[$IDX]}
CASE_TAG=${CASE_TAGS[$IDX]}

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_s2fix_${CASE_TAG}_j${JOB_TAG}_r${IDX}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

RUN_ID="j${JOB_TAG}_r${IDX}"
MODE_TAG="distill_fix_${CASE_TAG}"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

echo "============================================================"
echo "[Stage2 Distill-Fix Metadata]"
echo "stage: 2"
echo "training_mode: distillation (teacher enabled)"
echo "case_tag: ${CASE_TAG}"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: ${TEACHER_BACKEND}"
echo "teacher_proxy_ckpt: ${TEACHER_PROXY_CKPT}"
echo "save_dir: ${SAVE_DIR}"
echo "dataset_root: ${DATA_ROOT}"
echo "dataset_type: ${DATASET_TYPE}"
if [[ "${DATASET_TYPE}" == "kitti3dobject" ]]; then
echo "kitti_split: ${KITTI_SPLIT}"
echo "kitti_imageset_file: ${KITTI_IMAGESET_FILE:-auto(ImageSets/<split>.txt)}"
echo "kitti_roi_classes: ${KITTI_ROI_CLASSES}"
fi
echo "loss_recipe: ${LOSS_RECIPE}"
echo "rate_loss_mode: ${RATE_LOSS_MODE}"
echo "importance_loss_mode: ${IMPORTANCE_LOSS_MODE}"
echo "distill_logit_loss: ${DISTILL_LOGIT_LOSS}"
echo "distill_feature_source: ${DISTILL_FEATURE_SOURCE}"
echo "distill_align_mode: ${DISTILL_ALIGN_MODE}"
echo "distill_align_hw: ${DISTILL_ALIGN_HW}"
echo "distill_teacher_score_min: ${DISTILL_TEACHER_SCORE_MIN}"
echo "distill_teacher_score_weight: ${DISTILL_TEACHER_SCORE_WEIGHT}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}, imp_sep=${L_IMP_SEP}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

DISTILL_SCORE_FLAG=()
if [[ "${DISTILL_TEACHER_SCORE_WEIGHT}" == "1" ]]; then
  DISTILL_SCORE_FLAG+=(--distill_teacher_score_weight)
fi

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
  --teacher_backend "$TEACHER_BACKEND"
  --teacher_proxy_ckpt "$TEACHER_PROXY_CKPT"
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
  --lambda_imp_separation "$L_IMP_SEP"
  --imp_separation_margin "$IMP_MARGIN"
  --distill_logit_loss "$DISTILL_LOGIT_LOSS"
  --distill_temperature "$DISTILL_TEMP"
  --distill_feature_weight "$DISTILL_FEATURE_WEIGHT"
  --distill_logit_weight "$DISTILL_LOGIT_WEIGHT"
  --distill_feature_source "$DISTILL_FEATURE_SOURCE"
  --distill_align_mode "$DISTILL_ALIGN_MODE"
  --distill_align_hw "$DISTILL_ALIGN_HW"
  --distill_teacher_score_min "$DISTILL_TEACHER_SCORE_MIN"
  --importance_head_type "$IMPORTANCE_HEAD_TYPE"
  --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS"
)

if [[ -n "${KITTI_IMAGESET_FILE}" ]]; then
  TRAIN_CMD+=(--kitti_imageset_file "${KITTI_IMAGESET_FILE}")
fi
TRAIN_CMD+=("${DISTILL_SCORE_FLAG[@]}")
"${TRAIN_CMD[@]}"

echo "Done Stage2 distill-fix ablation task ${IDX} (${CASE_TAG})"
