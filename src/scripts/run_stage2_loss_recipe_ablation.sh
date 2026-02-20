#!/bin/bash
# Stage2 ablation: two new loss recipes + stronger distill signal + richer importance head.
#
# Usage:
#   sbatch src/scripts/run_stage2_loss_recipe_ablation.sh
#
# Array layout:
#   0 -> balanced_v1 + basic head
#   1 -> balanced_v2 + multiscale head

#SBATCH --job-name=lidar_s2_lossrec
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1

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

LOSS_RECIPES=("balanced_v1" "balanced_v2")
RATE_MODES=("normalized_global" "normalized_bg")
IMP_MODES=("weighted_bce" "weighted_bce")
IMP_HEAD_TYPES=("basic" "multiscale")
LAMBDA_IMP_SEPS=(0.0 0.2)

IDX=${SLURM_ARRAY_TASK_ID}
LOSS_RECIPE=${LOSS_RECIPES[$IDX]}
RATE_MODE=${RATE_MODES[$IDX]}
IMP_MODE=${IMP_MODES[$IDX]}
HEAD_TYPE=${IMP_HEAD_TYPES[$IDX]}
L_IMP_SEP=${LAMBDA_IMP_SEPS[$IDX]}

BACKBONE=${BACKBONE:-darknet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}

L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.02}
L_DISTILL=${L_DISTILL:-0.1}
L_IMPORTANCE=${L_IMPORTANCE:-1.0}
IMP_MARGIN=${IMP_MARGIN:-0.05}
IMP_POS_WEIGHT_MODE=${IMP_POS_WEIGHT_MODE:-auto}
IMP_POS_WEIGHT=${IMP_POS_WEIGHT:-1.0}
IMP_POS_WEIGHT_MAX=${IMP_POS_WEIGHT_MAX:-50.0}
DISTILL_LOGIT_LOSS=${DISTILL_LOGIT_LOSS:-auto}
DISTILL_TEMP=${DISTILL_TEMP:-1.0}
DISTILL_FEATURE_WEIGHT=${DISTILL_FEATURE_WEIGHT:-1.0}
DISTILL_LOGIT_WEIGHT=${DISTILL_LOGIT_WEIGHT:-1.0}

if [[ ! -f "$TEACHER_PROXY_CKPT" ]]; then
  echo "Warning: teacher proxy checkpoint not found at $TEACHER_PROXY_CKPT"
fi

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_${LOSS_RECIPE}_head${HEAD_TYPE}_j${JOB_TAG}_r${IDX}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

RUN_ID="j${JOB_TAG}_r${IDX}"
MODE_TAG="distill_${LOSS_RECIPE}_head${HEAD_TYPE}"
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
echo "quantizer_mode: ${QUANTIZER_MODE}"
echo "quant_bits: ${QUANT_BITS}"
echo "roi_levels: ${ROI_LEVELS}"
echo "bg_levels: ${BG_LEVELS}"
echo "roi_target_mode: ${ROI_TARGET_MODE}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "loss_recipe: ${LOSS_RECIPE}"
echo "rate_loss_mode: ${RATE_MODE}"
echo "importance_loss_mode: ${IMP_MODE}"
echo "importance_pos_weight_mode: ${IMP_POS_WEIGHT_MODE}"
echo "importance_pos_weight: ${IMP_POS_WEIGHT}"
echo "importance_pos_weight_max: ${IMP_POS_WEIGHT_MAX}"
echo "lambda_imp_separation: ${L_IMP_SEP}"
echo "imp_separation_margin: ${IMP_MARGIN}"
echo "distill_logit_loss: ${DISTILL_LOGIT_LOSS}"
echo "distill_temperature: ${DISTILL_TEMP}"
echo "distill_feature_weight: ${DISTILL_FEATURE_WEIGHT}"
echo "distill_logit_weight: ${DISTILL_LOGIT_WEIGHT}"
echo "importance_head_type: ${HEAD_TYPE}"
echo "importance_hidden_channels: 32"
echo "teacher_proxy_ckpt: ${TEACHER_PROXY_CKPT}"
echo "teacher_score_topk_ratio: 0.01"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}, imp_separation=${L_IMP_SEP}"
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
    --teacher_proxy_ckpt "$TEACHER_PROXY_CKPT" \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --quantizer_mode "$QUANTIZER_MODE" \
    --quant_bits "$QUANT_BITS" \
    --roi_levels "$ROI_LEVELS" \
    --bg_levels "$BG_LEVELS" \
    --roi_target_mode "$ROI_TARGET_MODE" \
    --max_train_frames "$MAX_TRAIN_FRAMES" \
    --loss_recipe "$LOSS_RECIPE" \
    --rate_loss_mode "$RATE_MODE" \
    --importance_loss_mode "$IMP_MODE" \
    --importance_pos_weight_mode "$IMP_POS_WEIGHT_MODE" \
    --importance_pos_weight "$IMP_POS_WEIGHT" \
    --importance_pos_weight_max "$IMP_POS_WEIGHT_MAX" \
    --lambda_recon "$L_RECON" \
    --lambda_rate "$L_RATE" \
    --lambda_distill "$L_DISTILL" \
    --lambda_importance "$L_IMPORTANCE" \
    --lambda_imp_separation "$L_IMP_SEP" \
    --imp_separation_margin "$IMP_MARGIN" \
    --distill_logit_loss "$DISTILL_LOGIT_LOSS" \
    --distill_temperature "$DISTILL_TEMP" \
    --distill_feature_weight "$DISTILL_FEATURE_WEIGHT" \
    --distill_logit_weight "$DISTILL_LOGIT_WEIGHT" \
    --importance_head_type "$HEAD_TYPE" \
    --importance_hidden_channels 32

echo "Done Stage2 loss recipe ablation task ${IDX} (${LOSS_RECIPE}, ${HEAD_TYPE})"
