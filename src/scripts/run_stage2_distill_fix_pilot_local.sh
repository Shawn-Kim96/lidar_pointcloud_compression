#!/bin/bash
# Local quick pilot for Stage2 distill-fix ablation.
# Runs 3 short cases:
# - legacy_mean_resize
# - energy_pool_16x32
# - energy_pool_16x32_scoregate015

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

DATE_TAG=$(date +%y%m%d)
BACKBONE=${BACKBONE:-resnet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-128}
LR=${LR:-1e-4}

CASE_TAGS=("legacy_mean_resize" "energy_pool_16x32" "energy_pool_16x32_scoregate015")
DISTILL_FEATURE_SOURCES=("channel_mean" "energy_map" "energy_map")
DISTILL_ALIGN_MODES=("resize" "adaptive_pool" "adaptive_pool")
DISTILL_ALIGN_HWS=("0,0" "16,32" "16,32")
DISTILL_TEACHER_SCORE_MINS=("0.0" "0.0" "0.15")
DISTILL_TEACHER_SCORE_WEIGHTS=("0" "1" "1")

for i in 0 1 2; do
  CASE_TAG=${CASE_TAGS[$i]}
  DISTILL_FEATURE_SOURCE=${DISTILL_FEATURE_SOURCES[$i]}
  DISTILL_ALIGN_MODE=${DISTILL_ALIGN_MODES[$i]}
  DISTILL_ALIGN_HW=${DISTILL_ALIGN_HWS[$i]}
  DISTILL_TEACHER_SCORE_MIN=${DISTILL_TEACHER_SCORE_MINS[$i]}
  DISTILL_TEACHER_SCORE_WEIGHT=${DISTILL_TEACHER_SCORE_WEIGHTS[$i]}

  RUN_ID="local_s2fix_r${i}"
  SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_${CASE_TAG}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"
  LOG_PREFIX="${DATE_TAG}_${BACKBONE}_pilot_${CASE_TAG}_r${i}"

  DISTILL_SCORE_FLAG=()
  if [[ "${DISTILL_TEACHER_SCORE_WEIGHT}" == "1" ]]; then
    DISTILL_SCORE_FLAG+=(--distill_teacher_score_weight)
  fi

  echo "============================================================"
  echo "[Stage2 Distill-Fix Pilot]"
  echo "case: ${CASE_TAG}"
  echo "save_dir: ${SAVE_DIR}"
  echo "python_env: ${PYTHON_ENV_DESC}"
  echo "============================================================"

  "${PYTHON_RUNNER[@]}" src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "$BACKBONE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_train_frames "$MAX_TRAIN_FRAMES" \
    --teacher_backend "$TEACHER_BACKEND" \
    --teacher_proxy_ckpt "$TEACHER_PROXY_CKPT" \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --quantizer_mode adaptive \
    --quant_bits 8 \
    --roi_levels 256 \
    --bg_levels 16 \
    --roi_target_mode maxpool \
    --loss_recipe balanced_v2 \
    --rate_loss_mode normalized_bg \
    --importance_loss_mode weighted_bce \
    --importance_pos_weight_mode auto \
    --importance_pos_weight 1.0 \
    --importance_pos_weight_max 50.0 \
    --lambda_recon 1.0 \
    --lambda_rate 0.02 \
    --lambda_distill 0.1 \
    --lambda_importance 1.0 \
    --lambda_imp_separation 0.2 \
    --imp_separation_margin 0.05 \
    --distill_logit_loss auto \
    --distill_temperature 1.0 \
    --distill_feature_weight 1.0 \
    --distill_logit_weight 1.0 \
    --distill_feature_source "$DISTILL_FEATURE_SOURCE" \
    --distill_align_mode "$DISTILL_ALIGN_MODE" \
    --distill_align_hw "$DISTILL_ALIGN_HW" \
    --distill_teacher_score_min "$DISTILL_TEACHER_SCORE_MIN" \
    "${DISTILL_SCORE_FLAG[@]}" \
    --importance_head_type pp_lite \
    --importance_hidden_channels 64 \
    > "logs/${LOG_PREFIX}.out" 2> "logs/${LOG_PREFIX}.err"
done

echo "Stage2 distill-fix pilot runs completed."
