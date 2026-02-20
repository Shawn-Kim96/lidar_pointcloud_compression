#!/bin/bash
# Quick local pilot for the updated objective + teacher + high-capacity importance head.
# Runs one Stage1 and one Stage2 experiment with matched backbone/settings.

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
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-1e-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-128}

ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
IMPORTANCE_HEAD_TYPE=${IMPORTANCE_HEAD_TYPE:-pp_lite}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}

if [[ ! -f "${TEACHER_PROXY_CKPT}" ]]; then
  echo "Warning: teacher checkpoint not found at ${TEACHER_PROXY_CKPT}"
fi

echo "[Pilot] python_env=${PYTHON_ENV_DESC}"

RUN1_ID=localpp_stage1
RUN1_SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_pp20_stage1_lr${LR}_bs${BATCH_SIZE}_${RUN1_ID}"
RUN1_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_pp20_stage1.out"

"${PYTHON_RUNNER[@]}" src/main_train.py \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "$BACKBONE" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --no_teacher \
  --run_id "$RUN1_ID" \
  --save_dir "$RUN1_SAVE_DIR" \
  --quantizer_mode "$QUANTIZER_MODE" \
  --quant_bits "$QUANT_BITS" \
  --roi_levels "$ROI_LEVELS" \
  --bg_levels "$BG_LEVELS" \
  --roi_target_mode "$ROI_TARGET_MODE" \
  --max_train_frames "$MAX_TRAIN_FRAMES" \
  --loss_recipe "balanced_v1" \
  --rate_loss_mode "normalized_global" \
  --importance_loss_mode "weighted_bce" \
  --importance_pos_weight_mode "auto" \
  --importance_pos_weight 1.0 \
  --importance_pos_weight_max 50.0 \
  --lambda_recon 1.0 \
  --lambda_rate 0.02 \
  --lambda_distill 0.0 \
  --lambda_importance 1.0 \
  --lambda_imp_separation 0.0 \
  --imp_separation_margin 0.05 \
  --distill_logit_loss "auto" \
  --distill_temperature 1.0 \
  --distill_feature_weight 1.0 \
  --distill_logit_weight 1.0 \
  --importance_head_type "$IMPORTANCE_HEAD_TYPE" \
  --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS" \
  > "$RUN1_LOG" 2>&1

RUN2_ID=localpp_stage2
RUN2_SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_pp20_stage2_lr${LR}_bs${BATCH_SIZE}_${RUN2_ID}"
RUN2_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_pp20_stage2.out"

"${PYTHON_RUNNER[@]}" src/main_train.py \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "$BACKBONE" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --teacher_backend "$TEACHER_BACKEND" \
  --teacher_proxy_ckpt "$TEACHER_PROXY_CKPT" \
  --run_id "$RUN2_ID" \
  --save_dir "$RUN2_SAVE_DIR" \
  --quantizer_mode "$QUANTIZER_MODE" \
  --quant_bits "$QUANT_BITS" \
  --roi_levels "$ROI_LEVELS" \
  --bg_levels "$BG_LEVELS" \
  --roi_target_mode "$ROI_TARGET_MODE" \
  --max_train_frames "$MAX_TRAIN_FRAMES" \
  --loss_recipe "balanced_v2" \
  --rate_loss_mode "normalized_bg" \
  --importance_loss_mode "weighted_bce" \
  --importance_pos_weight_mode "auto" \
  --importance_pos_weight 1.0 \
  --importance_pos_weight_max 50.0 \
  --lambda_recon 1.0 \
  --lambda_rate 0.02 \
  --lambda_distill 0.1 \
  --lambda_importance 1.0 \
  --lambda_imp_separation 0.2 \
  --imp_separation_margin 0.05 \
  --distill_logit_loss "auto" \
  --distill_temperature 1.0 \
  --distill_feature_weight 1.0 \
  --distill_logit_weight 1.0 \
  --importance_head_type "$IMPORTANCE_HEAD_TYPE" \
  --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS" \
  > "$RUN2_LOG" 2>&1

echo "Pilot completed:"
echo "  - $RUN1_LOG"
echo "  - $RUN2_LOG"
