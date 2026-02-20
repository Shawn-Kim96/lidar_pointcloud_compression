#!/bin/bash
# Local pilot:
# - Stage0: uniform baseline sweep (q6, q8)
# - Stage1: adaptive ROI (no teacher)
# - Stage2: adaptive ROI + distillation
# Then evaluate each run with native/oracle ROI analysis.

set -euo pipefail

mkdir -p logs notebooks

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
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-1e-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-128}
VAL_SEQ=${VAL_SEQ:-08}
VAL_MAX_FRAMES=${VAL_MAX_FRAMES:-64}

ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
IMPORTANCE_HEAD_TYPE=${IMPORTANCE_HEAD_TYPE:-pp_lite}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}

echo "[Pilot] python_env=${PYTHON_ENV_DESC}"
echo "[Pilot] backbone=${BACKBONE}, epochs=${EPOCHS}, max_train_frames=${MAX_TRAIN_FRAMES}"

run_train() {
  local run_id="$1"
  local save_dir="$2"
  local log_path="$3"
  shift 3
  "${PYTHON_RUNNER[@]}" src/main_train.py "$@" \
    --run_id "${run_id}" \
    --save_dir "${save_dir}" \
    > "${log_path}" 2>&1
}

run_eval() {
  local run_dir="$1"
  local tag="$2"
  "${PYTHON_RUNNER[@]}" src/train/evaluate_oracle_roi.py \
    --run_dir "${run_dir}" \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --val_seq "${VAL_SEQ}" \
    --max_frames "${VAL_MAX_FRAMES}" \
    --batch_size 1 \
    --num_workers 0 \
    --output_summary_csv "notebooks/oracle_eval_summary_${tag}.csv" \
    --output_detail_csv "notebooks/oracle_eval_detail_${tag}.csv"
}

# Stage0 q6
RUN0A_ID="pilot_s0_q6"
RUN0A_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_s0q6_lr${LR}_bs${BATCH_SIZE}_${RUN0A_ID}"
RUN0A_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_s0_q6.out"
run_train "${RUN0A_ID}" "${RUN0A_DIR}" "${RUN0A_LOG}" \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "${BACKBONE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_frames "${MAX_TRAIN_FRAMES}" \
  --no_teacher \
  --quantizer_mode "uniform" \
  --quant_bits 6 \
  --lambda_recon 1.0 \
  --lambda_rate 0.0 \
  --lambda_distill 0.0 \
  --lambda_importance 0.0
run_eval "${RUN0A_DIR}" "${DATE_TAG}_${BACKBONE}_pilot_s0q6"

# Stage0 q8
RUN0B_ID="pilot_s0_q8"
RUN0B_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_s0q8_lr${LR}_bs${BATCH_SIZE}_${RUN0B_ID}"
RUN0B_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_s0_q8.out"
run_train "${RUN0B_ID}" "${RUN0B_DIR}" "${RUN0B_LOG}" \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "${BACKBONE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_frames "${MAX_TRAIN_FRAMES}" \
  --no_teacher \
  --quantizer_mode "uniform" \
  --quant_bits 8 \
  --lambda_recon 1.0 \
  --lambda_rate 0.0 \
  --lambda_distill 0.0 \
  --lambda_importance 0.0
run_eval "${RUN0B_DIR}" "${DATE_TAG}_${BACKBONE}_pilot_s0q8"

# Stage1 adaptive
RUN1_ID="pilot_s1_adapt"
RUN1_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_s1adapt_lr${LR}_bs${BATCH_SIZE}_${RUN1_ID}"
RUN1_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_s1_adapt.out"
run_train "${RUN1_ID}" "${RUN1_DIR}" "${RUN1_LOG}" \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "${BACKBONE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_frames "${MAX_TRAIN_FRAMES}" \
  --no_teacher \
  --quantizer_mode "adaptive" \
  --quant_bits 8 \
  --roi_levels "${ROI_LEVELS}" \
  --bg_levels "${BG_LEVELS}" \
  --roi_target_mode "${ROI_TARGET_MODE}" \
  --loss_recipe "balanced_v2" \
  --rate_loss_mode "normalized_bg" \
  --importance_loss_mode "weighted_bce" \
  --importance_pos_weight_mode "auto" \
  --importance_pos_weight 1.0 \
  --importance_pos_weight_max 50.0 \
  --lambda_recon 1.0 \
  --lambda_rate 0.02 \
  --lambda_distill 0.0 \
  --lambda_importance 1.0 \
  --lambda_imp_separation 0.2 \
  --imp_separation_margin 0.05 \
  --distill_logit_loss "auto" \
  --distill_temperature 1.0 \
  --distill_feature_weight 1.0 \
  --distill_logit_weight 1.0 \
  --importance_head_type "${IMPORTANCE_HEAD_TYPE}" \
  --importance_hidden_channels "${IMPORTANCE_HIDDEN_CHANNELS}"
run_eval "${RUN1_DIR}" "${DATE_TAG}_${BACKBONE}_pilot_s1adapt"

# Stage2 adaptive + distill
if [[ ! -f "${TEACHER_PROXY_CKPT}" ]]; then
  echo "Warning: teacher checkpoint not found at ${TEACHER_PROXY_CKPT}"
fi

RUN2_ID="pilot_s2_distill"
RUN2_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_s2distill_lr${LR}_bs${BATCH_SIZE}_${RUN2_ID}"
RUN2_LOG="logs/${DATE_TAG}_${BACKBONE}_pilot_s2_distill.out"
run_train "${RUN2_ID}" "${RUN2_DIR}" "${RUN2_LOG}" \
  --data_root "data/dataset/semantickitti/dataset/sequences" \
  --backbone "${BACKBONE}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_train_frames "${MAX_TRAIN_FRAMES}" \
  --teacher_backend "${TEACHER_BACKEND}" \
  --teacher_proxy_ckpt "${TEACHER_PROXY_CKPT}" \
  --quantizer_mode "adaptive" \
  --quant_bits 8 \
  --roi_levels "${ROI_LEVELS}" \
  --bg_levels "${BG_LEVELS}" \
  --roi_target_mode "${ROI_TARGET_MODE}" \
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
  --importance_head_type "${IMPORTANCE_HEAD_TYPE}" \
  --importance_hidden_channels "${IMPORTANCE_HIDDEN_CHANNELS}"
run_eval "${RUN2_DIR}" "${DATE_TAG}_${BACKBONE}_pilot_s2distill"

"${PYTHON_RUNNER[@]}" src/utils/update_experiments_result.py

echo "Pilot completed."
echo "Logs:"
echo "  - ${RUN0A_LOG}"
echo "  - ${RUN0B_LOG}"
echo "  - ${RUN1_LOG}"
echo "  - ${RUN2_LOG}"
echo "Oracle summaries:"
echo "  - notebooks/oracle_eval_summary_${DATE_TAG}_${BACKBONE}_pilot_s0q6.csv"
echo "  - notebooks/oracle_eval_summary_${DATE_TAG}_${BACKBONE}_pilot_s0q8.csv"
echo "  - notebooks/oracle_eval_summary_${DATE_TAG}_${BACKBONE}_pilot_s1adapt.csv"
echo "  - notebooks/oracle_eval_summary_${DATE_TAG}_${BACKBONE}_pilot_s2distill.csv"
