#!/bin/bash
# Local pilot run (CPU/GPU auto) for quick verification of new loss recipes.
# This is intentionally short and uses subset training.

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
JOB_TAG="localpilot"

LOSS_RECIPES=("balanced_v1" "balanced_v2")
RATE_MODES=("normalized_global" "normalized_bg")
HEAD_TYPES=("basic" "multiscale")
LAMBDA_IMP_SEPS=(0.0 0.2)

BACKBONE=${BACKBONE:-resnet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-1e-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-128}

for i in 0 1; do
  LOSS_RECIPE=${LOSS_RECIPES[$i]}
  RATE_MODE=${RATE_MODES[$i]}
  HEAD_TYPE=${HEAD_TYPES[$i]}
  L_IMP_SEP=${LAMBDA_IMP_SEPS[$i]}

  RUN_ID="${JOB_TAG}_r${i}"
  SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_pilot_${LOSS_RECIPE}_head${HEAD_TYPE}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"
  LOG_PREFIX="${DATE_TAG}_${BACKBONE}_pilot_${LOSS_RECIPE}_head${HEAD_TYPE}_r${i}"

  echo "============================================================"
  echo "[Pilot Metadata]"
  echo "run: ${RUN_ID}"
  echo "python_env: ${PYTHON_ENV_DESC}"
  echo "save_dir: ${SAVE_DIR}"
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
    --quantizer_mode "adaptive" \
    --quant_bits 8 \
    --roi_levels 256 \
    --bg_levels 16 \
    --roi_target_mode "maxpool" \
    --max_train_frames "$MAX_TRAIN_FRAMES" \
    --loss_recipe "$LOSS_RECIPE" \
    --rate_loss_mode "$RATE_MODE" \
    --importance_loss_mode "weighted_bce" \
    --importance_pos_weight_mode "auto" \
    --importance_pos_weight 1.0 \
    --importance_pos_weight_max 50.0 \
    --lambda_recon 1.0 \
    --lambda_rate 0.02 \
    --lambda_distill 0.1 \
    --lambda_importance 1.0 \
    --lambda_imp_separation "$L_IMP_SEP" \
    --imp_separation_margin 0.05 \
    --distill_logit_loss "auto" \
    --distill_temperature 1.0 \
    --distill_feature_weight 1.0 \
    --distill_logit_weight 1.0 \
    --importance_head_type "$HEAD_TYPE" \
    --importance_hidden_channels 32 \
    > "logs/${LOG_PREFIX}.out" 2> "logs/${LOG_PREFIX}.err"

done

echo "Pilot runs completed."
