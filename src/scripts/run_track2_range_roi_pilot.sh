#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  PYTHON_RUNNER=("${CONDA_PREFIX_DIR}/bin/python")
elif command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import sys" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_RUNNER=(python)
fi

RUN_DIR=${RUN_DIR:-data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4}
CHECKPOINT=${CHECKPOINT:-}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
TRACK2_TAG=${TRACK2_TAG:-$(date +%y%m%d_%H%M%S)}
HEAD_TYPE=${HEAD_TYPE:-linear}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-2048}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-512}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-1}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-42}
LOG_INTERVAL=${LOG_INTERVAL:-20}
THRESHOLD=${THRESHOLD:-0.5}
POS_WEIGHT_CAP=${POS_WEIGHT_CAP:-50}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-1024}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/results/track2_range_roi}
RUN_NAME=${RUN_NAME:-${TRACK2_TAG}_${HEAD_TYPE}}

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT_DIR}/${RUN_DIR}"
fi
if [[ "${KITTI_ROOT}" != /* ]]; then
  KITTI_ROOT="${ROOT_DIR}/${KITTI_ROOT}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}"
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Error: RUN_DIR not found: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -d "${KITTI_ROOT}" ]]; then
  echo "Error: KITTI_ROOT not found: ${KITTI_ROOT}" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

declare -a CMD=(
  "${PYTHON_RUNNER[@]}"
  src/train/train_track2_range_roi.py
  --run_dir "${RUN_DIR}"
  --kitti_root "${KITTI_ROOT}"
  --head_type "${HEAD_TYPE}"
  --hidden_channels "${HIDDEN_CHANNELS}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_val_samples "${MAX_VAL_SAMPLES}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --log_interval "${LOG_INTERVAL}"
  --threshold "${THRESHOLD}"
  --pos_weight_cap "${POS_WEIGHT_CAP}"
  --img_h "${IMG_H}"
  --img_w "${IMG_W}"
  --fov_up_deg "${FOV_UP_DEG}"
  --fov_down_deg "${FOV_DOWN_DEG}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  CMD+=(--freeze_backbone)
fi

echo "============================================================"
echo "[Track2 ROI Run]"
echo "run_dir: ${RUN_DIR}"
echo "head_type: ${HEAD_TYPE}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "max_train_samples: ${MAX_TRAIN_SAMPLES}"
echo "max_val_samples: ${MAX_VAL_SAMPLES}"
echo "freeze_backbone: ${FREEZE_BACKBONE}"
echo "output_dir: ${OUTPUT_DIR}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

"${CMD[@]}"

echo "[track2-run] done"
