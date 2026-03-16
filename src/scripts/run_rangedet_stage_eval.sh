#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
CONDA_ENV="${CONDA_ENV:-lidarcomp311}"
CONDA_PYTHON="${CONDA_PYTHON:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"

COMPRESSION_RUN_DIR="${COMPRESSION_RUN_DIR:-}"
RECON_DATA_ROOT="${RECON_DATA_ROOT:-}"
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX:-$REPO_ROOT/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260304_0100/checkpoint}"
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH:-24}"
RANGEDET_CONFIG="${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}"
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_stage_eval}"
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR:-$REPO_ROOT/logs/rangedet_eval_outputs}"
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG:-$(basename "${RECON_DATA_ROOT}")}"

KITTI_ROOT="${KITTI_ROOT:-$REPO_ROOT/data/dataset/kitti3dobject}"
SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT:-$REPO_ROOT/data/dataset/rangedet_kitti_hq}"
SPLIT="${SPLIT:-validation}"
LIDAR_SUBDIR="${LIDAR_SUBDIR:-training}"
IMG_H="${IMG_H:-64}"
IMG_W="${IMG_W:-2048}"
FOV_UP_DEG="${FOV_UP_DEG:-3.0}"
FOV_DOWN_DEG="${FOV_DOWN_DEG:--25.0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${COMPRESSION_RUN_DIR}" ]]; then
  echo "Error: COMPRESSION_RUN_DIR is required." >&2
  exit 1
fi
if [[ -z "${RECON_DATA_ROOT}" ]]; then
  echo "Error: RECON_DATA_ROOT is required." >&2
  exit 1
fi
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  echo "Error: python not found: ${CONDA_PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

echo "============================================================"
echo "[RangeDet Stage Eval]"
echo "compression_run_dir: ${COMPRESSION_RUN_DIR}"
echo "recon_data_root: ${RECON_DATA_ROOT}"
echo "rangedet_model_prefix: ${RANGEDET_MODEL_PREFIX}"
echo "rangedet_test_epoch: ${RANGEDET_TEST_EPOCH}"
echo "rangedet_output_archive_dir: ${RANGEDET_OUTPUT_ARCHIVE_DIR}"
echo "rangedet_output_archive_tag: ${RANGEDET_OUTPUT_ARCHIVE_TAG}"
echo "split: ${SPLIT}"
echo "img_hxw: ${IMG_H}x${IMG_W}"
echo "export_device: ${EXPORT_DEVICE}"
echo "max_frames: ${MAX_FRAMES}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

EXPORT_ARGS=(
  src/scripts/export_rangedet_recon_kitti_split.py
  --run_dir "${COMPRESSION_RUN_DIR}"
  --kitti_root "${KITTI_ROOT}"
  --source_rangedet_root "${SOURCE_RANGEDET_ROOT}"
  --output_root "${RECON_DATA_ROOT}"
  --split "${SPLIT}"
  --lidar_subdir "${LIDAR_SUBDIR}"
  --img_h "${IMG_H}"
  --img_w "${IMG_W}"
  --fov_up_deg "${FOV_UP_DEG}"
  --fov_down_deg "${FOV_DOWN_DEG}"
  --device "${EXPORT_DEVICE}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  EXPORT_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${MAX_FRAMES}" != "0" ]]; then
  EXPORT_ARGS+=(--max_frames "${MAX_FRAMES}")
fi

"${CONDA_PYTHON}" "${EXPORT_ARGS[@]}"

REPO_ROOT="${REPO_ROOT}" \
RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
RANGEDET_DATA_ROOT="${RECON_DATA_ROOT}" \
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME}" \
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR}" \
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG}" \
bash src/scripts/run_rangedet_kitti_eval.sh

echo "[rangedet-stage-eval] done"
