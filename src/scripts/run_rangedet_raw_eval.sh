#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
RANGEDET_DATA_ROOT="${RANGEDET_DATA_ROOT:-$REPO_ROOT/data/dataset/rangedet_kitti_hq}"
RANGEDET_CONFIG="${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}"
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_raw_eval}"
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX:-$REPO_ROOT/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260304_0100/checkpoint}"
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH:-24}"
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR:-$REPO_ROOT/logs/rangedet_eval_outputs}"
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG:-raw}"

cd "${REPO_ROOT}"
mkdir -p logs "${RANGEDET_OUTPUT_ARCHIVE_DIR}"

echo "============================================================"
echo "[RangeDet Raw Eval]"
echo "data_root: ${RANGEDET_DATA_ROOT}"
echo "model_prefix: ${RANGEDET_MODEL_PREFIX}"
echo "test_epoch: ${RANGEDET_TEST_EPOCH}"
echo "archive_dir: ${RANGEDET_OUTPUT_ARCHIVE_DIR}"
echo "archive_tag: ${RANGEDET_OUTPUT_ARCHIVE_TAG}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

REPO_ROOT="${REPO_ROOT}" \
RANGEDET_DATA_ROOT="${RANGEDET_DATA_ROOT}" \
RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME}" \
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR}" \
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG}" \
bash src/scripts/run_rangedet_kitti_eval.sh
