#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
RANGEDET_REPO="${RANGEDET_REPO:-$REPO_ROOT/third_party/external_range_det/RangeDet}"
RANGEDET_ENV="${RANGEDET_ENV:-/home/018219422/miniconda3/envs/rangedet39}"
RANGEDET_PYTHON="${RANGEDET_PYTHON:-$RANGEDET_ENV/bin/python}"
RANGEDET_CONFIG="${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}"
RANGEDET_DATA_ROOT="${RANGEDET_DATA_ROOT:-$REPO_ROOT/data/dataset/rangedet_kitti_hq}"
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_eval}"
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX:-experiments/${RANGEDET_EXPERIMENT_NAME}/checkpoint}"
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH:-24}"
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR:-}"
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG:-}"

export PYTHONNOUSERSITE=1
export RANGEDET_DATA_ROOT
export RANGEDET_EXPERIMENT_NAME
export RANGEDET_MODEL_PREFIX
export RANGEDET_TEST_EPOCH
export MXNET_CUDNN_AUTOTUNE_DEFAULT="${MXNET_CUDNN_AUTOTUNE_DEFAULT:-0}"
export MXNET_CUDNN_LIB_CHECKING="${MXNET_CUDNN_LIB_CHECKING:-0}"

NVIDIA_SITE="$RANGEDET_ENV/lib/python3.9/site-packages/nvidia"
LD_ADD=()
if [[ -d "$NVIDIA_SITE" ]]; then
  while IFS= read -r libdir; do
    LD_ADD+=("$libdir")
  done < <(find "$NVIDIA_SITE" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
fi

LD_PATH="${LD_LIBRARY_PATH:-}"
for path in "${LD_ADD[@]}"; do
  if [[ -d "$path" ]]; then
    if [[ -n "$LD_PATH" ]]; then
      LD_PATH="${path}:${LD_PATH}"
    else
      LD_PATH="$path"
    fi
  fi
done
export LD_LIBRARY_PATH="$LD_PATH"

cd "$RANGEDET_REPO"
export PYTHONPATH="$RANGEDET_REPO${PYTHONPATH:+:$PYTHONPATH}"

"$RANGEDET_PYTHON" - <<'PY'
import os
import sys

sys.path.insert(0, os.getcwd())
import mxnet as mx
import processing_cxx

print("mxnet", mx.__version__)
print("num_gpus", mx.context.num_gpus())
print("model_prefix", os.environ["RANGEDET_MODEL_PREFIX"])
print("test_epoch", os.environ["RANGEDET_TEST_EPOCH"])
print("data_root", os.environ["RANGEDET_DATA_ROOT"])
print("processing_cxx_ok", bool(processing_cxx))
PY

"$RANGEDET_PYTHON" -u tools/test.py --config "$RANGEDET_CONFIG"

OUTPUT_PKL="${RANGEDET_MODEL_PREFIX}_output_dict_${RANGEDET_TEST_EPOCH}e.pkl"
if [[ -n "${RANGEDET_OUTPUT_ARCHIVE_DIR}" && -f "${OUTPUT_PKL}" ]]; then
  mkdir -p "${RANGEDET_OUTPUT_ARCHIVE_DIR}"
  archive_tag="${RANGEDET_OUTPUT_ARCHIVE_TAG:-$(date +%y%m%d_%H%M%S)}"
  archive_pkl="${RANGEDET_OUTPUT_ARCHIVE_DIR}/${archive_tag}_output_dict_${RANGEDET_TEST_EPOCH}e.pkl"
  cp -f "${OUTPUT_PKL}" "${archive_pkl}"
  echo "[rangedet-eval] archived_output=${archive_pkl}"
fi
