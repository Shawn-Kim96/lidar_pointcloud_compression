#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
RANGEDET_REPO="${RANGEDET_REPO:-$REPO_ROOT/third_party/external_range_det/RangeDet}"
RANGEDET_ENV="${RANGEDET_ENV:-/home/018219422/miniconda3/envs/rangedet39}"
RANGEDET_PYTHON="${RANGEDET_PYTHON:-$RANGEDET_ENV/bin/python}"
RANGEDET_CONFIG="${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}"
RANGEDET_DATA_ROOT="${RANGEDET_DATA_ROOT:-$REPO_ROOT/data/dataset/rangedet_kitti_hq}"
RANGEDET_NUM_EPOCHS="${RANGEDET_NUM_EPOCHS:-1}"
RANGEDET_SAMPLING_RATE="${RANGEDET_SAMPLING_RATE:-1}"
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_smoke_1e}"
RUN_TEST="${RUN_TEST:-0}"

export PYTHONNOUSERSITE=1
export RANGEDET_DATA_ROOT
export RANGEDET_NUM_EPOCHS
export RANGEDET_SAMPLING_RATE
export RANGEDET_EXPERIMENT_NAME
export RANGEDET_MODEL_PREFIX="experiments/${RANGEDET_EXPERIMENT_NAME}/checkpoint"
export RANGEDET_TEST_EPOCH="${RANGEDET_NUM_EPOCHS}"
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
mkdir -p "experiments/${RANGEDET_EXPERIMENT_NAME}"
export PYTHONPATH="$RANGEDET_REPO${PYTHONPATH:+:$PYTHONPATH}"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$RANGEDET_PYTHON"
  echo "config=$RANGEDET_CONFIG"
  echo "data_root=$RANGEDET_DATA_ROOT"
  echo "epochs=$RANGEDET_NUM_EPOCHS"
  echo "sampling_rate=$RANGEDET_SAMPLING_RATE"
  echo "run_test=$RUN_TEST"
} > "experiments/${RANGEDET_EXPERIMENT_NAME}/launch_env.txt"

nvidia-smi -L || true

"$RANGEDET_PYTHON" - <<'PY'
import os
import sys

print("cwd", os.getcwd())
print("python", sys.executable)
print("ld_library_path", os.environ.get("LD_LIBRARY_PATH", ""))
sys.path.insert(0, os.getcwd())

import mxnet as mx
import processing_cxx

print("mxnet", mx.__version__)
print("num_gpus", mx.context.num_gpus())
print("processing_cxx_ok", bool(processing_cxx))
PY

"$RANGEDET_PYTHON" -u tools/train.py --config "$RANGEDET_CONFIG"

if [[ "$RUN_TEST" == "1" ]]; then
  "$RANGEDET_PYTHON" -u tools/test.py --config "$RANGEDET_CONFIG"
fi
