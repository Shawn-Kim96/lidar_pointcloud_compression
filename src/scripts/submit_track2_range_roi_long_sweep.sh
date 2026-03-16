#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

BASE_TAG=${BASE_TAG:-$(date +%y%m%d)_track2long}
CONDA_ENV=${CONDA_ENV:-lidarcomp311}
RUN_DIR=${RUN_DIR:-data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
SBATCH_PARTITION=${SBATCH_PARTITION:-gpuqm}
SBATCH_CPUS=${SBATCH_CPUS:-4}
SBATCH_GRES=${SBATCH_GRES:-gpu:1}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-1}
DRY_RUN=${DRY_RUN:-0}

submit_cfg() {
  local tag="$1"
  local head="$2"
  local hidden="$3"
  local epochs="$4"
  local lr="$5"
  local wd="$6"

  TRACK2_TAG="${tag}" \
  HEAD_SWEEP="${head}" \
  HIDDEN_CHANNELS="${hidden}" \
  EPOCHS="${epochs}" \
  LR="${lr}" \
  WEIGHT_DECAY="${wd}" \
  MAX_TRAIN_SAMPLES=0 \
  MAX_VAL_SAMPLES=0 \
  BATCH_SIZE=2 \
  WORKERS=4 \
  FREEZE_BACKBONE=1 \
  CONDA_ENV="${CONDA_ENV}" \
  RUN_DIR="${RUN_DIR}" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SBATCH_PARTITION="${SBATCH_PARTITION}" \
  SBATCH_CPUS="${SBATCH_CPUS}" \
  SBATCH_GRES="${SBATCH_GRES}" \
  SBATCH_TIME="${SBATCH_TIME}" \
  SKIP_DOWNLOAD="${SKIP_DOWNLOAD}" \
  DRY_RUN="${DRY_RUN}" \
  bash "${ROOT_DIR}/src/scripts/submit_track2_range_roi_pilot.sh"
}

echo "============================================================"
echo "[Track2 Long Sweep]"
echo "base_tag: ${BASE_TAG}"
echo "run_dir: ${RUN_DIR}"
echo "policy: full train/val split, frozen compression backbone"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

submit_cfg "${BASE_TAG}_refine_h128_e120" refine 128 120 3e-4 1e-4
submit_cfg "${BASE_TAG}_refine_h192_e160" refine 192 160 2e-4 1e-4
submit_cfg "${BASE_TAG}_deep_h128_e120" deep 128 120 3e-4 1e-4
submit_cfg "${BASE_TAG}_deep_h192_e180" deep 192 180 2e-4 1e-4

echo "[track2-long-sweep] queued"
