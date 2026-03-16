#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

BASE_TAG=${BASE_TAG:-$(date +%y%m%d_%H%M%S)}
CONDA_ENV=${CONDA_ENV:-lidarcomp311}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}
TRAIN_EPOCHS_NOQUANT=${TRAIN_EPOCHS_NOQUANT:-180}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
PP_FT_EPOCHS=${PP_FT_EPOCHS:-20}
PP_FT_BATCH_SIZE=${PP_FT_BATCH_SIZE:-4}
PP_FT_WORKERS=${PP_FT_WORKERS:-4}
SUBMIT_ENHANCED=${SUBMIT_ENHANCED:-1}

choose_partition() {
  if command -v sinfo >/dev/null 2>&1; then
    local free_gpuql
    free_gpuql=$(sinfo -h -p gpuql -t idle,mix -o "%D" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [[ "${free_gpuql:-0}" -gt 0 ]]; then
      echo "gpuql"
      return 0
    fi
  fi
  echo "gpuqm"
}

PARTITION=${SBATCH_PARTITION:-$(choose_partition)}
if [[ "${PARTITION}" == "gpuql" ]]; then
  TIME_LIMIT=${SBATCH_TIME:-72:00:00}
else
  TIME_LIMIT=${SBATCH_TIME:-48:00:00}
fi

submit_one() {
  local label="$1"
  local variant="$2"
  local out="logs/$(date +%y%m%d)_${label}_%A.out"
  local err="logs/$(date +%y%m%d)_${label}_%A.err"
  sbatch --parsable \
    --job-name="${label}" \
    --output="${out}" \
    --error="${err}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=6 \
    --gres=gpu:1 \
    --time="${TIME_LIMIT}" \
    --exclude="g16,g18,g19" \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",TRACK_TAG="${BASE_TAG}",TRACK_LABEL="${label}",TRACK_VARIANT="${variant}",KITTI_ROOT="${KITTI_ROOT}",IMG_H="${IMG_H}",IMG_W="${IMG_W}",FOV_UP_DEG="${FOV_UP_DEG}",FOV_DOWN_DEG="${FOV_DOWN_DEG}",UNPROJECTION_MODE="${UNPROJECTION_MODE}",TRAIN_EPOCHS_NOQUANT="${TRAIN_EPOCHS_NOQUANT}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",TRAIN_WORKERS="${TRAIN_WORKERS}",PP_FT_EPOCHS="${PP_FT_EPOCHS}",PP_FT_BATCH_SIZE="${PP_FT_BATCH_SIZE}",PP_FT_WORKERS="${PP_FT_WORKERS}" \
    src/scripts/run_track1_noquant_chain.sh
}

echo "============================================================"
echo "[Submit Track1 NoQuant Chains]"
echo "base_tag: ${BASE_TAG}"
echo "partition: ${PARTITION}"
echo "time_limit: ${TIME_LIMIT}"
echo "projection: ${IMG_H}x${IMG_W} unprojection=${UNPROJECTION_MODE}"
echo "train_epochs_noquant: ${TRAIN_EPOCHS_NOQUANT}"
echo "============================================================"

JID_A=$(submit_one "track1nq_a" "baseline")
JID_B=""
if [[ "${SUBMIT_ENHANCED}" == "1" ]]; then
  JID_B=$(submit_one "track1nq_b" "enhanced")
fi

MANIFEST="logs/${BASE_TAG}_track1_noquant_manifest.csv"
if [[ -n "${JID_B}" ]]; then
  cat > "${MANIFEST}" <<CSV
submitted_at,job_id,track_label,track_variant,partition,time_limit,img_h,img_w,unprojection_mode,epochs_noquant
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_A},track1nq_a,baseline,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT}
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_B},track1nq_b,enhanced,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT}
CSV
else
  cat > "${MANIFEST}" <<CSV
submitted_at,job_id,track_label,track_variant,partition,time_limit,img_h,img_w,unprojection_mode,epochs_noquant
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_A},track1nq_a,baseline,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT}
CSV
fi

echo "track1nq_a job_id=${JID_A}"
if [[ -n "${JID_B}" ]]; then
  echo "track1nq_b job_id=${JID_B}"
fi
echo "manifest=${MANIFEST}"
