#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

COMPRESSION_RUN_DIR_STAGE0=${COMPRESSION_RUN_DIR_STAGE0:-}
COMPRESSION_RUN_DIR_STAGE1=${COMPRESSION_RUN_DIR_STAGE1:-}
if [[ -z "${COMPRESSION_RUN_DIR_STAGE0}" || -z "${COMPRESSION_RUN_DIR_STAGE1}" ]]; then
  echo "Error: COMPRESSION_RUN_DIR_STAGE0 and COMPRESSION_RUN_DIR_STAGE1 are required." >&2
  exit 1
fi

RANGEDET_MODEL_PREFIX=${RANGEDET_MODEL_PREFIX:-${ROOT_DIR}/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260304_0100/checkpoint}
RANGEDET_TEST_EPOCH=${RANGEDET_TEST_EPOCH:-24}
RANGEDET_CONFIG=${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}
KITTI_ROOT=${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}
SOURCE_RANGEDET_ROOT=${SOURCE_RANGEDET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_kitti_hq}

IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
MAX_FRAMES=${MAX_FRAMES:-0}
EXPORT_DEVICE=${EXPORT_DEVICE:-cuda}
SPLIT=${SPLIT:-validation}
LIDAR_SUBDIR=${LIDAR_SUBDIR:-training}
CHECKPOINT=${CHECKPOINT:-}

RECON_ROOT_BASE=${RECON_ROOT_BASE:-${ROOT_DIR}/data/dataset/rangedet_kitti_recon_track1a}
RECON_ROOT_STAGE0=${RECON_ROOT_STAGE0:-${RECON_ROOT_BASE}_stage0}
RECON_ROOT_STAGE1=${RECON_ROOT_STAGE1:-${RECON_ROOT_BASE}_stage1}

SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-6}
SBATCH_MEM=${SBATCH_MEM:-64G}
PREFERRED_PARTITION=${PREFERRED_PARTITION:-gpuql}
FALLBACK_PARTITION=${FALLBACK_PARTITION:-gpuqm}

TAG=${TAG:-$(date +%y%m%d_%H%M%S)}
manifest_csv="${ROOT_DIR}/logs/${TAG}_rangedet_stage_eval_chain.csv"
echo "job_id,job_name,dependency_type,dependency_job_id,partition,script,compression_run_dir,recon_data_root,rangedet_model_prefix,test_epoch,split,img_h,img_w" > "${manifest_csv}"

submit_stage_eval() {
  local job_name="$1"
  local compression_run_dir="$2"
  local recon_data_root="$3"
  local dependency="$4"

  local -a sbatch_extra=()
  if [[ -n "${dependency}" ]]; then
    sbatch_extra+=(--dependency="${dependency}")
  fi

  local partition=""
  local job_id=""
  local date_tag
  date_tag="$(date +%y%m%d)"
  local log_base="${ROOT_DIR}/logs/${date_tag}_${TAG}_${job_name}"
  local -a partitions=("${PREFERRED_PARTITION}" "${FALLBACK_PARTITION}")
  for p in "${partitions[@]}"; do
    set +e
    job_id="$(env \
      ROOT_DIR="${ROOT_DIR}" \
      COMPRESSION_RUN_DIR="${compression_run_dir}" \
      RECON_DATA_ROOT="${recon_data_root}" \
      RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
      RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
      RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
      KITTI_ROOT="${KITTI_ROOT}" \
      SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
      SPLIT="${SPLIT}" \
      LIDAR_SUBDIR="${LIDAR_SUBDIR}" \
      IMG_H="${IMG_H}" \
      IMG_W="${IMG_W}" \
      FOV_UP_DEG="${FOV_UP_DEG}" \
      FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
      MAX_FRAMES="${MAX_FRAMES}" \
      EXPORT_DEVICE="${EXPORT_DEVICE}" \
      CHECKPOINT="${CHECKPOINT}" \
      sbatch --parsable \
        --partition="${p}" \
        --job-name="${job_name}" \
        --output="${log_base}_%j.out" \
        --error="${log_base}_%j.err" \
        --cpus-per-task="${SBATCH_CPUS}" \
        --mem="${SBATCH_MEM}" \
        --gres=gpu:1 \
        --time="${SBATCH_TIME}" \
        "${sbatch_extra[@]}" \
        src/scripts/run_rangedet_stage_eval.sh 2>/tmp/rangedet_stage_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${job_id}" =~ ^[0-9]+$ ]]; then
      partition="${p}"
      break
    fi
    cat /tmp/rangedet_stage_submit.err >&2 || true
  done
  rm -f /tmp/rangedet_stage_submit.err

  if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to submit ${job_name}" >&2
    exit 1
  fi

  local dep_type=""
  local dep_job=""
  if [[ -n "${dependency}" ]]; then
    dep_type="${dependency%%:*}"
    dep_job="${dependency#*:}"
  fi
  echo "${job_id},${job_name},${dep_type},${dep_job},${partition},src/scripts/run_rangedet_stage_eval.sh,${compression_run_dir},${recon_data_root},${RANGEDET_MODEL_PREFIX},${RANGEDET_TEST_EPOCH},${SPLIT},${IMG_H},${IMG_W}" >> "${manifest_csv}"
  echo "${job_id}"
}

job_stage0="$(submit_stage_eval "rdt_s0_eval" "${COMPRESSION_RUN_DIR_STAGE0}" "${RECON_ROOT_STAGE0}" "")"
job_stage1="$(submit_stage_eval "rdt_s1_eval" "${COMPRESSION_RUN_DIR_STAGE1}" "${RECON_ROOT_STAGE1}" "afterok:${job_stage0}")"

echo "[rangedet-stage-chain] stage0_job=${job_stage0}"
echo "[rangedet-stage-chain] stage1_job=${job_stage1}"
echo "[rangedet-stage-chain] manifest=${manifest_csv}"
