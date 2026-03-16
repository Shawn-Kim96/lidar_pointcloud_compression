#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

BASE_TAG=${BASE_TAG:-$(date +%y%m%d_%H%M%S)}
KITTI_ROOT=${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}
SOURCE_RANGEDET_ROOT=${SOURCE_RANGEDET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_kitti_hq}
RANGEDET_MODEL_PREFIX=${RANGEDET_MODEL_PREFIX:-${ROOT_DIR}/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260304_0100/checkpoint}
RANGEDET_TEST_EPOCH=${RANGEDET_TEST_EPOCH:-24}
RANGEDET_CONFIG=${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
SPLIT=${SPLIT:-validation}
EXPORT_DEVICE=${EXPORT_DEVICE:-cuda}
SBATCH_CPUS=${SBATCH_CPUS:-6}
SBATCH_MEM=${SBATCH_MEM:-64G}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}

RUN_NOQUANT_A=${RUN_NOQUANT_A:-${ROOT_DIR}/data/results/experiments/260304_resnet_noquant_hires_track1nq_a_baseline_260304_230724_nq}
RUN_NOQUANT_B=${RUN_NOQUANT_B:-${ROOT_DIR}/data/results/experiments/260304_resnet_noquant_hires_track1nq_b_enhanced_260304_230724_nq}
RUN_UNIFORM_A=${RUN_UNIFORM_A:-${ROOT_DIR}/data/results/experiments/260303_resnet_uniform_q6_hires_track1a_hires_baseline_260303_225625_s0}
RUN_UNIFORM_B=${RUN_UNIFORM_B:-${ROOT_DIR}/data/results/experiments/260303_resnet_uniform_q6_hires_track1b_hires_enhanced_260303_225625_s0}

ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/logs/rangedet_eval_outputs}

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

submit_job() {
  local job_name="$1"
  local script="$2"
  local dependency="$3"
  shift 3

  local -a dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args+=(--dependency="${dependency}")
  fi

  local date_tag
  date_tag="$(date +%y%m%d)"
  local log_base="${ROOT_DIR}/logs/${date_tag}_${BASE_TAG}_${job_name}"
  local jid=""
  local part=""
  local -a parts=("${SBATCH_PARTITION:-$(choose_partition)}")
  if [[ "${parts[0]}" != "gpuqm" ]]; then
    parts+=("gpuqm")
  fi

  for p in "${parts[@]}"; do
    set +e
    jid="$(env "$@" sbatch --parsable \
      --job-name="${job_name}" \
      --output="${log_base}_%j.out" \
      --error="${log_base}_%j.err" \
      --partition="${p}" \
      --nodes=1 --ntasks=1 \
      --cpus-per-task="${SBATCH_CPUS}" \
      --mem="${SBATCH_MEM}" \
      --gres=gpu:1 \
      --time="${SBATCH_TIME}" \
      "${dep_args[@]}" \
      "${script}" 2>/tmp/track2_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${jid}" =~ ^[0-9]+$ ]]; then
      part="${p}"
      echo "${jid},${job_name},${part}" >> "${MANIFEST_TMP}"
      echo "${jid}"
      return 0
    fi
  done

  cat /tmp/track2_submit.err >&2 || true
  rm -f /tmp/track2_submit.err
  echo "Error: failed to submit ${job_name}" >&2
  exit 1
}

mkdir -p logs "${ARCHIVE_DIR}"
MANIFEST_TMP="$(mktemp)"

echo "============================================================"
echo "[Submit Track2 RangeDet Raw/NoQuant/Uniform]"
echo "base_tag: ${BASE_TAG}"
echo "img_hxw: ${IMG_H}x${IMG_W}"
echo "rangedet_test_epoch: ${RANGEDET_TEST_EPOCH}"
echo "archive_dir: ${ARCHIVE_DIR}"
echo "============================================================"

JID_RAW=$(submit_job "rd_raw_eval" src/scripts/run_rangedet_raw_eval.sh "" \
  REPO_ROOT="${ROOT_DIR}" \
  RANGEDET_DATA_ROOT="${SOURCE_RANGEDET_ROOT}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_raw_eval_${BASE_TAG}" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_raw")

JID_NQA=$(submit_job "rd_nqa_eval" src/scripts/run_rangedet_stage_eval.sh "" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_NOQUANT_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqa" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_nqa" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_nqa" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_NQB=$(submit_job "rd_nqb_eval" src/scripts/run_rangedet_stage_eval.sh "" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_NOQUANT_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqb" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_nqb" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_nqb" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_UA=$(submit_job "rd_uqa_eval" src/scripts/run_rangedet_stage_eval.sh "afterok:${JID_NQA}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_UNIFORM_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqa" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_uqa" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_uqa" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_UB=$(submit_job "rd_uqb_eval" src/scripts/run_rangedet_stage_eval.sh "afterok:${JID_NQB}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_UNIFORM_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqb" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_uqb" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_uqb" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

MANIFEST="logs/${BASE_TAG}_track2_rangedet_raw_noquant_uniform_manifest.csv"
{
  echo "submitted_at,job_id,job_name,partition,mode,compression_run_dir,recon_data_root,archive_tag,dependency"
  while IFS=, read -r jid job_name partition; do
    case "${job_name}" in
      rd_raw_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},raw,${SOURCE_RANGEDET_ROOT},${SOURCE_RANGEDET_ROOT},${BASE_TAG}_raw,"
        ;;
      rd_nqa_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},noquant_a,${RUN_NOQUANT_A},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqa,${BASE_TAG}_nqa,"
        ;;
      rd_nqb_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},noquant_b,${RUN_NOQUANT_B},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqb,${BASE_TAG}_nqb,"
        ;;
      rd_uqa_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},uniform_a,${RUN_UNIFORM_A},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqa,${BASE_TAG}_uqa,afterok:${JID_NQA}"
        ;;
      rd_uqb_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},uniform_b,${RUN_UNIFORM_B},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqb,${BASE_TAG}_uqb,afterok:${JID_NQB}"
        ;;
    esac
  done < "${MANIFEST_TMP}"
} > "${MANIFEST}"
rm -f "${MANIFEST_TMP}" /tmp/track2_submit.err

echo "rd_raw_eval job_id=${JID_RAW}"
echo "rd_nqa_eval job_id=${JID_NQA}"
echo "rd_nqb_eval job_id=${JID_NQB}"
echo "rd_uqa_eval job_id=${JID_UA}"
echo "rd_uqb_eval job_id=${JID_UB}"
echo "manifest=${MANIFEST}"
