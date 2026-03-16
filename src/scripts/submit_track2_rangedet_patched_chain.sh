#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run on a Slurm login node." >&2
  exit 1
fi

BASE_TAG=${BASE_TAG:-$(date +%y%m%d_%H%M%S)}
DATE_TAG=$(date +%y%m%d)
KITTI_ROOT=${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}
SOURCE_RANGEDET_ROOT=${SOURCE_RANGEDET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_kitti_hq}
RANGEDET_CONFIG=${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}
RANGEDET_NUM_EPOCHS=${RANGEDET_NUM_EPOCHS:-24}
RANGEDET_TEST_EPOCH=${RANGEDET_TEST_EPOCH:-24}
RANGEDET_SAMPLING_RATE=${RANGEDET_SAMPLING_RATE:-1}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
SPLIT=${SPLIT:-validation}
EXPORT_DEVICE=${EXPORT_DEVICE:-cuda}
SBATCH_CPUS=${SBATCH_CPUS:-8}
SBATCH_MEM=${SBATCH_MEM:-64G}
TRAIN_TIME=${TRAIN_TIME:-12:00:00}
EVAL_TIME=${EVAL_TIME:-24:00:00}
ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/logs/rangedet_eval_outputs}

RUN_NOQUANT_A=${RUN_NOQUANT_A:-${ROOT_DIR}/data/results/experiments/260304_resnet_noquant_hires_track1nq_a_baseline_260304_230724_nq}
RUN_NOQUANT_B=${RUN_NOQUANT_B:-${ROOT_DIR}/data/results/experiments/260304_resnet_noquant_hires_track1nq_b_enhanced_260304_230724_nq}
RUN_UNIFORM_A=${RUN_UNIFORM_A:-${ROOT_DIR}/data/results/experiments/260303_resnet_uniform_q6_hires_track1a_hires_baseline_260303_225625_s0}
RUN_UNIFORM_B=${RUN_UNIFORM_B:-${ROOT_DIR}/data/results/experiments/260303_resnet_uniform_q6_hires_track1b_hires_enhanced_260303_225625_s0}

RANGEDET_EXPERIMENT_NAME=${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_raw_24e_${BASE_TAG}_patched2048}
RANGEDET_MODEL_PREFIX=${RANGEDET_MODEL_PREFIX:-${ROOT_DIR}/third_party/external_range_det/RangeDet/experiments/${RANGEDET_EXPERIMENT_NAME}/checkpoint}

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
  local time_limit="$3"
  local dependency="$4"
  shift 4

  local -a dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args+=(--dependency="${dependency}")
  fi

  local log_base="${ROOT_DIR}/logs/${DATE_TAG}_${BASE_TAG}_${job_name}"
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
      --time="${time_limit}" \
      "${dep_args[@]}" \
      "${script}" 2>/tmp/rangedet_chain_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${jid}" =~ ^[0-9]+$ ]]; then
      part="${p}"
      echo "${jid},${job_name},${part},${dependency}" >> "${MANIFEST_TMP}"
      echo "${jid}"
      return 0
    fi
  done

  cat /tmp/rangedet_chain_submit.err >&2 || true
  rm -f /tmp/rangedet_chain_submit.err
  echo "Error: failed to submit ${job_name}" >&2
  exit 1
}

mkdir -p logs "${ARCHIVE_DIR}"
MANIFEST_TMP="$(mktemp)"

echo "============================================================"
echo "[Submit Track2 RangeDet patched raw/stage chain]"
echo "base_tag: ${BASE_TAG}"
echo "experiment_name: ${RANGEDET_EXPERIMENT_NAME}"
echo "model_prefix: ${RANGEDET_MODEL_PREFIX}"
echo "img_hxw: ${IMG_H}x${IMG_W}"
echo "============================================================"

JID_TRAIN=$(submit_job "rd_raw24_patch" src/scripts/run_rangedet_kitti_train.sh "${TRAIN_TIME}" "" \
  REPO_ROOT="${ROOT_DIR}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_DATA_ROOT="${SOURCE_RANGEDET_ROOT}" \
  RANGEDET_NUM_EPOCHS="${RANGEDET_NUM_EPOCHS}" \
  RANGEDET_SAMPLING_RATE="${RANGEDET_SAMPLING_RATE}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME}" \
  RUN_TEST="0")

DEP_TRAIN="afterok:${JID_TRAIN}"
JID_RAW=$(submit_job "rd_raw_eval_p" src/scripts/run_rangedet_raw_eval.sh "${EVAL_TIME}" "${DEP_TRAIN}" \
  REPO_ROOT="${ROOT_DIR}" \
  RANGEDET_DATA_ROOT="${SOURCE_RANGEDET_ROOT}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_raw_eval_${BASE_TAG}_patched" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_raw_patched")

JID_NQA=$(submit_job "rd_nqa_eval_p" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "${DEP_TRAIN}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_NOQUANT_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqa_patched" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_nqa_patched" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_nqa_patched" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_NQB=$(submit_job "rd_nqb_eval_p" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "afterok:${JID_RAW}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_NOQUANT_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqb_patched" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_nqb_patched" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_nqb_patched" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_UQA=$(submit_job "rd_uqa_eval_p" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "afterok:${JID_NQA}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_UNIFORM_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqa_patched" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_uqa_patched" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_uqa_patched" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_UQB=$(submit_job "rd_uqb_eval_p" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "afterok:${JID_NQB}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_UNIFORM_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqb_patched" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_uqb_patched" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_uqb_patched" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

MANIFEST="logs/${BASE_TAG}_track2_rangedet_patched_chain_manifest.csv"
{
  echo "submitted_at,job_id,job_name,partition,dependency,experiment_name,model_prefix,mode,compression_run_dir,recon_data_root,archive_tag"
  while IFS=, read -r jid job_name partition dependency; do
    case "${job_name}" in
      rd_raw24_patch)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},train,${SOURCE_RANGEDET_ROOT},,"
        ;;
      rd_raw_eval_p)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},raw,${SOURCE_RANGEDET_ROOT},${SOURCE_RANGEDET_ROOT},${BASE_TAG}_raw_patched"
        ;;
      rd_nqa_eval_p)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},noquant_a,${RUN_NOQUANT_A},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqa_patched,${BASE_TAG}_nqa_patched"
        ;;
      rd_nqb_eval_p)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},noquant_b,${RUN_NOQUANT_B},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_nqb_patched,${BASE_TAG}_nqb_patched"
        ;;
      rd_uqa_eval_p)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},uniform_a,${RUN_UNIFORM_A},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqa_patched,${BASE_TAG}_uqa_patched"
        ;;
      rd_uqb_eval_p)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RANGEDET_EXPERIMENT_NAME},${RANGEDET_MODEL_PREFIX},uniform_b,${RUN_UNIFORM_B},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_uqb_patched,${BASE_TAG}_uqb_patched"
        ;;
    esac
  done < "${MANIFEST_TMP}"
} > "${MANIFEST}"
rm -f "${MANIFEST_TMP}" /tmp/rangedet_chain_submit.err

echo "rd_raw24_patch job_id=${JID_TRAIN}"
echo "rd_raw_eval_p job_id=${JID_RAW}"
echo "rd_nqa_eval_p job_id=${JID_NQA}"
echo "rd_nqb_eval_p job_id=${JID_NQB}"
echo "rd_uqa_eval_p job_id=${JID_UQA}"
echo "rd_uqb_eval_p job_id=${JID_UQB}"
echo "manifest=${MANIFEST}"
