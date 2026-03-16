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
RANGEDET_TEST_EPOCH=${RANGEDET_TEST_EPOCH:-24}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
SPLIT=${SPLIT:-validation}
EXPORT_DEVICE=${EXPORT_DEVICE:-cuda}
SBATCH_CPUS=${SBATCH_CPUS:-8}
SBATCH_MEM=${SBATCH_MEM:-64G}
EVAL_TIME=${EVAL_TIME:-24:00:00}
COMPARE_TIME=${COMPARE_TIME:-02:00:00}
ARCHIVE_DIR=${ARCHIVE_DIR:-${ROOT_DIR}/logs/rangedet_eval_outputs}

RUN_SKIP_A=${RUN_SKIP_A:-${ROOT_DIR}/data/results/experiments/260315_resnet_noquant_hires_t1nq_pillar_skip_a_pillar_skip_a_260315_t1skipmain_nq}
RUN_SKIP_B=${RUN_SKIP_B:-${ROOT_DIR}/data/results/experiments/260315_resnet_noquant_hires_t1nq_pillar_skip_b_pillar_skip_b_260315_t1skipmain_nq}
RANGEDET_MODEL_PREFIX=${RANGEDET_MODEL_PREFIX:-${ROOT_DIR}/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260315_rddecodefixfull_patched2048/checkpoint}
RAW_ARCHIVE=${RAW_ARCHIVE:-${ROOT_DIR}/logs/rangedet_eval_outputs/260315_rddecodefixfull_raw_patched_output_dict_24e.pkl}
OLD_NQA_ARCHIVE=${OLD_NQA_ARCHIVE:-${ROOT_DIR}/logs/rangedet_eval_outputs/260315_rddecodefixfull_nqa_patched_output_dict_24e.pkl}
OLD_NQB_ARCHIVE=${OLD_NQB_ARCHIVE:-${ROOT_DIR}/logs/rangedet_eval_outputs/260315_rddecodefixfull_nqb_patched_output_dict_24e.pkl}

DEP_SKIP_A=${DEP_SKIP_A:-25892}
DEP_SKIP_B=${DEP_SKIP_B:-25894}

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
      "${script}" 2>/tmp/rangedet_skip_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${jid}" =~ ^[0-9]+$ ]]; then
      part="${p}"
      echo "${jid},${job_name},${part},${dependency}" >> "${MANIFEST_TMP}"
      echo "${jid}"
      return 0
    fi
  done

  cat /tmp/rangedet_skip_submit.err >&2 || true
  rm -f /tmp/rangedet_skip_submit.err
  echo "Error: failed to submit ${job_name}" >&2
  exit 1
}

mkdir -p logs "${ARCHIVE_DIR}"
MANIFEST_TMP="$(mktemp)"

echo "============================================================"
echo "[Submit Track2 RangeDet skip Stage0 chain]"
echo "base_tag: ${BASE_TAG}"
echo "rangedet_model_prefix: ${RANGEDET_MODEL_PREFIX}"
echo "run_skip_a: ${RUN_SKIP_A}"
echo "run_skip_b: ${RUN_SKIP_B}"
echo "============================================================"

JID_SKIP_A=$(submit_job "rd_skipa_eval" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "afterok:${DEP_SKIP_A}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_SKIP_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_skipa" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_skipa" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_skipa" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

JID_SKIP_B=$(submit_job "rd_skipb_eval" src/scripts/run_rangedet_stage_eval.sh "${EVAL_TIME}" "afterok:${DEP_SKIP_B}" \
  REPO_ROOT="${ROOT_DIR}" \
  COMPRESSION_RUN_DIR="${RUN_SKIP_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_skipb" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" \
  RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_CONFIG="${RANGEDET_CONFIG}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_kitti_eval_${BASE_TAG}_skipb" \
  RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" \
  RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_skipb" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" \
  SPLIT="${SPLIT}" \
  IMG_H="${IMG_H}" IMG_W="${IMG_W}" \
  FOV_UP_DEG="${FOV_UP_DEG}" FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
  EXPORT_DEVICE="${EXPORT_DEVICE}")

COMPARE_DEP="afterok:${JID_SKIP_A}:${JID_SKIP_B}"
ARCHIVES_CSV="${RAW_ARCHIVE},${OLD_NQA_ARCHIVE},${OLD_NQB_ARCHIVE},${ARCHIVE_DIR}/${BASE_TAG}_skipa_output_dict_${RANGEDET_TEST_EPOCH}e.pkl,${ARCHIVE_DIR}/${BASE_TAG}_skipb_output_dict_${RANGEDET_TEST_EPOCH}e.pkl"
JID_COMPARE=$(submit_job "rd_skip_compare" src/scripts/run_rangedet_archive_compare.sh "${COMPARE_TIME}" "${COMPARE_DEP}" \
  REPO_ROOT="${ROOT_DIR}" \
  ARCHIVES_CSV="${ARCHIVES_CSV}" \
  OUTPUT_TAG="${BASE_TAG}_skip_compare" \
  KITTI_ROOT="${KITTI_ROOT}" \
  SOURCE_ROIDB="${SOURCE_RANGEDET_ROOT}/validation/part-0000.roidb")

MANIFEST="logs/${BASE_TAG}_track2_rangedet_skip_stage0_manifest.csv"
{
  echo "submitted_at,job_id,job_name,partition,dependency,compression_run_dir,recon_data_root,archive_tag,rangedet_model_prefix"
  while IFS=, read -r jid job_name partition dependency; do
    case "${job_name}" in
      rd_skipa_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RUN_SKIP_A},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_skipa,${BASE_TAG}_skipa,${RANGEDET_MODEL_PREFIX}"
        ;;
      rd_skipb_eval)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},${RUN_SKIP_B},${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_skipb,${BASE_TAG}_skipb,${RANGEDET_MODEL_PREFIX}"
        ;;
      rd_skip_compare)
        echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency},compare,compare,${BASE_TAG}_skip_compare,${RANGEDET_MODEL_PREFIX}"
        ;;
    esac
  done < "${MANIFEST_TMP}"
} > "${MANIFEST}"
rm -f "${MANIFEST_TMP}" /tmp/rangedet_skip_submit.err

echo "rd_skipa_eval job_id=${JID_SKIP_A}"
echo "rd_skipb_eval job_id=${JID_SKIP_B}"
echo "rd_skip_compare job_id=${JID_COMPARE}"
echo "manifest=${MANIFEST}"
