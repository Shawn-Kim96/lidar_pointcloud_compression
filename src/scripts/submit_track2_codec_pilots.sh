#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found." >&2
  exit 1
fi

BASE_TAG=${BASE_TAG:-$(date +%y%m%d_%H%M%S)_t2pilot}
DATE_TAG=$(date +%y%m%d)
KITTI_ROOT=${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}
SOURCE_RANGEDET_ROOT=${SOURCE_RANGEDET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_kitti_hq}
RANGEDET_MODEL_PREFIX=${RANGEDET_MODEL_PREFIX:-${ROOT_DIR}/third_party/external_range_det/RangeDet/experiments/rangedet_kitti_raw_24e_260315_rddecodefixfull_patched2048/checkpoint}
RANGEDET_CONFIG=${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}
RANGEDET_TEST_EPOCH=${RANGEDET_TEST_EPOCH:-24}
RAW_ARCHIVE=${RAW_ARCHIVE:-${ROOT_DIR}/logs/rangedet_eval_outputs/260315_rddecodefixfull_raw_patched_output_dict_24e.pkl}
OLD_STAGE0_ARCHIVE=${OLD_STAGE0_ARCHIVE:-${ROOT_DIR}/logs/rangedet_eval_outputs/260315_rddecodefixfull_nqa_patched_output_dict_24e.pkl}
TEACHER_TARGET_ROOT=${TEACHER_TARGET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_teacher_targets_raw24_260315}
TRAIN_FRAMES=${TRAIN_FRAMES:-1024}
VAL_FRAMES=${VAL_FRAMES:-512}
TRAIN_TIME=${TRAIN_TIME:-18:00:00}
EVAL_TIME=${EVAL_TIME:-08:00:00}
COMPARE_TIME=${COMPARE_TIME:-02:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-8}
SBATCH_MEM=${SBATCH_MEM:-64G}
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
  local time_limit="$3"
  local dependency="$4"
  shift 4
  local -a dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args+=(--dependency="${dependency}")
  fi
  local -a parts=("${SBATCH_PARTITION:-$(choose_partition)}")
  if [[ "${parts[0]}" != "gpuqm" ]]; then
    parts+=("gpuqm")
  fi
  local log_base="${ROOT_DIR}/logs/${DATE_TAG}_${BASE_TAG}_${job_name}"
  local jid=""
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
      "${script}" 2>/tmp/track2_pilot_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${jid}" =~ ^[0-9]+$ ]]; then
      echo "${jid},${job_name},${p},${dependency}" >> "${MANIFEST_TMP}"
      echo "${jid}"
      return 0
    fi
  done
  cat /tmp/track2_pilot_submit.err >&2 || true
  rm -f /tmp/track2_pilot_submit.err
  echo "Error: failed to submit ${job_name}" >&2
  exit 1
}

mkdir -p logs "${ARCHIVE_DIR}" "${TEACHER_TARGET_ROOT}"

python3 src/scripts/export_rangedet_teacher_maps.py \
  --source-roidb "${SOURCE_RANGEDET_ROOT}/validation/part-0000.roidb" \
  --archive "${RAW_ARCHIVE}" \
  --output-root "${TEACHER_TARGET_ROOT}"

MANIFEST_TMP="$(mktemp)"

RUN_A="${ROOT_DIR}/data/results/experiments/${DATE_TAG}_track2_codec_pilotA_${BASE_TAG}"
RUN_B="${ROOT_DIR}/data/results/experiments/${DATE_TAG}_track2_codec_pilotB_${BASE_TAG}"
RUN_C="${ROOT_DIR}/data/results/experiments/${DATE_TAG}_track2_codec_pilotC_${BASE_TAG}"
RUN_D="${ROOT_DIR}/data/results/experiments/${DATE_TAG}_track2_codec_pilotD_${BASE_TAG}"

JID_A=$(submit_job "t2p_mask_train" src/scripts/run_track2_codec_pilot.sh "${TRAIN_TIME}" "" \
  ROOT_DIR="${ROOT_DIR}" PILOT_TAG="${BASE_TAG}" PILOT_NAME="pilotA_mask" RUN_ID="pilotA_mask_${BASE_TAG}" \
  KITTI_ROOT="${KITTI_ROOT}" EPOCHS="40" MAX_TRAIN_FRAMES="${TRAIN_FRAMES}" SAVE_DIR="${RUN_A}" \
  MASK_HEAD="1" LAMBDA_VALID_MASK="1.0" LAMBDA_VALID_MASK_DICE="0.25" \
  RECON_LOSS_MODE="masked_channel_weighted" RECON_RANGE_WEIGHT="1.0" RECON_XYZ_WEIGHT="1.0" RECON_REMISSION_WEIGHT="0.5")
JID_AE=$(submit_job "t2p_mask_eval" src/scripts/run_track2_codec_pilot_eval.sh "${EVAL_TIME}" "afterok:${JID_A}" \
  ROOT_DIR="${ROOT_DIR}" COMPRESSION_RUN_DIR="${RUN_A}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_pilotA" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" RANGEDET_CONFIG="${RANGEDET_CONFIG}" RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_eval_${BASE_TAG}_pilotA" RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_pilotA" \
  KITTI_ROOT="${KITTI_ROOT}" SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" MAX_FRAMES="${VAL_FRAMES}" \
  SUMMARY_CSV="${ROOT_DIR}/logs/${BASE_TAG}_pilotA_eval.csv")

JID_B=$(submit_job "t2p_band_train" src/scripts/run_track2_codec_pilot.sh "${TRAIN_TIME}" "" \
  ROOT_DIR="${ROOT_DIR}" PILOT_TAG="${BASE_TAG}" PILOT_NAME="pilotB_band" RUN_ID="pilotB_band_${BASE_TAG}" \
  KITTI_ROOT="${KITTI_ROOT}" EPOCHS="40" MAX_TRAIN_FRAMES="${TRAIN_FRAMES}" SAVE_DIR="${RUN_B}" \
  DECODER_TYPE="skip_unet" DECODER_REFINE_BLOCKS="2" \
  RECON_LOSS_MODE="masked_channel_weighted" RECON_RANGE_WEIGHT="1.0" RECON_XYZ_WEIGHT="1.0" RECON_REMISSION_WEIGHT="0.5" \
  LAMBDA_RANGE_GRAD_ROW="0.5" LAMBDA_RANGE_GRAD_COL="0.5" LAMBDA_ROW_PROFILE="0.5")
JID_BE=$(submit_job "t2p_band_eval" src/scripts/run_track2_codec_pilot_eval.sh "${EVAL_TIME}" "afterok:${JID_B}" \
  ROOT_DIR="${ROOT_DIR}" COMPRESSION_RUN_DIR="${RUN_B}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_pilotB" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" RANGEDET_CONFIG="${RANGEDET_CONFIG}" RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_eval_${BASE_TAG}_pilotB" RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_pilotB" \
  KITTI_ROOT="${KITTI_ROOT}" SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" MAX_FRAMES="${VAL_FRAMES}" \
  SUMMARY_CSV="${ROOT_DIR}/logs/${BASE_TAG}_pilotB_eval.csv")

JID_C=$(submit_job "t2p_det_train" src/scripts/run_track2_codec_pilot.sh "${TRAIN_TIME}" "afterok:${JID_AE}" \
  ROOT_DIR="${ROOT_DIR}" PILOT_TAG="${BASE_TAG}" PILOT_NAME="pilotC_det" RUN_ID="pilotC_det_${BASE_TAG}" \
  KITTI_ROOT="${KITTI_ROOT}" EPOCHS="40" MAX_TRAIN_FRAMES="${TRAIN_FRAMES}" SAVE_DIR="${RUN_C}" \
  DETECTOR_AUX_HEAD="1" LAMBDA_DETECTOR_TARGET="1.0" TEACHER_TARGET_ROOT="${TEACHER_TARGET_ROOT}" \
  RECON_LOSS_MODE="masked_channel_weighted" RECON_RANGE_WEIGHT="1.0" RECON_XYZ_WEIGHT="1.0" RECON_REMISSION_WEIGHT="0.5")
JID_CE=$(submit_job "t2p_det_eval" src/scripts/run_track2_codec_pilot_eval.sh "${EVAL_TIME}" "afterok:${JID_C}" \
  ROOT_DIR="${ROOT_DIR}" COMPRESSION_RUN_DIR="${RUN_C}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_pilotC" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" RANGEDET_CONFIG="${RANGEDET_CONFIG}" RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_eval_${BASE_TAG}_pilotC" RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_pilotC" \
  KITTI_ROOT="${KITTI_ROOT}" SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" MAX_FRAMES="${VAL_FRAMES}" \
  SUMMARY_CSV="${ROOT_DIR}/logs/${BASE_TAG}_pilotC_eval.csv")

JID_D=$(submit_job "t2p_combo_train" src/scripts/run_track2_codec_pilot.sh "${TRAIN_TIME}" "afterok:${JID_BE}" \
  ROOT_DIR="${ROOT_DIR}" PILOT_TAG="${BASE_TAG}" PILOT_NAME="pilotD_combo" RUN_ID="pilotD_combo_${BASE_TAG}" \
  KITTI_ROOT="${KITTI_ROOT}" EPOCHS="40" MAX_TRAIN_FRAMES="${TRAIN_FRAMES}" SAVE_DIR="${RUN_D}" \
  DECODER_TYPE="skip_unet" DECODER_REFINE_BLOCKS="2" MASK_HEAD="1" \
  LAMBDA_VALID_MASK="1.0" LAMBDA_VALID_MASK_DICE="0.25" \
  LAMBDA_RANGE_GRAD_ROW="0.5" LAMBDA_RANGE_GRAD_COL="0.5" LAMBDA_ROW_PROFILE="0.5" \
  RECON_LOSS_MODE="masked_channel_weighted" RECON_RANGE_WEIGHT="1.0" RECON_XYZ_WEIGHT="1.0" RECON_REMISSION_WEIGHT="0.5")
JID_DE=$(submit_job "t2p_combo_eval" src/scripts/run_track2_codec_pilot_eval.sh "${EVAL_TIME}" "afterok:${JID_D}" \
  ROOT_DIR="${ROOT_DIR}" COMPRESSION_RUN_DIR="${RUN_D}" \
  RECON_DATA_ROOT="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_pilotD" \
  RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX}" RANGEDET_CONFIG="${RANGEDET_CONFIG}" RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH}" \
  RANGEDET_EXPERIMENT_NAME="rangedet_eval_${BASE_TAG}_pilotD" RANGEDET_OUTPUT_ARCHIVE_DIR="${ARCHIVE_DIR}" RANGEDET_OUTPUT_ARCHIVE_TAG="${BASE_TAG}_pilotD" \
  KITTI_ROOT="${KITTI_ROOT}" SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT}" MAX_FRAMES="${VAL_FRAMES}" \
  SUMMARY_CSV="${ROOT_DIR}/logs/${BASE_TAG}_pilotD_eval.csv")

COMPARE_DEP="afterok:${JID_AE}:${JID_BE}:${JID_CE}:${JID_DE}"
ARCHIVES_CSV="${RAW_ARCHIVE},${OLD_STAGE0_ARCHIVE},${ARCHIVE_DIR}/${BASE_TAG}_pilotA_output_dict_${RANGEDET_TEST_EPOCH}e.pkl,${ARCHIVE_DIR}/${BASE_TAG}_pilotB_output_dict_${RANGEDET_TEST_EPOCH}e.pkl,${ARCHIVE_DIR}/${BASE_TAG}_pilotC_output_dict_${RANGEDET_TEST_EPOCH}e.pkl,${ARCHIVE_DIR}/${BASE_TAG}_pilotD_output_dict_${RANGEDET_TEST_EPOCH}e.pkl"
JID_COMPARE=$(submit_job "t2p_compare" src/scripts/run_rangedet_archive_compare.sh "${COMPARE_TIME}" "${COMPARE_DEP}" \
  REPO_ROOT="${ROOT_DIR}" ARCHIVES_CSV="${ARCHIVES_CSV}" OUTPUT_TAG="${BASE_TAG}_pilot_compare" \
  SOURCE_ROIDB="${ROOT_DIR}/data/dataset/rangedet_kitti_recon_${BASE_TAG}_pilotA/validation/part-0000.roidb")

MANIFEST="logs/${BASE_TAG}_track2_codec_pilots_manifest.csv"
{
  echo "submitted_at,job_id,job_name,partition,dependency"
  while IFS=, read -r jid job_name partition dependency; do
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${jid},${job_name},${partition},${dependency}"
  done < "${MANIFEST_TMP}"
} > "${MANIFEST}"
rm -f "${MANIFEST_TMP}" /tmp/track2_pilot_submit.err

echo "pilotA train/eval: ${JID_A} / ${JID_AE}"
echo "pilotB train/eval: ${JID_B} / ${JID_BE}"
echo "pilotC train/eval: ${JID_C} / ${JID_CE}"
echo "pilotD train/eval: ${JID_D} / ${JID_DE}"
echo "compare: ${JID_COMPARE}"
echo "manifest=${MANIFEST}"
