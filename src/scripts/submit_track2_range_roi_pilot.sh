#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
RUN_DIR=${RUN_DIR:-data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
TRACK2_TAG=${TRACK2_TAG:-$(date +%y%m%d_%H%M%S)}
HEAD_SWEEP=${HEAD_SWEEP:-"linear refine"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-2048}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-512}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-1}
DEVICE=${DEVICE:-auto}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/results/track2_range_roi}
CHECKPOINT=${CHECKPOINT:-}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64}
SEED=${SEED:-42}
LOG_INTERVAL=${LOG_INTERVAL:-20}
THRESHOLD=${THRESHOLD:-0.5}
POS_WEIGHT_CAP=${POS_WEIGHT_CAP:-50}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-1024}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
SBATCH_PARTITION=${SBATCH_PARTITION:-gpuqm}
SBATCH_CPUS=${SBATCH_CPUS:-4}
SBATCH_GRES=${SBATCH_GRES:-gpu:1}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SBATCH_NODELIST=${SBATCH_NODELIST:-}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
DRY_RUN=${DRY_RUN:-0}
MANIFEST_PATH=${MANIFEST_PATH:-}

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  bash "${ROOT_DIR}/src/scripts/download_prbonn_range_backbones.sh"
fi

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT_DIR}/${RUN_DIR}"
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Error: RUN_DIR not found: ${RUN_DIR}" >&2
  exit 1
fi
if [[ -n "${MANIFEST_PATH}" && "${MANIFEST_PATH}" != /* ]]; then
  MANIFEST_PATH="${ROOT_DIR}/${MANIFEST_PATH}"
fi

ensure_manifest_header() {
  if [[ -z "${MANIFEST_PATH}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${MANIFEST_PATH}")"
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    cat > "${MANIFEST_PATH}" <<'EOF'
submitted_at,job_id,track2_tag,run_name,head_type,hidden_channels,epochs,lr,weight_decay,batch_size,max_train_samples,max_val_samples,freeze_backbone,device,run_dir,kitti_root,output_dir,slurm_log
EOF
  fi
}

append_manifest_row() {
  local job_id="$1"
  local run_name="$2"
  local head="$3"
  local output_dir="$4"
  if [[ -z "${MANIFEST_PATH}" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  ensure_manifest_header
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
    "${job_id}" \
    "${TRACK2_TAG}" \
    "${run_name}" \
    "${head}" \
    "${HIDDEN_CHANNELS}" \
    "${EPOCHS}" \
    "${LR}" \
    "${WEIGHT_DECAY}" \
    "${BATCH_SIZE}" \
    "${MAX_TRAIN_SAMPLES}" \
    "${MAX_VAL_SAMPLES}" \
    "${FREEZE_BACKBONE}" \
    "${DEVICE}" \
    "${RUN_DIR}" \
    "${KITTI_ROOT}" \
    "${output_dir}" \
    "${ROOT_DIR}/logs/slurm_${job_id}.out" >> "${MANIFEST_PATH}"
}

submit_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

echo "============================================================"
echo "[Track2 ROI Submit]"
echo "track2_tag: ${TRACK2_TAG}"
echo "run_dir: ${RUN_DIR}"
echo "head_sweep: ${HEAD_SWEEP}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "max_train_samples: ${MAX_TRAIN_SAMPLES}"
echo "max_val_samples: ${MAX_VAL_SAMPLES}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

for head in ${HEAD_SWEEP}; do
  run_name="${TRACK2_TAG}_${head}"
  output_dir="${OUTPUT_ROOT}"
  if [[ "${output_dir}" != /* ]]; then
    output_dir="${ROOT_DIR}/${output_dir}"
  fi
  output_dir="${output_dir}/${run_name}"
  declare -a SBATCH_CMD=(
    sbatch --parsable
    --job-name="track2_${head}"
    --output="logs/slurm_%A.out"
    --error="logs/slurm_%A.err"
    --partition="${SBATCH_PARTITION}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${SBATCH_CPUS}"
    --gres="${SBATCH_GRES}"
    --time="${SBATCH_TIME}"
  )
  if [[ -n "${SBATCH_NODELIST}" ]]; then
    SBATCH_CMD+=(--nodelist="${SBATCH_NODELIST}")
  fi
  SBATCH_CMD+=(
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",RUN_DIR="${RUN_DIR}",KITTI_ROOT="${KITTI_ROOT}",TRACK2_TAG="${TRACK2_TAG}",HEAD_TYPE="${head}",RUN_NAME="${run_name}",CHECKPOINT="${CHECKPOINT}",HIDDEN_CHANNELS="${HIDDEN_CHANNELS}",EPOCHS="${EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",WORKERS="${WORKERS}",LR="${LR}",WEIGHT_DECAY="${WEIGHT_DECAY}",MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES}",MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES}",FREEZE_BACKBONE="${FREEZE_BACKBONE}",DEVICE="${DEVICE}",SEED="${SEED}",LOG_INTERVAL="${LOG_INTERVAL}",THRESHOLD="${THRESHOLD}",POS_WEIGHT_CAP="${POS_WEIGHT_CAP}",IMG_H="${IMG_H}",IMG_W="${IMG_W}",FOV_UP_DEG="${FOV_UP_DEG}",FOV_DOWN_DEG="${FOV_DOWN_DEG}",OUTPUT_ROOT="${OUTPUT_ROOT}"
    --wrap="bash ${ROOT_DIR}/src/scripts/run_track2_range_roi_pilot.sh"
  )
  job_id=$(submit_or_echo "${SBATCH_CMD[@]}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    job_id="DRYRUN_${head}"
  fi

  append_manifest_row "${job_id}" "${run_name}" "${head}" "${output_dir}"
  echo "[track2-submit] head=${head} run_name=${run_name} job_id=${job_id}"
done

echo "[track2-submit] queued"
