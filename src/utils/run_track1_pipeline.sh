#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  PYTHON_RUNNER=("${CONDA_PREFIX_DIR}/bin/python")
elif command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import sys" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_RUNNER=(python)
fi

TRACK1_TAG=${TRACK1_TAG:-$(date +%y%m%d_%H%M%S)}
KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-data/dataset/kitti3dobject}
TRACK1_IMG_H=${TRACK1_IMG_H:-64}
TRACK1_IMG_W=${TRACK1_IMG_W:-2048}
TRACK1_FOV_UP_DEG=${TRACK1_FOV_UP_DEG:-3.0}
TRACK1_FOV_DOWN_DEG=${TRACK1_FOV_DOWN_DEG:--25.0}
TRACK1_UNPROJECTION_MODE=${TRACK1_UNPROJECTION_MODE:-ray}
KITTI_IDENTITY_ROOT=${KITTI_IDENTITY_ROOT:-data/dataset/kitti_identity_h${TRACK1_IMG_H}w${TRACK1_IMG_W}_${TRACK1_UNPROJECTION_MODE}}
FT_EPOCH_SWEEP=${FT_EPOCH_SWEEP:-"10 20"}
FT_BATCH_SIZE=${FT_BATCH_SIZE:-4}
FT_WORKERS=${FT_WORKERS:-4}
SKIP_EXPORT=${SKIP_EXPORT:-0}
DRY_RUN=${DRY_RUN:-0}
SBATCH_PARTITION=${SBATCH_PARTITION:-gpuqm}
SBATCH_CPUS=${SBATCH_CPUS:-4}
SBATCH_GRES=${SBATCH_GRES:-gpu:1}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
EVAL_EXCLUDE_NODES=${EVAL_EXCLUDE_NODES:-}

OPENPCDET_ROOT=${OPENPCDET_ROOT:-third_party/OpenPCDet}
OPENPCDET_CFG=${OPENPCDET_CFG:-tools/cfgs/kitti_models/pointpillar.yaml}
OPENPCDET_PRETRAIN_CKPT=${OPENPCDET_PRETRAIN_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}

mkdir -p logs

if [[ "${KITTI_ROOT_OFFICIAL}" != /* ]]; then
  KITTI_ROOT_OFFICIAL="${ROOT_DIR}/${KITTI_ROOT_OFFICIAL}"
fi
if [[ "${KITTI_IDENTITY_ROOT}" != /* ]]; then
  KITTI_IDENTITY_ROOT="${ROOT_DIR}/${KITTI_IDENTITY_ROOT}"
fi

if [[ ! -d "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL not found: ${KITTI_ROOT_OFFICIAL}" >&2
  exit 1
fi

declare -a DEFAULT_RUN_DIRS=(
  "data/results/experiments/260224_resnet_uniform_q6_lr1e-4_bs4_j22274_r4"
  "data/results/experiments/260301_resnet_solo_lr1e-4_bs4_j22968_r1"
  "data/results/experiments/260301_resnet_distill_lr1e-4_bs4_j22963_r2"
  "data/results/experiments/260301_resnet_distill_fix2_energy_pool_16x32_ld0p05_lr5em5_bs4_j22965_r3"
)

RUN_DIRS_INPUT=${RUN_DIRS:-}
declare -a RUN_DIR_ARRAY=()
if [[ -n "${RUN_DIRS_INPUT}" ]]; then
  IFS=',' read -r -a RUN_DIR_ARRAY <<< "${RUN_DIRS_INPUT}"
else
  for rel in "${DEFAULT_RUN_DIRS[@]}"; do
    if [[ -d "${ROOT_DIR}/${rel}" ]]; then
      RUN_DIR_ARRAY+=("${rel}")
    fi
  done
fi

if [[ "${#RUN_DIR_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: no evaluation run dirs were found. Set RUN_DIRS explicitly." >&2
  exit 1
fi

RUN_DIRS_FILE=${RUN_DIRS_FILE:-logs/track1_run_dirs_${TRACK1_TAG}.txt}
if [[ "${RUN_DIRS_FILE}" != /* ]]; then
  RUN_DIRS_FILE="${ROOT_DIR}/${RUN_DIRS_FILE}"
fi
printf "%s\n" "${RUN_DIR_ARRAY[@]}" > "${RUN_DIRS_FILE}"

echo "============================================================"
echo "[Track1 Pipeline]"
echo "track1_tag: ${TRACK1_TAG}"
echo "kitti_root_official: ${KITTI_ROOT_OFFICIAL}"
echo "kitti_identity_root: ${KITTI_IDENTITY_ROOT}"
echo "projection: ${TRACK1_IMG_H}x${TRACK1_IMG_W} fov_up=${TRACK1_FOV_UP_DEG} fov_down=${TRACK1_FOV_DOWN_DEG} unprojection=${TRACK1_UNPROJECTION_MODE}"
echo "ft_epoch_sweep: ${FT_EPOCH_SWEEP}"
echo "ft_batch_size: ${FT_BATCH_SIZE}"
echo "ft_workers: ${FT_WORKERS}"
echo "run_dirs_file: ${RUN_DIRS_FILE}"
echo "run_dirs_count: ${#RUN_DIR_ARRAY[@]}"
echo "dry_run: ${DRY_RUN}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  echo "[track1] exporting KITTI_Identity baseline ..."
  "${PYTHON_RUNNER[@]}" src/utils/export_kitti_identity_dataset.py \
    --source_root "${KITTI_ROOT_OFFICIAL}" \
    --output_root "${KITTI_IDENTITY_ROOT}" \
    --img_h "${TRACK1_IMG_H}" \
    --img_w "${TRACK1_IMG_W}" \
    --fov_up_deg "${TRACK1_FOV_UP_DEG}" \
    --fov_down_deg "${TRACK1_FOV_DOWN_DEG}" \
    --unprojection_mode "${TRACK1_UNPROJECTION_MODE}"
fi

submit_or_echo() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

for epochs in ${FT_EPOCH_SWEEP}; do
  EXTRA_TAG="track1_pp_identity_${TRACK1_TAG}_e${epochs}"

  FT_JOB_ID=$(submit_or_echo sbatch --parsable \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",KITTI_ROOT_OFFICIAL="${KITTI_IDENTITY_ROOT}",OPENPCDET_ROOT="${OPENPCDET_ROOT}",OPENPCDET_CFG="${OPENPCDET_CFG}",OPENPCDET_PRETRAIN_CKPT="${OPENPCDET_PRETRAIN_CKPT}",EPOCHS="${epochs}",BATCH_SIZE="${FT_BATCH_SIZE}",WORKERS="${FT_WORKERS}",EXTRA_TAG="${EXTRA_TAG}",RUN_FINAL_TEST=1 \
    src/scripts/run_pointpillar_kitti_finetune_sbatch.sh)

  if [[ "${DRY_RUN}" == "1" ]]; then
    FT_JOB_ID="DRYRUN_FT_${epochs}"
  fi

  EVAL_JOB_ID=$(submit_or_echo sbatch --parsable \
    --dependency="afterok:${FT_JOB_ID}" \
    --job-name="track1_eval_e${epochs}" \
    --output="logs/slurm_%A.out" \
    --error="logs/slurm_%A.err" \
    --partition="${SBATCH_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SBATCH_CPUS}" \
    --gres="${SBATCH_GRES}" \
    --time="${SBATCH_TIME}" \
    ${EVAL_EXCLUDE_NODES:+--exclude="${EVAL_EXCLUDE_NODES}"} \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",EXTRA_TAG="${EXTRA_TAG}",KITTI_ROOT_OFFICIAL="${KITTI_ROOT_OFFICIAL}",RUN_DIRS_FILE="${RUN_DIRS_FILE}",REFERENCE_MODE=identity,RUN_ORIGINAL_SANITY=0,IMG_H="${TRACK1_IMG_H}",IMG_W="${TRACK1_IMG_W}",FOV_UP_DEG="${TRACK1_FOV_UP_DEG}",FOV_DOWN_DEG="${TRACK1_FOV_DOWN_DEG}",UNPROJECTION_MODE="${TRACK1_UNPROJECTION_MODE}" \
    --wrap="bash ${ROOT_DIR}/src/utils/run_track1_eval_with_finetuned_ckpt.sh")

  if [[ "${DRY_RUN}" == "1" ]]; then
    EVAL_JOB_ID="DRYRUN_EVAL_${epochs}"
  fi

  echo "[track1] epochs=${epochs} extra_tag=${EXTRA_TAG} ft_job=${FT_JOB_ID} eval_job=${EVAL_JOB_ID}"
done

echo "[track1] queued"
