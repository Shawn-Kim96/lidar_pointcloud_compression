#!/bin/bash
#SBATCH --job-name=dl_kitti3dobj
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --partition=cpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"
mkdir -p logs

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import sys" >/dev/null 2>&1; then
  BASH_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" bash)
  PY_ENV_DESC="conda:${CONDA_ENV}"
else
  BASH_RUNNER=(bash)
  PY_ENV_DESC="system-bash"
fi

DATA_DIR=${DATA_DIR:-data/dataset/kitti3dobject}
ONLY=${ONLY:-essentials}
EXTRACTOR=${EXTRACTOR:-unzip}
RETRIES=${RETRIES:-5}
CHUNK_MB=${CHUNK_MB:-8}
DELETE_ZIPS=${DELETE_ZIPS:-0}

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_download_kitti3dobject_j${JOB_TAG}"

echo "============================================================"
echo "[KITTI 3D Object Download]"
echo "job_id: ${SLURM_JOB_ID:-n/a}"
echo "python_env: ${PY_ENV_DESC}"
echo "data_dir: ${DATA_DIR}"
echo "only: ${ONLY}"
echo "extractor: ${EXTRACTOR}"
echo "retries: ${RETRIES}"
echo "chunk_mb: ${CHUNK_MB}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

DEL_FLAG=()
if [[ "${DELETE_ZIPS}" == "1" ]]; then
  DEL_FLAG=(--delete-zips)
fi

"${BASH_RUNNER[@]}" src/scripts/download_kitti3dobject.sh \
  --data-dir "${DATA_DIR}" \
  --only "${ONLY}" \
  --extractor "${EXTRACTOR}" \
  --retries "${RETRIES}" \
  --chunk-mb "${CHUNK_MB}" \
  "${DEL_FLAG[@]}"

echo "[download-kitti3dobject] done"
