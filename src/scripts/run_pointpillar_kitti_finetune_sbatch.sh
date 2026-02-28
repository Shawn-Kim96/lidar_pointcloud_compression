#!/bin/bash
#SBATCH --job-name=pp_kitti_ft
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"
mkdir -p logs

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  BASH_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" bash)
  PY_ENV_DESC="conda:${CONDA_ENV}"
else
  BASH_RUNNER=(bash)
  PY_ENV_DESC="system-bash"
fi

KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-}
if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
WAIT_FOR_KITTI_SEC=${WAIT_FOR_KITTI_SEC:-14400}
WAIT_POLL_SEC=${WAIT_POLL_SEC:-60}
export WAIT_FOR_KITTI_SEC WAIT_POLL_SEC

echo "============================================================"
echo "[PointPillar Finetune SBATCH]"
echo "job_id: ${SLURM_JOB_ID:-n/a}"
echo "python_env: ${PY_ENV_DESC}"
echo "kitti_root_official: ${KITTI_ROOT_OFFICIAL}"
echo "wait_for_kitti_sec: ${WAIT_FOR_KITTI_SEC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

"${BASH_RUNNER[@]}" src/scripts/run_pointpillar_kitti_finetune.sh
