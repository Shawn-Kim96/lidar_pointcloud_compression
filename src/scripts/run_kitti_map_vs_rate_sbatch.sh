#!/bin/bash
#SBATCH --job-name=kitti_map_rate
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#
# Submit Track-B evaluation on a GPU node:
#  1) original KITTI sanity via OpenPCDet tools/test.py (Car AP(mod) > 0)
#  2) reconstructed-vs-original comparison with same protocol
#
# Required env:
#   KITTI_ROOT_OFFICIAL=/path/to/kitti_detection_root
#   RUN_DIRS=data/results/experiments/runA,data/results/experiments/runB
#
# Optional env:
#   CONDA_ENV=lidarcomp311
#   ROOT_DIR=/home/018219422/lidar_pointcloud_compression
#   OPENPCDET_ROOT=third_party/OpenPCDet
#   OPENPCDET_CFG=third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
#   OPENPCDET_CKPT=data/checkpoints/openpcdet_pointpillar_18M.pth
#   SETUP_OPENPCDET=0
#   REBUILD_KITTI_INFOS=0
#   RUN_ORIGINAL_SANITY=1
#   REQUIRE_CUDA=1
#   MAX_FRAMES=0
#   WORKERS=0
#   EVAL_METRIC=kitti
#   REFERENCE_MODE=original
#   RECON_QUANT_MODE=native
#   ORACLE_CLASSES=Car,Pedestrian,Cyclist
#   ORACLE_DILATE_PX=0
#   ADAPTIVE_BG_LEVELS_OVERRIDE=-1
#   ADAPTIVE_ROI_LEVELS_OVERRIDE=-1
#   SUMMARY_CSV=notebooks/kitti_map_vs_rate_summary.csv
#   DETAIL_CSV=notebooks/kitti_map_vs_rate_detail.csv
#   DETECTOR_PAIR_CSV=notebooks/kitti_map_vs_rate_pairs.csv

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
RUN_DIRS=${RUN_DIRS:-}
OPENPCDET_ROOT=${OPENPCDET_ROOT:-third_party/OpenPCDet}
OPENPCDET_CFG=${OPENPCDET_CFG:-third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}
OPENPCDET_CKPT=${OPENPCDET_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}
SETUP_OPENPCDET=${SETUP_OPENPCDET:-0}
REBUILD_KITTI_INFOS=${REBUILD_KITTI_INFOS:-0}
RUN_ORIGINAL_SANITY=${RUN_ORIGINAL_SANITY:-1}
REQUIRE_CUDA=${REQUIRE_CUDA:-1}
MAX_FRAMES=${MAX_FRAMES:-0}
WORKERS=${WORKERS:-0}
EVAL_METRIC=${EVAL_METRIC:-kitti}
REFERENCE_MODE=${REFERENCE_MODE:-original}
RECON_QUANT_MODE=${RECON_QUANT_MODE:-native}
ORACLE_CLASSES=${ORACLE_CLASSES:-Car,Pedestrian,Cyclist}
ORACLE_DILATE_PX=${ORACLE_DILATE_PX:-0}
ADAPTIVE_BG_LEVELS_OVERRIDE=${ADAPTIVE_BG_LEVELS_OVERRIDE:--1}
ADAPTIVE_ROI_LEVELS_OVERRIDE=${ADAPTIVE_ROI_LEVELS_OVERRIDE:--1}
SUMMARY_CSV=${SUMMARY_CSV:-notebooks/kitti_map_vs_rate_summary.csv}
DETAIL_CSV=${DETAIL_CSV:-notebooks/kitti_map_vs_rate_detail.csv}
DETECTOR_PAIR_CSV=${DETECTOR_PAIR_CSV:-notebooks/kitti_map_vs_rate_pairs.csv}

if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
if [[ -z "${RUN_DIRS}" ]]; then
  echo "Error: RUN_DIRS is required." >&2
  exit 1
fi

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_kitti_map_vs_rate_j${JOB_TAG}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

echo "============================================================"
echo "[KITTI Map-vs-Rate SBATCH]"
echo "job_id: ${SLURM_JOB_ID:-n/a}"
echo "python_env: ${PY_ENV_DESC}"
echo "kitti_root_official: ${KITTI_ROOT_OFFICIAL}"
echo "run_dirs: ${RUN_DIRS}"
echo "openpcdet_cfg: ${OPENPCDET_CFG}"
echo "openpcdet_ckpt: ${OPENPCDET_CKPT}"
echo "workers: ${WORKERS}"
echo "max_frames: ${MAX_FRAMES}"
echo "eval_metric: ${EVAL_METRIC}"
echo "reference_mode: ${REFERENCE_MODE}"
echo "recon_quant_mode: ${RECON_QUANT_MODE}"
echo "oracle_classes: ${ORACLE_CLASSES}"
echo "oracle_dilate_px: ${ORACLE_DILATE_PX}"
echo "adaptive_bg_levels_override: ${ADAPTIVE_BG_LEVELS_OVERRIDE}"
echo "adaptive_roi_levels_override: ${ADAPTIVE_ROI_LEVELS_OVERRIDE}"
echo "slurm_job_nodelist: ${SLURM_JOB_NODELIST:-n/a}"
echo "slurm_job_gres: ${SLURM_JOB_GRES:-n/a}"
echo "slurm_job_gpus: ${SLURM_JOB_GPUS:-n/a}"
echo "slurm_step_gpus: ${SLURM_STEP_GPUS:-n/a}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

KITTI_ROOT_OFFICIAL="${KITTI_ROOT_OFFICIAL}" \
RUN_DIRS="${RUN_DIRS}" \
OPENPCDET_ROOT="${OPENPCDET_ROOT}" \
OPENPCDET_CFG="${OPENPCDET_CFG}" \
OPENPCDET_CKPT="${OPENPCDET_CKPT}" \
SETUP_OPENPCDET="${SETUP_OPENPCDET}" \
REBUILD_KITTI_INFOS="${REBUILD_KITTI_INFOS}" \
RUN_ORIGINAL_SANITY="${RUN_ORIGINAL_SANITY}" \
REQUIRE_CUDA="${REQUIRE_CUDA}" \
MAX_FRAMES="${MAX_FRAMES}" \
WORKERS="${WORKERS}" \
EVAL_METRIC="${EVAL_METRIC}" \
REFERENCE_MODE="${REFERENCE_MODE}" \
RECON_QUANT_MODE="${RECON_QUANT_MODE}" \
ORACLE_CLASSES="${ORACLE_CLASSES}" \
ORACLE_DILATE_PX="${ORACLE_DILATE_PX}" \
ADAPTIVE_BG_LEVELS_OVERRIDE="${ADAPTIVE_BG_LEVELS_OVERRIDE}" \
ADAPTIVE_ROI_LEVELS_OVERRIDE="${ADAPTIVE_ROI_LEVELS_OVERRIDE}" \
SUMMARY_CSV="${SUMMARY_CSV}" \
DETAIL_CSV="${DETAIL_CSV}" \
DETECTOR_PAIR_CSV="${DETECTOR_PAIR_CSV}" \
"${BASH_RUNNER[@]}" src/scripts/run_kitti_map_vs_rate.sh

echo "[kitti-map-sbatch] done"
