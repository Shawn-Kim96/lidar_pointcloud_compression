#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-}
RUN_DIRS=${RUN_DIRS:-}
OPENPCDET_CKPT=${OPENPCDET_CKPT:-}

if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
if [[ -z "${RUN_DIRS}" ]]; then
  echo "Error: RUN_DIRS is required." >&2
  exit 1
fi
if [[ -z "${OPENPCDET_CKPT}" ]]; then
  echo "Error: OPENPCDET_CKPT is required." >&2
  exit 1
fi

BG_LEVELS_CSV=${BG_LEVELS_CSV:-24,32,48,64}
ROI_LEVELS_OVERRIDE=${ROI_LEVELS_OVERRIDE:-256}
SWEEP_TAG=${SWEEP_TAG:-$(date +%y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/logs/track1_quant_diag/${SWEEP_TAG}}

REFERENCE_MODE=${REFERENCE_MODE:-identity}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}
RUN_ORIGINAL_SANITY=${RUN_ORIGINAL_SANITY:-0}
WORKERS=${WORKERS:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
EVAL_METRIC=${EVAL_METRIC:-kitti}
UPDATE_PAPER_TABLE=${UPDATE_PAPER_TABLE:-0}
ORACLE_CLASSES=${ORACLE_CLASSES:-Car,Pedestrian,Cyclist}
ORACLE_DILATE_PX=${ORACLE_DILATE_PX:-0}

mkdir -p "${OUTPUT_DIR}"

echo "[track1-bg-sweep] output_dir=${OUTPUT_DIR}"
echo "[track1-bg-sweep] run_dirs=${RUN_DIRS}"
echo "[track1-bg-sweep] bg_levels_csv=${BG_LEVELS_CSV}"
echo "[track1-bg-sweep] roi_levels_override=${ROI_LEVELS_OVERRIDE}"
echo "[track1-bg-sweep] projection=${IMG_H}x${IMG_W} unprojection=${UNPROJECTION_MODE}"

IFS=',' read -r -a BG_ARR <<< "${BG_LEVELS_CSV}"
for raw_bg in "${BG_ARR[@]}"; do
  bg="${raw_bg//[[:space:]]/}"
  if [[ -z "${bg}" ]]; then
    continue
  fi
  if ! [[ "${bg}" =~ ^[0-9]+$ ]]; then
    echo "[track1-bg-sweep][warn] skipping invalid bg level: ${raw_bg}" >&2
    continue
  fi

  summary_csv="${OUTPUT_DIR}/summary_bg${bg}.csv"
  detail_csv="${OUTPUT_DIR}/detail_bg${bg}.csv"
  pair_csv="${OUTPUT_DIR}/pairs_bg${bg}.csv"
  log_file="${OUTPUT_DIR}/run_bg${bg}.log"

  echo "[track1-bg-sweep] running bg=${bg} -> ${summary_csv}"
  (
    KITTI_ROOT_OFFICIAL="${KITTI_ROOT_OFFICIAL}" \
    RUN_DIRS="${RUN_DIRS}" \
    OPENPCDET_CKPT="${OPENPCDET_CKPT}" \
    REFERENCE_MODE="${REFERENCE_MODE}" \
    RECON_QUANT_MODE="native" \
    ORACLE_CLASSES="${ORACLE_CLASSES}" \
    ORACLE_DILATE_PX="${ORACLE_DILATE_PX}" \
    ADAPTIVE_BG_LEVELS_OVERRIDE="${bg}" \
    ADAPTIVE_ROI_LEVELS_OVERRIDE="${ROI_LEVELS_OVERRIDE}" \
    IMG_H="${IMG_H}" \
    IMG_W="${IMG_W}" \
    UNPROJECTION_MODE="${UNPROJECTION_MODE}" \
    RUN_ORIGINAL_SANITY="${RUN_ORIGINAL_SANITY}" \
    WORKERS="${WORKERS}" \
    MAX_FRAMES="${MAX_FRAMES}" \
    EVAL_METRIC="${EVAL_METRIC}" \
    UPDATE_PAPER_TABLE="${UPDATE_PAPER_TABLE}" \
    SUMMARY_CSV="${summary_csv}" \
    DETAIL_CSV="${detail_csv}" \
    DETECTOR_PAIR_CSV="${pair_csv}" \
    bash src/scripts/run_kitti_map_vs_rate.sh
  ) | tee "${log_file}"
done

echo "[track1-bg-sweep] done"
