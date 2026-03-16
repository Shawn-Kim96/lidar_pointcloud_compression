#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
CONDA_ENV="${CONDA_ENV:-lidarcomp311}"
CONDA_PYTHON="${CONDA_PYTHON:-$HOME/miniconda3/envs/${CONDA_ENV}/bin/python}"
ARCHIVES_CSV="${ARCHIVES_CSV:-}"
OUTPUT_TAG="${OUTPUT_TAG:-rangedet_compare}"
SOURCE_ROIDB="${SOURCE_ROIDB:-$REPO_ROOT/data/dataset/rangedet_kitti_hq/validation/part-0000.roidb}"
KITTI_ROOT="${KITTI_ROOT:-$REPO_ROOT/data/dataset/kitti3dobject}"
KITTI_INFOS="${KITTI_INFOS:-$REPO_ROOT/data/dataset/kitti3dobject/kitti_infos_val.pkl}"

if [[ -z "${ARCHIVES_CSV}" ]]; then
  echo "Error: ARCHIVES_CSV is required." >&2
  exit 1
fi
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  echo "Error: python not found: ${CONDA_PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

IFS=',' read -r -a ARCHIVES <<< "${ARCHIVES_CSV}"
if [[ "${#ARCHIVES[@]}" -eq 0 ]]; then
  echo "Error: no archives parsed from ARCHIVES_CSV=${ARCHIVES_CSV}" >&2
  exit 1
fi

CUSTOM_CSV="logs/${OUTPUT_TAG}_archive_car_ap_summary.csv"
OFFICIAL_CSV="logs/${OUTPUT_TAG}_kitti_official_summary.csv"

echo "============================================================"
echo "[RangeDet Archive Compare]"
echo "output_tag: ${OUTPUT_TAG}"
echo "source_roidb: ${SOURCE_ROIDB}"
echo "kitti_root: ${KITTI_ROOT}"
echo "kitti_infos: ${KITTI_INFOS}"
echo "archives:"
for a in "${ARCHIVES[@]}"; do
  echo "  - ${a}"
done
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

"${CONDA_PYTHON}" src/scripts/eval_rangedet_archive_car_ap.py \
  --source-roidb "${SOURCE_ROIDB}" \
  --archives "${ARCHIVES[@]}" \
  --output-csv "${CUSTOM_CSV}"

"${CONDA_PYTHON}" src/scripts/eval_rangedet_archive_kitti_official.py \
  --source-roidb "${SOURCE_ROIDB}" \
  --kitti-root "${KITTI_ROOT}" \
  --kitti-infos "${KITTI_INFOS}" \
  --archives "${ARCHIVES[@]}" \
  --output-csv "${OFFICIAL_CSV}"

echo "[rangedet-archive-compare] custom_csv=${CUSTOM_CSV}"
echo "[rangedet-archive-compare] official_csv=${OFFICIAL_CSV}"
