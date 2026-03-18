#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

REPO_ROOT="${ROOT_DIR}" \
COMPRESSION_RUN_DIR="${COMPRESSION_RUN_DIR:?COMPRESSION_RUN_DIR is required}" \
RECON_DATA_ROOT="${RECON_DATA_ROOT:?RECON_DATA_ROOT is required}" \
RANGEDET_MODEL_PREFIX="${RANGEDET_MODEL_PREFIX:?RANGEDET_MODEL_PREFIX is required}" \
RANGEDET_TEST_EPOCH="${RANGEDET_TEST_EPOCH:-24}" \
RANGEDET_CONFIG="${RANGEDET_CONFIG:-config/rangedet/rangedet_kitti_car_24e.py}" \
RANGEDET_EXPERIMENT_NAME="${RANGEDET_EXPERIMENT_NAME:-rangedet_kitti_stage_eval}" \
RANGEDET_OUTPUT_ARCHIVE_DIR="${RANGEDET_OUTPUT_ARCHIVE_DIR:-${ROOT_DIR}/logs/rangedet_eval_outputs}" \
RANGEDET_OUTPUT_ARCHIVE_TAG="${RANGEDET_OUTPUT_ARCHIVE_TAG:-$(basename "${RECON_DATA_ROOT}")}" \
KITTI_ROOT="${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}" \
SOURCE_RANGEDET_ROOT="${SOURCE_RANGEDET_ROOT:-${ROOT_DIR}/data/dataset/rangedet_kitti_hq}" \
SPLIT="${SPLIT:-validation}" \
IMG_H="${IMG_H:-64}" IMG_W="${IMG_W:-2048}" \
FOV_UP_DEG="${FOV_UP_DEG:-3.0}" FOV_DOWN_DEG="${FOV_DOWN_DEG:--25.0}" \
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}" \
MAX_FRAMES="${MAX_FRAMES:-512}" \
bash src/scripts/run_rangedet_stage_eval.sh

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  CONDA_PYTHON=python
fi

ARCHIVE_PATH="${RANGEDET_OUTPUT_ARCHIVE_DIR}/${RANGEDET_OUTPUT_ARCHIVE_TAG}_output_dict_${RANGEDET_TEST_EPOCH}e.pkl"
SUMMARY_CSV="${SUMMARY_CSV:-${ROOT_DIR}/logs/${RANGEDET_OUTPUT_ARCHIVE_TAG}_pilot_eval.csv}"

"${CONDA_PYTHON}" src/scripts/eval_rangedet_archive_car_ap.py \
  --source-roidb "${RECON_DATA_ROOT}/${SPLIT}/part-0000.roidb" \
  --archives "${ARCHIVE_PATH}" \
  --output-csv "${SUMMARY_CSV}"

echo "[track2-codec-pilot-eval] summary_csv=${SUMMARY_CSV}"
