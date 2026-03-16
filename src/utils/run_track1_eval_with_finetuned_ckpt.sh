#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

EXTRA_TAG=${EXTRA_TAG:-}
KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-}
RUN_DIRS_FILE=${RUN_DIRS_FILE:-}

if [[ -z "${EXTRA_TAG}" ]]; then
  echo "Error: EXTRA_TAG is required." >&2
  exit 1
fi
if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
if [[ -z "${RUN_DIRS_FILE}" || ! -f "${RUN_DIRS_FILE}" ]]; then
  echo "Error: RUN_DIRS_FILE is required and must exist." >&2
  exit 1
fi

OPENPCDET_OUTPUT_ROOT=${OPENPCDET_OUTPUT_ROOT:-third_party/OpenPCDet/output}
REFERENCE_MODE=${REFERENCE_MODE:-identity}
RUN_ORIGINAL_SANITY=${RUN_ORIGINAL_SANITY:-0}
SUMMARY_CSV=${SUMMARY_CSV:-notebooks/kitti_map_vs_rate_summary_${EXTRA_TAG}.csv}
DETAIL_CSV=${DETAIL_CSV:-notebooks/kitti_map_vs_rate_detail_${EXTRA_TAG}.csv}
DETECTOR_PAIR_CSV=${DETECTOR_PAIR_CSV:-notebooks/kitti_map_vs_rate_pairs_${EXTRA_TAG}.csv}

if [[ "${KITTI_ROOT_OFFICIAL}" != /* ]]; then
  KITTI_ROOT_OFFICIAL="${ROOT_DIR}/${KITTI_ROOT_OFFICIAL}"
fi
if [[ "${RUN_DIRS_FILE}" != /* ]]; then
  RUN_DIRS_FILE="${ROOT_DIR}/${RUN_DIRS_FILE}"
fi

mapfile -t RUN_DIR_ARRAY < <(sed '/^[[:space:]]*$/d' "${RUN_DIRS_FILE}")
if [[ "${#RUN_DIR_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: RUN_DIRS_FILE is empty: ${RUN_DIRS_FILE}" >&2
  exit 1
fi
RUN_DIRS=$(IFS=,; echo "${RUN_DIR_ARRAY[*]}")

SEARCH_ROOT="${OPENPCDET_OUTPUT_ROOT}"
if [[ "${SEARCH_ROOT}" != /* ]]; then
  SEARCH_ROOT="${ROOT_DIR}/${SEARCH_ROOT}"
fi
CKPT_DIR="$(find "${SEARCH_ROOT}" -type d -path "*/${EXTRA_TAG}/ckpt" | head -n1)"
if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "Error: could not locate ckpt dir for EXTRA_TAG=${EXTRA_TAG} under ${SEARCH_ROOT}" >&2
  exit 1
fi
CKPT_DIR="$(cd "${CKPT_DIR}" && pwd -P)"

if [[ -f "${CKPT_DIR}/latest_model.pth" ]]; then
  CKPT="${CKPT_DIR}/latest_model.pth"
else
  CKPT="$(ls -1t "${CKPT_DIR}"/*.pth 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Error: could not locate fine-tuned checkpoint file in ${CKPT_DIR}" >&2
  exit 1
fi
CKPT="$(cd "$(dirname "${CKPT}")" && pwd -P)/$(basename "${CKPT}")"

echo "============================================================"
echo "[Track1 Dependent Eval]"
echo "extra_tag: ${EXTRA_TAG}"
echo "reference_mode: ${REFERENCE_MODE}"
echo "kitti_root_official: ${KITTI_ROOT_OFFICIAL}"
echo "run_dirs_file: ${RUN_DIRS_FILE}"
echo "openpcdet_ckpt: ${CKPT}"
echo "summary_csv: ${SUMMARY_CSV}"
echo "detail_csv: ${DETAIL_CSV}"
echo "pair_csv: ${DETECTOR_PAIR_CSV}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

KITTI_ROOT_OFFICIAL="${KITTI_ROOT_OFFICIAL}" \
RUN_DIRS="${RUN_DIRS}" \
OPENPCDET_CKPT="${CKPT}" \
REFERENCE_MODE="${REFERENCE_MODE}" \
RUN_ORIGINAL_SANITY="${RUN_ORIGINAL_SANITY}" \
SUMMARY_CSV="${SUMMARY_CSV}" \
DETAIL_CSV="${DETAIL_CSV}" \
DETECTOR_PAIR_CSV="${DETECTOR_PAIR_CSV}" \
bash src/scripts/run_kitti_map_vs_rate.sh
