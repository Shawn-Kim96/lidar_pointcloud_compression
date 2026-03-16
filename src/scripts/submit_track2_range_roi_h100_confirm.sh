#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

BASE_TAG=${BASE_TAG:-$(date +%y%m%d)_track2h100}
CONDA_ENV=${CONDA_ENV:-lidarcomp311}
RUN_DIR=${RUN_DIR:-data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
MANIFEST_PATH=${MANIFEST_PATH:-logs/${BASE_TAG}_manifest.csv}
RESULTS_CSV=${RESULTS_CSV:-logs/${BASE_TAG}_results.csv}
RESULTS_MD=${RESULTS_MD:-logs/${BASE_TAG}_results.md}
PLAN_MD=${PLAN_MD:-docs/report/${BASE_TAG}_plan.md}

if [[ "${MANIFEST_PATH}" != /* ]]; then
  MANIFEST_PATH="${ROOT_DIR}/${MANIFEST_PATH}"
fi
if [[ "${RESULTS_CSV}" != /* ]]; then
  RESULTS_CSV="${ROOT_DIR}/${RESULTS_CSV}"
fi
if [[ "${RESULTS_MD}" != /* ]]; then
  RESULTS_MD="${ROOT_DIR}/${RESULTS_MD}"
fi
if [[ "${PLAN_MD}" != /* ]]; then
  PLAN_MD="${ROOT_DIR}/${PLAN_MD}"
fi

mkdir -p "$(dirname "${MANIFEST_PATH}")" "$(dirname "${PLAN_MD}")"

cat > "${PLAN_MD}" <<EOF
# Track 2 H100 Confirmatory Sweep

- Base tag: \`${BASE_TAG}\`
- Goal: remove heterogeneous GPU-memory confounding and the 24h walltime bottleneck from the broad grid.
- Partition: \`gpuql\`
- Nodes: \`g16,g18,g19\` (H100-only)
- Requested walltime: \`48:00:00\`
- Batch size: \`2\`
- Manifest: \`${MANIFEST_PATH}\`
- Results CSV: \`${RESULTS_CSV}\`
- Results MD: \`${RESULTS_MD}\`

## Jobs

- refine: \`h96/e120\`, \`h128/e120\`, \`h192/e120\`, \`h128/e180\`, \`h192/e180\`
- deep: \`h96/e120\`, \`h128/e120\`, \`h192/e120\`
EOF

submit_one() {
  local track2_tag="$1"
  local head="$2"
  local hidden="$3"
  local epochs="$4"

  TRACK2_TAG="${track2_tag}" \
  HEAD_SWEEP="${head}" \
  HIDDEN_CHANNELS="${hidden}" \
  EPOCHS="${epochs}" \
  LR=1e-4 \
  WEIGHT_DECAY=1e-4 \
  BATCH_SIZE=2 \
  WORKERS=4 \
  MAX_TRAIN_SAMPLES=0 \
  MAX_VAL_SAMPLES=0 \
  FREEZE_BACKBONE=1 \
  DEVICE=auto \
  OUTPUT_ROOT=data/results/track2_range_roi \
  SBATCH_PARTITION=gpuql \
  SBATCH_TIME=48:00:00 \
  SBATCH_NODELIST=g16,g18,g19 \
  SKIP_DOWNLOAD=1 \
  MANIFEST_PATH="${MANIFEST_PATH}" \
  CONDA_ENV="${CONDA_ENV}" \
  RUN_DIR="${RUN_DIR}" \
  KITTI_ROOT="${KITTI_ROOT}" \
  bash "${ROOT_DIR}/src/scripts/submit_track2_range_roi_pilot.sh"
}

echo "============================================================"
echo "[Track2 H100 Confirm]"
echo "base_tag: ${BASE_TAG}"
echo "manifest: ${MANIFEST_PATH}"
echo "plan_md: ${PLAN_MD}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

submit_one "${BASE_TAG}_refine_h96_e120" refine 96 120
submit_one "${BASE_TAG}_refine_h128_e120" refine 128 120
submit_one "${BASE_TAG}_refine_h192_e120" refine 192 120
submit_one "${BASE_TAG}_refine_h128_e180" refine 128 180
submit_one "${BASE_TAG}_refine_h192_e180" refine 192 180
submit_one "${BASE_TAG}_deep_h96_e120" deep 96 120
submit_one "${BASE_TAG}_deep_h128_e120" deep 128 120
submit_one "${BASE_TAG}_deep_h192_e120" deep 192 120

python3 src/scripts/collect_track2_range_roi_results.py \
  --manifest "${MANIFEST_PATH}" \
  --output_csv "${RESULTS_CSV}" \
  --output_md "${RESULTS_MD}"

echo "[track2-h100] queued"
echo "[track2-h100] manifest=${MANIFEST_PATH}"
