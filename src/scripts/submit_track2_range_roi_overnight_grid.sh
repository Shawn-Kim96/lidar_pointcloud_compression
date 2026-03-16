#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

BASE_TAG=${BASE_TAG:-$(date +%y%m%d)_track2grid}
CONDA_ENV=${CONDA_ENV:-lidarcomp311}
RUN_DIR=${RUN_DIR:-data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/results/track2_range_roi}
SBATCH_PARTITION=${SBATCH_PARTITION:-gpuqm}
SBATCH_CPUS=${SBATCH_CPUS:-4}
SBATCH_GRES=${SBATCH_GRES:-gpu:1}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-1}
DRY_RUN=${DRY_RUN:-0}

MANIFEST_PATH=${MANIFEST_PATH:-logs/${BASE_TAG}_manifest.csv}
RESULTS_CSV=${RESULTS_CSV:-logs/${BASE_TAG}_results.csv}
RESULTS_MD=${RESULTS_MD:-logs/${BASE_TAG}_results.md}
PLAN_MD=${PLAN_MD:-docs/report/${BASE_TAG}_submission_plan.md}

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
# Track 2 Overnight Grid

- Base tag: \`${BASE_TAG}\`
- Run dir: \`${RUN_DIR}\`
- KITTI root: \`${KITTI_ROOT}\`
- Manifest: \`${MANIFEST_PATH}\`
- Results CSV: \`${RESULTS_CSV}\`
- Results MD: \`${RESULTS_MD}\`
- Policy: full train/val split, frozen compression backbone

## Sweep

- Heads: \`refine\`, \`deep\`
- Hidden channels: \`96\`, \`128\`, \`192\`
- Learning rates: \`1e-4\`, \`2e-4\`
- Epochs:
  - \`refine\`: \`120\`, \`180\`
  - \`deep\`: \`120\`, \`180\`

Total jobs: \`24\`

## Result Collection

\`\`\`bash
python3 src/scripts/collect_track2_range_roi_results.py \\
  --manifest ${MANIFEST_PATH} \\
  --output_csv ${RESULTS_CSV} \\
  --output_md ${RESULTS_MD}
\`\`\`
EOF

submit_one() {
  local track2_tag="$1"
  local head="$2"
  local hidden="$3"
  local epochs="$4"
  local lr="$5"

  TRACK2_TAG="${track2_tag}" \
  HEAD_SWEEP="${head}" \
  HIDDEN_CHANNELS="${hidden}" \
  EPOCHS="${epochs}" \
  LR="${lr}" \
  WEIGHT_DECAY=1e-4 \
  MAX_TRAIN_SAMPLES=0 \
  MAX_VAL_SAMPLES=0 \
  BATCH_SIZE=2 \
  WORKERS=4 \
  FREEZE_BACKBONE=1 \
  CONDA_ENV="${CONDA_ENV}" \
  RUN_DIR="${RUN_DIR}" \
  KITTI_ROOT="${KITTI_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  SBATCH_PARTITION="${SBATCH_PARTITION}" \
  SBATCH_CPUS="${SBATCH_CPUS}" \
  SBATCH_GRES="${SBATCH_GRES}" \
  SBATCH_TIME="${SBATCH_TIME}" \
  SKIP_DOWNLOAD="${SKIP_DOWNLOAD}" \
  DRY_RUN="${DRY_RUN}" \
  MANIFEST_PATH="${MANIFEST_PATH}" \
  bash "${ROOT_DIR}/src/scripts/submit_track2_range_roi_pilot.sh"
}

echo "============================================================"
echo "[Track2 Overnight Grid]"
echo "base_tag: ${BASE_TAG}"
echo "manifest: ${MANIFEST_PATH}"
echo "plan_md: ${PLAN_MD}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

for head in refine deep; do
  for hidden in 96 128 192; do
    for lr in 1e-4 2e-4; do
      for epochs in 120 180; do
        tag="${BASE_TAG}_${head}_h${hidden}_lr${lr//-/m}_e${epochs}"
        submit_one "${tag}" "${head}" "${hidden}" "${epochs}" "${lr}"
      done
    done
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 src/scripts/collect_track2_range_roi_results.py \
    --manifest "${MANIFEST_PATH}" \
    --output_csv "${RESULTS_CSV}" \
    --output_md "${RESULTS_MD}"
fi

echo "[track2-grid] queued"
echo "[track2-grid] manifest=${MANIFEST_PATH}"
echo "[track2-grid] results_csv=${RESULTS_CSV}"
echo "[track2-grid] results_md=${RESULTS_MD}"
