#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  PYTHON_RUNNER=("${CONDA_PREFIX_DIR}/bin/python")
  BASH_RUNNER=(bash)
  PY_ENV_DESC="python:${CONDA_PREFIX_DIR}/bin/python"
elif command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  BASH_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" bash)
  PY_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  BASH_RUNNER=(bash)
  PY_ENV_DESC="python:$(command -v python)"
fi

TRACK_TAG=${TRACK_TAG:-$(date +%y%m%d_%H%M%S)}
TRACK_VARIANT=${TRACK_VARIANT:-baseline}
TRACK_LABEL=${TRACK_LABEL:-track1_chain}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}

TRAIN_EPOCHS_STAGE0=${TRAIN_EPOCHS_STAGE0:-40}
TRAIN_EPOCHS_STAGE1=${TRAIN_EPOCHS_STAGE1:-60}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
TRAIN_LR=${TRAIN_LR:-1e-4}

PP_FT_EPOCHS=${PP_FT_EPOCHS:-20}
PP_FT_BATCH_SIZE=${PP_FT_BATCH_SIZE:-4}
PP_FT_WORKERS=${PP_FT_WORKERS:-4}

OPENPCDET_ROOT=${OPENPCDET_ROOT:-third_party/OpenPCDet}
OPENPCDET_CFG=${OPENPCDET_CFG:-tools/cfgs/kitti_models/pointpillar.yaml}
OPENPCDET_PRETRAIN_CKPT=${OPENPCDET_PRETRAIN_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "${LOG_DIR}" notebooks docs/report

if [[ "${KITTI_ROOT}" != /* ]]; then
  KITTI_ROOT="${ROOT_DIR}/${KITTI_ROOT}"
fi
if [[ ! -d "${KITTI_ROOT}" ]]; then
  echo "Error: KITTI_ROOT not found: ${KITTI_ROOT}" >&2
  exit 1
fi

IDENTITY_ROOT=${IDENTITY_ROOT:-${ROOT_DIR}/data/dataset/kitti_identity_h${IMG_H}w${IMG_W}_${UNPROJECTION_MODE}}

DATE_TAG=$(date +%y%m%d)
VARIANT_TAG="${TRACK_LABEL}_${TRACK_VARIANT}_${TRACK_TAG}"
SAVE_ROOT="${ROOT_DIR}/data/results/experiments"
STAGE0_SAVE_DIR="${SAVE_ROOT}/${DATE_TAG}_resnet_uniform_q6_hires_${VARIANT_TAG}_s0"
STAGE1_SAVE_DIR="${SAVE_ROOT}/${DATE_TAG}_resnet_adaptive_hires_${VARIANT_TAG}_s1"
RUN_DIRS_FILE="${ROOT_DIR}/${LOG_DIR}/${VARIANT_TAG}_run_dirs.txt"
PP_EXTRA_TAG="pp_ft_${VARIANT_TAG}"

FUSION_VARIANT="none"
FUSION_HIDDEN="128"
DECODER_REFINE_BLOCKS="0"
IMP_HEAD_TYPE="basic"
IMP_HIDDEN="64"
if [[ "${TRACK_VARIANT}" == "enhanced" ]]; then
  FUSION_VARIANT="rangeformer"
  FUSION_HIDDEN="160"
  DECODER_REFINE_BLOCKS="3"
  IMP_HEAD_TYPE="rangeformer"
  IMP_HIDDEN="96"
fi

echo "============================================================"
echo "[Track1 Hires Chain]"
echo "track_label: ${TRACK_LABEL}"
echo "track_variant: ${TRACK_VARIANT}"
echo "track_tag: ${TRACK_TAG}"
echo "python_env: ${PY_ENV_DESC}"
echo "kitti_root: ${KITTI_ROOT}"
echo "identity_root: ${IDENTITY_ROOT}"
echo "projection: ${IMG_H}x${IMG_W} fov_up=${FOV_UP_DEG} fov_down=${FOV_DOWN_DEG} unprojection=${UNPROJECTION_MODE}"
echo "stage0_save_dir: ${STAGE0_SAVE_DIR}"
echo "stage1_save_dir: ${STAGE1_SAVE_DIR}"
echo "fusion_variant: ${FUSION_VARIANT}"
echo "fusion_hidden: ${FUSION_HIDDEN}"
echo "decoder_refine_blocks: ${DECODER_REFINE_BLOCKS}"
echo "importance_head: ${IMP_HEAD_TYPE} hidden=${IMP_HIDDEN}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

COMMON_ARGS=(
  --backbone resnet
  --dataset_type kitti3dobject
  --data_root "${KITTI_ROOT}"
  --kitti_split train
  --img_h "${IMG_H}"
  --img_w "${IMG_W}"
  --fov_up_deg "${FOV_UP_DEG}"
  --fov_down_deg "${FOV_DOWN_DEG}"
  --batch_size "${TRAIN_BATCH_SIZE}"
  --num_workers "${TRAIN_WORKERS}"
  --lr "${TRAIN_LR}"
  --feature_fusion_variant "${FUSION_VARIANT}"
  --feature_fusion_hidden_channels "${FUSION_HIDDEN}"
  --decoder_post_refine_blocks "${DECODER_REFINE_BLOCKS}"
)

echo "[track1-chain] stage0 uniform training ..."
"${PYTHON_RUNNER[@]}" src/main_train.py \
  "${COMMON_ARGS[@]}" \
  --quantizer_mode uniform \
  --quant_bits 6 \
  --epochs "${TRAIN_EPOCHS_STAGE0}" \
  --no_teacher \
  --lambda_rate 0.0 \
  --lambda_distill 0.0 \
  --lambda_importance 0.0 \
  --loss_recipe legacy \
  --importance_head_type basic \
  --importance_hidden_channels 32 \
  --run_id "${VARIANT_TAG}_s0" \
  --save_dir "${STAGE0_SAVE_DIR}"

echo "[track1-chain] stage1 adaptive training ..."
"${PYTHON_RUNNER[@]}" src/main_train.py \
  "${COMMON_ARGS[@]}" \
  --quantizer_mode adaptive \
  --epochs "${TRAIN_EPOCHS_STAGE1}" \
  --no_teacher \
  --lambda_rate 0.02 \
  --lambda_distill 0.0 \
  --lambda_importance 1.0 \
  --lambda_imp_separation 0.2 \
  --loss_recipe balanced_v2 \
  --rate_loss_mode normalized_bg \
  --importance_loss_mode weighted_bce \
  --importance_pos_weight_mode auto \
  --importance_pos_weight_max 50.0 \
  --importance_head_type "${IMP_HEAD_TYPE}" \
  --importance_hidden_channels "${IMP_HIDDEN}" \
  --run_id "${VARIANT_TAG}_s1" \
  --save_dir "${STAGE1_SAVE_DIR}"

printf "%s\n%s\n" "${STAGE0_SAVE_DIR}" "${STAGE1_SAVE_DIR}" > "${RUN_DIRS_FILE}"

echo "[track1-chain] exporting identity dataset ..."
"${PYTHON_RUNNER[@]}" src/utils/export_kitti_identity_dataset.py \
  --source_root "${KITTI_ROOT}" \
  --output_root "${IDENTITY_ROOT}" \
  --img_h "${IMG_H}" \
  --img_w "${IMG_W}" \
  --fov_up_deg "${FOV_UP_DEG}" \
  --fov_down_deg "${FOV_DOWN_DEG}" \
  --unprojection_mode "${UNPROJECTION_MODE}"

echo "[track1-chain] PointPillars fine-tune on identity domain ..."
KITTI_ROOT_OFFICIAL="${IDENTITY_ROOT}" \
CONDA_ENV="${CONDA_ENV}" \
ROOT_DIR="${ROOT_DIR}" \
OPENPCDET_ROOT="${OPENPCDET_ROOT}" \
OPENPCDET_CFG="${OPENPCDET_CFG}" \
OPENPCDET_PRETRAIN_CKPT="${OPENPCDET_PRETRAIN_CKPT}" \
EPOCHS="${PP_FT_EPOCHS}" \
BATCH_SIZE="${PP_FT_BATCH_SIZE}" \
WORKERS="${PP_FT_WORKERS}" \
EXTRA_TAG="${PP_EXTRA_TAG}" \
RUN_FINAL_TEST=1 \
"${BASH_RUNNER[@]}" src/scripts/run_pointpillar_kitti_finetune.sh

TRACK1_SUMMARY_CSV="notebooks/kitti_map_vs_rate_summary_${VARIANT_TAG}.csv"
TRACK1_DETAIL_CSV="notebooks/kitti_map_vs_rate_detail_${VARIANT_TAG}.csv"
TRACK1_PAIR_CSV="notebooks/kitti_map_vs_rate_pairs_${VARIANT_TAG}.csv"

echo "[track1-chain] evaluating reconstructed endpoint ..."
ROOT_DIR="${ROOT_DIR}" \
EXTRA_TAG="${PP_EXTRA_TAG}" \
KITTI_ROOT_OFFICIAL="${KITTI_ROOT}" \
RUN_DIRS_FILE="${RUN_DIRS_FILE}" \
REFERENCE_MODE="identity" \
RUN_ORIGINAL_SANITY=0 \
IMG_H="${IMG_H}" \
IMG_W="${IMG_W}" \
FOV_UP_DEG="${FOV_UP_DEG}" \
FOV_DOWN_DEG="${FOV_DOWN_DEG}" \
UNPROJECTION_MODE="${UNPROJECTION_MODE}" \
SUMMARY_CSV="${TRACK1_SUMMARY_CSV}" \
DETAIL_CSV="${TRACK1_DETAIL_CSV}" \
DETECTOR_PAIR_CSV="${TRACK1_PAIR_CSV}" \
"${BASH_RUNNER[@]}" src/utils/run_track1_eval_with_finetuned_ckpt.sh

echo "[track1-chain] done"
