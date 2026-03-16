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
TRACK_LABEL=${TRACK_LABEL:-track1_noquant}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}

TRAIN_EPOCHS_NOQUANT=${TRAIN_EPOCHS_NOQUANT:-180}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
TRAIN_LR=${TRAIN_LR:-1e-4}
LATENT_CHANNELS=${LATENT_CHANNELS:-64}
BASE_CHANNELS=${BASE_CHANNELS:-64}
BLOCKS_PER_STAGE=${BLOCKS_PER_STAGE:-1}
RECON_LOSS_MODE=${RECON_LOSS_MODE:-mse}
RECON_RANGE_WEIGHT=${RECON_RANGE_WEIGHT:-1.0}
RECON_XYZ_WEIGHT=${RECON_XYZ_WEIGHT:-1.0}
RECON_REMISSION_WEIGHT=${RECON_REMISSION_WEIGHT:-1.0}
POSITION_BRANCH=${POSITION_BRANCH:-0}
POSITION_LATENT_CHANNELS=${POSITION_LATENT_CHANNELS:-32}
DECODER_TYPE=${DECODER_TYPE:-deconv}
IMPLICIT_HIDDEN_CHANNELS=${IMPLICIT_HIDDEN_CHANNELS:-128}
LAMBDA_RAY_CONSISTENCY=${LAMBDA_RAY_CONSISTENCY:-0.0}
PILLAR_SIDE_BRANCH=${PILLAR_SIDE_BRANCH:-0}
PILLAR_MAX_RAW_POINTS=${PILLAR_MAX_RAW_POINTS:-150000}
PILLAR_SIZE_X=${PILLAR_SIZE_X:-0.24}
PILLAR_SIZE_Y=${PILLAR_SIZE_Y:-0.24}
PILLAR_PFN_HIDDEN_CHANNELS=${PILLAR_PFN_HIDDEN_CHANNELS:-64}
PILLAR_PFN_OUT_CHANNELS=${PILLAR_PFN_OUT_CHANNELS:-128}
PILLAR_BEV_CHANNELS=${PILLAR_BEV_CHANNELS:-128,128,192,256}
PILLAR_BEV_BLOCKS=${PILLAR_BEV_BLOCKS:-2,2,2,3}
PILLAR_FPN_CHANNELS=${PILLAR_FPN_CHANNELS:-128}
PILLAR_BEV_CHANNELS_ARG=${PILLAR_BEV_CHANNELS//-/,}
PILLAR_BEV_BLOCKS_ARG=${PILLAR_BEV_BLOCKS//-/,}

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
NOQUANT_SAVE_DIR="${SAVE_ROOT}/${DATE_TAG}_resnet_noquant_hires_${VARIANT_TAG}_nq"
RUN_DIRS_FILE="${ROOT_DIR}/${LOG_DIR}/${VARIANT_TAG}_run_dirs.txt"
PP_EXTRA_TAG="pp_ft_${VARIANT_TAG}"

FUSION_VARIANT=${FUSION_VARIANT:-}
FUSION_HIDDEN=${FUSION_HIDDEN:-}
DECODER_REFINE_BLOCKS=${DECODER_REFINE_BLOCKS:-}
if [[ -z "${FUSION_VARIANT}" || -z "${FUSION_HIDDEN}" || -z "${DECODER_REFINE_BLOCKS}" ]]; then
  FUSION_VARIANT="none"
  FUSION_HIDDEN="128"
  DECODER_REFINE_BLOCKS="0"
  if [[ "${TRACK_VARIANT}" == "enhanced" ]]; then
    FUSION_VARIANT="rangeformer"
    FUSION_HIDDEN="160"
    DECODER_REFINE_BLOCKS="3"
  fi
fi

echo "============================================================"
echo "[Track1 NoQuant Chain]"
echo "track_label: ${TRACK_LABEL}"
echo "track_variant: ${TRACK_VARIANT}"
echo "track_tag: ${TRACK_TAG}"
echo "python_env: ${PY_ENV_DESC}"
echo "kitti_root: ${KITTI_ROOT}"
echo "identity_root: ${IDENTITY_ROOT}"
echo "projection: ${IMG_H}x${IMG_W} fov_up=${FOV_UP_DEG} fov_down=${FOV_DOWN_DEG} unprojection=${UNPROJECTION_MODE}"
echo "noquant_save_dir: ${NOQUANT_SAVE_DIR}"
echo "train_epochs_noquant: ${TRAIN_EPOCHS_NOQUANT}"
echo "latent_channels: ${LATENT_CHANNELS}"
echo "base_channels: ${BASE_CHANNELS}"
echo "blocks_per_stage: ${BLOCKS_PER_STAGE}"
echo "recon_loss_mode: ${RECON_LOSS_MODE}"
echo "recon_channel_weights: range=${RECON_RANGE_WEIGHT} xyz=${RECON_XYZ_WEIGHT} remission=${RECON_REMISSION_WEIGHT}"
echo "position_branch: ${POSITION_BRANCH}"
echo "position_latent_channels: ${POSITION_LATENT_CHANNELS}"
echo "decoder_type: ${DECODER_TYPE}"
echo "implicit_hidden_channels: ${IMPLICIT_HIDDEN_CHANNELS}"
echo "lambda_ray_consistency: ${LAMBDA_RAY_CONSISTENCY}"
echo "pillar_side_branch: ${PILLAR_SIDE_BRANCH}"
echo "pillar_max_raw_points: ${PILLAR_MAX_RAW_POINTS}"
echo "pillar_size: (${PILLAR_SIZE_X}, ${PILLAR_SIZE_Y})"
echo "pillar_pfn_hidden_channels: ${PILLAR_PFN_HIDDEN_CHANNELS}"
echo "pillar_pfn_out_channels: ${PILLAR_PFN_OUT_CHANNELS}"
echo "pillar_bev_channels: ${PILLAR_BEV_CHANNELS_ARG}"
echo "pillar_bev_blocks: ${PILLAR_BEV_BLOCKS_ARG}"
echo "pillar_fpn_channels: ${PILLAR_FPN_CHANNELS}"
echo "fusion_variant: ${FUSION_VARIANT}"
echo "fusion_hidden: ${FUSION_HIDDEN}"
echo "decoder_refine_blocks: ${DECODER_REFINE_BLOCKS}"
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
  --latent_channels "${LATENT_CHANNELS}"
  --base_channels "${BASE_CHANNELS}"
  --blocks_per_stage "${BLOCKS_PER_STAGE}"
  --recon_loss_mode "${RECON_LOSS_MODE}"
  --recon_range_weight "${RECON_RANGE_WEIGHT}"
  --recon_xyz_weight "${RECON_XYZ_WEIGHT}"
  --recon_remission_weight "${RECON_REMISSION_WEIGHT}"
  --position_latent_channels "${POSITION_LATENT_CHANNELS}"
  --decoder_type "${DECODER_TYPE}"
  --implicit_hidden_channels "${IMPLICIT_HIDDEN_CHANNELS}"
  --lambda_ray_consistency "${LAMBDA_RAY_CONSISTENCY}"
  --pillar_max_raw_points "${PILLAR_MAX_RAW_POINTS}"
  --pillar_size_x "${PILLAR_SIZE_X}"
  --pillar_size_y "${PILLAR_SIZE_Y}"
  --pillar_pfn_hidden_channels "${PILLAR_PFN_HIDDEN_CHANNELS}"
  --pillar_pfn_out_channels "${PILLAR_PFN_OUT_CHANNELS}"
  --pillar_bev_channels "${PILLAR_BEV_CHANNELS_ARG}"
  --pillar_bev_blocks "${PILLAR_BEV_BLOCKS_ARG}"
  --pillar_fpn_channels "${PILLAR_FPN_CHANNELS}"
  --feature_fusion_variant "${FUSION_VARIANT}"
  --feature_fusion_hidden_channels "${FUSION_HIDDEN}"
  --decoder_post_refine_blocks "${DECODER_REFINE_BLOCKS}"
)
if [[ "${POSITION_BRANCH}" == "1" ]]; then
  COMMON_ARGS+=(--position_branch)
fi
if [[ "${PILLAR_SIDE_BRANCH}" == "1" ]]; then
  COMMON_ARGS+=(--pillar_side_branch)
fi

echo "[track1-noquant] pure encoder-decoder training ..."
"${PYTHON_RUNNER[@]}" src/main_train.py \
  "${COMMON_ARGS[@]}" \
  --quantizer_mode none \
  --epochs "${TRAIN_EPOCHS_NOQUANT}" \
  --no_teacher \
  --lambda_rate 0.0 \
  --lambda_distill 0.0 \
  --lambda_importance 0.0 \
  --lambda_imp_separation 0.0 \
  --loss_recipe legacy \
  --importance_head_type basic \
  --importance_hidden_channels 32 \
  --run_id "${VARIANT_TAG}_nq" \
  --save_dir "${NOQUANT_SAVE_DIR}"

printf "%s\n" "${NOQUANT_SAVE_DIR}" > "${RUN_DIRS_FILE}"

echo "[track1-noquant] exporting identity dataset ..."
"${PYTHON_RUNNER[@]}" src/utils/export_kitti_identity_dataset.py \
  --source_root "${KITTI_ROOT}" \
  --output_root "${IDENTITY_ROOT}" \
  --img_h "${IMG_H}" \
  --img_w "${IMG_W}" \
  --fov_up_deg "${FOV_UP_DEG}" \
  --fov_down_deg "${FOV_DOWN_DEG}" \
  --unprojection_mode "${UNPROJECTION_MODE}"

echo "[track1-noquant] PointPillars fine-tune on identity domain ..."
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

echo "[track1-noquant] evaluating reconstructed endpoint ..."
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

echo "[track1-noquant] done"
