#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  PYTHON_RUNNER=("${CONDA_PREFIX_DIR}/bin/python")
  PY_ENV_DESC="python:${CONDA_PREFIX_DIR}/bin/python"
else
  PYTHON_RUNNER=(python)
  PY_ENV_DESC="python:$(command -v python)"
fi

PILOT_TAG=${PILOT_TAG:-$(date +%y%m%d_%H%M%S)}
PILOT_NAME=${PILOT_NAME:-track2_pilot}
RUN_ID=${RUN_ID:-${PILOT_NAME}_${PILOT_TAG}}
KITTI_ROOT=${KITTI_ROOT:-${ROOT_DIR}/data/dataset/kitti3dobject}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-1024}
LATENT_CHANNELS=${LATENT_CHANNELS:-64}
BASE_CHANNELS=${BASE_CHANNELS:-64}
BLOCKS_PER_STAGE=${BLOCKS_PER_STAGE:-1}
DECODER_TYPE=${DECODER_TYPE:-deconv}
DECODER_REFINE_BLOCKS=${DECODER_REFINE_BLOCKS:-0}
FEATURE_FUSION_VARIANT=${FEATURE_FUSION_VARIANT:-none}
FEATURE_FUSION_HIDDEN=${FEATURE_FUSION_HIDDEN:-128}
RECON_LOSS_MODE=${RECON_LOSS_MODE:-masked_channel_weighted}
RECON_RANGE_WEIGHT=${RECON_RANGE_WEIGHT:-1.0}
RECON_XYZ_WEIGHT=${RECON_XYZ_WEIGHT:-1.0}
RECON_REMISSION_WEIGHT=${RECON_REMISSION_WEIGHT:-0.5}
LAMBDA_RAY_CONSISTENCY=${LAMBDA_RAY_CONSISTENCY:-0.0}
MASK_HEAD=${MASK_HEAD:-0}
MASK_HEAD_HIDDEN_CHANNELS=${MASK_HEAD_HIDDEN_CHANNELS:-32}
MASK_HEAD_GATE_THRESHOLD=${MASK_HEAD_GATE_THRESHOLD:-0.5}
LAMBDA_VALID_MASK=${LAMBDA_VALID_MASK:-0.0}
LAMBDA_VALID_MASK_DICE=${LAMBDA_VALID_MASK_DICE:-0.0}
LAMBDA_RANGE_GRAD_ROW=${LAMBDA_RANGE_GRAD_ROW:-0.0}
LAMBDA_RANGE_GRAD_COL=${LAMBDA_RANGE_GRAD_COL:-0.0}
LAMBDA_ROW_PROFILE=${LAMBDA_ROW_PROFILE:-0.0}
DETECTOR_AUX_HEAD=${DETECTOR_AUX_HEAD:-0}
DETECTOR_AUX_HIDDEN_CHANNELS=${DETECTOR_AUX_HIDDEN_CHANNELS:-32}
LAMBDA_DETECTOR_TARGET=${LAMBDA_DETECTOR_TARGET:-0.0}
TEACHER_TARGET_ROOT=${TEACHER_TARGET_ROOT:-}
SAVE_ROOT=${SAVE_ROOT:-${ROOT_DIR}/data/results/experiments}
DATE_TAG=$(date +%y%m%d)
SAVE_DIR=${SAVE_DIR:-${SAVE_ROOT}/${DATE_TAG}_track2_codec_${RUN_ID}}

mkdir -p "${SAVE_ROOT}" logs

echo "============================================================"
echo "[Track2 Codec Pilot]"
echo "pilot_name: ${PILOT_NAME}"
echo "pilot_tag: ${PILOT_TAG}"
echo "python_env: ${PY_ENV_DESC}"
echo "save_dir: ${SAVE_DIR}"
echo "decoder_type: ${DECODER_TYPE}"
echo "mask_head: ${MASK_HEAD}"
echo "detector_aux_head: ${DETECTOR_AUX_HEAD}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "teacher_target_root: ${TEACHER_TARGET_ROOT:-none}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

ARGS=(
  src/main_train.py
  --backbone resnet
  --quantizer_mode none
  --dataset_type kitti3dobject
  --data_root "${KITTI_ROOT}"
  --kitti_split train
  --img_h "${IMG_H}"
  --img_w "${IMG_W}"
  --fov_up_deg "${FOV_UP_DEG}"
  --fov_down_deg "${FOV_DOWN_DEG}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${WORKERS}"
  --lr "${LR}"
  --max_train_frames "${MAX_TRAIN_FRAMES}"
  --latent_channels "${LATENT_CHANNELS}"
  --base_channels "${BASE_CHANNELS}"
  --blocks_per_stage "${BLOCKS_PER_STAGE}"
  --decoder_type "${DECODER_TYPE}"
  --decoder_post_refine_blocks "${DECODER_REFINE_BLOCKS}"
  --feature_fusion_variant "${FEATURE_FUSION_VARIANT}"
  --feature_fusion_hidden_channels "${FEATURE_FUSION_HIDDEN}"
  --recon_loss_mode "${RECON_LOSS_MODE}"
  --recon_range_weight "${RECON_RANGE_WEIGHT}"
  --recon_xyz_weight "${RECON_XYZ_WEIGHT}"
  --recon_remission_weight "${RECON_REMISSION_WEIGHT}"
  --lambda_ray_consistency "${LAMBDA_RAY_CONSISTENCY}"
  --lambda_valid_mask "${LAMBDA_VALID_MASK}"
  --lambda_valid_mask_dice "${LAMBDA_VALID_MASK_DICE}"
  --lambda_range_grad_row "${LAMBDA_RANGE_GRAD_ROW}"
  --lambda_range_grad_col "${LAMBDA_RANGE_GRAD_COL}"
  --lambda_row_profile "${LAMBDA_ROW_PROFILE}"
  --lambda_detector_target "${LAMBDA_DETECTOR_TARGET}"
  --mask_head_hidden_channels "${MASK_HEAD_HIDDEN_CHANNELS}"
  --mask_head_gate_threshold "${MASK_HEAD_GATE_THRESHOLD}"
  --detector_aux_hidden_channels "${DETECTOR_AUX_HIDDEN_CHANNELS}"
  --save_dir "${SAVE_DIR}"
  --run_id "${RUN_ID}"
  --no_teacher
)

if [[ "${MASK_HEAD}" == "1" ]]; then
  ARGS+=(--mask_head)
fi
if [[ "${DETECTOR_AUX_HEAD}" == "1" ]]; then
  ARGS+=(--detector_aux_head)
fi
if [[ -n "${TEACHER_TARGET_ROOT}" ]]; then
  ARGS+=(--kitti_teacher_target_root "${TEACHER_TARGET_ROOT}")
fi

"${PYTHON_RUNNER[@]}" "${ARGS[@]}"

echo "[track2-codec-pilot] done"
