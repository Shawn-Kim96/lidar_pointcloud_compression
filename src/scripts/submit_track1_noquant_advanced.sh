#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

BASE_TAG=${BASE_TAG:-$(date +%y%m%d_%H%M%S)}
CONDA_ENV=${CONDA_ENV:-lidarcomp311}
KITTI_ROOT=${KITTI_ROOT:-data/dataset/kitti3dobject}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
FOV_UP_DEG=${FOV_UP_DEG:-3.0}
FOV_DOWN_DEG=${FOV_DOWN_DEG:--25.0}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}
TRAIN_EPOCHS_NOQUANT=${TRAIN_EPOCHS_NOQUANT:-220}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
PP_FT_EPOCHS=${PP_FT_EPOCHS:-20}
PP_FT_BATCH_SIZE=${PP_FT_BATCH_SIZE:-4}
PP_FT_WORKERS=${PP_FT_WORKERS:-4}

choose_partition() {
  if command -v sinfo >/dev/null 2>&1; then
    local free_gpuql
    free_gpuql=$(sinfo -h -p gpuql -t idle,mix -o "%D" 2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [[ "${free_gpuql:-0}" -gt 0 ]]; then
      echo "gpuql"
      return 0
    fi
  fi
  echo "gpuqm"
}

PARTITION=${SBATCH_PARTITION:-$(choose_partition)}
TIME_LIMIT=${SBATCH_TIME:-72:00:00}

submit_variant() {
  local label="$1"
  local variant="$2"
  local latent="$3"
  local base="$4"
  local blocks="$5"
  local fusion="$6"
  local fusion_hidden="$7"
  local refine="$8"
  local recon_mode="$9"
  local range_w="${10}"
  local xyz_w="${11}"
  local rem_w="${12}"

  local date_tag
  date_tag="$(date +%y%m%d)"
  local out="logs/${date_tag}_${label}_%j.out"
  local err="logs/${date_tag}_${label}_%j.err"
  sbatch --parsable \
    --job-name="${label}" \
    --output="${out}" \
    --error="${err}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=6 \
    --gres=gpu:1 \
    --time="${TIME_LIMIT}" \
    --exclude="g16,g18,g19" \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",TRACK_TAG="${BASE_TAG}",TRACK_LABEL="${label}",TRACK_VARIANT="${variant}",KITTI_ROOT="${KITTI_ROOT}",IMG_H="${IMG_H}",IMG_W="${IMG_W}",FOV_UP_DEG="${FOV_UP_DEG}",FOV_DOWN_DEG="${FOV_DOWN_DEG}",UNPROJECTION_MODE="${UNPROJECTION_MODE}",TRAIN_EPOCHS_NOQUANT="${TRAIN_EPOCHS_NOQUANT}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",TRAIN_WORKERS="${TRAIN_WORKERS}",PP_FT_EPOCHS="${PP_FT_EPOCHS}",PP_FT_BATCH_SIZE="${PP_FT_BATCH_SIZE}",PP_FT_WORKERS="${PP_FT_WORKERS}",LATENT_CHANNELS="${latent}",BASE_CHANNELS="${base}",BLOCKS_PER_STAGE="${blocks}",FUSION_VARIANT="${fusion}",FUSION_HIDDEN="${fusion_hidden}",DECODER_REFINE_BLOCKS="${refine}",RECON_LOSS_MODE="${recon_mode}",RECON_RANGE_WEIGHT="${range_w}",RECON_XYZ_WEIGHT="${xyz_w}",RECON_REMISSION_WEIGHT="${rem_w}" \
    src/scripts/run_track1_noquant_chain.sh
}

echo "============================================================"
echo "[Submit Track1 Advanced NoQuant]"
echo "base_tag: ${BASE_TAG}"
echo "partition: ${PARTITION}"
echo "time_limit: ${TIME_LIMIT}"
echo "projection: ${IMG_H}x${IMG_W} unprojection=${UNPROJECTION_MODE}"
echo "train_epochs_noquant: ${TRAIN_EPOCHS_NOQUANT}"
echo "============================================================"

JID_RF=$(submit_variant "t1nq_geo_rf" "geo_rf" 128 96 2 rangeformer 192 4 masked_channel_weighted 1.5 4.0 0.25)
JID_FR=$(submit_variant "t1nq_geo_fr" "geo_fr" 128 96 2 frnet 192 4 masked_channel_weighted 1.5 4.0 0.25)

MANIFEST="logs/${BASE_TAG}_track1_noquant_advanced_manifest.csv"
cat > "${MANIFEST}" <<CSV
submitted_at,job_id,label,variant,partition,time_limit,img_h,img_w,unprojection_mode,epochs_noquant,latent_channels,base_channels,blocks_per_stage,fusion_variant,fusion_hidden,decoder_refine_blocks,recon_loss_mode,recon_range_weight,recon_xyz_weight,recon_remission_weight
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_RF},t1nq_geo_rf,geo_rf,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT},128,96,2,rangeformer,192,4,masked_channel_weighted,1.5,4.0,0.25
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_FR},t1nq_geo_fr,geo_fr,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT},128,96,2,frnet,192,4,masked_channel_weighted,1.5,4.0,0.25
CSV

echo "t1nq_geo_rf job_id=${JID_RF}"
echo "t1nq_geo_fr job_id=${JID_FR}"
echo "manifest=${MANIFEST}"
