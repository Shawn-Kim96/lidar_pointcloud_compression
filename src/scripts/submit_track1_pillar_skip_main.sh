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
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
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
TIME_LIMIT=${SBATCH_TIME:-120:00:00}

submit_variant() {
  local label="$1"
  local variant="$2"
  local latent="$3"
  local base="$4"
  local blocks="$5"
  local fusion="$6"
  local fusion_hidden="$7"
  local refine="$8"
  local range_w="$9"
  local xyz_w="${10}"
  local rem_w="${11}"
  local ray_w="${12}"
  local position_branch="${13}"
  local decoder_type="${14}"
  local implicit_hidden="${15}"
  local pillar_size_x="${16}"
  local pillar_size_y="${17}"
  local pillar_pfn_hidden="${18}"
  local pillar_pfn_out="${19}"
  local pillar_bev_channels="${20}"
  local pillar_bev_blocks="${21}"
  local pillar_fpn_channels="${22}"

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
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --time="${TIME_LIMIT}" \
    --exclude="g16,g18,g19" \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",TRACK_TAG="${BASE_TAG}",TRACK_LABEL="${label}",TRACK_VARIANT="${variant}",KITTI_ROOT="${KITTI_ROOT}",IMG_H="${IMG_H}",IMG_W="${IMG_W}",FOV_UP_DEG="${FOV_UP_DEG}",FOV_DOWN_DEG="${FOV_DOWN_DEG}",UNPROJECTION_MODE="${UNPROJECTION_MODE}",TRAIN_EPOCHS_NOQUANT="${TRAIN_EPOCHS_NOQUANT}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",TRAIN_WORKERS="${TRAIN_WORKERS}",PP_FT_EPOCHS="${PP_FT_EPOCHS}",PP_FT_BATCH_SIZE="${PP_FT_BATCH_SIZE}",PP_FT_WORKERS="${PP_FT_WORKERS}",LATENT_CHANNELS="${latent}",BASE_CHANNELS="${base}",BLOCKS_PER_STAGE="${blocks}",FUSION_VARIANT="${fusion}",FUSION_HIDDEN="${fusion_hidden}",DECODER_REFINE_BLOCKS="${refine}",RECON_LOSS_MODE="masked_channel_weighted",RECON_RANGE_WEIGHT="${range_w}",RECON_XYZ_WEIGHT="${xyz_w}",RECON_REMISSION_WEIGHT="${rem_w}",LAMBDA_RAY_CONSISTENCY="${ray_w}",POSITION_BRANCH="${position_branch}",DECODER_TYPE="${decoder_type}",IMPLICIT_HIDDEN_CHANNELS="${implicit_hidden}",PILLAR_SIDE_BRANCH=1,PILLAR_MAX_RAW_POINTS=150000,PILLAR_SIZE_X="${pillar_size_x}",PILLAR_SIZE_Y="${pillar_size_y}",PILLAR_PFN_HIDDEN_CHANNELS="${pillar_pfn_hidden}",PILLAR_PFN_OUT_CHANNELS="${pillar_pfn_out}",PILLAR_BEV_CHANNELS="${pillar_bev_channels}",PILLAR_BEV_BLOCKS="${pillar_bev_blocks}",PILLAR_FPN_CHANNELS="${pillar_fpn_channels}" \
    src/scripts/run_track1_noquant_chain.sh
}

echo "============================================================"
echo "[Submit Track1 Pillar Skip Main]"
echo "base_tag: ${BASE_TAG}"
echo "partition: ${PARTITION}"
echo "time_limit: ${TIME_LIMIT}"
echo "projection: ${IMG_H}x${IMG_W} unprojection=${UNPROJECTION_MODE}"
echo "train_epochs_noquant: ${TRAIN_EPOCHS_NOQUANT}"
echo "train_batch_size: ${TRAIN_BATCH_SIZE}"
echo "============================================================"

JID_A=$(submit_variant "t1nq_pillar_skip_a" "pillar_skip_a" 160 128 2 frnet 224 4 1.5 5.5 0.25 2.5 1 skip_coord_conditioned 224 0.32 0.32 96 160 160-192-256-320 2-2-3-3 160)
JID_B=$(submit_variant "t1nq_pillar_skip_b" "pillar_skip_b" 192 128 3 rangeformer 256 4 1.5 6.0 0.25 3.0 1 skip_coord_conditioned 256 0.24 0.24 128 192 192-224-288-384 2-2-3-3 192)

MANIFEST="logs/${BASE_TAG}_track1_pillar_skip_main_manifest.csv"
cat > "${MANIFEST}" <<CSV
submitted_at,job_id,label,variant,partition,time_limit,img_h,img_w,unprojection_mode,epochs_noquant,train_batch_size,latent_channels,base_channels,blocks_per_stage,fusion_variant,fusion_hidden,decoder_refine_blocks,recon_range_weight,recon_xyz_weight,recon_remission_weight,lambda_ray_consistency,decoder_type,pillar_size_x,pillar_size_y,pillar_pfn_hidden,pillar_pfn_out,pillar_bev_channels,pillar_bev_blocks,pillar_fpn_channels
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_A},t1nq_pillar_skip_a,pillar_skip_a,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT},${TRAIN_BATCH_SIZE},160,128,2,frnet,224,4,1.5,5.5,0.25,2.5,skip_coord_conditioned,0.32,0.32,96,160,"160,192,256,320","2,2,3,3",160
$(date '+%Y-%m-%d %H:%M:%S %Z'),${JID_B},t1nq_pillar_skip_b,pillar_skip_b,${PARTITION},${TIME_LIMIT},${IMG_H},${IMG_W},${UNPROJECTION_MODE},${TRAIN_EPOCHS_NOQUANT},${TRAIN_BATCH_SIZE},192,128,3,rangeformer,256,4,1.5,6.0,0.25,3.0,skip_coord_conditioned,0.24,0.24,128,192,"192,224,288,384","2,2,3,3",192
CSV

echo "t1nq_pillar_skip_a job_id=${JID_A}"
echo "t1nq_pillar_skip_b job_id=${JID_B}"
echo "manifest=${MANIFEST}"
