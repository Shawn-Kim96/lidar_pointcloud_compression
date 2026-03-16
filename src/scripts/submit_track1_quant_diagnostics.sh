#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Run this on a Slurm login node." >&2
  exit 1
fi

KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-${ROOT_DIR}/data/dataset/kitti3dobject}
RUN_DIRS=${RUN_DIRS:-${ROOT_DIR}/data/results/experiments/260303_resnet_adaptive_hires_track1a_hires_baseline_260303_225625_s1,${ROOT_DIR}/data/results/experiments/260303_resnet_adaptive_hires_track1b_hires_enhanced_260303_225625_s1}
OPENPCDET_CKPT=${OPENPCDET_CKPT:-${ROOT_DIR}/third_party/OpenPCDet/output/home/018219422/lidar_pointcloud_compression/logs/pointpillar_cfg_no_roadplane_pp_ft_track1b_hires_enhanced_260303_225625/pp_ft_track1b_hires_enhanced_260303_225625/ckpt/latest_model.pth}
OPENPCDET_CFG=${OPENPCDET_CFG:-${ROOT_DIR}/third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}

REFERENCE_MODE=${REFERENCE_MODE:-identity}
IMG_H=${IMG_H:-64}
IMG_W=${IMG_W:-2048}
UNPROJECTION_MODE=${UNPROJECTION_MODE:-ray}
EVAL_METRIC=${EVAL_METRIC:-kitti}
MAX_FRAMES=${MAX_FRAMES:-0}
WORKERS=${WORKERS:-0}
RUN_ORIGINAL_SANITY=${RUN_ORIGINAL_SANITY:-0}
UPDATE_PAPER_TABLE=${UPDATE_PAPER_TABLE:-0}
ORACLE_CLASSES=${ORACLE_CLASSES:-Car,Pedestrian,Cyclist}
ORACLE_DILATE_PX=${ORACLE_DILATE_PX:-1}
ROI_LEVELS_OVERRIDE=${ROI_LEVELS_OVERRIDE:-256}
BG_LEVELS_CSV=${BG_LEVELS_CSV:-24,32,48,64}

SBATCH_TIME=${SBATCH_TIME:-24:00:00}
SBATCH_CPUS=${SBATCH_CPUS:-4}
SBATCH_MEM=${SBATCH_MEM:-64G}
PREFERRED_PARTITION=${PREFERRED_PARTITION:-gpuql}
FALLBACK_PARTITION=${FALLBACK_PARTITION:-gpuqm}

TAG=${TAG:-$(date +%y%m%d_%H%M%S)}
OUT_DIR="${ROOT_DIR}/logs/track1_quant_diag/${TAG}"
mkdir -p "${OUT_DIR}"
manifest_csv="${OUT_DIR}/manifest.csv"
echo "submitted_at,job_id,job_name,partition,mode,recon_quant_mode,bg_override,roi_override,oracle_dilate_px,summary_csv,detail_csv,pair_csv" > "${manifest_csv}"

if [[ ! -f "${OPENPCDET_CKPT}" ]]; then
  echo "Error: OPENPCDET_CKPT not found: ${OPENPCDET_CKPT}" >&2
  exit 1
fi

submit_one() {
  local job_name="$1"
  local run_script="$2"
  local mode="$3"
  local recon_quant_mode="$4"
  local bg_override="$5"
  local roi_override="$6"
  local oracle_dilate="$7"
  local summary_csv="$8"
  local detail_csv="$9"
  local pair_csv="${10}"
  shift 10
  local -a extra_env=("$@")

  local partition=""
  local job_id=""
  local date_tag
  date_tag="$(date +%y%m%d)"
  local log_base="${ROOT_DIR}/logs/${date_tag}_${TAG}_${job_name}"
  local -a partitions=("${PREFERRED_PARTITION}" "${FALLBACK_PARTITION}")
  for p in "${partitions[@]}"; do
    set +e
    job_id="$(env \
      ROOT_DIR="${ROOT_DIR}" \
      KITTI_ROOT_OFFICIAL="${KITTI_ROOT_OFFICIAL}" \
      RUN_DIRS="${RUN_DIRS}" \
      OPENPCDET_CFG="${OPENPCDET_CFG}" \
      OPENPCDET_CKPT="${OPENPCDET_CKPT}" \
      REFERENCE_MODE="${REFERENCE_MODE}" \
      IMG_H="${IMG_H}" \
      IMG_W="${IMG_W}" \
      UNPROJECTION_MODE="${UNPROJECTION_MODE}" \
      EVAL_METRIC="${EVAL_METRIC}" \
      MAX_FRAMES="${MAX_FRAMES}" \
      WORKERS="${WORKERS}" \
      RUN_ORIGINAL_SANITY="${RUN_ORIGINAL_SANITY}" \
      UPDATE_PAPER_TABLE="${UPDATE_PAPER_TABLE}" \
      RECON_QUANT_MODE="${recon_quant_mode}" \
      ORACLE_CLASSES="${ORACLE_CLASSES}" \
      ORACLE_DILATE_PX="${oracle_dilate}" \
      ADAPTIVE_BG_LEVELS_OVERRIDE="${bg_override}" \
      ADAPTIVE_ROI_LEVELS_OVERRIDE="${roi_override}" \
      SUMMARY_CSV="${summary_csv}" \
      DETAIL_CSV="${detail_csv}" \
      DETECTOR_PAIR_CSV="${pair_csv}" \
      "${extra_env[@]}" \
      sbatch --parsable \
        --partition="${p}" \
        --job-name="${job_name}" \
        --output="${log_base}_%j.out" \
        --error="${log_base}_%j.err" \
        --cpus-per-task="${SBATCH_CPUS}" \
        --mem="${SBATCH_MEM}" \
        --gres=gpu:1 \
        --time="${SBATCH_TIME}" \
        "${run_script}" 2>/tmp/track1_diag_submit.err)"
    status=$?
    set -e
    if [[ ${status} -eq 0 && "${job_id}" =~ ^[0-9]+$ ]]; then
      partition="${p}"
      break
    fi
    cat /tmp/track1_diag_submit.err >&2 || true
  done
  rm -f /tmp/track1_diag_submit.err

  if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: failed to submit ${job_name}" >&2
    exit 1
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${job_id},${job_name},${partition},${mode},${recon_quant_mode},${bg_override},${roi_override},${oracle_dilate},${summary_csv},${detail_csv},${pair_csv}" >> "${manifest_csv}"
  echo "${job_id}"
}

uniform_summary="${OUT_DIR}/summary_uniform_only.csv"
uniform_detail="${OUT_DIR}/detail_uniform_only.csv"
uniform_pairs="${OUT_DIR}/pairs_uniform_only.csv"
uniform_job="$(submit_one \
  "t1_uniform" \
  "src/scripts/run_kitti_map_vs_rate_sbatch.sh" \
  "uniform_only" \
  "native" \
  "${ROI_LEVELS_OVERRIDE}" \
  "${ROI_LEVELS_OVERRIDE}" \
  "0" \
  "${uniform_summary}" \
  "${uniform_detail}" \
  "${uniform_pairs}")"

oracle_summary="${OUT_DIR}/summary_oracle_roi.csv"
oracle_detail="${OUT_DIR}/detail_oracle_roi.csv"
oracle_pairs="${OUT_DIR}/pairs_oracle_roi.csv"
oracle_job="$(submit_one \
  "t1_oracle" \
  "src/scripts/run_kitti_map_vs_rate_sbatch.sh" \
  "oracle_roi" \
  "oracle_roi" \
  "-1" \
  "-1" \
  "${ORACLE_DILATE_PX}" \
  "${oracle_summary}" \
  "${oracle_detail}" \
  "${oracle_pairs}")"

sweep_env=(
  "BG_LEVELS_CSV=${BG_LEVELS_CSV}"
  "ROI_LEVELS_OVERRIDE=${ROI_LEVELS_OVERRIDE}"
  "SWEEP_TAG=${TAG}_bg"
  "OUTPUT_DIR=${OUT_DIR}/bg_sweep"
)
sweep_job="$(submit_one \
  "t1_bg_sweep" \
  "src/scripts/run_track1_kitti_bg_sweep.sh" \
  "bg_levels_sweep" \
  "native" \
  "-1" \
  "${ROI_LEVELS_OVERRIDE}" \
  "0" \
  "${OUT_DIR}/bg_sweep/summary_bg_sweep_manifest.csv" \
  "${OUT_DIR}/bg_sweep/detail_bg_sweep_manifest.csv" \
  "${OUT_DIR}/bg_sweep/pairs_bg_sweep_manifest.csv" \
  "${sweep_env[@]}")"

echo "[track1-diag-submit] tag=${TAG}"
echo "[track1-diag-submit] uniform_job=${uniform_job}"
echo "[track1-diag-submit] oracle_job=${oracle_job}"
echo "[track1-diag-submit] bg_sweep_job=${sweep_job}"
echo "[track1-diag-submit] manifest=${manifest_csv}"
