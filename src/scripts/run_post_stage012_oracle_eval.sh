#!/bin/bash
# Post-processing for Stage0/1/2 batch:
# 1) collect run directories from slurm logs of parent jobs
# 2) run native/oracle evaluation for each run
# 3) build bitrate-matched pairs table
#
# Required env:
#   PARENT_JOB_IDS_CSV="21804,21805,21806,21820"
#
# Optional env:
#   CONDA_ENV=lidarcomp311
#   VAL_SEQ=08
#   EVAL_MAX_FRAMES=64
#   EVAL_BATCH_SIZE=1
#   EVAL_NUM_WORKERS=0
#   MATCH_METRIC=bpp_entropy_mean
#   BATCH_TAG=custom_tag

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

PARENT_JOB_IDS_CSV=${PARENT_JOB_IDS_CSV:-}
if [[ -z "${PARENT_JOB_IDS_CSV}" ]]; then
  echo "Error: PARENT_JOB_IDS_CSV is required." >&2
  exit 1
fi

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  PY_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  PY_ENV_DESC="python:$(command -v python)"
fi

VAL_SEQ=${VAL_SEQ:-08}
EVAL_MAX_FRAMES=${EVAL_MAX_FRAMES:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}
MATCH_METRIC=${MATCH_METRIC:-bpp_entropy_mean}
BATCH_TAG=${BATCH_TAG:-$(date +%y%m%d)_stage012_post}

mkdir -p notebooks logs

echo "[post] python_env=${PY_ENV_DESC}"
echo "[post] parent_job_ids=${PARENT_JOB_IDS_CSV}"
echo "[post] val_seq=${VAL_SEQ}, max_frames=${EVAL_MAX_FRAMES}"
echo "[post] match_metric=${MATCH_METRIC}, batch_tag=${BATCH_TAG}"

IFS=',' read -r -a PARENT_IDS <<< "${PARENT_JOB_IDS_CSV}"

declare -a RUN_DIRS_RAW=()
for parent_id in "${PARENT_IDS[@]}"; do
  while IFS= read -r slurm_out; do
    [[ -z "${slurm_out}" ]] && continue
    while IFS= read -r save_dir; do
      [[ -z "${save_dir}" ]] && continue
      if [[ -d "${save_dir}" ]]; then
        RUN_DIRS_RAW+=("${save_dir}")
      fi
    done < <(sed -n 's/^save_dir: //p' "${slurm_out}")
  done < <(ls -1 "logs/slurm_${parent_id}_"*.out 2>/dev/null || true)
done

if [[ ${#RUN_DIRS_RAW[@]} -eq 0 ]]; then
  echo "Error: no run directories found from parent jobs." >&2
  exit 1
fi

mapfile -t RUN_DIRS < <(printf "%s\n" "${RUN_DIRS_RAW[@]}" | awk '!seen[$0]++')

echo "[post] collected run dirs: ${#RUN_DIRS[@]}"

declare -a SUMMARY_CSVS=()
for run_dir in "${RUN_DIRS[@]}"; do
  tag=$(basename "${run_dir}")
  summary_csv="notebooks/oracle_eval_summary_${tag}.csv"
  detail_csv="notebooks/oracle_eval_detail_${tag}.csv"
  echo "[post] eval run_dir=${run_dir}"
  "${PYTHON_RUNNER[@]}" src/train/evaluate_oracle_roi.py \
    --run_dir "${run_dir}" \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --val_seq "${VAL_SEQ}" \
    --max_frames "${EVAL_MAX_FRAMES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${EVAL_NUM_WORKERS}" \
    --output_summary_csv "${summary_csv}" \
    --output_detail_csv "${detail_csv}"
  SUMMARY_CSVS+=("${summary_csv}")
done

MATCH_OUT="notebooks/matched_bitrate_pairs_${BATCH_TAG}.csv"
"${PYTHON_RUNNER[@]}" src/utils/match_bitrate_budget.py \
  --inputs "${SUMMARY_CSVS[@]}" \
  --metric "${MATCH_METRIC}" \
  --output_csv "${MATCH_OUT}"

echo "[post] done."
echo "[post] matched pairs csv: ${MATCH_OUT}"
