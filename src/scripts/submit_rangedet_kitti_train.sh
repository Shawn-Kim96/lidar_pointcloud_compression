#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/018219422/lidar_pointcloud_compression}"
RUN_SCRIPT="${RUN_SCRIPT:-$REPO_ROOT/src/scripts/run_rangedet_kitti_train.sh}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
JOB_NAME="${JOB_NAME:-rangedet_kitti}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"
PREFERRED_PARTITION="${PREFERRED_PARTITION:-gpuql}"
FALLBACK_PARTITION="${FALLBACK_PARTITION:-gpuqm}"

mkdir -p "$LOG_DIR"

submit_once() {
  local partition="$1"
  local -a args
  args=(
    --parsable
    --partition="$partition"
    --job-name="$JOB_NAME"
    --output="$LOG_DIR/slurm_%j.out"
    --export=ALL
    --gres=gpu:1
    --cpus-per-task="$SBATCH_CPUS"
    --mem="$SBATCH_MEM"
    --time="$SBATCH_TIME"
  )

  if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    args+=(--dependency="${DEPENDENCY_TYPE}:${DEPENDENCY_JOB_ID}")
  fi

  sbatch "${args[@]}" "$RUN_SCRIPT"
}

set +e
job_id="$(submit_once "$PREFERRED_PARTITION" 2>/tmp/rangedet_submit_err.$$)"
status=$?
set -e

if [[ $status -ne 0 ]]; then
  cat "/tmp/rangedet_submit_err.$$" >&2 || true
  rm -f "/tmp/rangedet_submit_err.$$"
  job_id="$(submit_once "$FALLBACK_PARTITION")"
else
  rm -f "/tmp/rangedet_submit_err.$$"
fi

echo "$job_id"
