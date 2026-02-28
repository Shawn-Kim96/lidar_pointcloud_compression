#!/bin/bash
set -euo pipefail
ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "$ROOT_DIR"
JOB_IDS=(22255 22256 22257 22258 22281)
LOG_PATH=${LOG_PATH:-logs/night_jobs_monitor_$(date +%Y%m%d_%H%M%S).log}
INTERVAL_SEC=${INTERVAL_SEC:-120}

echo "[monitor] started_at=$(date '+%Y-%m-%d %H:%M:%S %Z') jobs=${JOB_IDS[*]}" | tee -a "$LOG_PATH"

while true; do
  {
    echo "\n[monitor] tick=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")" -o '%.18i %.22j %.8T %.10M %.20S %R' || true

    # quick error scan on relevant logs only
    rg -n "Traceback|FileNotFound|RuntimeError|Exception|ERROR conda|Segmentation fault|CUDA out of memory" \
      logs/slurm_22255_*.out logs/slurm_22255_*.err \
      logs/slurm_22256_*.out logs/slurm_22256_*.err \
      logs/slurm_22257_*.out logs/slurm_22257_*.err \
      logs/slurm_22258_*.out logs/slurm_22258_*.err \
      logs/slurm_22281.out logs/slurm_22281.err \
      logs/*j22255_r*.out logs/*j22255_r*.err \
      logs/*j22256_r*.out logs/*j22256_r*.err \
      logs/*j22257_r*.out logs/*j22257_r*.err \
      logs/*j22258_r*.out logs/*j22258_r*.err \
      2>/dev/null || echo "[monitor] no error signatures"
  } | tee -a "$LOG_PATH"

  # stop when none of tracked jobs are in queue
  if ! squeue -j "$(IFS=,; echo "${JOB_IDS[*]}")" -h | grep -q .; then
    echo "[monitor] all tracked jobs left queue at $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_PATH"
    break
  fi

  sleep "$INTERVAL_SEC"
done
