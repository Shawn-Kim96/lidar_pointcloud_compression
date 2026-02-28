#!/bin/bash
# Submit Stage2 distill-fix two-experiment multi-parameter sweep
# and schedule automatic result aggregation.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

mkdir -p logs notebooks docs/report

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
BACKBONE=${BACKBONE:-resnet}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
PARTITION=${PARTITION:-gpuqm}
TIME_LIMIT=${TIME_LIMIT:-36:00:00}

TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}

submit_out=$(sbatch \
  -p "${PARTITION}" \
  -t "${TIME_LIMIT}" \
  --export=ALL,CONDA_ENV="${CONDA_ENV}",BACKBONE="${BACKBONE}",EPOCHS="${EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",NUM_WORKERS="${NUM_WORKERS}",MAX_TRAIN_FRAMES="${MAX_TRAIN_FRAMES}",TEACHER_BACKEND="${TEACHER_BACKEND}",TEACHER_PROXY_CKPT="${TEACHER_PROXY_CKPT}" \
  src/scripts/run_stage2_distill_fix_twoexp_sweep.sh)

train_job_id=$(echo "${submit_out}" | awk '{print $4}')
if [[ -z "${train_job_id}" ]]; then
  echo "Failed to parse job id from sbatch output: ${submit_out}" >&2
  exit 1
fi

post_cmd="cd ${ROOT_DIR} && "
post_cmd+="python src/utils/update_experiments_result.py && "
post_cmd+="python src/utils/summarize_stage2_distill_fix_sweep.py --job_id ${train_job_id} "
post_cmd+="--output_csv notebooks/stage2_distill_fix_twoexp_summary_${train_job_id}.csv "
post_cmd+="--output_md docs/report/stage2_distill_fix_twoexp_summary_${train_job_id}.md"

post_out=$(sbatch \
  -p "${PARTITION}" \
  --dependency=afterany:${train_job_id} \
  -t 00:30:00 \
  -J stage2_fix2_post \
  -o logs/stage2_fix2_post_%j.out \
  -e logs/stage2_fix2_post_%j.err \
  --wrap "${post_cmd}")

post_job_id=$(echo "${post_out}" | awk '{print $4}')

echo "train_job_id=${train_job_id}"
echo "post_job_id=${post_job_id}"
echo "summary_csv=notebooks/stage2_distill_fix_twoexp_summary_${train_job_id}.csv"
echo "summary_md=docs/report/stage2_distill_fix_twoexp_summary_${train_job_id}.md"
