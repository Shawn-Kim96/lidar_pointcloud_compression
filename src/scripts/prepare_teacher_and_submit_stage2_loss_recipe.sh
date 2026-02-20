#!/bin/bash
# Login-node helper:
# 1) download PointPillars checkpoint (for OpenPCDet teacher path)
# 2) submit Stage2 loss-recipe ablation jobs

set -euo pipefail

mkdir -p logs

# Pre-download detector checkpoint before entering GPU queue.
if bash src/scripts/download_pointpillar_checkpoint.sh; then
  echo "[prepare_submit] PointPillars checkpoint ready."
else
  echo "[prepare_submit] PointPillars download failed. Continuing with proxy-teacher path."
fi

export TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
export TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}

JOB_ID=$(sbatch src/scripts/run_stage2_loss_recipe_ablation.sh | awk '{print $4}')
echo "[prepare_submit] Submitted Stage2 loss-recipe ablation: job_id=${JOB_ID}"
echo "[prepare_submit] Monitor with: squeue -j ${JOB_ID}"
