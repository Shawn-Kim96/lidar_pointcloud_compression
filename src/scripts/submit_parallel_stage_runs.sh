#!/bin/bash
# Submit Stage0(uniform), Stage1(adaptive no teacher), Stage2(adaptive with teacher)
# in parallel as independent Slurm jobs.
#
# Usage:
#   bash src/scripts/submit_parallel_stage_runs.sh

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch command not found. Run this on a Slurm environment." >&2
  exit 1
fi

mkdir -p logs

jid_uniform=$(sbatch src/scripts/run_uniform_baseline.sh | awk '{print $4}')
jid_stage1=$(sbatch src/scripts/run_stage1.sh | awk '{print $4}')
jid_stage2=$(sbatch src/scripts/run_stage2.sh | awk '{print $4}')

echo "Submitted jobs:"
echo "  uniform_baseline : ${jid_uniform}"
echo "  stage1_adaptive  : ${jid_stage1}"
echo "  stage2_distill   : ${jid_stage2}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${jid_uniform},${jid_stage1},${jid_stage2}"
