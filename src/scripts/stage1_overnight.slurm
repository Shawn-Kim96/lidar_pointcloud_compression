#!/bin/bash
#SBATCH -J stage1_overnight
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -o logs/stage1_overnight_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs

export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "SLURM job ${SLURM_JOB_ID} starting in ${REPO_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

run_one() {
  local run_name="$1"
  local epochs="$2"
  local noise_std="$3"
  local batch_size="$4"
  local lr="$5"

  local run_id="s1_${run_name}_j${SLURM_JOB_ID}_q8_ns${noise_std}_lr${lr}_bs${batch_size}_e${epochs}_sd42_bc64_lc64_st4_bp1_nmbatch_acrelu_do0"

  echo "==== RUN ${run_name} (epochs=${epochs}, noise=${noise_std}, batch=${batch_size}, lr=${lr}) ===="
  python -u src/train/train.py \
    --data_dir data/dataset/semantickitti/dataset/sequences \
    --train_seq 00 \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --noise_std "${noise_std}" \
    --lr "${lr}" \
    --run_id "${run_id}" \
    --save_name "${run_id}.pth"

  python -u src/train/evaluate.py \
    --model "data/results/checkpoints/${run_id}.pth" \
    --quant_bits 8 \
    --val_seq 08 \
    --max_frames 512 \
    --log_path logs/stage1_eval_overnight.out

  python -u src/train/evaluate.py \
    --model "data/results/checkpoints/${run_id}.pth" \
    --quant_bits 4 \
    --val_seq 08 \
    --max_frames 512 \
    --log_path logs/stage1_eval_overnight.out
}

run_one "ae_n010" 12 0.10 4 1e-3
run_one "ae_n005" 12 0.05 4 1e-3
run_one "ae_n000" 12 0.00 4 1e-3

echo "Done time: $(date '+%Y-%m-%d %H:%M:%S')"
