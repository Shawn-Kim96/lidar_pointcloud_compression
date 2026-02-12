#!/bin/bash
#SBATCH -J stage2_1_overnight
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/stage2_1_overnight_%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

echo "SLURM job ${SLURM_JOB_ID} starting in ${REPO_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

NUM_RUNS="$(python - <<'PY'
from pathlib import Path
lines = [ln for ln in Path("src/scripts/sweep_stage2_1.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
print(len(lines))
PY
)"

echo "Configured runs: ${NUM_RUNS}"

for ((idx=0; idx<NUM_RUNS; idx++)); do
  echo "=============================="
  echo "Run index: ${idx}"
  CONFIG_LINE="$(RUN_IDX="${idx}" python - <<'PY'
from pathlib import Path
import os
idx = int(os.environ["RUN_IDX"])
lines = [ln for ln in Path("src/scripts/sweep_stage2_1.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
print(lines[idx])
PY
)"

  RUN_IDX="${idx}" CONFIG_LINE="${CONFIG_LINE}" python - <<'PY' 2>&1 | tee -a "logs/stage2_1_overnight_${SLURM_JOB_ID}_py.out"
import json
import os
import shlex
import subprocess

cfg = json.loads(os.environ["CONFIG_LINE"])
idx = int(os.environ["RUN_IDX"])
job_id = os.environ.get("SLURM_JOB_ID", "na")

run_id = (
    f"s2_1_{cfg.get('run_name','full')}"
    f"_j{job_id}_r{idx}"
    f"_ns{cfg['noise_std']}_lr{cfg['lr']}_bs{cfg['batch_size']}_e{cfg['epochs']}_sd{cfg['seed']}"
    f"_bc{cfg.get('base_channels',64)}_lc{cfg.get('latent_channels',64)}_st{cfg.get('num_stages',4)}_bp{cfg.get('blocks_per_stage',1)}"
    f"_ld{cfg.get('lambda_distill',1.0)}_li{cfg.get('lambda_importance',0.2)}_be{cfg.get('beta_rate',0.005)}"
)

train_cmd = [
    "python", "-u", "src/train/train_stage2_1.py",
    "--data_dir", "data/dataset/semantickitti/dataset/sequences",
    "--train_seqs", *cfg["train_seqs"],
    "--val_seq", cfg["val_seq"],
    "--epochs", str(cfg["epochs"]),
    "--batch_size", str(cfg["batch_size"]),
    "--num_workers", str(cfg["num_workers"]),
    "--lr", str(cfg["lr"]),
    "--noise_std", str(cfg["noise_std"]),
    "--beta_rate", str(cfg["beta_rate"]),
    "--lambda_distill", str(cfg["lambda_distill"]),
    "--lambda_importance", str(cfg["lambda_importance"]),
    "--importance_supervision", str(cfg["importance_supervision"]),
    "--teacher_backend", str(cfg["teacher_backend"]),
    "--teacher_ckpt", str(cfg["teacher_ckpt"]),
    "--seed", str(cfg["seed"]),
    "--base_channels", str(cfg.get("base_channels", 64)),
    "--latent_channels", str(cfg.get("latent_channels", 64)),
    "--num_stages", str(cfg.get("num_stages", 4)),
    "--blocks_per_stage", str(cfg.get("blocks_per_stage", 1)),
    "--norm", str(cfg.get("norm", "batch")),
    "--activation", str(cfg.get("activation", "relu")),
    "--dropout", str(cfg.get("dropout", 0.0)),
    "--roi_levels", str(cfg.get("roi_levels", 256)),
    "--bg_levels", str(cfg.get("bg_levels", 16)),
    "--run_name", str(cfg.get("run_name", "s2_1")),
    "--run_id", run_id,
    "--save_name", f"{run_id}.pth",
    "--alias_name", "stage2_1_latest.pth",
    "--no_labels",
]

eval_cmd = [
    "python", "-u", "src/train/evaluate_stage2_1.py",
    "--stage1_model", "data/results/checkpoints/stage1_baseline.pth",
    "--stage2_1_ckpt", f"data/results/checkpoints/{run_id}.pth",
    "--val_seq", cfg["val_seq"],
    "--max_frames", "4071",
    "--batch_size", "4",
    "--num_workers", "2",
    "--teacher_backend", str(cfg["teacher_backend"]),
    "--teacher_ckpt", str(cfg["teacher_ckpt"]),
    "--output_plot", f"data/results/visualization/stage2_1_rd_curve_{run_id}.png",
    "--output_log", f"logs/stage2_1_eval_{run_id}.out",
]

score_cmd = [
    "python", "-u", "src/train/evaluate_teacher_scores.py",
    "--model_stage", "stage2_1",
    "--model_ckpt", f"data/results/checkpoints/{run_id}.pth",
    "--val_seq", cfg["val_seq"],
    "--max_frames", "4071",
    "--batch_size", "4",
    "--num_workers", "2",
    "--teacher_backend", str(cfg["teacher_backend"]),
    "--teacher_ckpt", str(cfg["teacher_ckpt"]),
    "--no_labels",
    "--output_csv", f"logs/stage2_1_teacher_scores_{run_id}.csv",
]

def run(cmd, tag):
    print(f"[{tag}] CMD:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)

run(train_cmd, "TRAIN")
run(eval_cmd, "EVAL")
run(score_cmd, "SCORE")
PY
done

echo "Done time: $(date '+%Y-%m-%d %H:%M:%S')"
