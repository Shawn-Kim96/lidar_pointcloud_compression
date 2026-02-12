#!/bin/bash
#SBATCH -J s1_arch
#SBATCH -p gpuqm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/sweep_s1_arch_%A_%a.out
#SBATCH --array=0-5

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p logs data/results/checkpoints data/results/visualization data/results/runs
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

source /home/018219422/miniconda3/etc/profile.d/conda.sh
conda activate lidarcomp311

# Config line for this array index.
CONFIG_LINE="$(python - <<'PY'
from pathlib import Path
idx = int(__import__("os").environ["SLURM_ARRAY_TASK_ID"])
lines = Path("src/scripts/sweep_stage1_arch.jsonl").read_text(encoding="utf-8").splitlines()
print(lines[idx])
PY
)"

CONFIG_LINE="$CONFIG_LINE" python - <<'PY'
import json, os, shlex, subprocess
cfg = json.loads(os.environ["CONFIG_LINE"])
train_seqs = cfg["train_seqs"]
run_id = (
    f"s1_{cfg.get('run_name','arch')}"
    f"_j{os.environ.get('SLURM_ARRAY_JOB_ID','na')}"
    f"_a{os.environ.get('SLURM_ARRAY_TASK_ID','na')}"
    f"_q{cfg['quant_bits']}_ns{cfg['noise_std']}_lr{cfg['lr']}_bs{cfg['batch_size']}_e{cfg['epochs']}_sd{cfg['seed']}"
    f"_bc{cfg.get('base_channels',64)}_lc{cfg.get('latent_channels',64)}_st{cfg.get('num_stages',4)}_bp{cfg.get('blocks_per_stage',1)}"
    f"_nm{cfg.get('norm','batch')}_ac{cfg.get('activation','relu')}_do{cfg.get('dropout',0.0)}"
)
cmd = [
    "python", "-u", "src/train/train.py",
    "--data_dir", "data/dataset/semantickitti/dataset/sequences",
    "--train_seqs", *train_seqs,
    "--epochs", str(cfg["epochs"]),
    "--batch_size", str(cfg["batch_size"]),
    "--num_workers", str(cfg["num_workers"]),
    "--lr", str(cfg["lr"]),
    "--noise_std", str(cfg["noise_std"]),
    "--quant_bits", str(cfg["quant_bits"]),
    "--seed", str(cfg["seed"]),
    "--run_name", cfg.get("run_name","s1"),
    "--run_id", run_id,
    "--save_name", f"{run_id}.pth",
    "--base_channels", str(cfg.get("base_channels", 64)),
    "--latent_channels", str(cfg.get("latent_channels", 64)),
    "--num_stages", str(cfg.get("num_stages", 4)),
    "--blocks_per_stage", str(cfg.get("blocks_per_stage", 1)),
    "--norm", str(cfg.get("norm", "batch")),
    "--activation", str(cfg.get("activation", "relu")),
    "--dropout", str(cfg.get("dropout", 0.0)),
]
print("CMD:", " ".join(shlex.quote(x) for x in cmd))
subprocess.check_call(cmd)
PY
  2>&1 | tee -a "logs/sweep_s1_arch_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

