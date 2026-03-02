#!/bin/bash
# Stage3-style ablation over modern multi-scale ROI heads.
#
# Head variants:
#   0: bifpn
#   1: deformable_msa
#   2: dynamic
#   3: rangeformer
#   4: frnet
#
# Usage:
#   sbatch src/scripts/run_stage3_multiscale_heads.sh

#SBATCH --job-name=lidar_stage3
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4

set -euo pipefail

mkdir -p logs

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  PYTHON_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  PYTHON_ENV_DESC="python:$(command -v python)"
fi

HEAD_TYPES=("bifpn" "deformable_msa" "dynamic" "rangeformer" "frnet")
HEAD_TYPE=${HEAD_TYPES[$SLURM_ARRAY_TASK_ID]}

BACKBONE=${BACKBONE:-resnet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
LR=${LR:-1e-4}
MIN_GPU_MEM_GB=${MIN_GPU_MEM_GB:-30}

ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.02}
L_DISTILL=${L_DISTILL:-0.1}
L_IMPORTANCE=${L_IMPORTANCE:-1.0}
L_IMP_SEPARATION=${L_IMP_SEPARATION:-0.2}
IMP_SEPARATION_MARGIN=${IMP_SEPARATION_MARGIN:-0.05}

LOSS_RECIPE=${LOSS_RECIPE:-balanced_v2}
RATE_LOSS_MODE=${RATE_LOSS_MODE:-normalized_bg}
IMPORTANCE_LOSS_MODE=${IMPORTANCE_LOSS_MODE:-weighted_bce}
IMPORTANCE_POS_WEIGHT_MODE=${IMPORTANCE_POS_WEIGHT_MODE:-auto}
IMPORTANCE_POS_WEIGHT=${IMPORTANCE_POS_WEIGHT:-1.0}
IMPORTANCE_POS_WEIGHT_MAX=${IMPORTANCE_POS_WEIGHT_MAX:-50.0}
DISTILL_LOGIT_LOSS=${DISTILL_LOGIT_LOSS:-auto}
DISTILL_TEMPERATURE=${DISTILL_TEMPERATURE:-1.0}
DISTILL_FEATURE_WEIGHT=${DISTILL_FEATURE_WEIGHT:-1.0}
DISTILL_LOGIT_WEIGHT=${DISTILL_LOGIT_WEIGHT:-1.0}

if [[ ! -f "${TEACHER_PROXY_CKPT}" ]]; then
  echo "Warning: teacher checkpoint not found at ${TEACHER_PROXY_CKPT}"
fi

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_JOB_ID:-local}"
LOG_PREFIX="${DATE_TAG}_${BACKBONE}_stage3_head${HEAD_TYPE}_j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

RUN_ID="j${JOB_TAG}_r${SLURM_ARRAY_TASK_ID}"
MODE_TAG="stage3_head${HEAD_TYPE}"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR}_bs${BATCH_SIZE}_${RUN_ID}"

echo "============================================================"
echo "[Experiment Metadata]"
echo "stage: 3"
echo "training_mode: stage3 multiscale head ablation (teacher enabled)"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: ${TEACHER_BACKEND}"
echo "teacher_proxy_ckpt: ${TEACHER_PROXY_CKPT}"
echo "run_id: ${RUN_ID}"
echo "save_dir: ${SAVE_DIR}"
echo "dataset_root: data/dataset/semantickitti/dataset/sequences"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "lr: ${LR}"
echo "quantizer_mode: ${QUANTIZER_MODE}"
echo "quant_bits: ${QUANT_BITS}"
echo "roi_levels: ${ROI_LEVELS}"
echo "bg_levels: ${BG_LEVELS}"
echo "roi_target_mode: ${ROI_TARGET_MODE}"
echo "loss_recipe: ${LOSS_RECIPE}"
echo "rate_loss_mode: ${RATE_LOSS_MODE}"
echo "importance_loss_mode: ${IMPORTANCE_LOSS_MODE}"
echo "importance_pos_weight_mode: ${IMPORTANCE_POS_WEIGHT_MODE}"
echo "importance_pos_weight: ${IMPORTANCE_POS_WEIGHT}"
echo "importance_pos_weight_max: ${IMPORTANCE_POS_WEIGHT_MAX}"
echo "lambda_imp_separation: ${L_IMP_SEPARATION}"
echo "imp_separation_margin: ${IMP_SEPARATION_MARGIN}"
echo "distill_logit_loss: ${DISTILL_LOGIT_LOSS}"
echo "distill_temperature: ${DISTILL_TEMPERATURE}"
echo "distill_feature_weight: ${DISTILL_FEATURE_WEIGHT}"
echo "distill_logit_weight: ${DISTILL_LOGIT_WEIGHT}"
echo "importance_head_type: ${HEAD_TYPE}"
echo "importance_hidden_channels: ${IMPORTANCE_HIDDEN_CHANNELS}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}, imp_separation=${L_IMP_SEPARATION}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

set +e
GPU_GUARD_OUT="$("${PYTHON_RUNNER[@]}" - "${MIN_GPU_MEM_GB}" <<'PY' 2>&1
import sys
import torch

min_gb = float(sys.argv[1])
if not torch.cuda.is_available():
    print("[stage3] cuda_available=0")
    raise SystemExit(2)

props = torch.cuda.get_device_properties(0)
mem_gb = props.total_memory / (1024 ** 3)
print(f"[stage3] gpu_name={props.name} gpu_mem_gb={mem_gb:.2f} min_required_gb={min_gb:.2f}")
if mem_gb + 1e-6 < min_gb:
    raise SystemExit(3)
PY
)"
gpu_guard_status=$?
set -e
echo "${GPU_GUARD_OUT}"
if [[ "${gpu_guard_status}" == "2" ]]; then
  echo "Error: CUDA unavailable for stage3 run." >&2
  exit 1
fi
if [[ "${gpu_guard_status}" == "3" ]]; then
  echo "Error: insufficient GPU memory for stage3 head ablation." >&2
  echo "Error: rerun on high-memory GPU (e.g., --gres=gpu:a100:1)." >&2
  exit 1
fi

"${PYTHON_RUNNER[@]}" src/main_train.py \
    --data_root "data/dataset/semantickitti/dataset/sequences" \
    --backbone "$BACKBONE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_train_frames "$MAX_TRAIN_FRAMES" \
    --teacher_backend "$TEACHER_BACKEND" \
    --teacher_proxy_ckpt "$TEACHER_PROXY_CKPT" \
    --run_id "$RUN_ID" \
    --save_dir "$SAVE_DIR" \
    --quantizer_mode "$QUANTIZER_MODE" \
    --quant_bits "$QUANT_BITS" \
    --roi_levels "$ROI_LEVELS" \
    --bg_levels "$BG_LEVELS" \
    --roi_target_mode "$ROI_TARGET_MODE" \
    --loss_recipe "$LOSS_RECIPE" \
    --rate_loss_mode "$RATE_LOSS_MODE" \
    --importance_loss_mode "$IMPORTANCE_LOSS_MODE" \
    --importance_pos_weight_mode "$IMPORTANCE_POS_WEIGHT_MODE" \
    --importance_pos_weight "$IMPORTANCE_POS_WEIGHT" \
    --importance_pos_weight_max "$IMPORTANCE_POS_WEIGHT_MAX" \
    --lambda_recon "$L_RECON" \
    --lambda_rate "$L_RATE" \
    --lambda_distill "$L_DISTILL" \
    --lambda_importance "$L_IMPORTANCE" \
    --lambda_imp_separation "$L_IMP_SEPARATION" \
    --imp_separation_margin "$IMP_SEPARATION_MARGIN" \
    --distill_logit_loss "$DISTILL_LOGIT_LOSS" \
    --distill_temperature "$DISTILL_TEMPERATURE" \
    --distill_feature_weight "$DISTILL_FEATURE_WEIGHT" \
    --distill_logit_weight "$DISTILL_LOGIT_WEIGHT" \
    --importance_head_type "$HEAD_TYPE" \
    --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS"

echo "Done Stage3 task ${SLURM_ARRAY_TASK_ID} (${HEAD_TYPE})"
