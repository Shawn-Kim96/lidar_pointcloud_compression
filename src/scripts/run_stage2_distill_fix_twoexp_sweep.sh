#!/bin/bash
#SBATCH --job-name=stage2_fix2_sweep
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --array=0-11

# Two updated Stage2 distill-fix experiments with parameter sweep.
# Cases (2):
#   0) energy_pool_16x32
#   1) energy_pool_16x32_scoregate015
# Parameters:
#   lr in {1e-4, 5e-5}
#   lambda_distill in {0.05, 0.1, 0.2}
# Total jobs: 2 * 2 * 3 = 12

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

IDX=${SLURM_ARRAY_TASK_ID:-0}

BACKBONE=${BACKBONE:-resnet}
TEACHER_BACKEND=${TEACHER_BACKEND:-pointpillars_zhulf}
TEACHER_PROXY_CKPT=${TEACHER_PROXY_CKPT:-data/checkpoints/pointpillars_epoch_160.pth}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MIN_GPU_MEM_GB=${MIN_GPU_MEM_GB:-0}

QUANTIZER_MODE=${QUANTIZER_MODE:-adaptive}
QUANT_BITS=${QUANT_BITS:-8}
ROI_LEVELS=${ROI_LEVELS:-256}
BG_LEVELS=${BG_LEVELS:-16}
ROI_TARGET_MODE=${ROI_TARGET_MODE:-maxpool}
LOSS_RECIPE=${LOSS_RECIPE:-balanced_v2}
RATE_LOSS_MODE=${RATE_LOSS_MODE:-normalized_bg}
IMPORTANCE_LOSS_MODE=${IMPORTANCE_LOSS_MODE:-weighted_bce}
IMPORTANCE_POS_WEIGHT_MODE=${IMPORTANCE_POS_WEIGHT_MODE:-auto}
IMPORTANCE_POS_WEIGHT=${IMPORTANCE_POS_WEIGHT:-1.0}
IMPORTANCE_POS_WEIGHT_MAX=${IMPORTANCE_POS_WEIGHT_MAX:-50.0}

L_RECON=${L_RECON:-1.0}
L_RATE=${L_RATE:-0.02}
L_IMPORTANCE=${L_IMPORTANCE:-1.0}
L_IMP_SEP=${L_IMP_SEP:-0.2}
IMP_MARGIN=${IMP_MARGIN:-0.05}

DISTILL_LOGIT_LOSS=${DISTILL_LOGIT_LOSS:-auto}
DISTILL_TEMP=${DISTILL_TEMP:-1.0}
DISTILL_FEATURE_WEIGHT=${DISTILL_FEATURE_WEIGHT:-1.0}
DISTILL_LOGIT_WEIGHT=${DISTILL_LOGIT_WEIGHT:-1.0}

IMPORTANCE_HEAD_TYPE=${IMPORTANCE_HEAD_TYPE:-pp_lite}
IMPORTANCE_HIDDEN_CHANNELS=${IMPORTANCE_HIDDEN_CHANNELS:-64}

if [[ "${MIN_GPU_MEM_GB}" == "0" ]]; then
  if [[ "${IMPORTANCE_HIDDEN_CHANNELS}" -ge 96 ]]; then
    MIN_GPU_MEM_GB=30
  else
    MIN_GPU_MEM_GB=14
  fi
fi

CASE_TAGS=("energy_pool_16x32" "energy_pool_16x32_scoregate015")
DISTILL_FEATURE_SOURCES=("energy_map" "energy_map")
DISTILL_ALIGN_MODES=("adaptive_pool" "adaptive_pool")
DISTILL_ALIGN_HWS=("16,32" "16,32")
DISTILL_TEACHER_SCORE_MINS=("0.0" "0.15")
DISTILL_TEACHER_SCORE_WEIGHTS=("1" "1")

LRS=("1e-4" "5e-5")
LAMBDA_DISTILLS=("0.05" "0.1" "0.2")

NUM_CASES=${#CASE_TAGS[@]}
NUM_LRS=${#LRS[@]}
NUM_LDS=${#LAMBDA_DISTILLS[@]}
NUM_COMBOS_PER_CASE=$((NUM_LRS * NUM_LDS))
TOTAL=$((NUM_CASES * NUM_COMBOS_PER_CASE))

if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Invalid array index ${IDX}. Expected [0, $((TOTAL-1))]." >&2
  exit 1
fi

CASE_IDX=$((IDX / NUM_COMBOS_PER_CASE))
INNER_IDX=$((IDX % NUM_COMBOS_PER_CASE))
LR_IDX=$((INNER_IDX / NUM_LDS))
LD_IDX=$((INNER_IDX % NUM_LDS))

CASE_TAG=${CASE_TAGS[$CASE_IDX]}
DISTILL_FEATURE_SOURCE=${DISTILL_FEATURE_SOURCES[$CASE_IDX]}
DISTILL_ALIGN_MODE=${DISTILL_ALIGN_MODES[$CASE_IDX]}
DISTILL_ALIGN_HW=${DISTILL_ALIGN_HWS[$CASE_IDX]}
DISTILL_TEACHER_SCORE_MIN=${DISTILL_TEACHER_SCORE_MINS[$CASE_IDX]}
DISTILL_TEACHER_SCORE_WEIGHT=${DISTILL_TEACHER_SCORE_WEIGHTS[$CASE_IDX]}

LR=${LRS[$LR_IDX]}
L_DISTILL=${LAMBDA_DISTILLS[$LD_IDX]}

DATE_TAG=$(date +%y%m%d)
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUN_ID="j${JOB_TAG}_r${IDX}"

LD_TAG=${L_DISTILL//./p}
LR_TAG=${LR//./p}
LR_TAG=${LR_TAG//-/m}
MODE_TAG="distill_fix2_${CASE_TAG}_ld${LD_TAG}"
SAVE_DIR="data/results/experiments/${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR_TAG}_bs${BATCH_SIZE}_${RUN_ID}"

LOG_PREFIX="${DATE_TAG}_${BACKBONE}_${MODE_TAG}_lr${LR_TAG}_j${JOB_TAG}_r${IDX}"
exec > >(tee -a "logs/${LOG_PREFIX}.out")
exec 2> >(tee -a "logs/${LOG_PREFIX}.err" >&2)

echo "============================================================"
echo "[Stage2 Distill-Fix TwoExp Sweep Metadata]"
echo "stage: 2"
echo "training_mode: distillation (teacher enabled)"
echo "case_tag: ${CASE_TAG}"
echo "sweep_param_lr: ${LR}"
echo "sweep_param_lambda_distill: ${L_DISTILL}"
echo "backbone: ${BACKBONE}"
echo "teacher_backend: ${TEACHER_BACKEND}"
echo "teacher_proxy_ckpt: ${TEACHER_PROXY_CKPT}"
echo "run_id: ${RUN_ID}"
echo "save_dir: ${SAVE_DIR}"
echo "loss_recipe: ${LOSS_RECIPE}"
echo "rate_loss_mode: ${RATE_LOSS_MODE}"
echo "importance_loss_mode: ${IMPORTANCE_LOSS_MODE}"
echo "distill_logit_loss: ${DISTILL_LOGIT_LOSS}"
echo "distill_feature_source: ${DISTILL_FEATURE_SOURCE}"
echo "distill_align_mode: ${DISTILL_ALIGN_MODE}"
echo "distill_align_hw: ${DISTILL_ALIGN_HW}"
echo "distill_teacher_score_min: ${DISTILL_TEACHER_SCORE_MIN}"
echo "distill_teacher_score_weight: ${DISTILL_TEACHER_SCORE_WEIGHT}"
echo "importance_head_type: ${IMPORTANCE_HEAD_TYPE}"
echo "importance_hidden_channels: ${IMPORTANCE_HIDDEN_CHANNELS}"
echo "loss_weights: recon=${L_RECON}, rate=${L_RATE}, distill=${L_DISTILL}, importance=${L_IMPORTANCE}, imp_separation=${L_IMP_SEP}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "num_workers: ${NUM_WORKERS}"
echo "max_train_frames: ${MAX_TRAIN_FRAMES}"
echo "python_env: ${PYTHON_ENV_DESC}"
echo "slurm_job_id: ${SLURM_JOB_ID:-n/a}"
echo "slurm_array_job_id: ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "slurm_array_task_id: ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

set +e
GPU_GUARD_OUT="$("${PYTHON_RUNNER[@]}" - "${MIN_GPU_MEM_GB}" <<'PY' 2>&1
import sys
import torch

min_gb = float(sys.argv[1])
if not torch.cuda.is_available():
    print("[stage2-fix] cuda_available=0")
    raise SystemExit(2)

props = torch.cuda.get_device_properties(0)
mem_gb = props.total_memory / (1024 ** 3)
print(f"[stage2-fix] gpu_name={props.name} gpu_mem_gb={mem_gb:.2f} min_required_gb={min_gb:.2f}")
if mem_gb + 1e-6 < min_gb:
    raise SystemExit(3)
PY
)"
gpu_guard_status=$?
set -e
echo "${GPU_GUARD_OUT}"
if [[ "${gpu_guard_status}" == "2" ]]; then
  echo "Error: CUDA unavailable for stage2-fix run." >&2
  exit 1
fi
if [[ "${gpu_guard_status}" == "3" ]]; then
  echo "Error: insufficient GPU memory for this config." >&2
  echo "Error: rerun on high-memory GPU (e.g., --gres=gpu:a100:1)." >&2
  exit 1
fi

DISTILL_SCORE_FLAG=()
if [[ "${DISTILL_TEACHER_SCORE_WEIGHT}" == "1" ]]; then
  DISTILL_SCORE_FLAG+=(--distill_teacher_score_weight)
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
  --lambda_imp_separation "$L_IMP_SEP" \
  --imp_separation_margin "$IMP_MARGIN" \
  --distill_logit_loss "$DISTILL_LOGIT_LOSS" \
  --distill_temperature "$DISTILL_TEMP" \
  --distill_feature_weight "$DISTILL_FEATURE_WEIGHT" \
  --distill_logit_weight "$DISTILL_LOGIT_WEIGHT" \
  --distill_feature_source "$DISTILL_FEATURE_SOURCE" \
  --distill_align_mode "$DISTILL_ALIGN_MODE" \
  --distill_align_hw "$DISTILL_ALIGN_HW" \
  --distill_teacher_score_min "$DISTILL_TEACHER_SCORE_MIN" \
  "${DISTILL_SCORE_FLAG[@]}" \
  --importance_head_type "$IMPORTANCE_HEAD_TYPE" \
  --importance_hidden_channels "$IMPORTANCE_HIDDEN_CHANNELS"

echo "Done Stage2 distill-fix twoexp sweep task ${IDX}"
