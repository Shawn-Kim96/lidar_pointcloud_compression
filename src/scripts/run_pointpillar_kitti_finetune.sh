#!/bin/bash
# Fine-tune OpenPCDet PointPillar on KITTI object detection dataset and record metrics.
#
# Required env:
#   KITTI_ROOT_OFFICIAL=/path/to/kitti_object_root
#
# Optional env:
#   CONDA_ENV=lidarcomp311
#   ROOT_DIR=/home/018219422/lidar_pointcloud_compression
#   OPENPCDET_ROOT=third_party/OpenPCDet
#   OPENPCDET_CFG=tools/cfgs/kitti_models/pointpillar.yaml
#   OPENPCDET_PRETRAIN_CKPT=data/checkpoints/openpcdet_pointpillar_18M.pth
#   EPOCHS=40
#   BATCH_SIZE=4
#   WORKERS=4
#   EXTRA_TAG=pointpillar_ft_kitti_<tag>
#   REBUILD_KITTI_INFOS=0
#   CKPT_SAVE_INTERVAL=1
#   MAX_CKPT_SAVE_NUM=60
#   FINETUNE_LR=   (optional override)
#   FINETUNE_WEIGHT_DECAY= (optional override)
#   RUN_FINAL_TEST=1
#   WAIT_FOR_KITTI_SEC=0  (if >0, wait up to N seconds for extracted KITTI layout)
#   WAIT_POLL_SEC=60
#   SUMMARY_CSV=notebooks/pointpillar_finetune_kitti_summary.csv
#   SUMMARY_MD=docs/report/pointpillar_finetune_kitti.md

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-}
if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
if [[ ! -d "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL not found: ${KITTI_ROOT_OFFICIAL}" >&2
  exit 1
fi

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  PYTHON_RUNNER=("${CONDA_PREFIX_DIR}/bin/python")
  PY_ENV_DESC="python:${CONDA_PREFIX_DIR}/bin/python"
elif command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  PY_ENV_DESC="conda:${CONDA_ENV}"
else
  PYTHON_RUNNER=(python)
  PY_ENV_DESC="python:$(command -v python)"
fi

OPENPCDET_ROOT=${OPENPCDET_ROOT:-third_party/OpenPCDet}
OPENPCDET_CFG=${OPENPCDET_CFG:-tools/cfgs/kitti_models/pointpillar.yaml}
OPENPCDET_PRETRAIN_CKPT=${OPENPCDET_PRETRAIN_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
EXTRA_TAG=${EXTRA_TAG:-}
REBUILD_KITTI_INFOS=${REBUILD_KITTI_INFOS:-0}
CKPT_SAVE_INTERVAL=${CKPT_SAVE_INTERVAL:-1}
MAX_CKPT_SAVE_NUM=${MAX_CKPT_SAVE_NUM:-60}
FINETUNE_LR=${FINETUNE_LR:-}
FINETUNE_WEIGHT_DECAY=${FINETUNE_WEIGHT_DECAY:-}
RUN_FINAL_TEST=${RUN_FINAL_TEST:-1}
WAIT_FOR_KITTI_SEC=${WAIT_FOR_KITTI_SEC:-0}
WAIT_POLL_SEC=${WAIT_POLL_SEC:-60}
SUMMARY_CSV=${SUMMARY_CSV:-notebooks/pointpillar_finetune_kitti_summary.csv}
SUMMARY_MD=${SUMMARY_MD:-docs/report/pointpillar_finetune_kitti.md}

if [[ ! -d "${OPENPCDET_ROOT}" ]]; then
  echo "Error: OPENPCDET_ROOT not found: ${OPENPCDET_ROOT}" >&2
  exit 1
fi

if [[ "${OPENPCDET_PRETRAIN_CKPT}" != /* ]]; then
  OPENPCDET_PRETRAIN_CKPT="${ROOT_DIR}/${OPENPCDET_PRETRAIN_CKPT}"
fi
if [[ ! -f "${OPENPCDET_PRETRAIN_CKPT}" ]]; then
  echo "Error: pretrained ckpt not found: ${OPENPCDET_PRETRAIN_CKPT}" >&2
  exit 1
fi

mkdir -p logs notebooks docs/report "${OPENPCDET_ROOT}/data"
ln -sfn "${KITTI_ROOT_OFFICIAL}" "${OPENPCDET_ROOT}/data/kitti"
KITTI_LINK="$(cd "${OPENPCDET_ROOT}/data/kitti" && pwd -P)"

REQUIRED_DIRS=(ImageSets training/velodyne training/label_2 training/calib)
if [[ "${WAIT_FOR_KITTI_SEC}" -gt 0 ]]; then
  deadline=$(( $(date +%s) + WAIT_FOR_KITTI_SEC ))
  while true; do
    ready=1
    for rel in "${REQUIRED_DIRS[@]}"; do
      if [[ ! -d "${KITTI_LINK}/${rel}" ]]; then
        ready=0
        break
      fi
    done
    if [[ "${ready}" == "1" && -f "${KITTI_LINK}/ImageSets/train.txt" && -f "${KITTI_LINK}/ImageSets/val.txt" ]]; then
      bin_count=$(find "${KITTI_LINK}/training/velodyne" -maxdepth 1 -name '*.bin' | wc -l | tr -d '[:space:]')
      if [[ "${bin_count}" -ge 1000 ]]; then
        break
      fi
    fi
    now=$(date +%s)
    if [[ "${now}" -ge "${deadline}" ]]; then
      echo "Error: KITTI layout did not become ready within WAIT_FOR_KITTI_SEC=${WAIT_FOR_KITTI_SEC}." >&2
      exit 1
    fi
    echo "[pp-ft] waiting for KITTI extraction ... (bin_count=${bin_count:-0})"
    sleep "${WAIT_POLL_SEC}"
  done
fi

for rel in "${REQUIRED_DIRS[@]}"; do
  if [[ ! -d "${KITTI_LINK}/${rel}" ]]; then
    echo "Error: missing ${KITTI_LINK}/${rel}" >&2
    exit 1
  fi
done

for split in train val trainval test; do
  f="${KITTI_LINK}/ImageSets/${split}.txt"
  if [[ -f "${f}" ]]; then
    echo "[pp-ft] split_${split}=$(wc -l < "${f}")"
  else
    echo "[pp-ft] split_${split}=missing"
  fi
done

if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  TORCH_LIB="$("${CONDA_PREFIX_DIR}/bin/python" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  NVHPC_ROOT=${NVHPC_ROOT:-/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11}
  export LD_LIBRARY_PATH="${TORCH_LIB}:${NVHPC_ROOT}/cuda/11.8/lib64:${NVHPC_ROOT}/math_libs/11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT_DIR}/${OPENPCDET_ROOT}:${PYTHONPATH:-}"

"${PYTHON_RUNNER[@]}" - <<'PY'
import torch
import pcdet  # noqa: F401
from pcdet.ops.iou3d_nms import iou3d_nms_cuda  # noqa: F401
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda  # noqa: F401
print(f"[pp-ft] torch={torch.__version__} cuda={torch.version.cuda} is_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for PointPillar fine-tuning.")
print("[pp-ft] OpenPCDet import check OK")
PY

CFG_REL="${OPENPCDET_CFG}"
if [[ "${CFG_REL}" == /* ]]; then
  if [[ "${CFG_REL}" == "${ROOT_DIR}/${OPENPCDET_ROOT}/"* ]]; then
    CFG_REL="${CFG_REL#${ROOT_DIR}/${OPENPCDET_ROOT}/}"
  else
    echo "Error: OPENPCDET_CFG absolute path must be under OPENPCDET_ROOT." >&2
    exit 1
  fi
fi
CFG_FOR_TRAIN="${CFG_REL}"
CFG_FOR_TEST="${CFG_REL#tools/}"
if [[ ! -f "${OPENPCDET_ROOT}/${CFG_FOR_TRAIN}" ]]; then
  echo "Error: cfg not found: ${OPENPCDET_ROOT}/${CFG_FOR_TRAIN}" >&2
  exit 1
fi
CFG_FOR_TRAIN_TOOLS="${CFG_FOR_TRAIN#tools/}"

if [[ -z "${EXTRA_TAG}" ]]; then
  DATE_TAG=$(date +%y%m%d)
  JOB_TAG=${SLURM_JOB_ID:-local}
  EXTRA_TAG="pointpillar_ft_kitti_${DATE_TAG}_j${JOB_TAG}"
fi

CFG_FOR_RUN="${CFG_FOR_TRAIN_TOOLS}"
CFG_FOR_EVAL="${CFG_FOR_TEST}"
if [[ ! -d "${KITTI_LINK}/training/planes" ]]; then
  CFG_PATCHED="${ROOT_DIR}/logs/pointpillar_cfg_no_roadplane_${EXTRA_TAG}.yaml"
  sed 's/USE_ROAD_PLANE:[[:space:]]*True/USE_ROAD_PLANE: False/g' \
    "${OPENPCDET_ROOT}/${CFG_FOR_TRAIN}" > "${CFG_PATCHED}"
  CFG_FOR_RUN="${CFG_PATCHED}"
  CFG_FOR_EVAL="${CFG_PATCHED}"
  echo "[pp-ft] training/planes missing -> disable USE_ROAD_PLANE via ${CFG_PATCHED}"
fi

echo "============================================================"
echo "[PointPillar Finetune]"
echo "python_env: ${PY_ENV_DESC}"
echo "openpcdet_root: ${OPENPCDET_ROOT}"
echo "cfg: ${CFG_FOR_TRAIN}"
echo "cfg_runtime: ${CFG_FOR_RUN}"
echo "pretrained_ckpt: ${OPENPCDET_PRETRAIN_CKPT}"
echo "kitti_root: ${KITTI_ROOT_OFFICIAL}"
echo "extra_tag: ${EXTRA_TAG}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "workers: ${WORKERS}"
echo "started_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"

INFO_TRAIN="${KITTI_LINK}/kitti_infos_train.pkl"
INFO_VAL="${KITTI_LINK}/kitti_infos_val.pkl"
if [[ "${REBUILD_KITTI_INFOS}" == "1" || ! -f "${INFO_TRAIN}" || ! -f "${INFO_VAL}" ]]; then
  INFO_LOG="logs/pointpillar_kitti_infos_${EXTRA_TAG}_$(date +%Y%m%d_%H%M%S).log"
  (
    cd "${OPENPCDET_ROOT}"
    "${PYTHON_RUNNER[@]}" -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml \
      2>&1 | tee "${ROOT_DIR}/${INFO_LOG}"
  )
fi
if [[ ! -f "${INFO_TRAIN}" || ! -f "${INFO_VAL}" ]]; then
  echo "Error: failed to create KITTI infos." >&2
  exit 1
fi

TRAIN_LOG="logs/pointpillar_finetune_train_${EXTRA_TAG}_$(date +%Y%m%d_%H%M%S).log"
SET_CFG_ARGS=(DATA_CONFIG.DATA_PATH "${KITTI_LINK}")
if [[ -n "${FINETUNE_LR}" ]]; then
  SET_CFG_ARGS+=(OPTIMIZATION.LR "${FINETUNE_LR}")
fi
if [[ -n "${FINETUNE_WEIGHT_DECAY}" ]]; then
  SET_CFG_ARGS+=(OPTIMIZATION.WEIGHT_DECAY "${FINETUNE_WEIGHT_DECAY}")
fi

(
  cd "${OPENPCDET_ROOT}/tools"
  "${PYTHON_RUNNER[@]}" train.py \
    --cfg_file "${CFG_FOR_RUN}" \
    --pretrained_model "${OPENPCDET_PRETRAIN_CKPT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --extra_tag "${EXTRA_TAG}" \
    --ckpt_save_interval "${CKPT_SAVE_INTERVAL}" \
    --max_ckpt_save_num "${MAX_CKPT_SAVE_NUM}" \
    --set "${SET_CFG_ARGS[@]}" \
    2>&1 | tee "${ROOT_DIR}/${TRAIN_LOG}"
)

CKPT_DIR="$(find "${OPENPCDET_ROOT}/output" -type d -path "*/${EXTRA_TAG}/ckpt" | head -n1)"
if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "Error: could not locate ckpt dir for extra_tag=${EXTRA_TAG}" >&2
  exit 1
fi
LATEST_CKPT="$(ls -1t "${CKPT_DIR}"/*.pth 2>/dev/null | head -n1)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "Error: no checkpoint produced under ${CKPT_DIR}" >&2
  exit 1
fi

TEST_LOG=""
CAR_MOD="nan"
PED_MOD="nan"
CYC_MOD="nan"
MAP_MOD="nan"
if [[ "${RUN_FINAL_TEST}" == "1" ]]; then
  TEST_LOG="logs/pointpillar_finetune_eval_${EXTRA_TAG}_$(date +%Y%m%d_%H%M%S).log"
  (
    cd "${OPENPCDET_ROOT}/tools"
    "${PYTHON_RUNNER[@]}" test.py \
      --cfg_file "${CFG_FOR_EVAL}" \
      --ckpt "${LATEST_CKPT}" \
      --batch_size 1 \
      --set DATA_CONFIG.DATA_PATH "${KITTI_LINK}" \
      2>&1 | tee "${ROOT_DIR}/${TEST_LOG}"
  )

  read -r CAR_MOD PED_MOD CYC_MOD MAP_MOD < <("${PYTHON_RUNNER[@]}" - "${ROOT_DIR}/${TEST_LOG}" <<'PY'
import math
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")

def parse_mod(cls):
    pat = re.compile(rf"{cls}\s+AP@.*?3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)", re.IGNORECASE | re.DOTALL)
    ms = list(pat.finditer(text))
    if not ms:
        return float("nan")
    return float(ms[-1].group(2))

car = parse_mod("Car")
ped = parse_mod("Pedestrian")
cyc = parse_mod("Cyclist")
vals = [v for v in (car, ped, cyc) if not math.isnan(v)]
m = sum(vals) / len(vals) if vals else float("nan")
print(f"{car} {ped} {cyc} {m}")
PY
)
fi

SUMMARY_ABS="${SUMMARY_CSV}"
if [[ "${SUMMARY_ABS}" != /* ]]; then
  SUMMARY_ABS="${ROOT_DIR}/${SUMMARY_ABS}"
fi
mkdir -p "$(dirname "${SUMMARY_ABS}")"
if [[ ! -f "${SUMMARY_ABS}" ]]; then
  cat > "${SUMMARY_ABS}" <<'CSV'
timestamp,extra_tag,kitti_root,epochs,batch_size,workers,pretrained_ckpt,finetuned_ckpt,car_ap3d_mod,ped_ap3d_mod,cyc_ap3d_mod,map3d_mod_mean,train_log,eval_log
CSV
fi
echo "$(date '+%Y-%m-%d %H:%M:%S %Z'),${EXTRA_TAG},${KITTI_ROOT_OFFICIAL},${EPOCHS},${BATCH_SIZE},${WORKERS},${OPENPCDET_PRETRAIN_CKPT},${LATEST_CKPT},${CAR_MOD},${PED_MOD},${CYC_MOD},${MAP_MOD},${TRAIN_LOG},${TEST_LOG}" >> "${SUMMARY_ABS}"

SUMMARY_MD_ABS="${SUMMARY_MD}"
if [[ "${SUMMARY_MD_ABS}" != /* ]]; then
  SUMMARY_MD_ABS="${ROOT_DIR}/${SUMMARY_MD_ABS}"
fi
mkdir -p "$(dirname "${SUMMARY_MD_ABS}")"
if [[ ! -f "${SUMMARY_MD_ABS}" ]]; then
  cat > "${SUMMARY_MD_ABS}" <<'MD'
# PointPillar KITTI Finetune Log

| Timestamp | Extra tag | Epochs | Batch | Car AP3D(mod) | Ped AP3D(mod) | Cyc AP3D(mod) | mAP3D(mod) | Train log | Eval log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
MD
fi
echo "| $(date '+%Y-%m-%d %H:%M:%S %Z') | ${EXTRA_TAG} | ${EPOCHS} | ${BATCH_SIZE} | ${CAR_MOD} | ${PED_MOD} | ${CYC_MOD} | ${MAP_MOD} | ${TRAIN_LOG} | ${TEST_LOG} |" >> "${SUMMARY_MD_ABS}"

echo "[pp-ft] done"
echo "[pp-ft] latest_ckpt=${LATEST_CKPT}"
echo "[pp-ft] summary_csv=${SUMMARY_ABS}"
echo "[pp-ft] summary_md=${SUMMARY_MD_ABS}"
