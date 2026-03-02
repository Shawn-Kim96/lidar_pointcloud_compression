#!/bin/bash
# Evaluate detector endpoint on KITTI with strict protocol:
#  1) official KITTI root validation + OpenPCDet info generation
#  2) original PointPillar sanity check (Car AP(mod) must be > 0)
#  3) reconstructed-vs-original comparison (same cfg/ckpt/split/metric)
#
# Required env:
#   KITTI_ROOT_OFFICIAL=/path/to/official_kitti_detection_format
#   RUN_DIRS="data/results/experiments/runA,data/results/experiments/runB"
#
# Optional env:
#   KITTI_ROOT=third_party/OpenPCDet/data/kitti
#   CONDA_ENV=lidarcomp311
#   OPENPCDET_ROOT=third_party/OpenPCDet
#   OPENPCDET_CFG=third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
#   OPENPCDET_CKPT=data/checkpoints/openpcdet_pointpillar_18M.pth
#   SETUP_OPENPCDET=0
#   REBUILD_KITTI_INFOS=0
#   RUN_ORIGINAL_SANITY=1
#   SANITY_BATCH_SIZE=1
#   REQUIRE_CUDA=1
#   KITTI_MAX_SPLIT_SAMPLES=10000
#   KITTI_MAX_TRAIN_FILES=10000
#   KITTI_MIN_TRAIN_SAMPLES=1000
#   KITTI_MIN_VAL_SAMPLES=1000
#   REQUIRE_DISJOINT_TRAIN_VAL=1
#   BATCH_SIZE=1
#   WORKERS=0
#   MAX_FRAMES=0
#   SPLIT=val
#   SPLIT_MANIFEST=/path/to/ImageSets/val.txt
#   EVAL_METRIC=kitti
#   TEACHER_AP3D_MOD_CAR_MIN=55.0
#   BITRATE_MATCH_METRIC=bpp_entropy_mean
#   BITRATE_PAIR_MAX_GAP=0.05
#   ALLOW_INVALID_RECON_FALLBACK=0   (debug only)
#   ALLOW_AP_BELOW_GATE=0            (debug only)
#   SUMMARY_CSV=notebooks/kitti_map_vs_rate_summary.csv
#   DETAIL_CSV=notebooks/kitti_map_vs_rate_detail.csv
#   DETECTOR_PAIR_CSV=notebooks/kitti_map_vs_rate_pairs.csv
#   UPDATE_PAPER_TABLE=1

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

RUN_DIRS=${RUN_DIRS:-}
if [[ -z "${RUN_DIRS}" ]]; then
  echo "Error: RUN_DIRS is required (comma-separated run dirs)." >&2
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
OPENPCDET_CFG=${OPENPCDET_CFG:-${OPENPCDET_ROOT}/tools/cfgs/kitti_models/pointpillar.yaml}
OPENPCDET_CKPT=${OPENPCDET_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}
SETUP_OPENPCDET=${SETUP_OPENPCDET:-0}
REBUILD_KITTI_INFOS=${REBUILD_KITTI_INFOS:-0}
RUN_ORIGINAL_SANITY=${RUN_ORIGINAL_SANITY:-1}
SANITY_BATCH_SIZE=${SANITY_BATCH_SIZE:-1}
REQUIRE_CUDA=${REQUIRE_CUDA:-1}
KITTI_MAX_SPLIT_SAMPLES=${KITTI_MAX_SPLIT_SAMPLES:-10000}
KITTI_MAX_TRAIN_FILES=${KITTI_MAX_TRAIN_FILES:-10000}
KITTI_MIN_TRAIN_SAMPLES=${KITTI_MIN_TRAIN_SAMPLES:-1000}
KITTI_MIN_VAL_SAMPLES=${KITTI_MIN_VAL_SAMPLES:-1000}
REQUIRE_DISJOINT_TRAIN_VAL=${REQUIRE_DISJOINT_TRAIN_VAL:-1}

KITTI_ROOT_OFFICIAL=${KITTI_ROOT_OFFICIAL:-${KITTI_ROOT:-}}
if [[ -z "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL is required." >&2
  exit 1
fi
if [[ ! -d "${KITTI_ROOT_OFFICIAL}" ]]; then
  echo "Error: KITTI_ROOT_OFFICIAL not found: ${KITTI_ROOT_OFFICIAL}" >&2
  exit 1
fi

if [[ "${OPENPCDET_CFG}" != /* ]]; then
  OPENPCDET_CFG="${ROOT_DIR}/${OPENPCDET_CFG}"
fi
if [[ "${OPENPCDET_CKPT}" != /* ]]; then
  OPENPCDET_CKPT="${ROOT_DIR}/${OPENPCDET_CKPT}"
fi

if [[ "${SETUP_OPENPCDET}" == "1" ]]; then
  CONDA_ENV="${CONDA_ENV}" \
  OPENPCDET_ROOT="${OPENPCDET_ROOT}" \
  OPENPCDET_PP18M_CKPT="${OPENPCDET_CKPT}" \
  bash src/scripts/setup_openpcdet.sh
fi

if [[ ! -d "${OPENPCDET_ROOT}" ]]; then
  echo "Error: OpenPCDet root not found: ${OPENPCDET_ROOT}" >&2
  exit 1
fi

# Runtime library paths for OpenPCDet CUDA extensions.
if [[ -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  TORCH_LIB="$("${CONDA_PREFIX_DIR}/bin/python" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  NVHPC_ROOT=${NVHPC_ROOT:-/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11}
  export LD_LIBRARY_PATH="${TORCH_LIB}:${NVHPC_ROOT}/cuda/11.8/lib64:${NVHPC_ROOT}/math_libs/11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p notebooks logs "${OPENPCDET_ROOT}/data"
ln -sfn "${KITTI_ROOT_OFFICIAL}" "${OPENPCDET_ROOT}/data/kitti"
KITTI_ROOT_LINK="${OPENPCDET_ROOT}/data/kitti"
KITTI_ROOT=${KITTI_ROOT:-${KITTI_ROOT_LINK}}

if [[ "${KITTI_ROOT}" != /* ]]; then
  KITTI_ROOT="${ROOT_DIR}/${KITTI_ROOT}"
fi

echo "[kitti-map] python_env=${PY_ENV_DESC}"
echo "[kitti-map] kitti_root_official=${KITTI_ROOT_OFFICIAL}"
echo "[kitti-map] kitti_root_link=${KITTI_ROOT_LINK}"
echo "[kitti-map] kitti_root_eval=${KITTI_ROOT}"
echo "[kitti-map] run_dirs=${RUN_DIRS}"
echo "[kitti-map] cfg=${OPENPCDET_CFG}"
echo "[kitti-map] ckpt=${OPENPCDET_CKPT}"

# Sanity check expected KITTI detection format.
MISSING_LAYOUT=0
for rel in ImageSets training/velodyne training/label_2 training/calib; do
  if [[ ! -d "${KITTI_ROOT_LINK}/${rel}" ]]; then
    echo "Error: missing required directory: ${KITTI_ROOT_LINK}/${rel}" >&2
    MISSING_LAYOUT=1
  fi
done
if [[ "${MISSING_LAYOUT}" != "0" ]]; then
  echo "Error: KITTI root is not in official detection format." >&2
  exit 1
fi

for split in train val trainval test; do
  split_file="${KITTI_ROOT_LINK}/ImageSets/${split}.txt"
  if [[ -f "${split_file}" ]]; then
    split_count="$(wc -l < "${split_file}")"
    echo "[kitti-map] split_count_${split}=${split_count}"
    if [[ "${split_count}" -gt "${KITTI_MAX_SPLIT_SAMPLES}" ]]; then
      echo "Error: split ${split} has ${split_count} samples, exceeds KITTI_MAX_SPLIT_SAMPLES=${KITTI_MAX_SPLIT_SAMPLES}." >&2
      echo "Error: dataset likely not official KITTI detection format/split." >&2
      exit 1
    fi
  else
    echo "[kitti-map] split_count_${split}=missing"
  fi
done

if [[ -f "${KITTI_ROOT_LINK}/ImageSets/train.txt" ]]; then
  train_count="$(wc -l < "${KITTI_ROOT_LINK}/ImageSets/train.txt")"
  if [[ "${train_count}" -lt "${KITTI_MIN_TRAIN_SAMPLES}" ]]; then
    echo "Error: train split too small (${train_count} < ${KITTI_MIN_TRAIN_SAMPLES})." >&2
    echo "Error: likely subset root, not full KITTI protocol for Track-B sanity." >&2
    exit 1
  fi
fi
if [[ -f "${KITTI_ROOT_LINK}/ImageSets/val.txt" ]]; then
  val_count="$(wc -l < "${KITTI_ROOT_LINK}/ImageSets/val.txt")"
  if [[ "${val_count}" -lt "${KITTI_MIN_VAL_SAMPLES}" ]]; then
    echo "Error: val split too small (${val_count} < ${KITTI_MIN_VAL_SAMPLES})." >&2
    echo "Error: likely subset root, not full KITTI protocol for Track-B sanity." >&2
    exit 1
  fi
fi

if [[ "${REQUIRE_DISJOINT_TRAIN_VAL}" == "1" && -f "${KITTI_ROOT_LINK}/ImageSets/train.txt" && -f "${KITTI_ROOT_LINK}/ImageSets/val.txt" ]]; then
  overlap_count="$("${PYTHON_RUNNER[@]}" - "${KITTI_ROOT_LINK}/ImageSets/train.txt" "${KITTI_ROOT_LINK}/ImageSets/val.txt" <<'PY'
import sys
train = {x.strip() for x in open(sys.argv[1], "r", encoding="utf-8") if x.strip()}
val = {x.strip() for x in open(sys.argv[2], "r", encoding="utf-8") if x.strip()}
print(len(train & val))
PY
)"
  if [[ "${overlap_count}" != "0" ]]; then
    echo "Error: train/val overlap detected (${overlap_count} shared ids)." >&2
    echo "Error: invalid split for fair original-vs-reconstructed comparison." >&2
    exit 1
  fi
fi

train_velodyne_count=$(find -L "${KITTI_ROOT_LINK}/training/velodyne" -maxdepth 1 -type f -name '*.bin' | wc -l)
train_label_count=$(find -L "${KITTI_ROOT_LINK}/training/label_2" -maxdepth 1 -type f -name '*.txt' | wc -l)
echo "[kitti-map] train_velodyne_files=${train_velodyne_count}"
echo "[kitti-map] train_label_files=${train_label_count}"
if [[ "${train_velodyne_count}" -gt "${KITTI_MAX_TRAIN_FILES}" || "${train_label_count}" -gt "${KITTI_MAX_TRAIN_FILES}" ]]; then
  echo "Error: training file count is too large for official KITTI detection." >&2
  echo "Error: velodyne=${train_velodyne_count}, label_2=${train_label_count}, max=${KITTI_MAX_TRAIN_FILES}" >&2
  exit 1
fi

# Validate OpenPCDet/CUDA extension imports before evaluation.
export PYTHONPATH="${ROOT_DIR}/${OPENPCDET_ROOT}:${PYTHONPATH:-}"
"${PYTHON_RUNNER[@]}" - <<'PY'
import os
import torch
print(f"[kitti-map] torch={torch.__version__} cuda={torch.version.cuda} is_available={torch.cuda.is_available()}")
import pcdet  # noqa: F401
from pcdet.ops.iou3d_nms import iou3d_nms_cuda  # noqa: F401
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda  # noqa: F401
print("[kitti-map] OpenPCDet import check OK")
PY

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  CUDA_OK="$("${PYTHON_RUNNER[@]}" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
  if [[ "${CUDA_OK}" != "1" ]]; then
    echo "Error: CUDA is required but torch.cuda.is_available() is False." >&2
    exit 1
  fi
fi

INFO_TRAIN="${KITTI_ROOT_LINK}/kitti_infos_train.pkl"
INFO_VAL="${KITTI_ROOT_LINK}/kitti_infos_val.pkl"
if [[ "${REBUILD_KITTI_INFOS}" == "1" || ! -f "${INFO_TRAIN}" || ! -f "${INFO_VAL}" ]]; then
  INFO_LOG="logs/kitti_create_infos_$(date +%Y%m%d_%H%M%S).log"
  echo "[kitti-map] generating kitti infos -> ${INFO_LOG}"
  (
    cd "${OPENPCDET_ROOT}"
    "${PYTHON_RUNNER[@]}" -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml \
      2>&1 | tee "${ROOT_DIR}/${INFO_LOG}"
  )
fi
if [[ ! -f "${INFO_TRAIN}" || ! -f "${INFO_VAL}" ]]; then
  echo "Error: kitti_infos generation failed (${INFO_TRAIN}, ${INFO_VAL})." >&2
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-1}
WORKERS=${WORKERS:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
SPLIT=${SPLIT:-val}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-}
EVAL_METRIC=${EVAL_METRIC:-kitti}
TEACHER_AP3D_MOD_CAR_MIN=${TEACHER_AP3D_MOD_CAR_MIN:-55.0}
BITRATE_MATCH_METRIC=${BITRATE_MATCH_METRIC:-bpp_entropy_mean}
BITRATE_PAIR_MAX_GAP=${BITRATE_PAIR_MAX_GAP:-0.05}
ALLOW_INVALID_RECON_FALLBACK=${ALLOW_INVALID_RECON_FALLBACK:-0}
ALLOW_AP_BELOW_GATE=${ALLOW_AP_BELOW_GATE:-0}
SUMMARY_CSV=${SUMMARY_CSV:-notebooks/kitti_map_vs_rate_summary.csv}
DETAIL_CSV=${DETAIL_CSV:-notebooks/kitti_map_vs_rate_detail.csv}
DETECTOR_PAIR_CSV=${DETECTOR_PAIR_CSV:-notebooks/kitti_map_vs_rate_pairs.csv}
UPDATE_PAPER_TABLE=${UPDATE_PAPER_TABLE:-1}

if [[ "${RUN_ORIGINAL_SANITY}" == "1" ]]; then
  CAR_AP3D_MOD=""
  SANITY_LOG="logs/kitti_original_sanity_$(date +%Y%m%d_%H%M%S).log"
  echo "[kitti-map] running original sanity (tools/test.py) -> ${SANITY_LOG}"
  (
    cd "${OPENPCDET_ROOT}/tools"
    "${PYTHON_RUNNER[@]}" test.py \
      --cfg_file "${OPENPCDET_CFG}" \
      --ckpt "${OPENPCDET_CKPT}" \
      --batch_size "${SANITY_BATCH_SIZE}" \
      2>&1 | tee "${ROOT_DIR}/${SANITY_LOG}"
  )

  set +e
  CAR_AP3D_MOD="$("${PYTHON_RUNNER[@]}" - "${ROOT_DIR}/${SANITY_LOG}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
pat = re.compile(r"Car\s+AP@.*?3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)", flags=re.IGNORECASE | re.DOTALL)
matches = list(pat.finditer(text))
if not matches:
    print("nan")
    raise SystemExit(2)
car_mod = float(matches[-1].group(2))
print(f"{car_mod:.6f}")
if car_mod <= 0.0:
    raise SystemExit(3)
PY
)"
  parse_status=$?
  set -e
  if [[ "${parse_status}" == "2" ]]; then
    echo "Error: failed to parse Car 3D AP(mod) from sanity log (${SANITY_LOG})." >&2
    exit 1
  fi
  if [[ "${parse_status}" == "3" ]]; then
    echo "Error: original sanity AP check failed (Car 3D AP(mod)=${CAR_AP3D_MOD} <= 0)." >&2
    exit 1
  fi
  if [[ "${parse_status}" != "0" ]]; then
    echo "Error: sanity parser failed with status=${parse_status}." >&2
    exit 1
  fi
  echo "[kitti-map] original Car 3D AP(mod)=${CAR_AP3D_MOD} (>0 confirmed)"
fi

TABLE_FLAG=()
if [[ "${UPDATE_PAPER_TABLE}" == "1" ]]; then
  TABLE_FLAG=(--update_paper_table)
fi

SPLIT_MANIFEST_FLAG=()
if [[ -n "${SPLIT_MANIFEST}" ]]; then
  SPLIT_MANIFEST_FLAG=(--split_manifest "${SPLIT_MANIFEST}")
fi

DEBUG_EVAL_FLAGS=()
if [[ "${ALLOW_INVALID_RECON_FALLBACK}" == "1" ]]; then
  DEBUG_EVAL_FLAGS+=(--allow_invalid_reconstruction_fallback)
fi
if [[ "${ALLOW_AP_BELOW_GATE}" == "1" ]]; then
  DEBUG_EVAL_FLAGS+=(--allow_ap_below_gate)
fi

"${PYTHON_RUNNER[@]}" src/train/evaluate_kitti_map_vs_rate.py \
  --kitti_root "${KITTI_ROOT}" \
  --run_dirs_csv "${RUN_DIRS}" \
  --openpcdet_cfg "${OPENPCDET_CFG}" \
  --openpcdet_ckpt "${OPENPCDET_CKPT}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --max_frames "${MAX_FRAMES}" \
  --split "${SPLIT}" \
  "${SPLIT_MANIFEST_FLAG[@]}" \
  --eval_metric "${EVAL_METRIC}" \
  --teacher_ap3d_mod_car_min "${TEACHER_AP3D_MOD_CAR_MIN}" \
  --bitrate_match_metric "${BITRATE_MATCH_METRIC}" \
  --bitrate_pair_max_gap "${BITRATE_PAIR_MAX_GAP}" \
  "${DEBUG_EVAL_FLAGS[@]}" \
  --output_summary_csv "${SUMMARY_CSV}" \
  --output_detail_csv "${DETAIL_CSV}" \
  "${TABLE_FLAG[@]}"

WRAPPER_ORIG_AP="$("${PYTHON_RUNNER[@]}" - "${SUMMARY_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
ap_vals = []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if str(row.get("mode", "")).strip().lower() != "original":
            continue
        try:
            ap_vals.append(float(row.get("ap3d_car_mod", "nan")))
        except Exception:
            pass
if not ap_vals:
    print("nan")
else:
    print(f"{ap_vals[0]:.6f}")
PY
)"
echo "[kitti-map] wrapper original Car 3D AP(mod)=${WRAPPER_ORIG_AP}"

if [[ "${RUN_ORIGINAL_SANITY}" == "1" ]]; then
  DIAG_STATUS="$("${PYTHON_RUNNER[@]}" - "${CAR_AP3D_MOD}" "${WRAPPER_ORIG_AP}" <<'PY'
import math
import sys

try:
    sanity = float(sys.argv[1])
except Exception:
    sanity = float("nan")
try:
    wrapper = float(sys.argv[2])
except Exception:
    wrapper = float("nan")

def finite(x):
    return x == x and not math.isinf(x)

if finite(sanity) and sanity > 0 and (not finite(wrapper) or wrapper <= 0):
    print("branch_a")
elif finite(sanity) and sanity <= 0:
    print("branch_b")
else:
    print("ok")
PY
)"
  if [[ "${DIAG_STATUS}" == "branch_a" ]]; then
    echo "Error: branch-A detected (tools/test.py non-zero AP, wrapper original AP is zero/NaN)." >&2
    echo "Error: inspect AP parsing or wrapper protocol alignment in evaluate_kitti_map_vs_rate.py." >&2
    exit 1
  fi
  if [[ "${DIAG_STATUS}" == "branch_b" ]]; then
    echo "Error: branch-B detected (tools/test.py original AP is zero)." >&2
    echo "Error: inspect KITTI root/protocol/checkpoint consistency first." >&2
    exit 1
  fi
fi

"${PYTHON_RUNNER[@]}" src/utils/match_bitrate_budget_detector.py \
  --inputs "${SUMMARY_CSV}" \
  --metric "${BITRATE_MATCH_METRIC}" \
  --max_gap "${BITRATE_PAIR_MAX_GAP}" \
  --output_csv "${DETECTOR_PAIR_CSV}"

echo "[kitti-map] done"
echo "[kitti-map] summary_csv=${SUMMARY_CSV}"
echo "[kitti-map] detail_csv=${DETAIL_CSV}"
echo "[kitti-map] pair_csv=${DETECTOR_PAIR_CSV}"
