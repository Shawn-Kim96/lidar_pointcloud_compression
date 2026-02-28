#!/bin/bash
# Optional teacher fine-tune branch for KITTI detection.
#
# Required env:
#   OPENPCDET_ROOT=/path/to/OpenPCDet
#   TEACHER_TRAIN_CFG=/path/to/openpcdet_train_cfg.yaml
#   KITTI_ROOT=/path/to/kitti
#
# Optional env:
#   AP_GATE_CSV=notebooks/kitti_map_vs_rate_summary.csv
#   TEACHER_AP3D_MOD_CAR_MIN=55.0
#   PRETRAIN_CKPT=/path/to/pointpillars_pretrained.pth
#   BATCH_SIZE=4
#   WORKERS=4
#   EPOCHS=80
#   EXTRA_TAG=teacher_ft_kitti

set -euo pipefail

OPENPCDET_ROOT=${OPENPCDET_ROOT:-}
TEACHER_TRAIN_CFG=${TEACHER_TRAIN_CFG:-}
KITTI_ROOT=${KITTI_ROOT:-}

if [[ -z "${OPENPCDET_ROOT}" ]]; then
  echo "Error: OPENPCDET_ROOT is required." >&2
  exit 1
fi
if [[ -z "${TEACHER_TRAIN_CFG}" ]]; then
  echo "Error: TEACHER_TRAIN_CFG is required." >&2
  exit 1
fi
if [[ -z "${KITTI_ROOT}" ]]; then
  echo "Error: KITTI_ROOT is required." >&2
  exit 1
fi

AP_GATE_CSV=${AP_GATE_CSV:-}
TEACHER_AP3D_MOD_CAR_MIN=${TEACHER_AP3D_MOD_CAR_MIN:-55.0}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-80}
EXTRA_TAG=${EXTRA_TAG:-teacher_ft_kitti}

if [[ ! -d "${OPENPCDET_ROOT}" ]]; then
  echo "Error: OPENPCDET_ROOT does not exist: ${OPENPCDET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_TRAIN_CFG}" ]]; then
  echo "Error: TEACHER_TRAIN_CFG not found: ${TEACHER_TRAIN_CFG}" >&2
  exit 1
fi
if [[ ! -d "${KITTI_ROOT}" ]]; then
  echo "Error: KITTI_ROOT does not exist: ${KITTI_ROOT}" >&2
  exit 1
fi

if [[ -n "${AP_GATE_CSV}" && -f "${AP_GATE_CSV}" ]]; then
  current_ap=$(python - <<'PY' "${AP_GATE_CSV}"
import csv, sys
path = sys.argv[1]
best = float("nan")
with open(path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if str(row.get("mode", "")).lower() != "original":
            continue
        try:
            v = float(row.get("ap3d_car_mod", "nan"))
        except Exception:
            continue
        if v == v:
            best = v
            break
print(best)
PY
)
  if python - <<'PY' "${current_ap}" "${TEACHER_AP3D_MOD_CAR_MIN}"
import sys
ap = float(sys.argv[1])
thr = float(sys.argv[2])
sys.exit(0 if (ap == ap and ap >= thr) else 1)
PY
  then
    echo "[teacher-ft] gate passed (ap3d_car_mod=${current_ap} >= ${TEACHER_AP3D_MOD_CAR_MIN}), skipping fine-tune."
    exit 0
  fi
  echo "[teacher-ft] gate failed (ap3d_car_mod=${current_ap}), running fine-tune."
else
  echo "[teacher-ft] gate CSV not provided or missing; running fine-tune by request."
fi

cd "${OPENPCDET_ROOT}"

cmd=(python tools/train.py
  --cfg_file "${TEACHER_TRAIN_CFG}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --epochs "${EPOCHS}"
  --extra_tag "${EXTRA_TAG}"
  --set DATA_CONFIG.DATA_PATH "${KITTI_ROOT}"
)

if [[ -n "${PRETRAIN_CKPT}" ]]; then
  if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Error: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2
    exit 1
  fi
  cmd+=(--ckpt "${PRETRAIN_CKPT}")
fi

echo "[teacher-ft] running: ${cmd[*]}"
"${cmd[@]}"

echo "[teacher-ft] done. Use OpenPCDet output checkpoint as distill teacher."
