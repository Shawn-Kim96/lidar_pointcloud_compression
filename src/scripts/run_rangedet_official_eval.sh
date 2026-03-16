#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/018219422/lidar_pointcloud_compression}
PY=${PY:-$HOME/miniconda3/envs/lidarcomp311/bin/python}
OUTDIR=${OUTDIR:-$ROOT/logs/rangedet_kitti_official_eval_native}
mkdir -p "$OUTDIR"
cd "$ROOT"

ARCHIVES=(
  logs/rangedet_eval_outputs/260312_195110_rdpatch12h_raw_patched_output_dict_24e.pkl
  logs/rangedet_eval_outputs/260312_195110_rdpatch12h_nqa_patched_output_dict_24e.pkl
  logs/rangedet_eval_outputs/260312_195110_rdpatch12h_nqb_patched_output_dict_24e.pkl
  logs/rangedet_eval_outputs/260312_195110_rdpatch12h_uqa_patched_output_dict_24e.pkl
  logs/rangedet_eval_outputs/260312_195110_rdpatch12h_uqb_patched_output_dict_24e.pkl
)

"$PY" src/scripts/eval_rangedet_archive_kitti_official.py \
  --archives "${ARCHIVES[@]}" \
  --output-csv logs/260315_rangedet_kitti_official_summary_native.csv \
  --output-dir "$OUTDIR/native" \
  --box-mode native

"$PY" src/scripts/eval_rangedet_archive_kitti_official.py \
  --archives logs/rangedet_eval_outputs/260312_195110_rdpatch12h_raw_patched_output_dict_24e.pkl \
  --output-csv logs/260315_rangedet_kitti_official_raw_boxmode_swap_lw.csv \
  --output-dir "$OUTDIR/swap_lw" \
  --box-mode swap_lw

"$PY" src/scripts/eval_rangedet_archive_kitti_official.py \
  --archives logs/rangedet_eval_outputs/260312_195110_rdpatch12h_raw_patched_output_dict_24e.pkl \
  --output-csv logs/260315_rangedet_kitti_official_raw_boxmode_swap_lw_yaw90.csv \
  --output-dir "$OUTDIR/swap_lw_yaw90" \
  --box-mode swap_lw_yaw90
