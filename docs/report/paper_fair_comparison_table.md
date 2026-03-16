# Paper-Facing Fair Comparison Table (Dual-Track)

## Scope and Fairness Protocol
- Cutoff snapshot: `2026-02-25`
- Eval subset: `16` validation frames (pilot subset)
- Fairness axis for cross-objective comparison:
  - primary: `bpp_entropy_mean` (bitrate proxy)
  - secondary: `all_mse_mean`, `roi_mse_mean`, `bg_mse_mean`
- Source files:
  - `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q6.csv`
  - `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q8.csv`
  - `notebooks/oracle_eval_summary_260219_resnet_pilot_s1adapt.csv`
  - `notebooks/oracle_eval_summary_260219_resnet_pilot_s2distill.csv`
  - `notebooks/oracle_eval_summary_260223_fix2_no_scoregate_best.csv`
  - `notebooks/oracle_eval_summary_260223_fix2_scoregate_best.csv`
  - `notebooks/matched_bitrate_pairs_260219_pilot.csv`
  - `notebooks/kitti_map_vs_rate_summary.csv` (official detector endpoint track)

## Track Boundary (Claim Policy)
- `Track-A (SemanticKITTI)`: codec/ROI behavior (`bpp`, `all/roi/bg MSE`, ROI overlap metrics).
- `Track-B (KITTI detection)`: official detector endpoint (`3D AP/mAP` on original vs reconstructed clouds).
- We do **not** mix Track-A proxy metrics and Track-B official detector metrics into one claim.

## Why Stage2 Did Not Train Well (Root Causes)
- `Problem 1: teacher/task mismatch`
  - PointPillars target is 3D bounding-box detection, while current SemanticKITTI supervision is point-wise semantic labels.
  - A proxy distillation path can improve optimization stability, but it does not guarantee detector-endpoint improvement.
- `Problem 2: resize-only distillation alignment`
  - Previous feature distillation effectively matched student/teacher maps after direct spatial resize.
  - With different backbone resolutions and semantics, naive resize can create weak or misleading alignment signals.

## Implemented Fix Strategy
- `Fix A: dual-track endpoint separation`
  - Keep SemanticKITTI for ROI-aware codec diagnostics (Track-A).
  - Use KITTI + OpenPCDet for official detector AP endpoint (Track-B).
- `Fix B: distillation alignment upgrade`
  - Added `distill_align_mode`:
    - `resize` (legacy),
    - `adaptive_pool` (recommended; pools both maps to common grid).
  - Added `distill_feature_source`:
    - `channel_mean` (legacy),
    - `energy_map` (recommended; channel-agnostic spatial energy),
    - `none` (disable feature distill ablation).
  - Added optional teacher quality gate:
    - `distill_teacher_score_min` + `distill_teacher_score_weight`.
    - suppress distillation on low-confidence teacher samples.
- `Fix C: runnable ablation scripts`
  - `src/scripts/run_stage2_distill_fix_pilot_local.sh`
  - `src/scripts/run_stage2_distill_fix_ablation.sh`

## Table-A1. Bitrate-Matched Native Comparison (SemanticKITTI)


| Reference family (uniform) | Candidate family | Frames | Reference `bpp_entropy` | Candidate `bpp_entropy` | abs `delta_bpp` | Reference ROI MSE | Candidate ROI MSE | ROI MSE gain vs ref | Reference All MSE | Candidate All MSE | All MSE gain vs ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform Baseline (ResNet, q6) | Adaptive ROI Student (ResNet, no distill) | 16 | 0.6668 | 0.8064 | 0.1396 | 228.0958 | 227.8031 | +0.1283% | 161.3999 | 161.2016 | +0.1229% |
| Uniform Baseline (ResNet, q8) | Adaptive Distilled Student (ResNet, ld=0.1) | 16 | 0.9598 | 0.8687 | 0.0910 | 227.3080 | 227.6069 | -0.1315% | 161.1836 | 161.0156 | +0.1043% |


Interpretation:
- Current matched pairs are nearest-neighbor matches, not exact bitrate matches.
- Pair-1 budget gap is still large (`~20.9%` of reference bitrate), so conclusions are directional only.
- Pair-2 is tighter (`~9.5%` gap) and shows similar global distortion but no clear ROI-distortion win yet.

## Table-A2. Native vs Oracle ROI Decomposition (SemanticKITTI)

- `native`: quantizer uses the model-predicted importance map.
- `oracle_roi`: quantizer uses GT ROI mask directly (head prediction bypassed), to isolate quantizer upper-bound potential.

| Family | Mode | Frames | `bpp_entropy_mean` | ROI MSE mean | All MSE mean | Precision | Recall | IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive ROI Student (ResNet, no distill) | Native importance | 16 | 0.8064 | 227.8031 | 161.2016 | 0.0083 | 1.0000 | 0.0083 |
| Adaptive ROI Student (ResNet, no distill) | Oracle ROI (GT-forced) | 16 | 0.4177 | 227.8031 | 161.2015 | 0.0083 | 1.0000 | 0.0083 |
| Adaptive Distilled Student (ResNet, ld=0.1) | Native importance | 16 | 0.8687 | 227.6069 | 161.0156 | 0.0089 | 1.0000 | 0.0089 |
| Adaptive Distilled Student (ResNet, ld=0.1) | Oracle ROI (GT-forced) | 16 | 0.4492 | 227.6073 | 161.0152 | 0.0089 | 1.0000 | 0.0089 |

Interpretation:
- Oracle forcing roughly halves bitrate (`-48.2%` to `-48.3%`) at nearly unchanged MSE.
- This indicates quantizer potential exists, while native importance calibration remains the main bottleneck.

## Table-A3. Stage2 Distill-Fix Best-vs-Best (Track-A Quick Check, 2026-02-23)

- Scope:
  - `native` mode only, `16` frames.
  - compare Stage2 distill-fix best runs against nearest available Stage0 uniform bitrate anchor.
  - anchor used: `Uniform Baseline (ResNet, q8)` from `2026-02-19` pilot.
  - `Importance Precision/Recall/IoU` below are importance-vs-ROI overlap metrics (not detector AP).

| Reference family (uniform) | Candidate family (distill-fix) | Frames | Reference `bpp_entropy` | Candidate `bpp_entropy` | abs `delta_bpp` | Reference ROI MSE | Candidate ROI MSE | ROI MSE gain vs ref | Reference All MSE | Candidate All MSE | All MSE gain vs ref | Importance Precision | Importance Recall | Importance IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform Baseline (ResNet, q8) | `energy_pool_16x32` best (`j22129_r0`) | 16 | 0.9598 | 0.8752 | 0.0845 | 227.3080 | 6.5502 | +97.1184% | 161.1836 | 3.1224 | +98.0628% | 0.0568 | 0.5375 | 0.0542 |
| Uniform Baseline (ResNet, q8) | `energy_pool_16x32_scoregate015` best (`j22137_r8`) | 16 | 0.9598 | 0.8986 | 0.0612 | 227.3080 | 4.7754 | +97.8992% | 161.1836 | 2.5997 | +98.3871% | 0.0707 | 0.6810 | 0.0684 |

Interpretation:
- With currently available anchors, score-gated case is closer to reference bitrate and better than non-gated case on ROI/all MSE and overlap.
- Importance overlap is still weak in absolute terms (high FP), so Track-A calibration bottleneck remains.
- This is a provisional quick comparison across snapshots; final claim should be regenerated with fully matched protocol/data split.

## Current Limitation for Paper Claim
- These tables are fairer than raw loss comparison, but still pilot-scale (`16` frames).
- Track-B detector endpoint currently reports `AP3D=0` in `notebooks/kitti_map_vs_rate_summary.csv`.
- For thesis-grade claim, add:
  - tighter bitrate matching (e.g., uniform `q7` + more bitrate anchor points),
  - full validation split,
  - detector-level endpoint metric (`3D mAP`) on reconstructed point clouds.

## PointPillars 3D mAP Comparison Feasibility
- Short answer: yes, it is feasible and recommended for final claim.
- Current repo status:
  - PointPillars checkpoint download is available (`src/scripts/download_pointpillar_checkpoint.sh`).
  - Distillation adapter path is still proxy/zhulf-specific (`openpcdet` backend in `src/utils/teacher_adapter.py` remains `NotImplementedError`).
  - Dedicated detector-endpoint evaluation path is now provided separately:
    - `src/train/evaluate_kitti_map_vs_rate.py`
    - `src/scripts/run_kitti_map_vs_rate.sh`
  - Current score in logs/notebooks is detector-proxy score, not official KITTI/SemanticKITTI 3D mAP.
- What is needed for real 3D mAP:
  - run detector inference on original point clouds and reconstructed point clouds,
  - evaluate with official 3D detection metric pipeline (class-wise AP/mAP),
  - report `mAP vs bitrate` for Uniform vs Adaptive ROI vs Distilled families at matched budgets.

<!-- BEGIN_TABLE_B -->

## Table B. KITTI Detector Endpoint (Official 3D AP, Reference vs Reconstructed Pair)

- `mode=identity`: detector AP on the chosen reference baseline.
- `mode=reconstructed`: detector AP on compression-reconstructed point clouds.
- Rows are paired by the same `run_dir` to make `reference vs reconstructed` comparison explicit.

| Model family | Frames | Identity Car 3D AP (mod) | Reconstructed Car 3D AP (mod) | Identity mAP3D(mod) | Reconstructed mAP3D(mod) | map_drop_vs_reference | Reconstructed `bpp_entropy_mean` | fairness_tag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| No-Quant Autoencoder (ResNet) | 3769 | 69.88 | 4.33 | 52.74 | 2.19 | 50.55 | 0.0000 | unmatched |

<!-- END_TABLE_B -->
