# Stage2 Distill-Fix TwoExp Sweep Summary (`job 22127`)

## Research Goal and Claim Boundary
- Goal:
  - verify whether ROI-aware LiDAR compression improves reconstruction quality and downstream detection at similar bitrate.
- Dual-track evaluation policy:
  - `Track-A` (SemanticKITTI): codec/ROI metrics (`all/roi/bg MSE`, ROI overlap, bitrate proxy).
  - `Track-B` (KITTI + OpenPCDet): detector endpoint (`3D AP/mAP`).
- Claim rule:
  - Track-A gains are not used as detector-quality claims unless Track-B improves.

## Current Stage2 Setup (Core)
- Pipeline:
  - `range image -> encoder (resnet/darknet) -> latent -> importance head -> adaptive quantization -> decoder`
- Distillation upgrades used in this sweep:
  - `distill_align_mode=adaptive_pool`
  - `distill_feature_source=energy_map`
  - `distill_teacher_score_min`-based score gating

## Why Stage2 Previously Failed
- teacher task-space and supervision-space mismatch.
- resize-only distillation alignment produced weak signals.
- objective conflict among rate / importance / distill destabilized importance calibration.

## Sweep Summary (`2026-02-23`, 12 runs)
- Total matched runs: `12`

### Best Run Per Case

| Case | run_id | lr | lambda_distill | best_loss | final_loss | final_imp_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| energy_pool_16x32 | j22129_r0 | 0.0001 | 0.05 | 1.1291 | 1.1292 | 0.1432 |
| energy_pool_16x32_scoregate015 | j22137_r8 | 0.0001 | 0.2 | 0.8168 | 0.8168 | 0.1472 |

### Full Sweep

| Case | run_id | lr | lambda_distill | score_min | best_loss | final_loss | final_rate_proxy | final_imp_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| energy_pool_16x32 | j22132_r3 | 5e-05 | 0.05 | 0.0 | 1.1801 | 1.1801 | 50.1937 | 0.1438 |
| energy_pool_16x32 | j22129_r0 | 0.0001 | 0.05 | 0.0 | 1.1291 | 1.1292 | 50.0323 | 0.1432 |
| energy_pool_16x32 | j22133_r4 | 5e-05 | 0.1 | 0.0 | 1.3184 | 1.3184 | 48.8957 | 0.1384 |
| energy_pool_16x32 | j22130_r1 | 0.0001 | 0.1 | 0.0 | 1.2803 | 1.2803 | 48.7390 | 0.1378 |
| energy_pool_16x32 | j22134_r5 | 5e-05 | 0.2 | 0.0 | 1.5556 | 1.5556 | 46.6746 | 0.1292 |
| energy_pool_16x32 | j22131_r2 | 0.0001 | 0.2 | 0.0 | 1.4993 | 1.5017 | 46.6525 | 0.1291 |
| energy_pool_16x32_scoregate015 | j22138_r9 | 5e-05 | 0.05 | 0.15 | 0.9256 | 0.9268 | 51.4168 | 0.1488 |
| energy_pool_16x32_scoregate015 | j22135_r6 | 0.0001 | 0.05 | 0.15 | 0.8297 | 0.8307 | 51.1406 | 0.1476 |
| energy_pool_16x32_scoregate015 | j22139_r10 | 5e-05 | 0.1 | 0.15 | 0.9269 | 0.9269 | 51.2087 | 0.1479 |
| energy_pool_16x32_scoregate015 | j22136_r7 | 0.0001 | 0.1 | 0.15 | 0.8172 | 0.8172 | 51.0634 | 0.1473 |
| energy_pool_16x32_scoregate015 | j22127_r11 | 5e-05 | 0.2 | 0.15 | 0.9310 | 0.9310 | 51.2721 | 0.1482 |
| energy_pool_16x32_scoregate015 | j22137_r8 | 0.0001 | 0.2 | 0.15 | 0.8168 | 0.8168 | 51.0490 | 0.1472 |

## Quick ROI Validation (16 frames, native)

| Best config | all_mse | roi_mse | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-scoregate (`j22129_r0`) | 3.122 | 6.550 | 0.0542 | 0.0568 | 0.5375 |
| scoregate (`j22137_r8`) | 2.600 | 4.775 | 0.0684 | 0.0707 | 0.6810 |

- Quick read:
  - score-gated case outperformed non-gated case on both reconstruction and ROI overlap metrics.

## Open Issue (Track-B)
- File:
  - `notebooks/kitti_map_vs_rate_summary.csv`
- Current status:
  - KITTI detector endpoint AP is still `0` (`Car/Ped/Cyc`, `mAP3D(mod)=0`).
- Implication:
  - Track-B pipeline connectivity is confirmed, but detector-quality claim remains unavailable.

## Reliable Conclusion at This Point
- Distillation training stability is partially resolved.
- Current best combination:
  - `adaptive_pool + energy_map + score-gate`
- Final claim on `3D AP` improvement is still pending.
