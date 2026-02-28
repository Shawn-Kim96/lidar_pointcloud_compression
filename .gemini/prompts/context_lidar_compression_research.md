# Project Context: ROI-Aware LiDAR Compression

## Research Goal
- Validate whether ROI-aware LiDAR compression improves:
  - reconstruction quality at same or similar bitrate,
  - downstream 3D detection robustness.

## Evaluation Tracks
- Track-A (diagnostic, fast iteration):
  - all/roi/bg MSE, ROI overlap (precision/recall/IoU), bitrate proxies.
- Track-B (claim-critical):
  - KITTI + OpenPCDet endpoint metrics (AP3D/mAP).
  - compare original vs reconstructed under identical protocol.

## Current Architecture
- Range image -> encoder (resnet/darknet) -> latent -> importance head
- Importance-conditioned quantization (uniform or adaptive)
- Decoder reconstructs 5ch range representation.

## Stage Semantics
- Stage0: uniform baseline.
- Stage1: adaptive ROI student without teacher distillation.
- Stage2: Stage1 + teacher distillation.

## Stage2 Distill-Fix Notes
- Distill alignment added: adaptive pooling with shared grid (`16x32`).
- Distill feature source added: `energy_map` alongside legacy `channel_mean`.
- Teacher confidence gate added: `distill_teacher_score_min`.
- Best recent stability combo:
  - `adaptive_pool + energy_map + score-gate`.

## Important Findings To Respect
- Track-B AP zero runs are not detector-gain evidence.
- Stage0/Stage1 RCA indicates a major representation bottleneck:
  - `raw -> fixed 64x1024 range -> points` causes heavy sparsification.
  - observed point retention around 0.4169 in a 200-frame check.
  - detector drop appears even in identity project/unproject path.

## Claim Guardrails
- Keep Track-A improvements and Track-B claims separate.
- Do not claim detection improvement unless:
  1. original AP sanity is non-zero on official protocol,
  2. reconstructed comparison is run with same cfg/ckpt/split/metric.

## Key Paths
- Detector endpoint eval:
  - `src/scripts/run_kitti_map_vs_rate.sh`
  - `src/train/evaluate_kitti_map_vs_rate.py`
- Reconstruction export:
  - `src/utils/recon_pointcloud_export.py`
- Core report ledger:
  - `docs/report/experiments.md`
  - `docs/notes/research_journal_en.md`
  - `docs/notes/research_progress.md`

