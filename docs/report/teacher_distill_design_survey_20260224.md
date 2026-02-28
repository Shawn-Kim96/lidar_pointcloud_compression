# Teacher Distillation Design Survey (2026-02-24)

## Scope
- Question:
  - Should we keep `TeacherAdapter` with PointPillar as a direct teacher for 2D range-image importance maps?
- Goal:
  - Define a technically consistent teacher strategy for Stage2 distillation under KITTI endpoint constraints.

## Key Findings

### 1) PointPillar is a strong detector teacher, but not a direct pixel-space range-image teacher
- PointPillar encodes point clouds into pillarized BEV features (not native range-image feature grids).
- Using PointPillar score maps as direct supervision for range-image importance pixels introduces modality/view mismatch.
- Implication:
  - PointPillar is valid for endpoint AP evaluation and pseudo-label quality checks.
  - PointPillar is weak as a direct feature-map teacher for range-image pixel-level distillation.

### 2) If feature distillation is needed, use modality-aligned teachers
- Range-view detectors explicitly operate on LiDAR range images and are structurally aligned with range-image students.
- Prior range-view detector families:
  - LaserNet (range-view detector).
  - RangeDet (range-view detector with strong Waymo results).
  - FCOS-LiDAR (range-image one-stage detector).
- Implication:
  - For per-pixel/per-grid distillation losses, range-view teacher is more coherent than BEV teacher.

### 3) Cross-modality distillation generally needs a bridge (teaching assistant)
- Literature on teacher-assistant KD shows large teacher-student gaps degrade transfer quality.
- Cross-modal 3D detection KD literature (e.g., MonoTAKD) also reports direct transfer inefficiency due to representation gap.
- Implication:
  - If PointPillar (BEV) must remain teacher, introduce an intermediate aligned assistant:
    - BEV teacher -> aligned assistant -> range-image student.

### 4) Confidence-gated distillation is justified
- Distillation quality depends on teacher reliability.
- Prior KD work uses confidence-weighted transfer to suppress noisy teacher guidance.
- Implication:
  - Current `distill_teacher_score_min` / score weighting is directionally correct and should be retained.

### 5) Original-vs-reconstructed AP with same detector is a valid codec evaluation
- Task-aware compression literature evaluates downstream detection impact from compressed/reconstructed point clouds.
- Protocol requirement:
  - same detector weights, same split, same metric, same post-processing.
- This measures task fidelity under compression, not detector architecture superiority.

## Recommended Design Decision

### Decision A (short-term, practical)
- Keep PointPillar for Track-B endpoint only:
  - original AP sanity check
  - reconstructed AP comparison
- Do not use PointPillar feature maps as direct range-image pixel teacher.
- Stage2 distillation for compression model should use:
  - ROI target supervision (range-image aligned), and/or
  - confidence-gated weak task signals that do not require strict pixel alignment.

### Decision B (mid-term, stronger distillation)
- Replace/add teacher with range-view detector to align feature space:
  - distill from range-view objectness/feature maps to importance head.
- If keeping PointPillar teacher:
  - add a teaching assistant bridge network for representation alignment.

## Suggested Experimental Matrix
- E1: No distill (ROI-only) baseline on KITTI.
- E2: PointPillar endpoint-guided (no direct feature-map distill).
- E3: Range-view teacher feature distill.
- E4: PointPillar + assistant bridge distill.
- Compare by:
  - Track-A: all/roi/bg MSE + ROI overlap metrics
  - Track-B: AP3D(mod) original vs reconstructed, and AP drop.

## References
- PointPillars (CVPR 2019):
  - https://arxiv.org/abs/1812.05784
- OpenPCDet KITTI model zoo / PointPillar baseline:
  - https://github.com/open-mmlab/OpenPCDet
- OpenPCDet KITTI data layout and infos generation:
  - https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md
- LaserNet (range-view detector):
  - https://arxiv.org/abs/1903.08701
- RangeDet (range-view detector):
  - https://arxiv.org/abs/2103.10039
- FCOS-LiDAR (range-image detector):
  - https://arxiv.org/abs/2205.13764
- Teacher Assistant KD:
  - https://arxiv.org/abs/1902.03393
- MonoTAKD (cross-modal distillation with assistant):
  - https://arxiv.org/abs/2404.04910
- Confidence-weighted distillation example:
  - https://arxiv.org/abs/1908.00858
- Task-aware point cloud compression with detection objective:
  - https://arxiv.org/abs/2502.04804
  - https://arxiv.org/abs/2405.01750
