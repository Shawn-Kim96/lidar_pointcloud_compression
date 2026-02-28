# LiDAR Compression Research Progress

## 1) Problem We Are Solving
This project targets ROI-aware LiDAR range-image compression for downstream 3D perception.

Core question:
- Can we compress aggressively while preserving information that matters for detection-relevant regions (ROI)?

Practical constraints:
- The model must run as an end-to-end trainable compression pipeline.
- ROI importance should be reliable enough to control adaptive quantization.
- Results must be reproducible across large parameter sweeps.

---

## 2) Latest Model Architecture

Current training pipeline (latest):
1. Input:
   - 2D range image with 5 channels (`range`, `intensity`, `x`, `y`, `z`).
2. Encoder backbone:
   - `resnet` or `darknet`.
3. Latent projection:
   - backbone output projected to latent channels (default 64).
4. Importance estimation:
   - `importance_head_type=pp_lite` (PointPillars-inspired FPN-like head).
5. Adaptive quantization:
   - level map controlled by importance:
   - `level_map = bg_levels + importance * (roi_levels - bg_levels)`
6. Decoder:
   - reconstructs 5-channel range-image representation.
7. Optional teacher distillation (Stage2):
   - teacher backend: `pointpillars_zhulf`
   - checkpoint: `data/checkpoints/pointpillars_epoch_160.pth`

Current default objective recipes:
- Stage1: `balanced_v1` (`normalized_global` rate + weighted BCE importance)
- Stage2: `balanced_v2` (`normalized_bg` rate + weighted BCE + separation margin)

---

## 3) Stage Definitions
This work now uses the following stage semantics:

- Stage1:
  - Adaptive quantization + ROI supervision.
  - No teacher distillation.
  - Goal: validate pure ROI-aware compression learning.

- Stage2:
  - Stage1 + teacher distillation.
  - Goal: improve ROI/feature quality with external teacher signal.

- Stage3 (planned):
  - Detector-facing validation and calibration stage.
  - Focus: convert compression losses into robust detector-relevant behavior.
  - Includes threshold calibration, detector-level metrics, and deployment-facing operating-point selection.
  - Current status: multi-scale head ablation scripts are implemented and running; final acceptance protocol is still being refined.

---

## 4) Iterative Progress Log
The project has followed repeated cycles:
- Problem
- Proposed Fix
- Result

### Cycle A: Early Stage1/Stage2 baseline was unstable and hard to compare
Problem:
- Baseline logs were inconsistent (missing metadata).
- Stage-level loss comparison was confusing because objectives differed.

Proposed fix:
- Standardized experiment logging and ledger generation.
- Added explicit experiment table update flow:
  - `src/results/experiments_result.md`
  - `src/results/experiments_result.csv`
  - `src/utils/update_experiments_result.py`

Result:
- Reproducibility improved.
- Later ablation decisions could be made from consistent run metadata.

---

### Cycle B: Importance collapse (core blocker)
Problem:
- Importance maps collapsed near floor values in earlier settings.
- At fixed threshold (0.5), some runs produced empty ROI predictions (`TP=0, FP=0, FN>0`).

Proposed fix:
- Reworked loss design:
  - `balanced_v1`, `balanced_v2`
  - normalized rate loss (`normalized_global` / `normalized_bg`)
  - weighted BCE for importance
  - optional ROI/BG separation margin
- Reduced rate pressure and increased importance supervision weight:
  - moved away from legacy settings that favored low importance globally.

Result:
- Collapse behavior was reduced in many runs.
- Importance dynamic range improved in best runs.
- However, calibration instability remained across different distillation settings.

---

### Cycle C: Stage2 distillation signal quality was weak
Problem:
- Proxy/random teacher behavior limited Stage2 interpretability.
- 1-channel logit KL setup was weak in old setup.

Proposed fix:
- Connected real teacher path:
  - backend: `pointpillars_zhulf`
  - ckpt: `epoch_160.pth`
- Distill logit mode improved (`auto` chooses BCE for 1-channel logits).

Result:
- Stage2 now uses a real, reproducible teacher source.
- Distillation sweeps became meaningful.
- Distill weight sensitivity became clearly observable (`ld=0.1` best in recent runs).

---

### Cycle D: Sparse ROI supervision degraded at low resolution
Problem:
- With heavy downsampling, ROI positives were easily lost when target downsampling was too hard.

Proposed fix:
- Added configurable ROI target downsampling:
  - `nearest`, `maxpool`, `area`
- Set practical default to `maxpool`.

Result:
- Better retention of sparse positives.
- Improved training stability for ROI supervision branch.

---

### Cycle E: Importance head capacity was likely too small
Problem:
- Basic/multiscale heads were too lightweight relative to task complexity.

Proposed fix:
- Added `pp_lite` importance head.
- Set default width so capacity meets PP20 target:
  - `hidden_channels >= 64`
  - approx params: `1,020,609` (~21.1% of teacher params).

Result:
- High-capacity head integrated into default Stage1/Stage2 scripts.
- 150-epoch sweeps completed successfully with this head.

---

### Cycle F: 150-epoch full sweeps showed remaining calibration/selectivity issues
Problem:
- Even with improved losses and larger head, fixed-threshold behavior remained unstable across runs.
- Some settings still had very high FP; one high-distill setting collapsed at threshold 0.5.

Proposed fix:
- Run full 150-epoch sweeps and evaluate confusion metrics (`TP/TN/FP/FN`) + threshold sweeps.
- Compare run-by-run operating points rather than only final training loss.

Result:
- Stage1 best (resnet):
  - `260216_resnet_lr1e-4_j21424_r1.out` final loss `0.7224`
- Stage2 best (darknet sweep):
  - `260216_darknet_ld0.1_j21422_r2.out` final loss `2.1185`
- Distill trend:
  - `ld=0.1` better than `ld=0.5` and `ld=1.0` in this setup.
- Confusion diagnostics (16-frame subset):
  - many runs: high recall but large FP (poor precision/IoU).
  - `ld=1.0`: at threshold 0.5, `TP=0, FP=0, FN>0`; recoverable only with lower threshold.

Interpretation:
- Main remaining issue is not only “learning failure,” but also:
  - importance score calibration instability across runs,
  - insufficient ROI selectivity at a fixed operating threshold.

---

### Cycle G: Fair-budget comparison and oracle decomposition
Problem:
- Direct `uniform vs adaptive` claims were still weak because bitrate budgets were not explicitly matched.
- Needed to separate:
  - “importance head/training is weak”
  - vs “quantization method itself cannot exploit ROI.”

Proposed fix:
- Added rate-budget observability:
  - training logs now record `eq_bits` and `code_entropy`.
- Added native-vs-oracle diagnostic:
  - `native`: predicted importance drives adaptive quantization.
  - `oracle_roi`: GT ROI mask directly drives quantization (same checkpoint).
- Added budget matching helper:
  - nearest-pair matching by `bpp_entropy_mean`.

Result (short pilot, 1 epoch / 32 frames train, 16 frames eval):
- Stage1 native vs oracle:
  - `bpp_entropy` dropped `0.806 -> 0.418` with almost unchanged ROI MSE.
- Stage2 native vs oracle:
  - `bpp_entropy` dropped `0.869 -> 0.449` with almost unchanged ROI MSE.
- Interpretation:
  - adaptive quantization can become much cheaper at similar distortion when ROI signal is ideal.
  - therefore the immediate bottleneck is still importance-map quality/calibration, not only quantizer formula.

---

### Cycle H: Stage3 multi-scale head expansion for detector-facing calibration
Problem:
- Stage2 improvements still showed unstable fixed-threshold behavior and FP-heavy ROI predictions in multiple runs.
- Existing lightweight heads were not sufficient to test whether richer multi-scale context/attention improves ROI selectivity at deployment-like thresholds.

Proposed fix:
- Launch Stage3 architecture ablation with stronger multi-scale heads:
  - `bifpn`
  - `deformable_msa`
  - `dynamic`
  - `rangeformer`
  - `frnet`
- Run both backbones (`resnet`, `darknet`) under the same Stage3 loss recipe and quantization settings for fair comparison.
- Keep acceptance criteria detector-facing:
  - compare calibration behavior (threshold sweep),
  - confusion metrics (`TP/TN/FP/FN`),
  - and bitrate-quality tradeoff.

---

### Cycle I: Dual-track recovery for label-space mismatch (Semantic labels vs 3D boxes)
Problem:
- SemanticKITTI supervision is point-wise semantics, while final detector claim requires 3D bounding-box AP.
- Proxy task metrics were useful for rapid iteration but are not sufficient for detector-level thesis claims.

Proposed fix:
- Split evaluation into two explicit tracks:
  - Track-A: SemanticKITTI ROI-aware codec metrics.
  - Track-B: KITTI detection official 3D AP/mAP via OpenPCDet.
- Add KITTI detector endpoint tooling:
  - `src/train/evaluate_kitti_map_vs_rate.py`
  - `src/scripts/run_kitti_map_vs_rate.sh`
  - `src/utils/match_bitrate_budget_detector.py`
- Add standardized reconstruction-to-pointcloud utility:
  - `src/utils/recon_pointcloud_export.py`
- Add teacher quality gate:
  - if original `ap3d_car_mod` is below threshold, trigger optional fine-tune branch (`src/scripts/run_teacher_finetune_kitti.sh`).

Result:
- Infrastructure and protocol are now separated for claim-safe reporting.
- Track-A/Track-B boundaries are explicit in paper-facing tables.
- Detector endpoint now has a reproducible execution path (OpenPCDet dependency required).

---

### Cycle J: Stage2 distillation signal refinement (teacher mismatch + alignment issue)
Problem:
- Stage2 could underperform because teacher objective space and current supervision space are not perfectly matched.
- Legacy feature distillation used resize-only spatial alignment, which can be semantically weak across heterogeneous backbones.

Proposed fix:
- Add distillation alignment controls:
  - `distill_align_mode={resize,adaptive_pool}`
  - `distill_align_hw` for canonical pooled grid.
- Add feature-map source controls:
  - `distill_feature_source={channel_mean,energy_map,none}`.
- Add optional teacher confidence gate:
  - `distill_teacher_score_min` with `distill_teacher_score_weight`.
- Add dedicated runnable ablation scripts:
  - `run_stage2_distill_fix_pilot_local.sh`
  - `run_stage2_distill_fix_ablation.sh`

Result:
- Code path now supports explicit tests of the “resize-only distill” hypothesis.
- Distillation quality can be audited as a controllable variable, not a fixed hidden assumption.
- Local smoke execution succeeded for all three new distill-fix cases (1-epoch short run), confirming runnable infra before full GPU sweep.

---

### Cycle K: Detector endpoint collapse traced to projection-path sparsification
Problem:
- Stage0/Stage1 losses kept decreasing, but reconstructed-cloud detector behavior was still poor.
- This raised ambiguity between:
  - under-training (`epochs`),
  - insufficient importance-head capacity,
  - representation damage before codec optimization.

Proposed fix:
- Run explicit decomposition diagnostics:
  - `raw` vs `identity(project->unproject)` vs `reconstructed`.
- Measure geometry loss from projection itself on val frames.
- Re-check selected high-ROI frame with detector overlays.

Result:
- Training did not fail:
  - Stage0 (`j22255_r5`): `21.3739 -> 1.5752` (50 epochs)
  - Stage1 (`j22279_r1`): `21.7544 -> 1.7810` (50 epochs)
- Projection-only loss is large:
  - mean point-retention after projection: `0.4169` (200 val frames).
  - mean collision factor (`raw/occupied`): `2.399`.
- One-sample detector decomposition (`sample 003464`):
  - raw: `123,592` points, `44` preds
  - identity: `50,911` points, `27` preds
  - Stage0 recon: `50,767` points, `1` pred
  - Stage1 recon: `50,635` points, `0` pred
- Interpretation:
  - first-order bottleneck is representation sparsification in current fixed-grid single-return projection path.
  - epoch/head effects remain possible but are not the dominant first explanation from current evidence.

---

## 5) Current Status
What is working:
- Reproducible experiment system and parameter tracking.
- Real teacher distillation wiring.
- Stronger loss formulations.
- Larger importance head integrated and trained at scale (150 epochs).
- Fair-budget and oracle-decomposition tooling is now integrated in codebase.
- Dual-track reporting policy is codified (SemanticKITTI codec track vs KITTI detector track).

What is not fully solved:
- Robust threshold calibration across runs.
- FP-heavy ROI predictions in several operating regimes.
- Clear downstream detector-level proof of gain under fixed deployment threshold.
- Representation-preserving reconstruction path for detector endpoint:
  - current projection path discards many points before codec comparison (`~58%` average drop in the 200-frame check).

---

## 6) Immediate Next Step Direction (for Stage3)
Recommended Stage3 focus:
1. Make `raw vs identity(project->unproject) vs reconstructed` a mandatory detector diagnostic baseline.
2. Introduce calibrated threshold policy (global or run-adaptive).
3. Optimize for precision-recall operating point, not only final training loss.
4. Add detector-level evaluation protocol as primary acceptance metric.
5. Compare fixed-threshold vs calibrated-threshold behavior in the report.

## 7) Stage3 Head Variants Implemented
- Added Stage3 multi-scale ROI head options in code:
  - `bifpn`
  - `deformable_msa`
  - `dynamic`
  - `rangeformer`
  - `frnet`
- Added encoder multi-stage tap support:
  - `DarkNetEncoder.forward(..., return_features=True)`
  - `Encoder.forward(..., return_features=True)` (resnet-style encoder)
- Added Stage3 experiment runner:
  - `src/scripts/run_stage3_multiscale_heads.sh`
