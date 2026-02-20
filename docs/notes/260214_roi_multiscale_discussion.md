# 2026-02-14 ROI/Importance Head Design Discussion

## Context
- Goal: Improve ROI-aware adaptive quantization quality in Stage1/Stage2 pipeline.
- Current issue observed in notebook:
  - Predicted ROI at threshold 0.5 can be empty.
  - Importance map tends to collapse near lower bound (`~0.01`).

## Stage Semantics (Updated)
1. Stage0 (new): **uniform quantization baseline**
  - ROI-unaware compression baseline for fair comparison.
2. Stage1: **adaptive quantization without teacher**
  - importance head + ROI supervision only.
3. Stage2: **adaptive quantization with teacher/distillation**
  - same adaptive path plus distillation objective.

## Questions Raised
1. Is a simple 3-layer CNN importance head sufficient?
2. Is using only the final DarkNet stage (very low spatial resolution) fundamentally limiting ROI localization?
3. Should ROI estimation and/or quantization be done at multiple stages?

## Findings
- DarkNet path downsamples aggressively (up to 32x), so latent spatial map is very small.
- With sparse ROI, downsampling GT ROI mask by `nearest` can drop positives entirely in many cells.
- This weakens supervision and can push importance predictions to near-constant low values.

## Decisions (Immediate)
1. **ROI supervision target downsampling**
  - Add configurable modes: `nearest`, `maxpool`, `area(soft)`.
  - Default to `maxpool` to preserve sparse positives under heavy downsampling.
2. **Experiment logging**
  - Record `roi_target_mode` in experiment metadata and config.

## Decisions (Near-term Design Direction)
1. **Multi-stage ROI estimation is a valid direction**
  - Use higher-resolution intermediate features (e.g., stage3/4/5 fusion) for ROI prediction.
  - Keep quantization at single bottleneck first to isolate variables.
2. **Multi-stage quantization**
  - Potentially effective but higher complexity/risk (rate accounting, instability, artifact interaction).
  - Not first step; evaluate after single-bottleneck ROI prediction is stabilized.

## Risks / Limitations
- Even with better ROI target downsampling, class imbalance remains severe.
- Proxy teacher quality may limit Stage2 gains even after distillation wiring.

## 2026-02-14 Execution Update
1. Distillation path wiring (implemented)
  - `Trainer` now computes distill loss using:
    - feature distill: channel-averaged student latent vs teacher feature map,
    - logit distill: student importance logits vs teacher logits,
    - ROI/teacher importance weighting.
2. Uniform baseline implementation (implemented)
  - Quantizer mode parameterized as `adaptive` vs `uniform`.
  - Uniform baseline disables importance head (`head_config=None`) and uses `quant_bits` sweep.
  - Uniform quantizer updated with STE option for train-time gradient flow.
3. Parallel execution scripts (implemented)
  - Stage0/1/2 submit script for parallel launch:
    - `src/scripts/submit_parallel_stage_runs.sh`
  - Uniform sweep:
    - `src/scripts/run_uniform_baseline.sh`
4. Reliability fix for Slurm runs
  - Added `set -euo pipefail` and conda environment bootstrap to experiment scripts.
  - Added job-id-based log/save naming to avoid collisions across reruns on the same date.

## Next Actions
1. Re-run Stage2 with `roi_target_mode=maxpool` and compare:
  - ROI IoU/Precision/Recall (importance-thresholded),
  - `imp_roi_gap`,
  - reconstruction ROI MSE.
2. Add threshold sweep curve (IoU/PR vs threshold) to notebook.
3. Prototype multi-scale ROI head (feature fusion) while keeping quantization at one level.
