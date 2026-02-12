# Stage 2.1 Backbone Upgrade Proposal (Concrete)

## 1) Target architecture sketch

## Baseline control (unchanged for comparison)
- `RangeCompressionModel` with:
  - `base_channels=64`,
  - `latent_channels=64`,
  - `num_stages=4`,
  - `blocks_per_stage=1`.

## Proposed upgrade path (first implementation target)
- **Backbone depth increase**:
  - move to `blocks_per_stage=2` (same stage count and latent size).
- **Task-aware adaptive branch**:
  - `AdaptiveRangeCompressionModel` with:
    - `ImportanceHead` from latent features,
    - soft importance map `[0,1]`,
    - adaptive quantizer levels in `[bg_levels, roi_levels]`.
- **Teacher distillation**:
  - frozen teacher adapter outputs dense feature/logit maps,
  - student optimized with distillation + reconstruction + rate proxy.

Why this shape:
- keeps the base architecture family stable for fair comparison,
- adds deployable adaptive path (no labels at inference),
- avoids high-risk full recurrent codec rewrite in first iteration.

## 2) Quantization + rate modeling plan

## Current
- Per-sample affine quantization + entropy-estimated BPP proxy.

## Stage2.1 step
- Adaptive quantization now accepts soft importance map:
  - `levels = bg_levels + importance * (roi_levels - bg_levels)`.
- Keep explicit reporting label:
  - **estimated BPP** (not true bitstream BPP).

## Next step after this stage
- Add learnable entropy model over latent codes (hyperprior/context-lite path).
- Keep GPCC/Draco as external true-bitstream baselines for evaluation charts.

## 3) Ablation plan

Hold fixed:
- dataset split (`train: 00,01,02,03,04,05,06,07,09,10`, `val: 08`),
- optimizer family/lr schedule,
- input representation (range, intensity, xyz channels).

Compare:
1. `S1-uniform-bp1` (control baseline).
2. `S1-uniform-bp2` (capacity-only change).
3. `S2.1-importance-only` (adaptive quantization + importance head, no distill).
4. `S2.1-distill-only` (teacher distill + uniform quantization).
5. `S2.1-full` (importance + distill + adaptive quantization).

Minimal training protocol:
- debug smoke first (done),
- overnight sweeps for matched-rate comparisons.

## 4) Acceptance metrics and budget constraints

Primary metrics:
- rate: estimated BPP (and clearly labeled as such),
- distortion: PSNR (range/intensity), Chamfer Distance,
- task-aware: teacher score drop on reconstructed vs original (and detector metric when OpenPCDet backend is available).

Acceptance thresholds (relative to `S1-uniform-bp1` at similar BPP):
1. non-inferior distortion (no major PSNR/CD regression),
2. improved task metric (lower teacher-score drop),
3. inference path works with `--no_labels`.

Compute budget constraints:
- single-GPU overnight sweep compatible,
- avoid major memory growth beyond `bp2` family for first upgrade cycle.
