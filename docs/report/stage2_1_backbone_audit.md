# Stage 2.1 Backbone Audit (Stage1 AE)

## Purpose
Quantify Stage1 baseline capacity and compare a small set of architecture variants to support a concrete keep/upgrade decision.

## Utility used
- Script: `src/utils/backbone_audit.py`
- Reports:
  - parameter counts (encoder/decoder/total),
  - latent tensor shape,
  - approximate MACs per forward,
  - rough latency (environment dependent).

## Commands used

```bash
conda run -n lidarcomp311 env PYTHONPATH=src python src/utils/backbone_audit.py \
  --batch_size 1 --height 64 --width 1024 --base_channels 64 --latent_channels 64 --blocks_per_stage 1 \
  --measure_iters 5 --warmup_iters 2

conda run -n lidarcomp311 env PYTHONPATH=src python src/utils/backbone_audit.py \
  --batch_size 1 --height 64 --width 1024 --base_channels 64 --latent_channels 64 --blocks_per_stage 2 \
  --measure_iters 5 --warmup_iters 2
```

Additional single-pass checks were run for:
- `blocks_per_stage=3`,
- `base_channels=96, blocks_per_stage=2`.

## Results

| Variant | Encoder Params | Decoder Params | Total Params | Approx MACs / forward | Latent shape |
|---|---:|---:|---:|---:|---|
| `bc64/lc64/st4/bp1` (current Stage1 default) | 1,431,488 | 556,869 | 1,988,357 | 8,565,817,344 | `[1, 64, 4, 64]` |
| `bc64/lc64/st4/bp2` | 3,055,552 | 556,869 | 3,612,421 | 12,208,570,368 | `[1, 64, 4, 64]` |
| `bc64/lc64/st4/bp3` | 4,679,616 | 556,869 | 5,236,485 | 15,851,323,392 | `[1, 64, 4, 64]` |
| `bc96/lc64/st4/bp2` | 6,607,456 | 1,139,429 | 7,746,885 | 27,006,074,880 | `[1, 64, 4, 64]` |

Notes:
- MAC estimates are hook-based approximations over Conv/ConvTranspose/Linear layers.
- CPU latency is noisy on shared nodes; use it only as a rough trend signal.

## Interpretation
- Current Stage1 (`~2.0M params`) is lightweight relative to many learned image-compression baselines and recurrent projection codecs.
- Moving from `bp1 -> bp2` raises capacity materially (`~1.8x` total params) without changing latent resolution.
- Jumping to `bc96/bp2` is a much larger compute increase and should be gated by GPU budget.

## Decision
- Keep current Stage1 as internal control baseline.
- For Stage2.1 primary path, prefer a **moderate capacity step** (`bp2` family) plus improved task-aware objectives (teacher distillation + deployable importance map), before moving to very heavy channel scaling.

## 2026-02-14 Update (ROI/Importance Discussion)

### Problem signal
- Importance prediction can collapse to near-minimum values, producing empty ROI predictions at common thresholds.
- One major contributor is sparse ROI supervision being downsampled with `nearest` under aggressive spatial compression.

### Implemented change
- Added configurable ROI target downsampling for importance supervision:
  - `nearest`, `maxpool`, `area(soft)`.
- Default set to `maxpool` to preserve sparse positives at low latent resolution.

### Architecture direction (agreed)
0. Add explicit **uniform quantization baseline (Stage0)** for ROI-unaware reference.
1. Multi-stage ROI estimation (higher-resolution feature fusion) is worth pursuing.
2. Keep **single-bottleneck quantization** first for controlled ablation.
3. Consider multi-stage quantization only after ROI prediction quality is stabilized.

### Implementation status snapshot
- Stage0/1/2 parallel sweep scripts are wired.
- Stage2 distillation loss path is connected in trainer (feature + logit distill with ROI/teacher weighting).
- Uniform quantizer path supports STE for train-time optimization.
