# Track 2 Codec Failure Map

This note maps recent range-image LiDAR compression papers to the specific failure modes we now see in `Track 2`.

Current local observation:

- `raw/basic` RangeDet on the repaired raw RI is strong enough to serve as a valid reference.
- The collapse happens at `PC -> RI -> codec -> RI`, already at `Stage 0` before quantization becomes the main issue.
- Therefore the first recovery path should stay `RI-native` and focus on:
  - validity / occupancy preservation
  - decoder artifact reduction
  - detector-aware RI supervision

## 1. RIDDLE (CVPR 2022)

Source:

- https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_RIDDLE_Lidar_Data_Compression_With_Range_Image_Deep_Delta_Encoding_CVPR_2022_paper.html

Observed problem:

- Plain RI reconstruction can destroy local range structure even when the image still looks visually plausible.

Method idea:

- Model local residuals/deltas instead of relying only on raw pixel reconstruction.

What to borrow:

- Treat local RI structure as a first-class target.
- Explicitly measure row/column gradient mismatch and local artifact patterns.

What not to borrow directly:

- The exact delta codec path is not the first implementation step here.
- First we should verify whether our failure is mainly `valid mask`, `banding`, or `detector mismatch`.

## 2. FLiCR (2023)

Source:

- https://arxiv.org/abs/2307.15005

Observed problem:

- Projection-based LiDAR compression can be rate-efficient, but downstream perception quality depends on whether task-relevant structure survives the RI path.

Method idea:

- Use RI-domain processing with a downstream-aware evaluation framing.

What to borrow:

- Keep `Track 2` framed as `task-aware range-view LiDAR compression`.
- Judge success by native RI detection, not by generic pixel quality alone.

What not to borrow directly:

- FLiCR is evidence that RI-domain compression is publishable, but it does not remove our need to diagnose why our current decoder artifacts are so destructive.

## 3. Range Image-Based Implicit Neural Compression for LiDAR Point Clouds (2025)

Source:

- https://arxiv.org/abs/2504.17229

Observed problem:

- A fixed low-resolution latent plus standard image-style decoding can be too rigid for sensor geometry.

Method idea:

- Use implicit / coordinate-conditioned reconstruction to reduce projection bottlenecks.

What to borrow:

- Coordinate-conditioned decoding is a plausible anti-artifact direction.
- This supports using the new skip / coord-conditioned decoder family as a Track 2 branch.

What not to borrow directly:

- We should not jump immediately to a large implicit redesign before we know whether our main failure is mask integrity or decoder upsampling.

## 4. RENO (CVPR 2025)

Source:

- https://openaccess.thecvf.com/content/CVPR2025/papers/You_RENO_Real-Time_Neural_Compression_for_3D_LiDAR_Point_Clouds_CVPR_2025_paper.pdf

Observed problem:

- Projection-based codecs can fail to preserve geometry well enough for downstream 3D use, even when the representation is compact.

Method idea:

- Use a stronger structural representation than a plain image-like bottleneck when fidelity matters.

What to borrow:

- Do not assume wider/deeper 2D blocks alone will solve the failure.
- If all RI-only no-quant pilots fail, the next move should be representation redesign, not just more capacity.

What not to borrow directly:

- `Track 2` should still start with RI-native fixes because raw/basic RangeDet already proves the clean RI itself is usable.

## 5. ALICE-LRI (2025)

Source:

- https://arxiv.org/abs/2510.20708

Observed problem:

- Projection itself can be the source of information loss if the RI construction is not sufficiently sensor-aware and compact.

Method idea:

- Treat the projection representation as an object to optimize, not just the neural codec after it.

What to borrow:

- If `Stage 0` remains near-zero after mask-aware and anti-banding pilots, we should move to a better RI representation rather than continue codec-only tuning.

What not to borrow directly:

- This is the fallback plan after the current RI-native pilots, not the first implementation step.

## Practical interpretation for this repo

The recent literature supports three immediate Track 2 priorities:

1. `Validity preservation`
   - If the predicted valid mask diverges from the raw RI occupancy, RangeDet loses support points before localization even starts.

2. `Decoder artifact reduction`
   - Stripe / banding artifacts are not harmless cosmetic noise in RI space; they perturb beam-wise depth structure.

3. `Detector-aware supervision`
   - Pixel-level reconstruction alone is not enough.
   - The codec should also preserve the regions and patterns that the detector actually relies on.

## Immediate implementation choices

These literature-backed fixes map directly to the current pilot program:

- `Pilot A`: mask-aware decoding
- `Pilot B`: skip decoder + gradient / row-profile anti-banding loss
- `Pilot C`: detector-aware auxiliary target distilled from repaired raw/basic RangeDet outputs
- `Pilot D`: combine the best structural and objective changes

If none of these no-quant pilots recover a meaningful fraction of raw/basic `AP3D@0.3`, then the next Track 2 step should be a representation change, not another round of minor codec tuning.
