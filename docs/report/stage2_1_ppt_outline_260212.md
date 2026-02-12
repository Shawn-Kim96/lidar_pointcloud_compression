# Stage2.1 PPT Outline (2026-02-12)

## Slide 1. Problem and Goal
- Goal: compress LiDAR point cloud while preserving downstream detection utility.
- Stage2.1 focus:
  - deployable adaptive quantization (no label/teacher at inference),
  - teacher-guided task awareness during training.

## Slide 2. Method (one diagram)
- Input range image -> Encoder -> Latent.
- Importance head predicts soft importance map.
- Adaptive quantizer allocates more levels to high-importance areas.
- Decoder reconstructs range/intensity/xyz channels.

## Slide 3. Loss design (`ld`, `li`, `be`)
- `L_total = L_recon + be*L_rate + ld*L_distill + li*L_importance`
- `ld` (`lambda_distill`): teacher mimic strength.
- `li` (`lambda_importance`): importance-map supervision strength.
- `be` (`beta_rate`): bitrate penalty strength.

## Slide 4. Experiment timeline
- `20988`: 3 runs, `1 epoch` each (`ld/li` sweep), partial eval (512 frames).
- `20986`: `4 epoch` run (`r0`), partial eval (512 frames).
- `21013`: 3 runs, `20 epoch`, full-val (`seq08`, 4071 frames), completed.

## Slide 5. Core quantitative results (job 21013)
- q8:
  - `r0(ld1.0/li0.2)`: best PSNR/CD.
  - `r2(ld1.0/li0.5)`: best rate/drop.
- q4:
  - `r2(ld1.0/li0.5)`: best operating point (drop/rate/CD).
- Message: `li` increase helped compression-oriented operating points.

## Slide 6. Why 1-epoch looked bad
- 1 epoch runs (`20988`) had large positive teacher drop (`~+0.55`): undertrained.
- 20 epoch full-val runs (`21013`) flipped to stable negative drop (`~ -0.11 to -0.13`).
- Conclusion: Stage2.1 requires enough training horizon (>=20 epoch in current setup).

## Slide 7. Cautions and interpretation
- Negative teacher drop does not automatically mean better real detector AP.
- Proxy-teacher bias / denoising effect can inflate proxy metrics.
- Therefore, detection mAP and qualitative frame checks are mandatory.

## Slide 8. Next steps aligned with advisor feedback
- Add PointPillars detection evaluation (mAP/recall, class-wise AP).
- Add preprocessing/frame-quality audit and intermediate visualizations.
- Expand ROI-related hyperparameter sweep:
  - ROI size, moving window, step size, confidence threshold.
- Keep reproducible metric backups:
  - per-run CSV + hyperparameter-indexed summary table.
