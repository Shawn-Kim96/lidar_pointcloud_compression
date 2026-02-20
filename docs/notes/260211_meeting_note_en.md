# 2026-02-11 Meeting Note (English Translation)

## Questions
- Baseline is an internal control (autoencoder + simple decoder), not an external SOTA baseline.
- External baselines (GPCC/Draco + learned point-cloud codecs) should be added during paper-stage evaluation.
- Current Stage2 uses GT-derived ROI masks and a proxy task loss; this validates plumbing but is not fully detector-aware yet.
- Recommended direction: combine teacher distillation with a learned importance map for deployability.

## Feedback and Action Items
- Detection/segmentation evaluation should be included as a core criterion.
- When using PointPillars for detection:
  - Prioritize detection accuracy.
  - If accuracy is unstable, inspect preprocessing because point-cloud quality may be noisy.
  - Visually inspect several frames/key frames to confirm data quality.
- In Stage2, the integration between ROI estimator and encoder is critical.
  - ROI-related hyperparameters should be explicit:
    - ROI size
    - moving window
    - step size
    - thresholding rule for high-importance assignment
- Confirm writing/collaboration pipeline (e.g., Overleaf).
- For encoder/decoder results, evaluate multiple metrics, including ROI-specific accuracy.
- Keep metric backups for each hyperparameter setting.
- Always check intermediate outputs, not only final metrics.
  - There can be cases where final metrics look good but intermediate behavior is not reasonable.
- Add visual artifacts for intermediate outputs.
- Consider starting from alternative baselines when useful.

## Adaptive Quantization Notes
- Current normalization is per-sample global normalization.
  - This is simple/stable, but may be suboptimal when channel distributions differ.
- Open points to test:
  - `z_min/z_max`: global-per-sample vs per-channel (per-channel often improves RD).
  - importance map resize mode: bilinear vs nearest (nearest can preserve segmentation boundaries better).
  - safety guard after rounding: `clamp(min=2)` should remain.
  - level-map distribution: if heavily biased to background, ROI map definition/training needs inspection.
  - bitrate estimation: current gain is from level modulation only, so verify whether entropy coding actually benefits (e.g., distribution becomes more peaked).
