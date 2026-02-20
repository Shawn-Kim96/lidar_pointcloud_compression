# Multi-Scale ROI Head Design (Draft)

## 1. Why This Is Needed
- Current DarkNet path uses heavy downsampling (`32x`), so final latent map is small for sparse ROI localization.
- Single-scale ROI prediction at the smallest feature map is prone to:
  - positive signal loss after ROI downsampling,
  - near-constant low importance prediction,
  - weak spatial boundary recovery.

## 2. Design Goal
- Predict a more reliable ROI/importance map using higher-resolution intermediate features.
- Keep quantization at a single bottleneck first (for controlled ablation and stable rate interpretation).

## 3. Proposed Architecture

### 3.1 Feature Taps
- Tap DarkNet encoder outputs from multiple stages (recommended: stage3, stage4, stage5).
- Let:
  - `F3`: higher resolution, lower semantic abstraction.
  - `F4`: mid-level features.
  - `F5`: low resolution, high-level semantics.

### 3.2 Fusion Block (FPN-style)
1. `1x1 conv` on each stage to unify channels (`C_fuse`, e.g., 64).
2. Upsample lower-resolution maps to `F3` resolution.
3. Fuse by concat + `3x3 conv` (or weighted sum + `3x3 conv`).

### 3.3 ROI Head
- On fused feature map:
  - `Conv3x3 -> BN -> Act`
  - `Conv3x3 -> BN -> Act`
  - `Conv1x1 -> logits`
- Optional side outputs:
  - stage-wise auxiliary logits for deep supervision.

### 3.4 Quantizer Interface
- Convert ROI logits to importance map (`sigmoid`, clamped to `[min_imp, max_imp]`).
- Downsample/upsample importance map to quantizer latent spatial size as needed.
- Quantizer remains single-level bottleneck in first phase.

## 4. Training Strategy

### 4.1 Supervision Target
- Use `roi_target_mode`:
  - `nearest` (legacy),
  - `maxpool` (positive-preserving),
  - `area` (soft occupancy).
- For multi-scale auxiliary heads:
  - build per-scale targets with matching mode.

### 4.2 Loss
- Base:
  - `L_recon`
  - `L_rate`
  - `L_importance_main`
- Recommended additions:
  - class-balanced BCE (`pos_weight`) or BCE+Dice for imbalance.
  - auxiliary importance losses:
    - `L_imp_aux = sum_i alpha_i * L_imp_i`

### 4.3 Distillation
- Distillation should be connected in trainer for meaningful Stage2 sweeps.
- Optional: distill ROI logits/features per selected stage.

## 5. Ablation Plan

## Phase A (low risk)
1. Single-scale head + ROI target mode sweep (`nearest/maxpool/area`)
2. Add imbalance-aware loss

## Phase B (moderate)
1. Multi-scale ROI head (stage3/4/5 fusion)
2. Compare with single-scale at matched training settings

## Phase C (high complexity)
1. Multi-stage quantization (optional, after ROI estimator is stable)
2. Re-check rate proxy vs true entropy coding behavior

## 6. Evaluation Checklist
- ROI quality:
  - IoU / Precision / Recall vs threshold curve
  - `imp_roi_gap`
- Reconstruction:
  - all/ROI/BG MSE
- Compression:
  - level-map distribution, rate proxy
- Downstream:
  - proxy teacher drop (intermediate)
  - detector mAP/recall (final claim)

## 7. Risks
- Multi-scale fusion increases memory/compute.
- Improper scale balancing can overfit high-resolution noise.
- Multi-stage quantization can break interpretability of rate-distortion tradeoff.

## 8. Decision Log (Current)
- Adopt `roi_target_mode=maxpool` as new default.
- Implement multi-scale ROI head after validating target-mode and loss improvements.
- Keep single-bottleneck quantization in primary path until ROI prediction quality is stabilized.
