# Loss Specification for Stage0-Stage3

## Scope
This document is the loss-function single source of truth for Stage0-Stage3 in this repository.

Code references:
- `src/train/trainer.py`
- `src/loss/distill_loss.py`
- `src/models/quantization.py`
- `src/models/compression.py`
- `src/models/importance_head.py`
- `src/scripts/run_uniform_baseline.sh`
- `src/scripts/run_stage1.sh`
- `src/scripts/run_stage2.sh`
- `src/scripts/run_stage3_multiscale_heads.sh`

## Notation
- Input range image: `x in R^{B x 5 x H x W}`
- Reconstruction: `x_hat`
- Latent feature: `z`
- Quantized/dequantized latent: `z_hat`
- Importance logits: `s`
- Importance probability map: `p = sigmoid(s)` (then scaled to `[min_importance, max_importance]`)
- Full-resolution ROI mask from dataset: `m_raw`
- ROI target aligned to importance resolution: `m = D(m_raw)`
- Teacher feature/logit: `f_t, s_t`
- Student feature/logit used for distill: `f_s, s`

## Common Training Objective
The trainer uses:

`L_total = lambda_recon * L_recon + lambda_distill * L_distill + lambda_rate * L_rate + lambda_importance * L_imp + lambda_imp_separation * L_sep`

where each term is defined below.

## Loss Components

### 1) Reconstruction Loss
`L_recon = MSE(x_hat, x)`

Implementation:
- `nn.MSELoss()` in `Trainer`.

### 2) Adaptive Quantization and Rate Loss
Adaptive level map is built as:

`level_map = bg_levels + p * (roi_levels - bg_levels)`

with `p in [0,1]`, then:
- round and clamp: `level_map <- clamp(round(level_map), min=2)`
- quantization code range is `0 ... level_map-1`

Rate modes:
- `global_mean`:
  - `L_rate = mean(level_map)`
- `normalized_global`:
  - `level_norm = (level_map - bg_levels) / (roi_levels - bg_levels)`
  - `L_rate = mean(level_norm)`
- `normalized_bg`:
  - same `level_norm`
  - background weight:
    - if ROI target exists: `w_bg = 1 - m`
    - else if predicted importance exists: `w_bg = 1 - p_pred(detached)`
    - else: `w_bg = 1`
  - `L_rate = sum(level_norm * w_bg) / sum(w_bg)`

Important behavior:
- Higher importance gives higher `level_map`, which means finer quantization (smaller quantization step), not coarser.

Uniform quantization case:
- In uniform mode there is no `level_map` in `aux`.
- `_compute_rate_loss` returns `0`.

### 3) Importance Supervision Loss
Primary form:

`L_imp = BCEWithLogits(s, target, pos_weight)`

Target construction:
- If dataset ROI mask exists (`m_raw`), resize to logits shape using:
  - `nearest`, `area`, or `maxpool` (`adaptive_max_pool2d`)
- Else if teacher importance exists:
  - use teacher importance map resized with bilinear interpolation

Pos-weight for class imbalance:
- `fixed` mode:
  - `pos_weight = importance_pos_weight`
- `auto` mode:
  - `pos = sum(target)`
  - `neg = total_pixels - pos`
  - `ratio = neg / max(pos, 1)`
  - `pos_weight = clamp(ratio, 1, importance_pos_weight_max)`

### 4) ROI/BG Separation Loss
This term pushes mean ROI importance above mean BG importance:

- `p = sigmoid(s)`
- `roi_mean = mean(p over ROI)`
- `bg_mean = mean(p over BG)`
- `L_sep = ReLU(margin - (roi_mean - bg_mean))`

### 5) Distillation Loss
Distillation is composite:

`L_distill = alpha_feat * L_feat + alpha_logit * L_logit`

and is then multiplied again in total objective by `lambda_distill`.

Distillation weight map:
- Priority 1: `m` (ROI target) if available
- Priority 2: teacher importance map

Feature distillation:
- Student feature used: channel-mean of latent
  - `f_s = mean_c(z)`
- Teacher feature used: channel-mean of teacher feature map
  - `f_t = mean_c(teacher_features)`
- Default feature loss: weighted MSE

Logit distillation (`logit_loss_type`):
- `auto`:
  - if logit channels = 1: use BCE-with-logits distillation
  - else: use KL distillation over channels
- `kl`: weighted KL with temperature
- `bce`: weighted BCE-with-logits with teacher sigmoid target
- `mse`: weighted MSE on logits

## Recipe Defaults
If not explicitly overridden:

- `legacy`:
  - `rate_loss_mode = global_mean`
  - `importance_loss_mode = bce`
- `balanced_v1`:
  - `rate_loss_mode = normalized_global`
  - `importance_loss_mode = weighted_bce`
- `balanced_v2`:
  - `rate_loss_mode = normalized_bg`
  - `importance_loss_mode = weighted_bce`

## Stage-by-Stage Objective

## Stage0 (Uniform Baseline)
Purpose:
- ROI-unaware compression baseline.

Default script:
- `src/scripts/run_uniform_baseline.sh`

Default settings:
- `quantizer_mode=uniform`
- `no_teacher`
- `lambda_recon=1.0`
- `lambda_rate=0.0`
- `lambda_distill=0.0`
- `lambda_importance=0.0`

Effective objective:
- `L_total = L_recon`

Notes:
- Importance head is disabled in uniform mode.
- Even if `lambda_rate > 0`, current implementation yields `L_rate=0` for uniform because `level_map` is absent.

## Stage1 (Adaptive + ROI Supervision, No Teacher)
Purpose:
- Isolate effect of importance-aware adaptive quantization without teacher distillation.

Default script:
- `src/scripts/run_stage1.sh`

Default settings:
- `quantizer_mode=adaptive`
- `no_teacher`
- `loss_recipe=balanced_v1`
- `rate_loss_mode=normalized_global`
- `importance_loss_mode=weighted_bce`
- `roi_target_mode=maxpool`
- `lambda_recon=1.0`
- `lambda_rate=0.02`
- `lambda_distill=0.0`
- `lambda_importance=1.0`
- `lambda_imp_separation=0.0`

Effective objective:
- `L_total = 1.0*L_recon + 0.02*L_rate + 1.0*L_imp`

## Stage2 (Adaptive + ROI Supervision + Teacher Distillation)
Purpose:
- Add teacher guidance on top of Stage1.

Default script:
- `src/scripts/run_stage2.sh`

Default settings:
- `quantizer_mode=adaptive`
- `teacher_backend=pointpillars_zhulf`
- `loss_recipe=balanced_v2`
- `rate_loss_mode=normalized_bg`
- `importance_loss_mode=weighted_bce`
- `roi_target_mode=maxpool`
- `distill_logit_loss=auto`
- `lambda_recon=1.0`
- `lambda_rate=0.02`
- `lambda_distill in {0.5, 1.0, 0.1}` (array sweep)
- `lambda_importance=1.0`
- `lambda_imp_separation=0.2`

Effective objective:
- `L_total = 1.0*L_recon + lambda_distill*L_distill + 0.02*L_rate + 1.0*L_imp + 0.2*L_sep`

## Stage3 (Stage2 Objective + Advanced Multi-Scale Importance Heads)
Purpose:
- Keep Stage2 loss formulation and test stronger head designs for ROI selectivity/calibration.

Default script:
- `src/scripts/run_stage3_multiscale_heads.sh`

Default settings:
- Objective recipe is Stage2-like:
  - `loss_recipe=balanced_v2`
  - `rate_loss_mode=normalized_bg`
  - `importance_loss_mode=weighted_bce`
  - `lambda_recon=1.0`
  - `lambda_rate=0.02`
  - `lambda_distill=0.1`
  - `lambda_importance=1.0`
  - `lambda_imp_separation=0.2`
- Head type sweep:
  - `bifpn`, `deformable_msa`, `dynamic`, `rangeformer`, `frnet`

Effective objective:
- Same formula as Stage2.
- Difference is architecture (`importance_head_type`) and calibration/selection evaluation protocol.

## Practical Wording for Paper
Methods section can state:
- "Stage0 trains reconstruction-only uniform quantization baseline."
- "Stage1 adds ROI-supervised adaptive quantization (no distillation)."
- "Stage2 adds teacher distillation and ROI/BG separation regularization."
- "Stage3 keeps Stage2 loss and replaces the importance head with advanced multi-scale variants for calibration-focused ablation."
