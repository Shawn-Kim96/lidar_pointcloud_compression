# Experiments Result Ledger

- generated_at: `2026-02-19 20:16:36`
- source_logs_dir: `logs`
- total_runs: `57`

## Parameter Policy
- New hyperparameters must be tracked for all runs.
- If a parameter did not exist in legacy runs, backfill explicitly and mark source.
- `roi_target_mode` backfill rule in this ledger:
  - if logged in metadata -> use logged value
  - if missing in legacy run -> `nearest` (`backfilled_legacy_default`)
- `quantizer_mode` backfill rule in this ledger:
  - if logged in metadata -> use logged value
  - if missing in legacy run -> `adaptive` (`backfilled_legacy_default`)
- `quant_bits` backfill rule in this ledger:
  - if logged in metadata -> use logged value
  - if missing in legacy run -> `8` (`backfilled_legacy_default`)
- `loss_recipe` backfill rule in this ledger:
  - if logged in metadata -> use logged value
  - if missing in legacy run -> `legacy` (`backfilled_legacy_default`)
- `rate_loss_mode`/`importance_loss_mode` backfill rule:
  - if missing -> infer from `loss_recipe` default and mark `backfilled_from_recipe_default`

## Column Guide (Key Fields)

| column | meaning | value semantics |
|---|---|---|
| `quantizer_mode` | Quantization method used in the run. | `adaptive`: importance-aware levels vary spatially by ROI/importance map. `uniform`: fixed bit-depth quantization across latent tensor (ROI-unaware baseline). |
| `quant_bits` | Bit-depth field recorded in metadata. | In `uniform`, directly sets quantizer levels (`2^bits`). In `adaptive`, this field is compatibility metadata; effective granularity comes from `roi_levels` and `bg_levels`. |
| `lambda_distill` | Loss weight for teacher-student distillation term. | `0`: distillation disabled. `>0`: distillation contributes to total loss. Larger values emphasize teacher matching more strongly. |
| `lambda_importance` | Loss weight for importance/ROI supervision term. | Larger values force stronger alignment between predicted importance and target ROI/teacher importance. |
| `loss_recipe` | Composite objective recipe variant. | `legacy`: old objective. `balanced_v1`: normalized rate + weighted BCE. `balanced_v2`: bg-focused normalized rate + weighted BCE + ROI/BG separation margin. |
| `rate_loss_mode` | How rate proxy is computed from level map. | `global_mean`, `normalized_global`, or `normalized_bg` (background-focused). |
| `distill_logit_loss` | Distillation loss type for logits. | `auto` picks `bce` for 1-channel logits, `kl` otherwise. Can force `kl`/`bce`/`mse`. |
| `importance_head_type` | Importance head architecture variant. | `basic` (2 conv) or `multiscale` (dilated multi-branch fusion). |
| `final_eq_bits` | Final epoch effective quantization bits per latent symbol. | Adaptive: `mean(log2(level_map))`; Uniform: constant `quant_bits`. |
| `final_code_entropy` | Final epoch empirical entropy of latent quant codes. | Shannon entropy in bits/symbol from observed code histogram. |

## Experiment Table

| log_file | stage | backbone | quantizer_mode | quant_bits | loss_recipe | rate_loss_mode | importance_loss_mode | distill_logit_loss | importance_head_type | lr | lambda_distill | lambda_importance | lambda_imp_separation | roi_target_mode | roi_target_mode_source | epochs | final_loss | final_eq_bits | final_code_entropy | rel_improve_% | save_dir |
|---|---:|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| `260215_resnet_pilot_balanced_v1_headbasic_r0.out` | 2 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | basic | 0.0001 | 0.100 | 1.000 | 0.000 | maxpool | logged | 2 | 38.1165 | n/a | n/a | 31.25 | `data/results/experiments/260215_resnet_pilot_balanced_v1_headbasic_lr1e-4_bs2_localpilot_r0` |
| `260215_resnet_pilot_balanced_v2_headmultiscale_r1.out` | 2 | resnet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | multiscale | 0.0001 | 0.100 | 1.000 | 0.200 | maxpool | logged | 2 | 39.7356 | n/a | n/a | 31.30 | `data/results/experiments/260215_resnet_pilot_balanced_v2_headmultiscale_lr1e-4_bs2_localpilot_r1` |
| `260212_darknet_lr1e-4_r0.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 3.9214 | n/a | n/a | 67.91 | `data/results/experiments/260212_darknet_solo_lr1e-4_bs4_r0` |
| `260212_darknet_lr5e-5_r2.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 4.1602 | n/a | n/a | 73.95 | `data/results/experiments/260212_darknet_solo_lr5e-5_bs4_r2` |
| `260212_resnet_lr1e-4_r1.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 2.8335 | n/a | n/a | 74.41 | `data/results/experiments/260212_resnet_solo_lr1e-4_bs4_r1` |
| `260212_resnet_lr5e-5_r3.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 2.9149 | n/a | n/a | 78.79 | `data/results/experiments/260212_resnet_solo_lr5e-5_bs4_r3` |
| `260212_darknet_ld0.1_r2.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.100 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 3.8924 | n/a | n/a | 68.11 | `data/results/experiments/260212_darknet_distill_lr1e-4_bs4_r2` |
| `260212_darknet_ld0.5_r0.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.500 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 3.9236 | n/a | n/a | 68.05 | `data/results/experiments/260212_darknet_distill_lr1e-4_bs4_r0` |
| `260212_darknet_ld1.0_r1.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 1.000 | 0.500 | 0.000 | nearest | backfilled_legacy_default | 50 | 3.9096 | n/a | n/a | 68.74 | `data/results/experiments/260212_darknet_distill_lr1e-4_bs4_r1` |
| `260213_darknet_ld0.1_r2.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.100 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_r2` |
| `260213_darknet_ld0.5_r0.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.500 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_r0` |
| `260213_darknet_ld1.0_r1.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 1.000 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_r1` |
| `260213_darknet_lr1e-4_r0.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_solo_lr1e-4_bs4_r0` |
| `260213_darknet_lr5e-5_r2.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_solo_lr5e-5_bs4_r2` |
| `260213_darknet_uniform_q4_r0.out` | 0 | darknet | uniform | 4 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_uniform_q4_lr1e-4_bs4_r0` |
| `260213_darknet_uniform_q6_r1.out` | 0 | darknet | uniform | 6 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_uniform_q6_lr1e-4_bs4_r1` |
| `260213_darknet_uniform_q8_r2.out` | 0 | darknet | uniform | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_darknet_uniform_q8_lr1e-4_bs4_r2` |
| `260213_resnet_lr1e-4_r1.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_resnet_solo_lr1e-4_bs4_r1` |
| `260213_resnet_lr5e-5_r3.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_resnet_solo_lr5e-5_bs4_r3` |
| `260213_resnet_uniform_q4_r3.out` | 0 | resnet | uniform | 4 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_resnet_uniform_q4_lr1e-4_bs4_r3` |
| `260213_resnet_uniform_q6_r4.out` | 0 | resnet | uniform | 6 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_resnet_uniform_q6_lr1e-4_bs4_r4` |
| `260213_resnet_uniform_q8_r5.out` | 0 | resnet | uniform | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260213_resnet_uniform_q8_lr1e-4_bs4_r5` |
| `260213_darknet_ld0.1_j21254_r2.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.100 | 0.500 | 0.000 | maxpool | logged | 50 | 4.5592 | n/a | n/a | 64.44 | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_j21254_r2` |
| `260213_darknet_ld1.0_j21259_r1.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 1.000 | 0.500 | 0.000 | maxpool | logged | 50 | 4.6658 | n/a | n/a | 63.24 | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_j21259_r1` |
| `260213_darknet_lr5e-5_j21257_r2.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | 4.6562 | n/a | n/a | 68.88 | `data/results/experiments/260213_darknet_solo_lr5e-5_bs4_j21257_r2` |
| `260213_darknet_lr1e-4_j21255_r0.out` | 1 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | 4.4119 | n/a | n/a | 64.48 | `data/results/experiments/260213_darknet_solo_lr1e-4_bs4_j21255_r0` |
| `260213_resnet_lr1e-4_j21256_r1.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | 3.0736 | n/a | n/a | 71.69 | `data/results/experiments/260213_resnet_solo_lr1e-4_bs4_j21256_r1` |
| `260213_darknet_ld0.5_j21258_r0.out` | 2 | darknet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.500 | 0.500 | 0.000 | maxpool | logged | 50 | 4.6435 | n/a | n/a | 63.00 | `data/results/experiments/260213_darknet_distill_lr1e-4_bs4_j21258_r0` |
| `260213_resnet_lr5e-5_j21253_r3.out` | 1 | resnet | adaptive | 8 | legacy | global_mean | bce | auto | basic | 5e-5 | 0.000 | 0.500 | 0.000 | maxpool | logged | 50 | 3.1913 | n/a | n/a | 76.26 | `data/results/experiments/260213_resnet_solo_lr5e-5_bs4_j21253_r3` |
| `260213_darknet_uniform_q4_j21261_r0.out` | 0 | darknet | uniform | 4 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 2.1034 | n/a | n/a | 75.51 | `data/results/experiments/260213_darknet_uniform_q4_lr1e-4_bs4_j21261_r0` |
| `260213_darknet_uniform_q6_j21262_r1.out` | 0 | darknet | uniform | 6 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 1.9479 | n/a | n/a | 76.21 | `data/results/experiments/260213_darknet_uniform_q6_lr1e-4_bs4_j21262_r1` |
| `260213_resnet_uniform_q4_j21264_r3.out` | 0 | resnet | uniform | 4 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 0.9452 | n/a | n/a | 86.31 | `data/results/experiments/260213_resnet_uniform_q4_lr1e-4_bs4_j21264_r3` |
| `260213_resnet_uniform_q6_j21265_r4.out` | 0 | resnet | uniform | 6 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 0.7960 | n/a | n/a | 88.24 | `data/results/experiments/260213_resnet_uniform_q6_lr1e-4_bs4_j21265_r4` |
| `260213_darknet_uniform_q8_j21263_r2.out` | 0 | darknet | uniform | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 1.9299 | n/a | n/a | 76.36 | `data/results/experiments/260213_darknet_uniform_q8_lr1e-4_bs4_j21263_r2` |
| `260213_resnet_uniform_q8_j21260_r5.out` | 0 | resnet | uniform | 8 | legacy | global_mean | bce | auto | basic | 1e-4 | 0.000 | 0.000 | 0.000 | n/a | logged | 50 | 0.7701 | n/a | n/a | 89.15 | `data/results/experiments/260213_resnet_uniform_q8_lr1e-4_bs4_j21260_r5` |
| `260216_resnet_pilot_balanced_v1_headbasic_r0.out` | 2 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | basic | 0.0001 | 0.100 | 1.000 | 0.000 | maxpool | logged | 5 | 12.7122 | n/a | n/a | 78.40 | `data/results/experiments/260216_resnet_pilot_balanced_v1_headbasic_lr1e-4_bs2_localpilot_r0` |
| `260216_resnet_pilot_balanced_v2_headmultiscale_r1.out` | 2 | resnet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | multiscale | 0.0001 | 0.100 | 1.000 | 0.200 | maxpool | logged | 5 | 12.5746 | n/a | n/a | 78.47 | `data/results/experiments/260216_resnet_pilot_balanced_v2_headmultiscale_lr1e-4_bs2_localpilot_r1` |
| `260216_resnet_pilot_pp20_stage1.out` | 1 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 0.0001 | 0.000 | 1.000 | 0.000 | maxpool | logged | 3 | 26.9204 | n/a | n/a | 54.40 | `data/results/experiments/260216_resnet_pilot_pp20_stage1_lr1e-4_bs2_localpp_stage1` |
| `260216_resnet_pilot_pp20_stage2.out` | 2 | resnet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 0.0001 | 0.100 | 1.000 | 0.200 | maxpool | logged | 3 | 26.3631 | n/a | n/a | 55.00 | `data/results/experiments/260216_resnet_pilot_pp20_stage2_lr1e-4_bs2_localpp_stage2` |
| `260216_darknet_ld0.5_j21419_r0.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 0.500 | 1.000 | 0.200 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21419_r0` |
| `260216_darknet_ld1.0_j21420_r1.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 1.000 | 1.000 | 0.200 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21420_r1` |
| `260216_darknet_lr5e-5_j21418_r2.out` | 1 | darknet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 5e-5 | 0.000 | 1.000 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_darknet_solo_lr5e-5_bs4_j21418_r2` |
| `260216_resnet_lr1e-4_j21417_r1.out` | 1 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 1e-4 | 0.000 | 1.000 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_resnet_solo_lr1e-4_bs4_j21417_r1` |
| `260216_darknet_ld0.1_j21415_r2.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 0.100 | 1.000 | 0.200 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21415_r2` |
| `260216_resnet_lr5e-5_j21414_r3.out` | 1 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 5e-5 | 0.000 | 1.000 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_resnet_solo_lr5e-5_bs4_j21414_r3` |
| `260216_darknet_lr1e-4_j21416_r0.out` | 1 | darknet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 1e-4 | 0.000 | 1.000 | 0.000 | maxpool | logged | 50 | n/a | n/a | n/a | n/a | `data/results/experiments/260216_darknet_solo_lr1e-4_bs4_j21416_r0` |
| `260216_darknet_lr1e-4_j21423_r0.out` | 1 | darknet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 1e-4 | 0.000 | 1.000 | 0.000 | maxpool | logged | 150 | 1.6822 | n/a | n/a | 80.81 | `data/results/experiments/260216_darknet_solo_lr1e-4_bs4_j21423_r0` |
| `260216_darknet_ld0.5_j21426_r0.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 0.500 | 1.000 | 0.200 | maxpool | logged | 150 | 2.7526 | n/a | n/a | 70.89 | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21426_r0` |
| `260216_darknet_ld1.0_j21427_r1.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 1.000 | 1.000 | 0.200 | maxpool | logged | 150 | 3.1818 | n/a | n/a | 67.98 | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21427_r1` |
| `260216_darknet_lr5e-5_j21425_r2.out` | 1 | darknet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 5e-5 | 0.000 | 1.000 | 0.000 | maxpool | logged | 150 | 1.7948 | n/a | n/a | 82.73 | `data/results/experiments/260216_darknet_solo_lr5e-5_bs4_j21425_r2` |
| `260216_resnet_lr1e-4_j21424_r1.out` | 1 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 1e-4 | 0.000 | 1.000 | 0.000 | maxpool | logged | 150 | 0.7224 | n/a | n/a | 90.11 | `data/results/experiments/260216_resnet_solo_lr1e-4_bs4_j21424_r1` |
| `260216_darknet_ld0.1_j21422_r2.out` | 2 | darknet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 1e-4 | 0.100 | 1.000 | 0.200 | maxpool | logged | 150 | 2.1185 | n/a | n/a | 75.79 | `data/results/experiments/260216_darknet_distill_lr1e-4_bs4_j21422_r2` |
| `260216_resnet_lr5e-5_j21421_r3.out` | 1 | resnet | adaptive | 8 | balanced_v1 | normalized_global | weighted_bce | auto | pp_lite | 5e-5 | 0.000 | 1.000 | 0.000 | maxpool | logged | 150 | 0.7877 | n/a | n/a | 91.54 | `data/results/experiments/260216_resnet_solo_lr5e-5_bs4_j21421_r3` |
| `260219_resnet_pilot_s0_q6.out` | 0 | resnet | uniform | 6 | legacy | auto_by_recipe | auto_by_recipe | auto | basic | 0.0001 | 0.000 | 0.000 | 0.000 | maxpool | logged | 1 | 72.3707 | 6.000 | 2.560 | 0.00 | `data/results/experiments/260219_resnet_pilot_s0q6_lr1e-4_bs2_pilot_s0_q6` |
| `260219_resnet_pilot_s0_q8.out` | 0 | resnet | uniform | 8 | legacy | auto_by_recipe | auto_by_recipe | auto | basic | 0.0001 | 0.000 | 0.000 | 0.000 | maxpool | logged | 1 | 72.4441 | 8.000 | 3.724 | 0.00 | `data/results/experiments/260219_resnet_pilot_s0q8_lr1e-4_bs2_pilot_s0_q8` |
| `260219_resnet_pilot_s1_adapt.out` | 1 | resnet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 0.0001 | 0.000 | 1.000 | 0.200 | maxpool | logged | 1 | 72.1719 | 7.055 | 3.327 | 0.00 | `data/results/experiments/260219_resnet_pilot_s1adapt_lr1e-4_bs2_pilot_s1_adapt` |
| `260219_resnet_pilot_s2_distill.out` | 2 | resnet | adaptive | 8 | balanced_v2 | normalized_bg | weighted_bce | auto | pp_lite | 0.0001 | 0.100 | 1.000 | 0.200 | maxpool | logged | 1 | 73.0926 | 6.951 | 3.367 | 0.00 | `data/results/experiments/260219_resnet_pilot_s2distill_lr1e-4_bs2_pilot_s2_distill` |

## Stage Summary

| stage | runs | best_final_loss | best_log |
|---:|---:|---:|---|
| 0 | 14 | 0.7701 | `260213_resnet_uniform_q8_j21260_r5.out` |
| 1 | 22 | 0.7224 | `260216_resnet_lr1e-4_j21424_r1.out` |
| 2 | 21 | 2.1185 | `260216_darknet_ld0.1_j21422_r2.out` |

## Notes
- This ledger is intended to be updated continuously after each experiment.
- Recommended command:
  - `python src/utils/update_experiments_result.py`
