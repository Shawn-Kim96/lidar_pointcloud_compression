# Experiment Tracking Workflow

## Goal
- Keep experiment history in a continuously updated report-friendly ledger.
- Backfill newly introduced hyperparameters for legacy runs explicitly.

## Stage Semantics
- `Stage0`:
  - Uniform quantization baseline.
  - Teacher disabled.
  - No importance head supervision (`lambda_importance=0`).
  - Purpose: ROI-unaware compression reference for bitrate-vs-task comparison.
- `Stage1`:
  - Adaptive quantization with student importance head.
  - Teacher disabled.
  - ROI supervision enabled (`lambda_importance>0`) from dataset ROI mask.
  - Purpose: isolate gain from ROI-aware quantization without distillation.
- `Stage2`:
  - Same adaptive path as Stage1, plus teacher distillation (`lambda_distill>0`).
  - Purpose: measure incremental gain from teacher guidance on top of Stage1.
- `Stage3`:
  - Stage2 objective setup + advanced multi-scale importance head ablation.
  - Purpose: improve ROI selectivity/calibration and prepare detector-facing validation.

## Source of Truth
- Auto-generated ledger:
  - `src/results/experiments_result.md`
  - `src/results/experiments_result.csv`
- Generator script:
  - `src/utils/update_experiments_result.py`
- Loss specification (Stage0-Stage3):
  - `docs/report/loss_spec_stage0_stage3.md`

## Update Rule (After Every Experiment)
1. Finish run (logs are written under `logs/`).
2. Regenerate ledger:
   - `python src/utils/update_experiments_result.py`
3. Commit both:
   - `src/results/experiments_result.md`
   - `src/results/experiments_result.csv`

## Hyperparameter Backfill Policy
- If a newly introduced hyperparameter is missing in older logs, backfill and mark source.
- Current backfill rule:
  - `roi_target_mode`:
    - logged value if present
    - otherwise `nearest` + `backfilled_legacy_default`
  - `quantizer_mode`:
    - logged value if present
    - otherwise `adaptive` + `backfilled_legacy_default`
  - `quant_bits`:
    - logged value if present
    - otherwise `8` + `backfilled_legacy_default`
  - `loss_recipe`:
    - logged value if present
    - otherwise `legacy` + `backfilled_legacy_default`
  - `rate_loss_mode` / `importance_loss_mode`:
    - logged value if present
    - otherwise inferred from `loss_recipe` default + `backfilled_from_recipe_default`

## New Stage2 Loss-Recipe Ablation (2026-02-16)
- Motivation:
  - reduce importance-collapse caused by overly strong raw rate penalty.
  - improve distillation signal quality and head capacity.
- Key new parameters:
  - `loss_recipe`: `legacy`, `balanced_v1`, `balanced_v2`
  - `rate_loss_mode`: `global_mean`, `normalized_global`, `normalized_bg`
  - `importance_loss_mode`: `bce`, `weighted_bce`
  - `lambda_imp_separation` / `imp_separation_margin`
  - `distill_logit_loss`: `auto`, `kl`, `bce`, `mse`
  - `importance_head_type`: `basic`, `multiscale`, `pp_lite`, `bifpn`, `deformable_msa`, `dynamic`, `rangeformer`, `frnet`
  - `teacher_proxy_ckpt`: pretrained proxy teacher path
- Scripts:
  - login-node checkpoint preparation:
    - `src/scripts/download_pointpillar_checkpoint.sh`
  - login-node predownload + submit:
    - `src/scripts/prepare_teacher_and_submit_stage2_loss_recipe.sh`
  - GPU ablation array:
    - `src/scripts/run_stage2_loss_recipe_ablation.sh`
  - local quick pilot:
    - `src/scripts/run_stage2_loss_recipe_pilot_local.sh`

## ROI Target Mode Sweep Scripts
- Stage1 (parallel):
  - `src/scripts/run_stage1_roi_target_sweep.sh`
- Stage2 (parallel):
  - `src/scripts/run_stage2_roi_target_sweep.sh`

Both scripts run Slurm array jobs over:
- `nearest`
- `maxpool`
- `area`

## Uniform Quantization Baseline Sweep
- Script:
  - `src/scripts/run_uniform_baseline.sh`
- Purpose:
  - ROI-unaware compression baseline with uniform quantization.
- Array dimensions:
  - backbone: `darknet`, `resnet`
  - quant bits: `4`, `6`, `8`

## One-Command Parallel Submission
- Script:
  - `src/scripts/submit_parallel_stage_runs.sh`
- Submits in parallel:
  - uniform baseline (Stage0)
  - Stage1 adaptive
  - Stage2 adaptive+distill

## Stage3 Multi-Scale Head Ablation (New)
- Script:
  - `src/scripts/run_stage3_multiscale_heads.sh`
- Purpose:
  - Compare modern multi-scale ROI head variants under Stage3-style setup.
- Array head types:
  - `bifpn`
  - `deformable_msa`
  - `dynamic`
  - `rangeformer`
  - `frnet`
- Backbone support:
  - `resnet` and `darknet` (via `BACKBONE` env)
- Notes:
  - model now supports multi-stage feature taps from encoder (`return_features=True`) and routes them into Stage3 fusion heads.
  - `rangeformer` / `frnet` are Stage3 importance-head fusion variants (neck/head side), not encoder replacements.

## 2026-02-16: Default Path Hardening + PP20 Head Pilot
- Problem statement:
  - legacy defaults (`w_rate=0.1`, `w_importance=0.5`, raw/global rate pressure) can collapse importance toward floor values.
  - Stage2 default script previously used `proxy` backend by default.
  - existing `basic/multiscale` heads were too small relative to teacher capacity.

- Code-level updates:
  - `src/scripts/run_stage1.sh`
    - defaults switched to collapse-safe objective:
      - `loss_recipe=balanced_v1`
      - `rate_loss_mode=normalized_global`
      - `importance_loss_mode=weighted_bce`
      - `lambda_rate=0.02`, `lambda_importance=1.0`
    - default head upgraded:
      - `importance_head_type=pp_lite`
      - `importance_hidden_channels=64`
  - `src/scripts/run_stage2.sh`
    - defaults switched to:
      - `loss_recipe=balanced_v2`
      - `rate_loss_mode=normalized_bg`
      - `importance_loss_mode=weighted_bce`
      - `lambda_rate=0.02`, `lambda_importance=1.0`, `lambda_imp_separation=0.2`
    - teacher defaults switched to real checkpoint path:
      - `TEACHER_BACKEND=pointpillars_zhulf`
      - `TEACHER_PROXY_CKPT=data/checkpoints/pointpillars_epoch_160.pth`
    - default head upgraded:
      - `importance_head_type=pp_lite`
      - `importance_hidden_channels=64`
  - `src/models/importance_head.py`
    - added `pp_lite` head type (FPN-like, PointPillars-inspired dense importance head).
    - enforced `hidden_channels >= 64` for `pp_lite` to keep PP20 capacity target.
  - `src/main_train.py`
    - added CLI choice: `--importance_head_type pp_lite`
  - New quick pilot script:
    - `src/scripts/run_stage1_stage2_pp_lite_pilot_local.sh`

- Capacity check (`in_channels=64`):
  - teacher (`ZhulfPointPillarsTeacherNet`): `4,834,824` params
  - `pp_lite(hidden=64)`: `1,020,609` params
  - ratio: `21.1%` of teacher (meets >=20% requirement)

- New pilot runs executed (local, CPU):
  - `logs/260216_resnet_pilot_pp20_stage1.out`
    - stage1, `balanced_v1`, `pp_lite(64)`, `epochs=3`, `max_train_frames=128`
    - loss: `59.0393 -> 26.9204`
  - `logs/260216_resnet_pilot_pp20_stage2.out`
    - stage2, `balanced_v2`, `pp_lite(64)`, `teacher=pointpillars_zhulf`, `epochs=3`, `max_train_frames=128`
    - loss: `58.5874 -> 26.3631`

- Ledger update:
  - regenerated with:
    - `python src/utils/update_experiments_result.py`
  - new rows:
    - `260216_resnet_pilot_pp20_stage1.out`
    - `260216_resnet_pilot_pp20_stage2.out`

## 2026-02-17: 150-Epoch Full Runs (PP20 + Balanced Recipes)
- Run setup:
  - Stage1 (`run_stage1.sh`):
    - adaptive, `balanced_v1`, `normalized_global`, `weighted_bce`
    - `importance_head_type=pp_lite`, `importance_hidden_channels=64`
    - `lambda_rate=0.02`, `lambda_importance=1.0`
    - sweep: backbone `{darknet,resnet}` x lr `{1e-4,5e-5}`
  - Stage2 (`run_stage2.sh`):
    - adaptive, `balanced_v2`, `normalized_bg`, `weighted_bce`
    - `importance_head_type=pp_lite`, `importance_hidden_channels=64`
    - `teacher_backend=pointpillars_zhulf`
    - `teacher_proxy_ckpt=data/checkpoints/pointpillars_epoch_160.pth`
    - distill sweep: `lambda_distill={0.1,0.5,1.0}`
  - all runs finished at `epochs=150`.

- Stage1 summary (final loss):
  - `260216_resnet_lr1e-4_j21424_r1.out`: `0.7224` (best Stage1)
  - `260216_resnet_lr5e-5_j21421_r3.out`: `0.7877`
  - `260216_darknet_lr1e-4_j21423_r0.out`: `1.6822`
  - `260216_darknet_lr5e-5_j21425_r2.out`: `1.7948`
  - Observation: resnet remains clearly stronger than darknet under this objective/head setup.

- Stage2 summary (darknet, final loss):
  - `lambda_distill=0.1`: `260216_darknet_ld0.1_j21422_r2.out` -> `2.1185` (best Stage2)
  - `lambda_distill=0.5`: `260216_darknet_ld0.5_j21426_r0.out` -> `2.7526`
  - `lambda_distill=1.0`: `260216_darknet_ld1.0_j21427_r1.out` -> `3.1818`
  - Observation: stronger distill weight degraded convergence metric in this sweep (`0.1` best).

- Current ledger-best by stage (after update):
  - Stage1 best: `0.7224` (`260216_resnet_lr1e-4_j21424_r1.out`)
  - Stage2 best: `2.1185` (`260216_darknet_ld0.1_j21422_r2.out`)

- Note:
  - final loss is stage-objective-dependent; use ROI/importance and downstream detector metrics for true Stage1 vs Stage2 effectiveness comparison.

## 2026-02-19: Fair-Budget + Oracle-ROI Decomposition Pilot
- Motivation:
  - `uniform vs adaptive` cannot be interpreted fairly if bitrate budgets differ.
  - needed a direct diagnostic to separate:
    - importance-head/training bottleneck
    - quantizer-formulation bottleneck

- New code additions:
  - `src/train/trainer.py`
    - epoch log now includes:
      - `rate_proxy`
      - `eq_bits`
      - `code_entropy`
    - always saves `model_final.pth` (short pilots are now evaluable).
  - `src/train/evaluate_oracle_roi.py`
    - evaluates per run:
      - `native` (predicted importance)
      - `oracle_roi` (GT ROI forced into quantizer; adaptive only)
    - outputs:
      - `all/roi/bg mse`
      - `eq_bits`, `code_entropy`
      - `bpp_eq`, `bpp_entropy`
      - TP/FP/FN/TN + IoU/Precision/Recall from importance map.
  - `src/utils/match_bitrate_budget.py`
    - matches adaptive rows to nearest uniform baseline by selected bitrate metric (`bpp_entropy_mean` etc.).
  - `src/scripts/run_stage0_stage1_stage2_oracle_pilot_local.sh`
    - one-command local run:
      - Stage0 q6 + q8
      - Stage1 adaptive
      - Stage2 adaptive+distill
      - then oracle/native evaluation + ledger refresh.

- Pilot run command:
  - `EPOCHS=1 MAX_TRAIN_FRAMES=32 VAL_MAX_FRAMES=16 BATCH_SIZE=2 NUM_WORKERS=0 bash src/scripts/run_stage0_stage1_stage2_oracle_pilot_local.sh`

- Pilot artifacts:
  - logs:
    - `logs/260219_resnet_pilot_s0_q6.out`
    - `logs/260219_resnet_pilot_s0_q8.out`
    - `logs/260219_resnet_pilot_s1_adapt.out`
    - `logs/260219_resnet_pilot_s2_distill.out`
  - oracle summaries:
    - `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q6.csv`
    - `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q8.csv`
    - `notebooks/oracle_eval_summary_260219_resnet_pilot_s1adapt.csv`
    - `notebooks/oracle_eval_summary_260219_resnet_pilot_s2distill.csv`
  - matched-budget table:
    - `notebooks/matched_bitrate_pairs_260219_pilot.csv`

- Key pilot numbers (16-frame eval subset):
  - Stage0 q6 native:
    - `bpp_entropy_mean=0.6668`, `roi_mse_mean=228.0958`
  - Stage0 q8 native:
    - `bpp_entropy_mean=0.9598`, `roi_mse_mean=227.3080`
  - Stage1 adaptive:
    - native: `bpp_entropy_mean=0.8064`, `roi_mse_mean=227.8031`
    - oracle: `bpp_entropy_mean=0.4177`, `roi_mse_mean=227.8031`
  - Stage2 adaptive+distill:
    - native: `bpp_entropy_mean=0.8687`, `roi_mse_mean=227.6069`
    - oracle: `bpp_entropy_mean=0.4492`, `roi_mse_mean=227.6073`

- Matched-budget quick read (`metric=bpp_entropy_mean`):
  - Stage1 native matched to Stage0 q6:
    - ROI MSE gain vs uniform: `+0.128%`
  - Stage2 native matched to Stage0 q8:
    - ROI MSE gain vs uniform: `-0.132%`

- Interpretation:
  - This is a short-run pilot, not final evidence.
  - Oracle mode achieves much lower bitrate (`bpp_entropy`) at nearly unchanged MSE:
    - suggests current native importance map is not yet rate-efficient.
    - supports that next priority is importance quality/calibration (head + training), before claiming robust ROI-aware gains.

## 2026-02-19/20: Stage3 Dual-Backbone Multi-Scale Head Runs (Running)
- Submission status:
  - `resnet` Stage3 array submitted: `job 21834` (`5` head variants).
  - `darknet` Stage3 array submitted: `job 21840` (`5` head variants).
  - both arrays are currently running; no final metrics yet.
- Shared run configuration:
  - `epochs=150`, `lr=1e-4`, `batch_size=4`
  - `quantizer_mode=adaptive`, `roi_target_mode=maxpool`
  - `loss_recipe=balanced_v2`, `rate_loss_mode=normalized_bg`, `importance_loss_mode=weighted_bce`
  - `lambda_recon=1.0`, `lambda_rate=0.02`, `lambda_distill=0.1`, `lambda_importance=1.0`, `lambda_imp_separation=0.2`
  - `teacher_backend=pointpillars_zhulf`, `teacher_proxy_ckpt=data/checkpoints/pointpillars_epoch_160.pth`
- Head variants under test:
  - `bifpn`
  - `deformable_msa`
  - `dynamic`
  - `rangeformer`
  - `frnet`
- Script:
  - `src/scripts/run_stage3_multiscale_heads.sh`
- Log patterns:
  - `logs/slurm_21834_*.out`, `logs/slurm_21834_*.err`
  - `logs/slurm_21840_*.out`, `logs/slurm_21840_*.err`
  - `logs/260219_resnet_stage3_head*.out`
  - `logs/260219_darknet_stage3_head*.out`
- Note on metadata:
  - Stage3 wrapper prints `stage: 3` metadata.
  - `src/main_train.py` still labels internal mode as Stage2 (`distill`) because the training objective path is Stage2-compatible; Stage3 in this repo currently denotes the head/validation regimen.
