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
- Paper-facing fair comparison snapshot:
  - `docs/report/paper_fair_comparison_table.md`
  - `notebooks/paper_fair_comparison_20260220.csv`
  - `notebooks/kitti_map_vs_rate_summary.csv`
  - `notebooks/kitti_map_vs_rate_detail.csv`
  - `notebooks/kitti_map_vs_rate_pairs.csv`

## Dual-Track Evaluation Policy (New)
- `Track-A` (SemanticKITTI):
  - ROI-aware codec behavior and calibration:
    - `bpp_entropy`, `all/roi/bg MSE`, ROI overlap (`TP/FP/FN/TN`, `IoU`, `Precision`, `Recall`).
- `Track-B` (KITTI detection + OpenPCDet):
  - Official detector endpoint:
    - `AP3D` class metrics and `mAP vs bitrate`.
- Policy:
  - Do not mix Track-A proxy metrics and Track-B official detector metrics in a single claim.

## KITTI Detector Endpoint Tooling (New)
- Evaluation entrypoint:
  - `src/train/evaluate_kitti_map_vs_rate.py`
- Shell wrapper:
  - `src/scripts/run_kitti_map_vs_rate.sh`
- GPU sbatch wrapper:
  - `src/scripts/run_kitti_map_vs_rate_sbatch.sh`
- Bitrate fairness matcher for detector table:
  - `src/utils/match_bitrate_budget_detector.py`
- Optional teacher quality/fine-tune branch:
  - `src/scripts/run_teacher_finetune_kitti.sh`
- Reconstruction export standardization:
  - `src/utils/recon_pointcloud_export.py`

## KITTI Object Download + PointPillar Finetune (New)
- KITTI object downloader:
  - `src/scripts/download_kitti3dobject.sh`
  - python backend: `src/dataset/download_kitti3dobject.py`
  - sbatch wrapper: `src/scripts/download_kitti3dobject_sbatch.sh`
- PointPillar finetune scripts:
  - `src/scripts/run_pointpillar_kitti_finetune.sh`
  - `src/scripts/run_pointpillar_kitti_finetune_sbatch.sh`
- Finetune result artifacts:
  - summary csv: `notebooks/pointpillar_finetune_kitti_summary.csv`
  - summary markdown: `docs/report/pointpillar_finetune_kitti.md`

### PointPillar finetune run (sbatch)
```bash
sbatch \
  --export=ALL,KITTI_ROOT_OFFICIAL=/path/to/kitti3dobject,EPOCHS=40,BATCH_SIZE=4,WORKERS=4,WAIT_FOR_KITTI_SEC=14400 \
  src/scripts/run_pointpillar_kitti_finetune_sbatch.sh
```

### Download run (long-running, resumable)
```bash
bash src/scripts/download_kitti3dobject.sh \
  --data-dir data/dataset/kitti3dobject \
  --only essentials \
  --extractor unzip
```

### New CLI interfaces
- `python src/train/evaluate_kitti_map_vs_rate.py --kitti_root ... --run_dirs ... --openpcdet_cfg ... --openpcdet_ckpt ...`
- `bash src/scripts/run_kitti_map_vs_rate.sh`
  - required env:
    - `KITTI_ROOT_OFFICIAL` (official KITTI detection-format root)
    - `RUN_DIRS` (comma-separated)
  - default fixed protocol:
    - links `${KITTI_ROOT_OFFICIAL}` -> `third_party/OpenPCDet/data/kitti`
    - validates KITTI layout/split sizes (guards against non-official huge converted sets)
    - auto-generates `kitti_infos_{train,val}.pkl` when missing
    - runs `tools/test.py` sanity first and enforces `Car 3D AP(mod) > 0`
    - then runs reconstructed comparison with same cfg/ckpt/split/metric

### New defaults
- `teacher_ap3d_mod_car_min=55.0`
- `bitrate_match_metric=bpp_entropy_mean`
- `bitrate_pair_max_gap=0.05`
- `workers=0`, `max_frames=0`, `eval_metric=kitti`

### Track-B execution example (official protocol)
```bash
KITTI_ROOT_OFFICIAL=/path/to/kitti_detection \
RUN_DIRS=data/results/experiments/runA,data/results/experiments/runB \
OPENPCDET_CFG=third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
OPENPCDET_CKPT=data/checkpoints/openpcdet_pointpillar_18M.pth \
MAX_FRAMES=0 WORKERS=0 EVAL_METRIC=kitti \
bash src/scripts/run_kitti_map_vs_rate.sh
```

### Track-B on GPU node (sbatch)
```bash
sbatch \
  --export=ALL,KITTI_ROOT_OFFICIAL=/path/to/kitti_detection,RUN_DIRS=data/results/experiments/runA,data/results/experiments/runB \
  src/scripts/run_kitti_map_vs_rate_sbatch.sh
```

## Stage2 Distill-Fix Ablation (New)
- Problem being tested:
  - teacher/task mismatch under SemanticKITTI-only supervision,
  - weak resize-only feature alignment in distillation.
- New distillation controls:
  - `distill_align_mode`: `resize` or `adaptive_pool`
  - `distill_align_hw`: target pool size for `adaptive_pool` (e.g., `16,32`)
  - `distill_feature_source`: `channel_mean`, `energy_map`, or `none`
  - `distill_teacher_score_min` + `distill_teacher_score_weight`
- Scripts:
  - local pilot:
    - `src/scripts/run_stage2_distill_fix_pilot_local.sh`
  - slurm ablation:
    - `src/scripts/run_stage2_distill_fix_ablation.sh`

### Quick Run Commands
- Local smoke run (short):
  - `EPOCHS=1 MAX_TRAIN_FRAMES=8 BATCH_SIZE=1 NUM_WORKERS=0 BACKBONE=resnet src/scripts/run_stage2_distill_fix_pilot_local.sh`
- Slurm full ablation:
  - `sbatch src/scripts/run_stage2_distill_fix_ablation.sh`
  - optional overrides:
    - `sbatch --export=ALL,BACKBONE=resnet,EPOCHS=150,TEACHER_PROXY_CKPT=data/checkpoints/pointpillars_epoch_160.pth src/scripts/run_stage2_distill_fix_ablation.sh`

### Latest Smoke Execution Snapshot
- Date: `2026-02-24`
- Script:
  - `src/scripts/run_stage2_distill_fix_pilot_local.sh`
- Config:
  - `epochs=1`, `max_train_frames=8`, `batch_size=1`, `backbone=resnet`
- Cases and epoch-0 losses:
  - `legacy_mean_resize`: `81.9734`
  - `energy_pool_16x32`: `80.1464`
  - `energy_pool_16x32_scoregate015`: `80.8066`
- Logs:
  - `logs/260223_resnet_pilot_legacy_mean_resize_r0.out`
  - `logs/260223_resnet_pilot_energy_pool_16x32_r1.out`
  - `logs/260223_resnet_pilot_energy_pool_16x32_scoregate015_r2.out`

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
- Dataset defaults (updated):
  - `dataset_type=kitti3dobject`
  - `data_root=data/dataset/kitti3dobject`
  - ROI target from KITTI label boxes projected to range-image mask

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

## 2026-02-24: KITTI 3D BBox Endpoint (OpenPCDet) First Complete Run
- Objective:
  - verify Track-B end-to-end execution (`original` vs `reconstructed` point clouds) with official-style 3D AP reporting path.
- Run:
  - slurm job: `22123`
  - script: `src/scripts/run_kitti_map_vs_rate.sh`
  - detector: OpenPCDet PointPillars checkpoint (`pointpillar_7728.pth`)
  - split: `128`-frame KITTI-format subset (`data/dataset/kitti_eval_subset`)
- Outputs:
  - `notebooks/kitti_map_vs_rate_summary.csv`
  - `notebooks/kitti_map_vs_rate_detail.csv`
  - `notebooks/kitti_map_vs_rate_pairs.csv`
  - `docs/report/paper_fair_comparison_table.md` (Table-B auto-update)
- Result snapshot:
  - `original` AP3D (Car/Ped/Cyc, mod): all `0.0`
  - `reconstructed` AP3D (Car/Ped/Cyc, mod): all `0.0`
  - bitrate proxies were still computed (`bpp_entropy_mean` non-zero), so evaluation path is running.
- Interpretation:
  - Track-B infrastructure is now connected and reproducible.
  - detector quality gate failed (`teacher_ap3d_mod_car=0.0 < 55.0`), so this run is not valid as a detector-performance claim.
  - next step is protocol/domain correction (canonical KITTI val protocol and/or detector fine-tune before distill claim).

## 2026-02-24: Stage2 Distill-Fix Two-Experiment Multi-Parameter Sweep (Submitted)
- Objective:
  - run the two updated Stage2 distillation fixes for multiple epochs and parameter combinations.
  - compare robustness of:
    - `energy_pool_16x32`
    - `energy_pool_16x32_scoregate015`
- Sweep configuration:
  - epochs: `80`
  - backbone: `resnet`
  - learning rate: `{1e-4, 5e-5}`
  - lambda_distill: `{0.05, 0.1, 0.2}`
  - total runs: `12` (`2 x 2 x 3`)
- Submission:
  - train array job: `22127`
  - post aggregation job: `22140` (`afterany:22127`)
- Scripts:
  - train array: `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh`
  - submit helper: `src/scripts/submit_stage2_distill_fix_twoexp_sweep.sh`
  - summary exporter: `src/utils/summarize_stage2_distill_fix_sweep.py`
- Output targets (auto-generated after completion):
  - `src/results/experiments_result.csv`
  - `src/results/experiments_result.md`
  - `notebooks/stage2_distill_fix_twoexp_summary_22127.csv`
  - `docs/report/stage2_distill_fix_twoexp_summary_22127.md`

## 2026-02-24: Stage2 Distill-Fix Two-Experiment Sweep (Completed)
- Completion:
  - array training tasks finished (`12` runs total).
  - results are in:
    - `src/results/experiments_result.csv`
    - `notebooks/stage2_distill_fix_twoexp_summary_22127.csv`
    - `docs/report/stage2_distill_fix_twoexp_summary_22127.md`
- Best runs per case:
  - `energy_pool_16x32`:
    - `run_id=j22129_r0`, `lr=1e-4`, `lambda_distill=0.05`
    - `best_loss=1.1291`, `final_imp_mean=0.1432`
  - `energy_pool_16x32_scoregate015`:
    - `run_id=j22137_r8`, `lr=1e-4`, `lambda_distill=0.2`
    - `best_loss=0.8168`, `final_imp_mean=0.1472`
- Key trend:
  - score-gated variant (`distill_teacher_score_min=0.15`) consistently outperformed non-gated variant in this sweep.
  - non-gated case degraded as `lambda_distill` increased (`0.05` best, `0.2` worst).

## 2026-02-23: Stage2 Distill-Fix Quick ROI Validation (16 Frames, Native)
- Purpose:
  - validate whether the best run from the score-gated case also improves Track-A ROI metrics over the best non-gated run.
- Input summaries:
  - `notebooks/oracle_eval_summary_260223_fix2_no_scoregate_best.csv`
  - `notebooks/oracle_eval_summary_260223_fix2_scoregate_best.csv`
- Compared checkpoints:
  - non-gated best:
    - `j22129_r0` (`energy_pool_16x32`, `lr=1e-4`, `lambda_distill=0.05`)
  - score-gated best:
    - `j22137_r8` (`energy_pool_16x32_scoregate015`, `lr=1e-4`, `lambda_distill=0.2`)
- Native (16-frame) result:
  - non-gated best:
    - `all_mse=3.122`, `roi_mse=6.550`, `IoU=0.0542`, `Precision=0.0568`, `Recall=0.5375`
  - score-gated best:
    - `all_mse=2.600`, `roi_mse=4.775`, `IoU=0.0684`, `Precision=0.0707`, `Recall=0.6810`
- Interpretation:
  - score-gated configuration improves both reconstruction and ROI overlap metrics in Track-A.
  - this supports that distillation stability is improved, but it is still Track-A-only evidence.

## Current Claim Boundary (As of 2026-02-25)
- Research objective remains dual-track:
  - `Track-A`: SemanticKITTI codec/ROI metrics.
  - `Track-B`: KITTI + OpenPCDet detector endpoint (`AP3D/mAP`).
- Track-B status:
  - `notebooks/kitti_map_vs_rate_summary.csv` still shows `AP3D=0` for all families in the current run.
- Reliable conclusion:
  - partial fix achieved for Stage2 training stability with:
    - `adaptive_pool + energy_map + score-gate`
  - final detector-performance claim is pending protocol/domain alignment on Track-B.

## 2026-02-25/27: Stage0/Stage1 KITTI Detector Degradation Root-Cause Analysis (RCA)
- Objective:
  - explain why Stage0/Stage1 training loss decreased but reconstructed-cloud detector behavior was still poor.
  - decide whether the bottleneck is mainly `epochs`, `importance head capacity`, or earlier representation loss.

### Scope (runs and artifacts used)
- Stage0 reference run:
  - `data/results/experiments/260224_resnet_uniform_q8_lr1e-4_bs4_j22255_r5`
  - log: `logs/260224_resnet_uniform_q8_j22255_r5.out`
- Stage1 reference run:
  - `data/results/experiments/260224_resnet_solo_lr1e-4_bs4_j22279_r1`
  - log: `logs/260224_resnet_lr1e-4_j22279_r1.out`
- Visualization artifact:
  - `notebooks/stage0_stage1_kitti_pointpillar_visualization.executed.ipynb`
  - `notebooks/stage0_stage1_kitti_pointpillar_visualization.executed.html`
- Detector-side one-sample debug log:
  - `logs/slurm_22383.out`

### Evidence 1: training itself did not collapse
- Both runs trained on KITTI (`dataset_type=kitti3dobject`, `3712` train frames) for `50` epochs.
- Stage0 loss: `21.3739 -> 1.5752` (epoch `0 -> 49`).
- Stage1 loss: `21.7544 -> 1.7810` (epoch `0 -> 49`).
- Tail epochs still decreased, so this is not a simple divergence/crash story.

### Evidence 2: large irreversible sparsification happens before codec learning
- Reconstruction path projects point cloud to fixed `64x1024` range image and keeps one point per pixel on collision:
  - see `src/utils/recon_pointcloud_export.py` (`project_points_to_range_image`).
- Additional val-split measurement (`200` frames):
  - mean point-retention ratio after projection: `0.4169`.
  - mean raw points/frame: `119,502.2`.
  - mean occupied pixels/frame: `49,812.1`.
  - mean collision factor (`raw/occupied`): `2.399`.
- Interpretation:
  - about `58%` of points are dropped by projection-collision alone (model-independent).

### Evidence 3: detector drop appears even with identity projection/unprojection
- Sample `003464` debug (`logs/slurm_22383.out`):
  - raw points: `123,592`
  - identity projected/unprojected points (no codec): `50,911`
  - raw detector preds: `44`
  - identity detector preds: `27`
  - Stage0 reconstructed: `50,767` points, `pred_count=1`
  - Stage1 reconstructed: `50,635` points, `pred_count=0`
- Interpretation:
  - detector quality already degrades materially at representation conversion (`raw -> range -> points`) before learned quantization differences.

### Evidence 4: visualization frame also shows raw-vs-recon gap
- Selected high-ROI frame:
  - sample `003464`, ROI ratio `0.0856`, GT boxes `15`.
  - raw prediction boxes (`score>=0.30`): `14`.
  - Stage0 q8 reconstruction: `raw_pred=14 -> recon_pred=0`, `bpp_entropy=1.5378`.
  - Stage1 nearest reconstruction: `raw_pred=14 -> recon_pred=0`, `bpp_entropy=0.7692`.
- This reproduces the endpoint symptom consistently in notebook outputs.

### Diagnosis (priority-ranked)
1. Primary bottleneck: representation loss in current reconstruction path.
   - fixed-grid, single-return projection introduces heavy point loss before detector input.
2. Secondary bottleneck: objective mismatch.
   - training optimizes reconstruction/rate/importance losses, not detector AP directly.
3. Monitoring caveat:
   - Track-B summary table with AP all-zero (`kitti_map_vs_rate_summary.csv` current pilot) is a warning signal, but parsing/protocol alignment is still being hardened.
4. Lower-priority factors:
   - `epoch` and `importance head` are not ruled out, but current evidence does not support them as first-order cause.
   - Stage0 (no importance head supervision) already shows the same endpoint degradation trend.

### Decision and immediate next experiments
- Add mandatory baseline in Track-B reports:
  - `raw` vs `identity(project->unproject)` vs `reconstructed`.
  - without this baseline, codec effect and projection effect are confounded.
- Keep `original KITTI AP non-zero sanity` as hard gate before reconstructed comparison claims.
- For compression model iteration:
  - prioritize representation-preserving export path experiments (multi-return per angular bin or alternative point reconstruction policy),
  - then re-check Stage0/Stage1/Stage2 detector deltas under identical protocol.
