# LiDAR Compression Research Journal (English)

Purpose: keep a date-ordered record of what was tested, why it was tested, what failed, and what was decided next.

## 2026-02-11

### What was done
- Team meeting and thesis-direction alignment.
- Defined Stage-level framing for experiments (baseline vs adaptive/distill tracks).

### Why
- Establish a reproducible experiment strategy before large sweeps.
- Prevent confusion between internal baselines and external paper baselines.

### Problems / risks
- Evaluation pipeline was focused on reconstruction-only signals.
- Detector-awareness was not yet fully integrated in training.

### Decisions
- Keep current baseline as internal control.
- Add detector-related validation and ROI-focused metrics.
- Track intermediate outputs and save per-hyperparameter metrics.

### Source notes
- `docs/notes/260211_meeting_note.md`
- `docs/notes/260211_meeting_note_en.md`

---

## 2026-02-12

### What was done
- Ran Stage1 sweep (adaptive quantization, no teacher):
  - `260212_darknet_lr1e-4_r0.out`
  - `260212_darknet_lr5e-5_r2.out`
  - `260212_resnet_lr1e-4_r1.out`
  - `260212_resnet_lr5e-5_r3.out`
- Ran Stage2 sweep (adaptive quantization + proxy distillation):
  - `260212_darknet_ld0.1_r2.out`
  - `260212_darknet_ld0.5_r0.out`
  - `260212_darknet_ld1.0_r1.out`

### Why
- Measure baseline behavior of Stage1/Stage2 before architecture changes.
- Check sensitivity to learning rate and distillation weight.

### Key observations
- Stage1 best run (among listed): `resnet, lr=1e-4`, final loss `2.8335`.
- Stage2 best run (darknet distill sweep): `lambda_distill=0.1`, final loss `3.8924`.
- Stage2 did not show a strong gain pattern from larger distill weight in this early proxy setup.

### Problems / risks
- Legacy logs did not contain explicit `roi_target_mode`, `quantizer_mode`, `quant_bits`.
- ROI prediction quality concerns remained (importance map collapse risk reported later).

### Decisions
- Add explicit metadata/backfill policy for newly introduced hyperparameters.
- Continue with ROI/importance head diagnosis before multi-stage model expansion.

### Sources
- `src/results/experiments_result.md`
- `src/results/experiments_result.csv`

---

## 2026-02-13

### What was done
- Implemented and wired:
  - Uniform quantization baseline path (`Stage0`).
  - Distillation loss path in trainer (feature/logit distill).
  - Parallel run scripts for Stage0/Stage1/Stage2.
  - ROI target mode control (`nearest`, `maxpool`, `area`) in training.
- Added reliability fixes for Slurm scripts:
  - fail-fast shell settings
  - conda environment bootstrap
  - job-id-based run/log naming to avoid collisions
- Submitted parallel runs:
  - Stage1: job `21253` (array)
  - Stage2: job `21254` (array)
  - Stage0 uniform: job `21260` (array)

### Why
- A uniform baseline is required to prove whether adaptive importance allocation truly improves task/reconstruction trade-offs.
- Distillation needed to be actually connected before interpreting Stage2 trends.
- Reliable job execution and non-colliding logs are mandatory for thesis-grade tracking.

### Problems / risks
- Initial Slurm submissions failed because `torch` was unavailable in default Python environment.
- Uniform script initially ran on wrong partition due misplaced `SBATCH` directive block; fixed and resubmitted.

### Decisions
- Standardize experiment scripts to always select the expected conda environment.
- Keep job-id-tagged run IDs in logs/save paths as default policy.

### Sources
- `src/scripts/run_uniform_baseline.sh`
- `src/scripts/run_stage1.sh`
- `src/scripts/run_stage2.sh`
- `src/scripts/submit_parallel_stage_runs.sh`

---

## 2026-02-14

### What was done
- Reviewed ROI/importance failure mode and architecture constraints.
- Documented ROI supervision downsampling strategy and multi-scale direction.
- Confirmed Stage semantics in documentation:
  - Stage0: uniform baseline
  - Stage1: adaptive + ROI supervision
  - Stage2: Stage1 + distillation

### Why
- ROI quality at low latent resolution was a core blocker for importance-aware compression claims.
- Needed a clear ablation structure before scaling complexity.

### Key observations
- Final latent map resolution is highly compressed (DarkNet path), so sparse ROI positives are easily lost.
- `nearest` downsampling for ROI targets can collapse positives after aggressive spatial reduction.
- This can produce near-empty predicted ROI at common thresholds.

### Problems / risks
- Class imbalance remains severe even after target-mode improvements.
- Proxy teacher quality can limit practical Stage2 gains.

### Decisions
- Set `maxpool` as practical default for ROI-target downsampling under heavy downsampling.
- Keep quantization at a single bottleneck for controlled ablations.
- Move to multi-scale ROI head next, then revisit multi-stage quantization later.

### Sources
- `docs/notes/260214_roi_multiscale_discussion.md`
- `docs/report/stage2_1_backbone_audit.md`
- `docs/report/experiments.md`
- `src/results/experiments_result.md`
- `notebooks/analysis_log_summary.csv`

### 2026-02-14 Late Update (Result Refresh + Visualization Refresh)

#### What was done
- Regenerated experiment ledger from full `logs/`:
  - `python src/utils/update_experiments_result.py`
- Re-executed analysis notebook to refresh plots/tables:
  - `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
- Expanded notebook log parser scope from dated pattern to generic run pattern:
  - `LOG_DIR.glob(\"*_r*.out\")`

#### Snapshot after refresh
- Ledger totals:
  - total parsed runs: `33`
  - runs with valid final loss: `20`
- Best valid runs by stage:
  - Stage0: `260213_resnet_uniform_q8_j21260_r5.out` (`final_loss=0.7701`)
  - Stage1: `260212_resnet_lr1e-4_r1.out` (`final_loss=2.8335`)
  - Stage2: `260212_darknet_ld0.1_r2.out` (`final_loss=3.8924`)
- Current `j`-run mean final loss:
  - Stage0: `1.4154` (n=6)
  - Stage1: `3.8333` (n=4)
  - Stage2: `4.6228` (n=3)

#### Important interpretation notes
- Final loss is **not directly comparable across stages** because loss composition differs:
  - Stage0 uses reconstruction-focused objective (`lambda_rate=0`, `lambda_importance=0`, no teacher).
  - Stage1/Stage2 include additional penalties (rate/importance/distill).
- Some logs from failed earlier submissions are still present and appear with `nan` final loss (kept for auditability).
- Several successful `j` runs stopped before epoch 50 in logs; they are still valid for trend reading but not fully matched-length training.

---

## Update Rule
- After each experiment cycle:
  - append one dated entry to this file
  - regenerate ledger with:
    - `python src/utils/update_experiments_result.py`
  - cross-link new logs and report files

---

## 2026-02-16

### What was done
- Refactored and re-executed analysis notebook:
  - `notebooks/stage1_stage2_experiment_analysis.ipynb`
  - `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
- Fixed checkpoint resolution bug in notebook (`save_dir` relative path now resolved from repo root).
- Updated visualization naming for presentation clarity:
  - `S0 Uniform Baseline`
  - `S1 Adaptive ROI (no distill)`
  - `S2 Adaptive ROI + Distill`
- Added explicit Cell 9 diagnostics:
  - threshold sweep curve
  - automatic message when `max_importance < fixed_threshold`
- Added Stage0 into quick-eval table/plots for side-by-side context.

### Why
- Existing notebook outputs were confusing (internal stage labels, empty cell outputs when checkpoints were not found).
- Needed a clear explanation of why `TP=0, FP=0, FN>0` persisted even after many epochs.

### Key observations
- Cell 8 now loads checkpoints correctly and selects recent maxpool runs.
- Cell 9 still reports empty ROI at threshold `0.50` for representative Stage2 run:
  - `TP=0, FP=0, FN=70`
  - importance stats: `min=0.0100, max=0.0307, mean=0.0127`
  - since `max_importance < 0.50`, empty prediction is expected at that threshold.
- Threshold sweep confirms collapse behavior:
  - at very low threshold (`0.01`), recall becomes high but FP explodes (`FP=51312`), IoU remains extremely low.
- Quick-eval summary indicates ROI emphasis is not working:
  - `imp_roi_gap_mean` is negative for Stage1/Stage2 adaptive runs (ROI importance lower than BG on average).

### Problems / risks
- Importance head output dynamic range is severely compressed near the floor (`~0.01-0.03`).
- Current adaptive runs are not producing ROI-discriminative importance maps.
- Training loss comparisons across stages remain non-comparable without context (different objective composition).

### Decisions
- Treat current result as a confirmed **importance-collapse failure mode** rather than a plotting artifact.
- Use updated notebook outputs for communication, but avoid over-claiming ROI-aware gains until importance separation is fixed.
- Next experiments should prioritize importance calibration/separation (targeting, weighting, and head supervision diagnostics) before new architecture complexity.

---

## 2026-02-24

### What was done
- Added KITTI 3D object download pipeline:
  - `src/dataset/download_kitti3dobject.py`
  - `src/scripts/download_kitti3dobject.sh`
  - `src/scripts/download_kitti3dobject_sbatch.sh`
- Started dataset download to:
  - `data/dataset/kitti3dobject`
  - resumed large archive (`data_object_image_2.zip.part`) in persistent `tmux` session `dl_kitti3dobject`.
- Added KITTI range-image training loader with ROI mask generation from KITTI labels/calibration:
  - `src/dataset/kitti_object_loader.py`
- Updated training entrypoint to support dataset switching:
  - `src/main_train.py`
  - new args: `--dataset_type`, `--kitti_split`, `--kitti_imageset_file`, `--kitti_roi_classes`
- Updated Stage scripts to support KITTI object training path (Stage0/1/2 + Stage2 distill-fix ablation):
  - `src/scripts/run_uniform_baseline.sh`
  - `src/scripts/run_stage1.sh`
  - `src/scripts/run_stage2.sh`
  - `src/scripts/run_stage2_distill_fix_ablation.sh`
- Added PointPillar KITTI finetune workflow:
  - `src/scripts/run_pointpillar_kitti_finetune.sh`
  - `src/scripts/run_pointpillar_kitti_finetune_sbatch.sh`
  - auto summary outputs:
    - `notebooks/pointpillar_finetune_kitti_summary.csv`
    - `docs/report/pointpillar_finetune_kitti.md`

### Why
- Track-B claims require detector-level protocol sanity first, then same-protocol reconstructed comparison.
- Existing Stage scripts were hard-coded to SemanticKITTI; ROI-aware compression study needed KITTI-compatible training/eval path.
- A reproducible PointPillar finetune record is needed for thesis/paper audit trail.

### Key observations
- KITTI eval subset in repo (`kitti_eval_subset`) is only 128 frames and has train/val/test overlap; not suitable for full protocol claims.
- First finetune submit (`sbatch job 22250`) failed immediately because `data/dataset/kitti3dobject` download/extraction was incomplete (`ImageSets` missing at runtime).
- KITTI loader smoke test on subset succeeded:
  - output tensors matched expected shapes `[5,64,1024]`, mask `[64,1024]`, roi `[1,64,1024]`.

### 2026-02-24 Late update

#### What was done
- Hardened KITTI finetune sbatch flow:
  - `src/scripts/run_pointpillar_kitti_finetune.sh` now supports dataset-ready wait (`WAIT_FOR_KITTI_SEC`, `WAIT_POLL_SEC`).
  - `src/scripts/run_pointpillar_kitti_finetune_sbatch.sh` exports wait defaults and prints them in metadata.
- Re-submitted PointPillar finetune as scheduled job:
  - `sbatch job 22253` (`BeginTime` start) with:
    - `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject`
    - `EPOCHS=40`, `BATCH_SIZE=4`, `WORKERS=4`, `REBUILD_KITTI_INFOS=1`, `RUN_FINAL_TEST=1`.
- Validated Stage1 KITTI path end-to-end with smoke run:
  - command used `DATASET_TYPE=kitti3dobject`, `DATA_ROOT=data/dataset/kitti_eval_subset`, `EPOCHS=1`, `MAX_TRAIN_FRAMES=2`.
  - fixed loader robustness bug for nonstandard point file layout:
    - `src/dataset/kitti_object_loader.py` now accepts both `Nx4` and `Nx3` float32 point bins.

#### Key observations
- The full KITTI object archive download remains in progress in `tmux` session `dl_kitti3dobject`.
- `run_stage1.sh` KITTI mode now executes a complete train epoch after loader fix (no crash on subset bins).
- First immediate finetune job (`22251`) was canceled to avoid idle GPU waiting; resubmitted as delayed start (`22253`).

### 2026-02-24 Night update

#### What was done
- Added teacher-design survey note to clarify distillation strategy boundaries:
  - `docs/report/teacher_distill_design_survey_20260224.md`

#### Key observations
- Direct PointPillar -> range-image pixel distillation remains modality-mismatched.
- Recommended immediate policy:
  - keep PointPillar for Track-B endpoint AP sanity/comparison,
  - use aligned teacher space (range-view teacher or assistant bridge) for feature distillation.

### Decisions
- Keep long-running KITTI object download active and resumable.
- Re-run PointPillar finetune sbatch after downloader finishes and layout validation passes.
- Keep Track-B reporting strict:
  - original non-zero AP sanity first,
  - then reconstructed-vs-original comparison under identical protocol.

### Sources
- `notebooks/stage1_stage2_experiment_analysis.ipynb`
- `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
- `notebooks/quick_eval_summary_16frames.csv`
- `notebooks/quick_eval_detail_16frames.csv`

---

## 2026-02-16 (Loss/Distill/Head Upgrade)

### What was done
- Implemented two new loss recipes in training:
  - `balanced_v1`: normalized rate + weighted BCE importance.
  - `balanced_v2`: bg-focused normalized rate + weighted BCE + ROI/BG separation margin.
- Improved distillation logit objective:
  - `distill_logit_loss=auto` now uses BCE for 1-channel logits and KL otherwise.
- Added richer importance head option:
  - `importance_head_type=multiscale` (dilated multi-branch fusion).
- Added teacher-signal plumbing:
  - `teacher_proxy_ckpt` is now configurable from CLI/scripts.
  - Stage2 pilot uses `data/results/checkpoints/stage2_adaptive.pth` instead of random proxy initialization.
- Added checkpoint pre-download workflow scripts:
  - `src/scripts/download_pointpillar_checkpoint.sh`
  - `src/scripts/prepare_teacher_and_submit_stage2_loss_recipe.sh`
  - `src/scripts/run_stage2_loss_recipe_ablation.sh`
  - `src/scripts/run_stage2_loss_recipe_pilot_local.sh`

### Why
- Previous setup strongly incentivized low importance due to raw `mean(level_map)` rate term.
- Distill logit path with 1-channel KL was weak/degenerate.
- Needed a stronger head option and reproducible pre-GPU checkpoint preparation step.

### New pilot runs executed (local quick verification)
- `260215_resnet_pilot_balanced_v1_headbasic_r0.out`
  - stage2, `loss_recipe=balanced_v1`, `rate_loss_mode=normalized_global`, `importance_head_type=basic`
  - final loss: `38.1165`
- `260215_resnet_pilot_balanced_v2_headmultiscale_r1.out`
  - stage2, `loss_recipe=balanced_v2`, `rate_loss_mode=normalized_bg`, `importance_head_type=multiscale`
  - final loss: `39.7356`

### Notes / caveats
- Pilot runs used subset training (`max_train_frames=128`, `epochs=2`) for fast validation, so they are not final thesis numbers.
- PointPillars public checkpoint URL availability changed; downloader now supports override (`POINTPILLAR_CKPT_URL`) and gdown fallback.

### Sources
- `src/train/trainer.py`
- `src/loss/distill_loss.py`
- `src/models/importance_head.py`
- `src/main_train.py`
- `src/scripts/download_pointpillar_checkpoint.sh`
- `src/scripts/run_stage2_loss_recipe_ablation.sh`
- `src/scripts/run_stage2_loss_recipe_pilot_local.sh`
- `src/results/experiments_result.md`

---

## 2026-02-16 (Zhulf PointPillars Teacher Connected + Re-run)

### What was done
- Connected teacher distillation to `zhulf0804/PointPillars` checkpoint:
  - downloaded `pretrained/epoch_160.pth` into `data/checkpoints/pointpillars_epoch_160.pth`
  - added teacher backend `pointpillars_zhulf` in `src/utils/teacher_adapter.py`
- Updated Stage2 scripts to default to the new backend/checkpoint:
  - `TEACHER_BACKEND=pointpillars_zhulf`
  - `TEACHER_PROXY_CKPT=data/checkpoints/pointpillars_epoch_160.pth`
- Re-ran Stage2 local pilot with real teacher signals (`epochs=5`, `max_train_frames=128`):
  - `260216_resnet_pilot_balanced_v1_headbasic_r0.out`
  - `260216_resnet_pilot_balanced_v2_headmultiscale_r1.out`
- Regenerated experiment ledger and analysis notebook outputs:
  - `src/results/experiments_result.csv`
  - `src/results/experiments_result.md`
  - `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
  - `notebooks/analysis_log_summary.csv`
  - `notebooks/analysis_epoch_curves.csv`
  - `notebooks/quick_eval_summary_16frames.csv`
  - `notebooks/quick_eval_detail_16frames.csv`

### Why
- Previous distill runs used proxy/randomized teacher behavior, which weakened interpretation of Stage2 gains.
- Needed a reproducible baseline where teacher path is tied to an actual public PointPillars checkpoint.

### Key results
- `260216_resnet_pilot_balanced_v1_headbasic_r0.out`
  - first loss: `58.8617` -> final/best: `12.7122` (`+78.40%` relative improvement)
- `260216_resnet_pilot_balanced_v2_headmultiscale_r1.out`
  - first loss: `58.4008` -> final/best: `12.5746` (`+78.47%` relative improvement)

### Notes
- Current quick-eval table in notebook is still a curated fixed comparison set, so newly added pilot runs are reflected in `analysis_log_summary.csv`/`analysis_epoch_curves.csv` but not automatically added to the fixed `quick_eval_summary_16frames.csv` comparison rows unless the run list is expanded.

### Sources
- `src/utils/teacher_adapter.py`
- `src/scripts/download_pointpillar_checkpoint.sh`
- `src/scripts/run_stage2_loss_recipe_pilot_local.sh`
- `src/scripts/run_stage2_loss_recipe_ablation.sh`
- `src/results/experiments_result.csv`
- `src/results/experiments_result.md`
- `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`

---

## 2026-02-16 (Objective/Teacher Default Fix + PP20 Importance Head)

### What was done
- Updated default Stage scripts to avoid legacy-collapse settings:
  - `src/scripts/run_stage1.sh`
  - `src/scripts/run_stage2.sh`
- Added high-capacity importance head:
  - new head type `pp_lite` in `src/models/importance_head.py`
  - minimum width guard (`hidden_channels >= 64`) to preserve PP20 target
  - CLI exposure in `src/main_train.py`
- Added reproducible local pilot script for updated path:
  - `src/scripts/run_stage1_stage2_pp_lite_pilot_local.sh`
- Executed new pilot runs (resnet, 3 epochs, 128 frames):
  - `260216_resnet_pilot_pp20_stage1.out`
  - `260216_resnet_pilot_pp20_stage2.out`
- Regenerated experiment ledger:
  - `src/results/experiments_result.md`
  - `src/results/experiments_result.csv`
- Re-executed analysis notebook to refresh log-level summaries:
  - `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
  - `notebooks/analysis_log_summary.csv`
  - `notebooks/analysis_epoch_curves.csv`

### Why
- Needed to harden default scripts so that new runs do not silently revert to `legacy + high rate pressure` settings.
- Needed a practical head capacity target (`>=20%` teacher-scale) for ROI/importance discrimination tests.

### Key observations
- Capacity check:
  - teacher (`ZhulfPointPillarsTeacherNet`): `4,834,824` params
  - `pp_lite(hidden=64)`: `1,020,609` params
  - ratio: `21.1%` (meets requested minimum).
- Pilot loss trends:
  - Stage1 (`balanced_v1`, no teacher): `59.0393 -> 26.9204`
  - Stage2 (`balanced_v2`, zhulf teacher): `58.5874 -> 26.3631`
- Stage2 is slightly better than Stage1 under matched pilot settings, but this is still short-run evidence (3 epochs / 128 frames).

### Problems / risks
- Runs were CPU-based local pilots (no GPU in current node), so throughput is low and convergence claims are limited.
- ROI detection/IoU behavior still needs dedicated evaluation pass (not inferred from total loss alone).

### Decisions
- Keep `pp_lite(64)` as default high-capacity candidate for next controlled ablation.
- Use updated `run_stage1.sh` / `run_stage2.sh` defaults for future batch submissions.
- Next step is GPU longer-run verification with ROI threshold sweep and quick-eval integration.

### Sources
- `src/models/importance_head.py`
- `src/main_train.py`
- `src/scripts/run_stage1.sh`
- `src/scripts/run_stage2.sh`
- `src/scripts/run_stage1_stage2_pp_lite_pilot_local.sh`
- `logs/260216_resnet_pilot_pp20_stage1.out`
- `logs/260216_resnet_pilot_pp20_stage2.out`
- `src/results/experiments_result.md`
- `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`

---

## 2026-02-17 (150-Epoch Sweep Completion)

### What was done
- Re-ran full Stage1 and Stage2 sweeps with `epochs=150` using the updated defaults.
- Confirmed all array runs completed.
- Regenerated ledger:
  - `src/results/experiments_result.md`
  - `src/results/experiments_result.csv`

### Run configuration
- Stage1:
  - `balanced_v1`, `rate_loss_mode=normalized_global`, `importance_loss_mode=weighted_bce`
  - `importance_head_type=pp_lite`, `importance_hidden_channels=64`
  - backbone/lr sweep:
    - darknet `1e-4`, `5e-5`
    - resnet `1e-4`, `5e-5`
- Stage2:
  - `balanced_v2`, `rate_loss_mode=normalized_bg`, `importance_loss_mode=weighted_bce`
  - `teacher_backend=pointpillars_zhulf`
  - `teacher_proxy_ckpt=data/checkpoints/pointpillars_epoch_160.pth`
  - `importance_head_type=pp_lite`, `importance_hidden_channels=64`
  - distill sweep:
    - `lambda_distill=0.1`, `0.5`, `1.0`

### Key results
- Stage1 final losses:
  - `260216_resnet_lr1e-4_j21424_r1.out`: `0.7224` (best Stage1)
  - `260216_resnet_lr5e-5_j21421_r3.out`: `0.7877`
  - `260216_darknet_lr1e-4_j21423_r0.out`: `1.6822`
  - `260216_darknet_lr5e-5_j21425_r2.out`: `1.7948`
- Stage2 final losses:
  - `260216_darknet_ld0.1_j21422_r2.out`: `2.1185` (best Stage2)
  - `260216_darknet_ld0.5_j21426_r0.out`: `2.7526`
  - `260216_darknet_ld1.0_j21427_r1.out`: `3.1818`

### Interpretation
- In this 150-epoch setting, Stage1 (resnet) converges to much lower objective values than Stage2 (darknet distill sweep).
- Stage2 trend remains consistent with earlier findings: lower distill weight (`0.1`) performs best; higher distill weights worsen final loss.
- This does not yet prove detector-level gain/loss; ROI/importance quality and detector metrics must be checked directly.

### Sources
- `logs/260216_resnet_lr1e-4_j21424_r1.out`
- `logs/260216_resnet_lr5e-5_j21421_r3.out`
- `logs/260216_darknet_lr1e-4_j21423_r0.out`
- `logs/260216_darknet_lr5e-5_j21425_r2.out`
- `logs/260216_darknet_ld0.1_j21422_r2.out`
- `logs/260216_darknet_ld0.5_j21426_r0.out`
- `logs/260216_darknet_ld1.0_j21427_r1.out`
- `src/results/experiments_result.md`

---

## 2026-02-17 (Notebook Re-run + TP/TN/FP/FN Diagnostic)

### What was done
- Re-executed analysis notebook:
  - `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
- Computed confusion metrics (`TP/TN/FP/FN`) on 16-frame subset for representative 150-epoch runs at threshold `0.50`.
- Per-run threshold sweep performed to find IoU-optimal threshold.

### Key observations
- Stage1 resnet runs:
  - near-perfect recall (`~0.999`) but large FP (`~100k`) -> low precision (`~0.34-0.35`) at `thr=0.50`.
  - IoU improves when threshold is raised (best around `thr~0.70`).
- Stage2 darknet distill sweep:
  - `ld=0.1`: high recall, still heavy FP (`precision~0.213`).
  - `ld=0.5`: FP reduced and precision improved (`~0.263`), recall drops (`~0.971`).
  - `ld=1.0`: complete collapse at `thr=0.50` (`TP=0, FP=0, FN>0`) because `max_importance < 0.5`.
    - threshold sweep shows recoverable behavior at lower threshold (`best_thr~0.44`).

### Problem diagnosis
- The dominant issue is **importance calibration instability across runs**:
  - fixed threshold `0.50` is not robust to scale shifts induced by different distill weights.
- A second issue is **high FP regime**:
  - many runs over-activate ROI (good recall, poor precision), meaning ROI maps are not selective enough.
- Therefore, current objective improvements in training loss do not directly imply usable ROI segmentation quality at deployment threshold.

### Sources
- `notebooks/stage1_stage2_experiment_analysis.executed.ipynb`
- `notebooks/analysis_log_summary.csv`
- `logs/260216_resnet_lr1e-4_j21424_r1.out`
- `logs/260216_darknet_ld0.1_j21422_r2.out`
- `logs/260216_darknet_ld0.5_j21426_r0.out`
- `logs/260216_darknet_ld1.0_j21427_r1.out`

---

## 2026-02-19 (Stage0 Fair-Budget + Oracle-ROI Decomposition Pilot)

### What was done
- Added explicit compression-budget metrics to training logs:
  - `rate_proxy`
  - `eq_bits` (`mean(log2(level_map))` for adaptive, `quant_bits` for uniform)
  - `code_entropy` (empirical Shannon entropy of latent codes)
- Added always-save final checkpoint behavior for short pilots:
  - `model_final.pth` is now saved even when `epochs < 5`.
- Added native vs oracle evaluation script:
  - `src/train/evaluate_oracle_roi.py`
  - evaluates two modes for adaptive runs:
    - `native`: model-predicted importance
    - `oracle_roi`: GT ROI mask directly drives quantization
- Added bitrate-matching helper:
  - `src/utils/match_bitrate_budget.py`
  - matches adaptive rows to nearest uniform baseline by chosen bitrate metric.
- Added one-command local pilot:
  - `src/scripts/run_stage0_stage1_stage2_oracle_pilot_local.sh`
  - runs:
    - Stage0 uniform q6
    - Stage0 uniform q8
    - Stage1 adaptive
    - Stage2 adaptive + distill
  - then runs oracle/native evaluation and updates result ledger.

### Why
- Direct `uniform vs adaptive` comparison is invalid when compression budgets differ.
- Needed objective evidence for:
  - whether adaptive gains come from true ROI selectivity,
  - whether current bottleneck is head/training vs quantization design.
- Oracle-ROI provides a controlled upper-bound diagnostic for the same checkpoint.

### Pilot setup
- `epochs=1`, `max_train_frames=32`, `val_seq=08`, `eval_frames=16`.
- Backbone: `resnet`.
- Adaptive settings:
  - `loss_recipe=balanced_v2`
  - `rate_loss_mode=normalized_bg`
  - `importance_head_type=pp_lite(64)`
  - Stage2 uses `teacher_backend=pointpillars_zhulf`.

### Key results
- Stage0 q6 native:
  - `bpp_entropy_mean=0.6668`, `roi_mse_mean=228.0958`
- Stage0 q8 native:
  - `bpp_entropy_mean=0.9598`, `roi_mse_mean=227.3080`
- Stage1 adaptive native:
  - `bpp_entropy_mean=0.8064`, `roi_mse_mean=227.8031`
- Stage1 adaptive oracle:
  - `bpp_entropy_mean=0.4177`, `roi_mse_mean=227.8031`
- Stage2 adaptive native:
  - `bpp_entropy_mean=0.8687`, `roi_mse_mean=227.6069`
- Stage2 adaptive oracle:
  - `bpp_entropy_mean=0.4492`, `roi_mse_mean=227.6073`

### Matched-budget quick comparison (by `bpp_entropy_mean`)
- Stage1 native matched to Stage0 q6:
  - ROI MSE gain vs uniform: `+0.128%` (small improvement)
  - All MSE gain vs uniform: `+0.123%`
- Stage2 native matched to Stage0 q8:
  - ROI MSE gain vs uniform: `-0.132%` (slightly worse)
  - All MSE gain vs uniform: `+0.104%`

### Interpretation
- This 1-epoch pilot is not for final performance claims.
- However, oracle/native gap in bitrate is large while MSE is nearly unchanged:
  - indicates current native importance map is not rate-efficient yet.
  - supports that bottleneck is still mainly importance-map quality/calibration (head+training), not only quantizer equation itself.

### Sources
- `src/train/trainer.py`
- `src/train/evaluate_oracle_roi.py`
- `src/utils/update_experiments_result.py`
- `src/utils/match_bitrate_budget.py`
- `src/scripts/run_stage0_stage1_stage2_oracle_pilot_local.sh`
- `logs/260219_resnet_pilot_s0_q6.out`
- `logs/260219_resnet_pilot_s0_q8.out`
- `logs/260219_resnet_pilot_s1_adapt.out`
- `logs/260219_resnet_pilot_s2_distill.out`
- `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q6.csv`
- `notebooks/oracle_eval_summary_260219_resnet_pilot_s0q8.csv`
- `notebooks/oracle_eval_summary_260219_resnet_pilot_s1adapt.csv`
- `notebooks/oracle_eval_summary_260219_resnet_pilot_s2distill.csv`
- `notebooks/matched_bitrate_pairs_260219_pilot.csv`

## 2026-02-24: Dual-Track Evaluation Recovery (SemanticKITTI ROI + KITTI 3D mAP)

### Problem
- We had a claim-boundary mismatch:
  - semantic point labels were used for ROI supervision and proxy diagnostics,
  - but final detector-level claim requires official 3D box AP endpoint.
- Mixing these in one table made interpretation weak.

### Decision
- Lock a dual-track protocol:
  - Track-A: SemanticKITTI codec/ROI diagnostics.
  - Track-B: KITTI + OpenPCDet official detector endpoint.
- Keep teacher policy as freeze-first with a quality gate:
  - if `ap3d_car_mod` on original clouds is below threshold, prepare fine-tune branch.

### Implementation Added
- `src/utils/recon_pointcloud_export.py`
  - standardized reconstruction export (`[range,intensity,x,y,z] -> XYZI .bin`) with fixed validity rule.
- `src/train/evaluate_kitti_map_vs_rate.py`
  - evaluates original vs reconstructed clouds under OpenPCDet and merges AP with bitrate metrics.
- `src/scripts/run_kitti_map_vs_rate.sh`
  - one-command wrapper for detector endpoint evaluation and pair matching.
- `src/utils/match_bitrate_budget_detector.py`
  - bitrate matching for detector rows with fairness tagging (`exact/nearest`, `fair/low-fairness`).
- `src/scripts/run_teacher_finetune_kitti.sh`
  - optional teacher fine-tune branch guarded by AP threshold.

### Documentation Update
- `docs/report/paper_fair_comparison_table.md`
  - now explicitly separates Table-A (SemanticKITTI) and Table-B (KITTI detector endpoint).
  - preserves native/oracle definitions and adds auto-update marker block for Table-B.

### Notes
- This round focused on infra + eval spec; no large retraining was included.
- OpenPCDet availability is now a strict runtime requirement for Track-B execution.

## 2026-02-24 (Update): Stage2 Distillation Fix Hypothesis Implementation

### Problem statement
- Stage2 weak learning was linked to two concrete factors:
  - teacher/task mismatch between detector objective space and semantic supervision space,
  - resize-only map alignment in distillation across heterogeneous feature resolutions.

### Implemented fix knobs
- Distill alignment:
  - `distill_align_mode`: `resize` (legacy) vs `adaptive_pool` (new recommended path).
  - `distill_align_hw`: pooled target grid (e.g., `16,32`).
- Distill feature source:
  - `channel_mean` (legacy),
  - `energy_map` (channel-agnostic spatial response),
  - `none` (feature-distill off ablation).
- Distill teacher quality gate:
  - `distill_teacher_score_min`
  - `distill_teacher_score_weight` (enable sample-wise gating)

### Runnable experiments
- Local pilot:
  - `src/scripts/run_stage2_distill_fix_pilot_local.sh`
- Slurm array:
  - `src/scripts/run_stage2_distill_fix_ablation.sh`
  - cases:
    - `legacy_mean_resize`
    - `energy_pool_16x32`
    - `energy_pool_16x32_scoregate015`

### Code paths updated
- `src/loss/distill_loss.py`
- `src/train/trainer.py`
- `src/main_train.py`
- `src/scripts/run_stage2.sh`
- `src/scripts/run_stage2_loss_recipe_ablation.sh`
- `src/scripts/run_stage2_loss_recipe_pilot_local.sh`

## 2026-02-24 (Update): Stage2 Distill-Fix Pilot Smoke Run

### What was run
- Script:
  - `src/scripts/run_stage2_distill_fix_pilot_local.sh`
- Runtime overrides:
  - `EPOCHS=1`
  - `MAX_TRAIN_FRAMES=8`
  - `BATCH_SIZE=1`
  - `NUM_WORKERS=0`
  - `BACKBONE=resnet`

### Why
- Verify end-to-end runnability of new distillation controls before large GPU sweeps.
- Confirm that all three ablation cases are executable with current defaults/checkpoints.

### Outcome
- Script completed all cases without runtime failure.
- Case logs:
  - `logs/260223_resnet_pilot_legacy_mean_resize_r0.out`
  - `logs/260223_resnet_pilot_energy_pool_16x32_r1.out`
  - `logs/260223_resnet_pilot_energy_pool_16x32_scoregate015_r2.out`
- Epoch-0 snapshots:
  - `legacy_mean_resize`: loss `81.9734`, `imp_mean=0.4957`
  - `energy_pool_16x32`: loss `80.1464`, `imp_mean=0.5423`
  - `energy_pool_16x32_scoregate015`: loss `80.8066`, `imp_mean=0.5655`

### Follow-up
- Move to Slurm full-length ablation (`epochs=150`) using:
  - `src/scripts/run_stage2_distill_fix_ablation.sh`
- After completion, regenerate ledger:
  - `python src/utils/update_experiments_result.py`

## 2026-02-24 (Update): KITTI 3D BBox Endpoint Run Completed

### What was run
- Slurm job:
  - `22123`
- Script:
  - `src/scripts/run_kitti_map_vs_rate.sh`
- Inputs:
  - KITTI-format subset root: `data/dataset/kitti_eval_subset`
  - detector cfg: `third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`
  - detector ckpt: `data/checkpoints/openpcdet_pointpillar_18M.pth`
  - compression runs:
    - `Uniform Baseline (ResNet)`
    - `Adaptive ROI Student (ResNet)`
    - `Adaptive Distilled Student (ResNet)`

### Output artifacts
- `notebooks/kitti_map_vs_rate_summary.csv`
- `notebooks/kitti_map_vs_rate_detail.csv`
- `notebooks/kitti_map_vs_rate_pairs.csv`
- `docs/report/paper_fair_comparison_table.md` (Table-B refreshed)

### Observed outcome
- Track-B path executed end-to-end without crash.
- AP3D values were all `0.0` for both `original` and `reconstructed` clouds in this run.
- Teacher quality gate warning was triggered:
  - `teacher_ap3d_mod_car=0.0 < 55.0`.

### Interpretation
- Infrastructure status: connected and reproducible.
- Scientific status: detector-endpoint claim is not yet valid.
- Immediate priority:
  - fix protocol/domain alignment for detector endpoint before using Stage2 distillation claims (canonical KITTI val protocol and/or detector fine-tuning).

## 2026-02-24 (Update): Multi-Epoch Sweep for Two Updated Stage2 Fixes

### Requested run scope
- Two updated experiments:
  - `energy_pool_16x32`
  - `energy_pool_16x32_scoregate015`
- Multi-parameter sweep:
  - `lr in {1e-4, 5e-5}`
  - `lambda_distill in {0.05, 0.1, 0.2}`
- Epochs:
  - `80`

### Execution
- Submitted array:
  - train job `22127` (12 tasks total)
- Auto post-processing:
  - job `22140` (`afterany:22127`)
  - updates ledger + emits dedicated summary files.

### Output paths (post job)
- `src/results/experiments_result.csv`
- `src/results/experiments_result.md`
- `notebooks/stage2_distill_fix_twoexp_summary_22127.csv`
- `docs/report/stage2_distill_fix_twoexp_summary_22127.md`

### Result snapshot
- Total runs: `12` (`2 cases x 2 lrs x 3 lambda_distill`)
- Best non-score-gate case:
  - `energy_pool_16x32`, `lr=1e-4`, `lambda_distill=0.05`
  - `best_loss=1.1291`, `final_imp_mean=0.1432`
- Best score-gate case:
  - `energy_pool_16x32_scoregate015`, `lr=1e-4`, `lambda_distill=0.2`
  - `best_loss=0.8168`, `final_imp_mean=0.1472`
- Interpretation:
  - score-gating improved convergence and importance activation stability in this sweep.
  - high distill weight without gating still hurts (`lambda_distill=0.2` was worst in non-gated branch).

## 2026-02-25/27 (Update): Stage0/Stage1 Detector Gap RCA on KITTI

### What was done
- Re-executed focused visualization with selected high-ROI KITTI frame:
  - `notebooks/stage0_stage1_kitti_pointpillar_visualization.executed.ipynb`
  - selected sample: `003464`, ROI ratio `0.0856`.
- Ran detector debug comparing four inputs on the same sample:
  - raw point cloud
  - identity projection/unprojection (no codec)
  - Stage0 reconstructed cloud
  - Stage1 reconstructed cloud
  - log: `logs/slurm_22383.out`
- Measured representation loss from projection alone on val split (`200` frames):
  - command path used `src/utils/recon_pointcloud_export.py::project_points_to_range_image`.

### Why
- There was a contradiction:
  - Stage0/Stage1 training losses kept decreasing,
  - but reconstructed detector behavior stayed weak.
- Needed to separate three hypotheses:
  - insufficient epochs,
  - importance-head capacity issue,
  - geometric information loss before/around codec path.

### Key observations
- Training status (not a crash/divergence):
  - Stage0 (`j22255_r5`): `21.3739 -> 1.5752` over `50` epochs.
  - Stage1 (`j22279_r1`): `21.7544 -> 1.7810` over `50` epochs.
  - both trained on `dataset_type=kitti3dobject`, `3712` train frames.
- Projection-only loss is large before codec learning:
  - mean point-retention after projection: `0.4169` (200 val frames).
  - mean raw points/frame: `119,502.2`
  - mean occupied range pixels/frame: `49,812.1`
  - collision factor (`raw/occupied`): `2.399`.
- One-sample detector comparison (`sample 003464`):
  - raw points: `123,592`, raw preds: `44`
  - identity points: `50,911`, identity preds: `27`
  - Stage0 recon points: `50,767`, preds: `1`
  - Stage1 recon points: `50,635`, preds: `0`
- Visualization endpoint confirmation:
  - GT boxes: `15`
  - raw prediction boxes (`score>=0.30`): `14`
  - Stage0/Stage1 recon both: `0` boxes on the selected frame.

### Diagnosis
- First-order bottleneck is representation loss in the current `raw -> fixed 64x1024 range -> points` path:
  - single-return per pixel on collisions causes heavy irreversible sparsification.
- Second-order bottleneck is objective mismatch:
  - model optimizes reconstruction/rate/importance losses, not detector AP directly.
- Epoch/head are currently lower-priority explanations:
  - Stage0 (no importance supervision) shows similar endpoint degradation trend.
  - Stage1 with `pp_lite(64)` still collapses detector endpoint on reconstruction in this check.

### Decisions
- Make `raw vs identity(project->unproject) vs reconstructed` mandatory in Track-B diagnostics to deconfound projection-loss vs codec-loss.
- Keep detector claim gate strict:
  - original KITTI AP must be non-zero under fixed official protocol before reconstructed comparisons are used as claims.
- Prioritize representation-preserving reconstruction experiments next (multi-return or improved point restoration policy), then rerun Stage0/Stage1/Stage2 endpoint comparison.

### Sources
- `docs/report/experiments.md`
- `logs/260224_resnet_uniform_q8_j22255_r5.out`
- `logs/260224_resnet_lr1e-4_j22279_r1.out`
- `logs/slurm_22383.out`
- `notebooks/stage0_stage1_kitti_pointpillar_visualization.executed.ipynb`
- `src/utils/recon_pointcloud_export.py`
