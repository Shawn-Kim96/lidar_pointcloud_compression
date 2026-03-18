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

## 2026-03-16: Track 2 codec root-cause cycle launched

### Last Experiments Result
- The repaired `RangeDet` raw/basic baseline remained the correct Track 2 reference:
  - `AP3D@0.3 = 0.5700`
  - `AP3D@0.5 = 0.4979`
  - `AP3D@0.7 = 0.2435`
- Therefore the Track 2 collapse is now attributed to the codec path, not to the raw detector baseline.
- A new multi-frame Track 2 artifact analysis notebook and summary table were added:
  - `notebooks/track2_codec_root_cause.ipynb`
  - `notebooks/track2_codec_root_cause.executed.ipynb`
  - `logs/track2_codec_root_cause_summary.csv`
  - `logs/track2_codec_root_cause_figs/`
- The current artifact summary already shows a strong RI degradation pattern:
  - `valid_mask_iou` drops from `1.0` in raw/basic to about `0.78-0.81` in Stage0/Stage1
  - `row_profile_mae` increases strongly from `0.0` in raw/basic to about `2.45-3.40`
  - the dominant failure tag is currently mostly `banding`

### Result Discussion & Problem Setting
- The present Track 2 question is no longer “does RangeDet work on KITTI RI?”
- That question is now answered: yes, the repaired raw/basic baseline is usable.
- The current question is “which RI artifact created by the codec is destroying detection?”
- Based on the current artifact summary and qualitative notebook overlays, the leading failure modes are:
  - occupancy / valid-mask corruption
  - row-wise banding / decoder artifact
  - detector-agnostic training objective
- For Track 2, adding extra 3D side information is not the first move.
- Since `raw PC -> RI -> RangeDet` already works, the first recovery path should remain RI-native.

### Next Steps
- Implemented four no-quant Track 2 pilot branches:
  - `Pilot A`: mask-aware codec
  - `Pilot B`: anti-banding skip decoder
  - `Pilot C`: detector-aware auxiliary target using repaired raw/basic RangeDet prediction maps
  - `Pilot D`: combined skip decoder + mask-aware + anti-banding losses
- Exported detector-aware teacher maps from the repaired raw/basic archive:
  - `data/dataset/rangedet_teacher_targets_raw24_260315/`
- Submitted the pilot chain:
  - `26265` `t2p_mask_train`
  - `26266` `t2p_mask_eval`
  - `26267` `t2p_band_train`
  - `26268` `t2p_band_eval`
  - `26269` `t2p_det_train`
  - `26270` `t2p_det_eval`
  - `26271` `t2p_combo_train`
  - `26272` `t2p_combo_eval`
  - `26273` `t2p_compare`
- Current execution policy:
  - `Pilot A` and `Pilot B` run first
  - `Pilot C` waits on `Pilot A` eval
  - `Pilot D` waits on `Pilot B` eval
  - final compare waits on all four eval jobs

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


## 2026-03-02: Redesigned Experiment Status

### Summary
- The earlier Track-2 endpoint results show a clear mismatch: PointPillars remains valid on raw KITTI point clouds, but the detector collapses on reconstructed point clouds.
- The dominant confounder is currently `projection loss + detector domain shift`, not a proven failure of the encoder/decoder alone.
- The originally assumed "Track 2" run did not actually execute a native range-view detector. It only re-ran the PointPillars-based endpoint pipeline.

### Problem Definition
Two separate questions need to be isolated instead of being mixed together:

1. `SemanticKITTI ROI sanity check`
   - Why are the observed ROI regions so small?
   - Is the dataset parsing correct?
   - Are the ROI labels and GT association being built correctly?

2. `KITTI detector endpoint failure`
   - PointPillars works on raw 3D point clouds.
   - Does it fail only after the pipeline `point cloud -> range image -> reconstructed point cloud`?
   - If so, is the main issue projection loss, detector domain shift, or additional codec distortion?

### Experimental Split
The redesigned work is now separated into two tracks.

1. `Track 1: Domain-adapted PointPillars`
   - Test whether PointPillars can still detect objects after a pure `point cloud -> range image -> point cloud` identity round-trip.
   - If the identity round-trip is still usable, test `point cloud -> range image -> encoder -> decoder -> range image -> point cloud`.
   - Then test the quantized path `point cloud -> range image -> encoder -> quantize -> decoder -> range image -> point cloud`.
   - Fine-tune the detector on the identity-domain data before judging the compression models.

2. `Track 2: Native range-view detection`
   - Replace the point-cloud detector with a detector that reads the range image directly.
   - The immediate goal is not a full production detector, but a controlled pilot that can answer whether native range-view inference is more stable under reconstruction artifacts.

### Results So Far
`1-1. Identity baseline without learned compression`

- Even without learned compression, the identity path `raw point cloud -> range image -> point cloud` loses many points.
- In the earlier check, the point count dropped from roughly `120k` to `50k` points.
- The number of predicted detection boxes also dropped from `44` to `27`.
- This means the `64x1024` projection itself introduces a large representation change before any learned codec is involved.
- Therefore, projection loss must be treated as a primary confounder.

`1-2. Learned encoder/decoder and quantized reconstruction`

- With the current detector protocol, the reconstructed endpoint produced `mAP = 0`.
- The strongest current explanation is `detector domain shift`, but this is still a hypothesis under active test, not a final conclusion.
- PointPillars was trained on dense raw point clouds.
- The reconstructed clouds have a grid-like sampling pattern and a different density distribution.
- From the detector's perspective, this is out-of-domain input, even when the scene content is still partially preserved.

`Is the bitrate too aggressive?`

- Current runs are around `1.06 bpp`.
- This is not obviously outside the operating range commonly reported in LiDAR compression papers, but direct rate comparisons are only fair when the same rate definition, sensor setup, and projection protocol are matched.
- The current backbone is lighter than many recent published models, but the available evidence does not support the claim that the encoder/decoder is failing to learn reconstruction at all.
- At this stage, the data support a stronger claim about `representation mismatch` than about `backbone insufficiency`.

### Updated Interpretation
- It is too strong to say "this is not a codec problem." The more defensible statement is that `identity projection loss already causes major damage`, so the codec must be evaluated only after that baseline is separated.
- It is also too strong to say "the true reason is domain shift." The more defensible statement is that `domain shift is currently the leading hypothesis`, and Track 1 is designed to test that directly.
- Track 2 was not previously executed. The jobs that looked like Track 2 were still PointPillars-based evaluation jobs.

### New Experiment Plan
`Track 1 (running now)`

- Export a `KITTI_Identity` dataset using the projection-unprojection path only.
- Fine-tune OpenPCDet PointPillars on `KITTI_Identity`, not on raw KITTI, and use that detector as the new reference baseline.
- Two fine-tune jobs are running now:
  - `23308`: `10` epochs
  - `23310`: `20` epochs
- Two dependent evaluation jobs are already queued:
  - `23309` depends on `23308`
  - `23311` depends on `23310`
- In Slurm, `afterok` means the child job starts only if the parent job finishes successfully with exit code `0`.

`Track 1 follow-up decision rule`

- If the `KITTI_Identity` detector recovers to a meaningful non-zero AP, then detector domain shift is confirmed as a major bottleneck.
- If the identity detector recovers but compressed reconstruction remains near zero, then codec distortion is still too destructive after the domain shift is controlled.
- If the identity detector also fails, the next controlled test is to increase projection width from `64x1024` to `64x2048`.
- The `64x2048` experiment is a proposed next step, not a completed run.

`Track 2 (implemented today as a controlled pilot)`

- Added a minimal native range-view ROI detector pilot on top of the existing compression backbone.
- This pilot is intentionally simpler than a full LaserNet-style 3D box detector. The purpose is to test whether native range-view objectness survives reconstruction better than the point-cloud detector endpoint.
- Two detector heads are implemented:
  - `linear`: a minimal dense `1x1` objectness head on the latent range-image features
  - `refine`: a stronger convolutional refinement head before full-resolution prediction
- Training protocol:
  - Train the head on raw KITTI range images with ROI supervision.
  - Evaluate on raw validation range images.
  - Evaluate again on compression-reconstructed range images from the same frozen backbone and compare the gap.
- This gives a controlled `raw vs compressed` native range-view comparison before investing in a full box-regression detector.
- The first pilot completed successfully:
  - `linear` head: best `compressed_iou = 0.1433`, best `compressed_f1 = 0.2506`
  - `refine` head: best `compressed_iou = 0.3775`, best `compressed_f1 = 0.5480`
- Since `refine` is materially better than `linear`, the next longer sweep is focused on `refine` and a deeper `deep` head rather than on the weak linear baseline.

`Track 2 external checkpoints`

- Added a checkpoint download script for official PRBonn LiDAR-Bonnetal range-view backbones.
- Two official pretrained packages are now the first external checkpoint candidates:
  - `darknet53`
  - `darknet53-1024`
- These are useful as future imported range-view backbones, but they are not yet wired into the pilot training loop.
- For the overnight run, the immediate controlled comparison is the two in-repo Track 2 pilot heads (`linear` and `refine`) on top of the existing compression model.
- The longer Track 2 sweep is now queued with full train/val splits:
  - `23331`: `refine`, `hidden=128`, `epochs=120`
  - `23332`: `refine`, `hidden=192`, `epochs=160`
  - `23333`: `deep`, `hidden=128`, `epochs=120`
  - `23334`: `deep`, `hidden=192`, `epochs=180`
- A broader overnight Track 2 grid is also queued for coverage rather than manual monitoring:
  - `24` jobs total
  - Heads: `refine`, `deep`
  - Hidden channels: `96`, `128`, `192`
  - Learning rates: `1e-4`, `2e-4`
  - Epochs: `120`, `180`
- Submission and result-tracking artifacts for this grid:
  - Submission plan: `docs/report/260302_track2grid_submission_plan.md`
  - Manifest (job -> config -> output path): `logs/260302_track2grid_manifest.csv`
  - Rolling result table: `logs/260302_track2grid_results.md`
  - Machine-readable result CSV: `logs/260302_track2grid_results.csv`
- The result collector can be re-run at any time to refresh the table after jobs finish, so the overnight sweep remains auditable without manual log inspection.
- Some of the broader grid jobs failed immediately on heterogeneous GPUs (`CUDA OOM` or `device unavailable`), so the failed subset was re-queued with `batch_size=1` as a safety retry:
  - Retry plan: `docs/report/260302_track2grid_retrybs1_plan.md`
  - Retry manifest: `logs/260302_track2grid_retrybs1_manifest.csv`
  - Retry rolling result table: `logs/260302_track2grid_retrybs1_results.md`
  - Retry result CSV: `logs/260302_track2grid_retrybs1_results.csv`

### 2026-03-03 Morning Check

`Track 1`

- Both PointPillars fine-tune jobs (`10` and `20` epochs) completed their training loops and wrote checkpoints successfully.
- However, the training jobs still exited with failure because the automatic post-train evaluation inside OpenPCDet could not reload the saved checkpoints under the current PyTorch version.
- Root cause:
  - PyTorch `2.6+` changed the default `torch.load(..., weights_only=True)` behavior.
  - The OpenPCDet checkpoint loader was still relying on the older default.
- Fix applied:
  - Patched the local OpenPCDet checkpoint loader to call `torch.load(..., weights_only=False)` in a backward-compatible wrapper.
- Recovery action:
  - Re-submitted the Track 1 evaluation stage directly from the already-saved checkpoints, without re-running fine-tuning.
  - New recovery jobs:
    - `23400`: `e20` evaluation
    - `23409`: `e10` evaluation (H100 excluded)
- Additional hardware-specific issue:
  - One first recovery evaluation attempt landed on an H100 node and failed in the OpenPCDet `iou3d_nms` CUDA extension with `no kernel image is available for execution on the device`.
  - This indicates the locally built OpenPCDet CUDA ops are not compiled for `sm90`.
  - Operationally, Track 1 evaluation should avoid H100 nodes unless the extension is rebuilt with H100 support.

`Track 2`

- The broad overnight grid did not fail for a single reason; two separate resource issues appeared:
  - Immediate failures on some nodes: `CUDA OOM` or `device unavailable`
  - Long runtime risk: the successful batch-2 full-split jobs only reached roughly `48-55` epochs after about `10` hours
- Interpretation:
  - At the observed throughput, `120`-epoch jobs are borderline under the old `24h` request.
  - `180`-epoch jobs are unlikely to finish cleanly under the old `24h` request.
- Important scheduler note:
  - `gpuqm` and `gpuql` use the same GPU node pool on this cluster.
  - Moving from `gpuqm` to `gpuql` is therefore not a hardware upgrade; it only gives a longer queue class / longer allowed walltime.
- Fixes applied:
  - Kept the original broad grid running, since it already consumed substantial compute.
  - Re-queued the immediate-failure subset with `batch_size=1` as a safety retry.
  - Added per-epoch artifact writing to the Track 2 trainer so future long jobs keep a rolling `metrics.csv`, `summary.json`, and best checkpoint during training instead of only at the end.
- Next controlled sweep now queued:
  - Launched an H100-only confirmatory sweep on `gpuql` with `48:00:00` walltime to remove the mixed-GPU confounder and the too-short walltime.
  - Node target: `g16,g18,g19`
  - Manifest: `logs/260303_track2h100_manifest.csv`
  - Result table: `logs/260303_track2h100_results.md`
  - Result CSV: `logs/260303_track2h100_results.csv`
  - Submission plan: `docs/report/260303_track2h100_plan.md`

### 2026-03-03 Midday Correction

`Track 2 stop decision`

- The in-repo Track 2 ROI pilot (`track2_refine`, `track2_deep`) was stopped intentionally.
- Reason: the current Track 2 jobs were only training a lightweight ROI/objectness head on top of the frozen in-repo compression backbone.
- This is useful only as a proxy sanity check, but it is not the intended publishable endpoint.
- Specifically, it is not a well-known external range-view 3D detector with a public detector checkpoint.
- All currently running `track2_*` jobs were cancelled so that no more GPU time is spent on this proxy setup.

`Track 1 geometry correction`

- The earlier Track 1 implementation did not actually run the requested `64x2048` geometry change.
- The original submitted jobs still used the default `64x1024` projection.
- This has now been corrected in code:
  - `run_track1_pipeline.sh` now defaults to `64x2048`
  - the identity export path now records geometry in the dataset path name
  - a new `unprojection_mode` flag is threaded through the Track 1 export and evaluation path
- New default for future Track 1 geometry experiments:
  - projection: `64x2048`
  - unprojection: `ray`

`New unprojection ablation`

- Added two explicit unprojection modes for Track 1:
  - `decoded_xyz`: trust reconstructed `x,y,z` channels directly (old behavior)
  - `ray`: reconstruct `x,y,z` from the reconstructed range channel and the fixed per-pixel ray geometry
- The `ray` mode is the more defensible geometry-preserving option for the next Track 1 ablation because it enforces angular consistency instead of trusting noisy decoded Cartesian coordinates.

`External baselines pulled locally`

- `RENO` (CVPR 2025) was pulled into `third_party/external_codecs/RENO`.
- The repository includes public pretrained weights, including `model/KITTIDetection/ckpt.pt`.
- This is a credible stronger reconstruction baseline for Track 1, but it is not yet wired into the in-repo evaluation path.
- Immediate blocker: the current environment does not have `torchsparse`, `open3d`, or `torchac`, which RENO requires.

- Two external range-view detector repos were also pulled for Track 2 review:
  - `third_party/external_range_det/RangeDet` (ICCV 2021, official code)
  - `third_party/external_range_det/range-view-3d-detection` (CoRL 2024, open-source code)
- Current status:
  - `RangeDet` is a real external range-view detector and includes KITTI range-image tooling, but no public pretrained detector checkpoint is bundled in the official repo.
  - `range-view-3d-detection` is a newer open-source detector codebase, but the public repo targets Argoverse 2 / Waymo, not KITTI.
- Therefore, the next Track 2 step should be a deliberate external-detector integration, not another in-repo toy head sweep.

### 2026-03-03 Late Night Track Reset

`Track 1 execution`

- Submitted two corrected high-resolution Track 1 chains, both using `64x2048` range images and `ray` unprojection.
- Both jobs are long sequential chains that run on one GPU each:
  1. Stage0 uniform training
  2. Stage1 adaptive training
  3. `KITTI_Identity` export for the same geometry
  4. PointPillars fine-tuning on the identity-domain point cloud
  5. Endpoint evaluation on the Stage0 and Stage1 reconstructed point clouds
- Submission manifest:
  - `logs/260303_225625_track1_hires_manifest.csv`
- Running jobs:
  - `23537`: `track1a_hires` (baseline codec, no multiscale latent fusion)
  - `23538`: `track1b_hires` (enhanced codec with multiscale latent fusion + decoder refinement)

`Track 1B architecture change`

- The enhanced codec is no longer a last-stage-only latent bottleneck.
- Added an optional multiscale latent fusion block before quantization.
- The fusion reuses the existing Stage3 neck variants and is now wired directly into the compression model.
- The current enhanced Track 1B run uses:
  - feature fusion variant: `rangeformer`
  - fusion hidden channels: `160`
  - decoder post-refine residual blocks: `3`
- This makes Track 1B a real multiscale encoder/decoder ablation rather than a shallow architectural duplicate.

`Track 2 status`

- No new Track 2 jobs were submitted.
- Reason: `RangeDet` is still blocked at the runtime/environment layer.
- The official repository is present locally, but the current environment does not have `mxnet`, and the codebase also depends on custom compiled operators.
- Therefore, submitting a fake Track 2 job now would not be scientifically or operationally valid.
- The correct next step for Track 2 is a dedicated environment/integration pass for `RangeDet`, not another placeholder launch.

### 2026-03-03 RangeDet Integration Push

`Dedicated RangeDet environment`

- Built a separate `RangeDet` environment at:
  - `/home/018219422/miniconda3/envs/rangedet39`
- Installed a CUDA-compatible MXNet stack for this environment:
  - `mxnet-cu112==1.9.1`
  - `numpy==1.23.5`
  - `cmake`, `ninja`, `pybind11`, `eigen`, `openblas-devel`
  - Python dependencies required by the official code path
  - NVIDIA runtime libraries inside the env (`cudnn`, `cublas`, `cuda_nvrtc`)
- Important runtime note:
  - `mxnet-cu112` does not import on the login node because `libcuda.so.1` is absent there.
  - This is expected on the login node and is not evidence of failure on a real GPU node.

`Official RangeDet bring-up status`

- The official `RangeDet` repository remains at:
  - `third_party/external_range_det/RangeDet`
- Added compatibility patches so the original code can run in the current environment:
  - `horovod.mxnet` import is now optional
  - explicit `ctypes.CDLL('./operator_cxx/contrib/contrib_cxx.so')` loads are now optional in `tools/train.py` and `tools/test.py`
  - modernized CMake / Python / Eigen handling for the C++ extension build
- The main C++ extension now compiles successfully:
  - `processing_cxx.cpython-39-x86_64-linux-gnu.so`

`Custom operator diagnosis`

- The legacy `contrib_cxx.so` custom CUDA operator still does not build cleanly on the current toolchain.
- Repeated failures occur in `decode_3d_bbox.cu` with a large cascade of:
  - `this declaration may not have extern "C" linkage`
- This persisted after:
  - supplying MXNet source headers
  - fixing CUDA / BLAS / Eigen includes
  - trying multiple host compiler setups
  - removing the legacy `cuda_utils.h` include path from the operator source
- Therefore, `contrib_cxx.so` is not the current forward path.

`Practical workaround`

- Replaced the `Decode3DBbox` dependency inside the model graph with a symbolic MXNet implementation in:
  - `rangedet/symbol/head/builder.py`
- This keeps the official detector path intact while avoiding the broken legacy custom decode operator.
- Weighted NMS still uses the compiled `processing_cxx` path, which is now available.

`KITTI dataset conversion for RangeDet`

- Patched the official KITTI conversion script so it writes the fields the actual training loader expects.
- The same converter now also accepts `--pointcloud-source-dir`, so future `stage0` / `stage1` reconstructed `.bin` trees can reuse the raw KITTI labels and calibration while swapping only the point clouds.
- Converted KITTI into a local RangeDet-formatted dataset at:
  - `data/dataset/rangedet_kitti_hq`
- Confirmed training and validation split records exist:
  - `data/dataset/rangedet_kitti_hq/training/part-0000.roidb`
  - `data/dataset/rangedet_kitti_hq/validation/part-0000.roidb`
- Current counts:
  - `npz_trainval`: `7481`
  - `npz_test`: `607` so far while the test-split conversion is still in progress

`Track 2 submission status`

- Submitted a real `RangeDet` smoke job and chained the first full raw-KITTI training behind it:
  - `23542`: `rangedet_smoke`
    - dependency: `afterany:23537`
    - config: `config/rangedet/rangedet_kitti_car_24e.py`
    - `1` epoch, `sampling_rate=64`
    - runs training plus test
  - `23543`: `rangedet_raw24`
    - dependency: `afterok:23542`
    - same config
    - `24` epochs, `sampling_rate=1`
    - runs training plus test
- Both are queued on `gpuql`, so they will start only after the current Track 1 usage drops.
- Job-chain manifest:
  - `logs/260303_rangedet_kitti_chain.csv`

`Interpretation`

- Track 2 is no longer blocked by "MXNet is missing" as a passive blocker.
- The environment, build fixes, dataset conversion, and first real submission chain are now in place.
- The remaining open question is not setup in principle; it is whether the first GPU smoke run executes cleanly on the hardware once the dependency clears.

### 2026-03-04 RangeDet Smoke Debug Iteration

`23542 root cause and follow-up`

- The original first smoke job `23542` did run on a real A100 node, but it failed immediately during `mxnet` import.
- The failure chain was resolved step-by-step:
  1. missing `libnccl.so.2`
  2. missing `libcudart.so.11.0` and other CUDA runtime libraries (`cufft`, `cusolver`, `curand`, `nvtx`)
  3. missing Python dependencies (`requests`, `numba`, `pytz`)
  4. `PYTHONPATH` mismatch for `from utils import ...`
  5. missing `RotatedIOU` in this MXNet build
  6. pickle compatibility issue (`numpy._core` vs NumPy 1.23)
  7. deprecated NumPy API (`np.asscalar`)
- These were handled in code and environment rather than treated as stopping blockers.

`RangeDet compatibility fixes now in place`

- Runtime scripts now export all NVIDIA wheel-provided library paths from the dedicated `rangedet39` env.
- RangeDet runners now use `python -u` for unbuffered logs.
- `RotatedIOU` is no longer required for training on this build:
  - when unavailable, the training graph falls back to a binary positive-mask target for the classification branch
  - this preserves executability while keeping the detector structure intact
- The symbolic `Decode3DBbox` replacement was further patched to avoid `mx.sym.np.arctan2` so it works with legacy MXNet symbols during inference.
- `tools/train.py` and `tools/test.py` now alias `numpy._core` to support the existing pickled roidb files.
- `tools/test.py` now uses `np.asarray(...).item()` instead of removed `np.asscalar`.
- `tools/test.py` also marks the loader thread as daemon, so eval failures do not leave hanging jobs holding a node.

`Current smoke status`

- A full smoke run (`23585`) already proved that:
  - training runs successfully on the GPU node
  - the model trains for `1` epoch
  - a checkpoint is saved
- That run then failed during the eval phase on the deprecated `np.asscalar` API.
- After patching that issue, the training half no longer needs to be re-proven.

`Current active Track 2 chain`

- `23587`: `rangedet_eval_smk`
  - eval-only job
  - reuses the saved checkpoint from `experiments/rangedet_kitti_smoke_1e_260304_0100/checkpoint-0001.params`
  - purpose: validate the patched `test.py` path only
- `23589`: `rangedet_raw24`
  - queued with `afterok:23587`
  - if the eval-only smoke passes, the first full `24`-epoch raw-KITTI RangeDet training run starts automatically

`Interpretation`

- Track 2 has crossed the key threshold from environment triage into actual detector execution:
  - training is already confirmed to run
  - the remaining active gating step is the evaluation path
- If `23587` completes, then the first real raw-KITTI RangeDet baseline (`23589`) proceeds automatically.

### 2026-03-05 Geometry Bottleneck Confirmation and New Branches

## Last Experiments Result

- `Track 1: high-resolution endpoint diagnosis`
- We re-ran the endpoint at `64x2048` with `ray` unprojection and identity-domain PointPillars fine-tuning.
- The detector adaptation itself succeeded on the exported `KITTI_Identity` domain:
  - `pp_ft_track1nq_a_baseline_260304_230724`: `mAP3D(mod)=73.5935`
  - `pp_ft_track1nq_b_enhanced_260304_230724`: `mAP3D(mod)=73.5501`
- However, the reconstructed endpoint still collapsed.
- `No-quant` result:
  - baseline (`track1nq_a`): identity reference `mAP3D(mod)=51.5667`, reconstructed `0.0002507`
  - enhanced (`track1nq_b`): identity reference `54.3868`, reconstructed `0.0000`
- `Uniform / oracle / bg-level` diagnostic result at the reconstructed endpoint:
  - uniform-like native quant: baseline `0.000`, enhanced `0.001`
  - oracle ROI map: baseline `0.000`, enhanced `0.001`
  - background-level sweep (`24/32/48/64`): still `0.000 ~ 0.001`
- This means the endpoint failure persists even after:
  - increasing projection width to `64x2048`
  - changing unprojection to `ray`
  - adapting the detector to the identity-domain point cloud
  - removing quantization entirely
  - giving the codec an oracle ROI allocation

- `Track 2: official RangeDet execution`
- The official raw KITTI RangeDet training had already completed successfully with the `24e` KITTI car config.
- We then re-ran evaluation so that raw and reconstructed outputs are archived separately instead of overwriting the same `output_dict` file.
- Completed and archived:
  - `24095`: raw RangeDet eval -> `logs/rangedet_eval_outputs/260305_215136_raw_output_dict_24e.pkl`
  - `24096`: no-quant reconstructed eval (baseline codec) -> `.../260305_215136_nqa_output_dict_24e.pkl`
  - `24097`: no-quant reconstructed eval (enhanced codec) -> `.../260305_215136_nqb_output_dict_24e.pkl`
- Completed afterward:
  - `24098`: uniform reconstructed eval (baseline codec) -> `.../260305_215136_uqa_output_dict_24e.pkl`
  - `24099`: uniform reconstructed eval (enhanced codec) -> `.../260305_215136_uqb_output_dict_24e.pkl`
- We also added a reproducible archive evaluator:
  - `src/scripts/eval_rangedet_archive_car_ap.py`
  - summary CSV: `logs/260306_rangedet_archive_car_ap_summary.csv`
- Under this current car-only lidar-space evaluator, the comparison is:
  - raw: `AP3D@0.3=0.0464`, `APBEV@0.3=0.1744`, `meanBestIoU3D=0.1915`
  - no-quant baseline: `AP3D@0.3=0.0001`, `APBEV@0.3=0.0085`, `meanBestIoU3D=0.0390`
  - no-quant enhanced: `AP3D@0.3=0.0003`, `APBEV@0.3=0.0061`, `meanBestIoU3D=0.0329`
  - uniform baseline: `AP3D@0.3=0.0001`, `APBEV@0.3=0.0032`, `meanBestIoU3D=0.0305`
  - uniform enhanced: `AP3D@0.3=0.0000`, `APBEV@0.3=0.0035`, `meanBestIoU3D=0.0234`
- At stricter thresholds (`0.5`, `0.7`), all reconstructed Track 2 runs are effectively zero, and even the current raw RangeDet baseline is already near-zero at `AP3D@0.5` and `AP3D@0.7`.

- `Track 1: new no-quant geometry-aware training now running`
- Two stronger no-quant codec runs are active:
  - `24093`: `t1nq_geo_rf`
  - `24094`: `t1nq_geo_fr`
- Both use:
  - wider/deeper encoder (`latent=128`, `base=96`, `blocks_per_stage=2`)
  - `masked_channel_weighted` reconstruction emphasizing `xyz`
  - decoder refinement
  - multiscale latent fusion
- Early training behavior is healthy:
  - `24093`: loss has dropped from the high teens to about `8.66` by epoch `28`
  - `24094`: loss has dropped from `51.32` to about `9.52` by epoch `8`

- `Track 1: new implicit / position-aware branch now running`
- We added a new `RENO/implicit-inspired` branch to the in-repo codec.
- This is not a full external-paper reproduction; it is a targeted geometry-aware implementation for our current bottleneck.
- New components:
  - `position_branch`: separate `xyz` side encoder
  - `coord_conditioned` decoder: sensor ray and pixel coordinate conditioning
  - `ray_consistency` loss: predicted `range` and predicted `xyz` must agree geometrically
- Active runs:
  - `24103`: `t1nq_impl_a`
  - `24104`: `t1nq_impl_b`
- Early training behavior is also healthy:
  - `24103`: loss `274.23 -> 31.08 -> 13.09 -> 9.87 -> 8.67`
  - `24104`: loss `291.22 -> 40.47 -> 18.46` in the first epochs

## Result Discussion & Problem Setting

- The main Track 1 result is now much clearer than before:
  - the reconstructed point-cloud endpoint does not fail mainly because of `stage1 adaptive quantization`
  - it already fails in the simpler `no-quant encoder/decoder` setting
- Therefore, the current leading explanation is:
  - the bottleneck is `geometry preservation in the RI -> latent -> RI path`
  - not merely `uniform vs adaptive bit allocation`
- More concretely:
  - a simple 2D RI autoencoder can reduce reconstruction loss while still destroying the geometric cues that a point-cloud detector uses
  - `PointPillars` needs the reconstructed cloud to preserve density, angular consistency, and object-support geometry well enough for voxelization and pillar statistics
  - our current baseline/enhanced codec families are still mostly `2D RI reconstruction` systems, even when they use multiscale 2D fusion
- The `no-quant` result is especially important:
  - if `no-quant` is already near zero AP, then stage1 quantization is not the first problem to solve
  - this justifies a temporary shift away from more `adaptive-vs-uniform` sweeps and toward stronger geometry-aware codec design

- There is still one protocol caveat:
  - the exported `KITTI_Identity` fine-tune summary reports `~73.55-73.59` mAP3D(mod)
  - the wrapper-side identity reference inside the no-quant endpoint CSV is lower (`~51.57-54.39`)
  - this means there is still an unresolved evaluation-protocol mismatch between the two identity baselines
  - the qualitative conclusion is still stable (`reconstructed endpoint collapses`), but this mismatch should be cleaned up before a final paper table is written

- For Track 2, the research framing is now more concrete:
  - if native range-view detection remains stable on reconstructed RI while point-cloud detection collapses on reconstructed PC,
  - then the right claim is not `generic point cloud preservation`
  - the right claim is `task-aware / range-view-aware LiDAR compression`
- This is still scientifically meaningful because the compressed signal still originates from the LiDAR scan.
- But the scope must be stated honestly:
  - Track 2 is a `native RI downstream` story
  - Track 1 is the stronger but harder `reconstructed point-cloud downstream` story
- However, the newly extracted Track 2 numbers add an important caution:
  - under our current KITTI conversion + raw `24e` RangeDet setup, the `raw` detector itself is not yet a strong baseline
  - therefore the current Track 2 table does show a large raw -> reconstructed degradation,
  - but it does **not** yet support a strong end claim about a competitive native RI detector pipeline
- In other words:
  - the engineering integration is solved
  - the current scientific baseline is still weak
  - Track 2 now needs either a stronger detector protocol or a corrected training/evaluation setup before it can become a publishable main result

## Next Steps

- `Track 1 immediate next step`
- Let the four active no-quant runs finish first:
  - `24093`, `24094` for stronger multiscale geometry-aware autoencoders
  - `24103`, `24104` for the new implicit / position-aware branch
- Then run the full endpoint again with the identity-domain PointPillars detector and compare:
  - old no-quant baseline
  - stronger no-quant multiscale branch
  - new implicit / position-aware branch
- Decision rule:
  - if the reconstructed endpoint is still near zero, then the next codec change should be more structural than just “wider 2D”
  - likely directions are a voxel/point hybrid side representation or a more lossless projection representation

- `Track 2 immediate next step`
- The raw / no-quant / uniform comparison table is now extracted.
- The next Track 2 task is no longer metric extraction; it is baseline repair:
  - verify whether the current KITTI conversion, calibration convention, or RangeDet training setup is suppressing raw performance
  - if the raw detector baseline cannot be made strong, Track 2 should not move on to more codec ablations yet
- Only after the raw RangeDet baseline is credible should we decide whether a detector-aware codec loss is needed for Track 2.

- `Documentation / paper framing next`
- Keep documenting Track 1 and Track 2 separately.
- The current storyline is now:
  - identity projection already damages the point-cloud endpoint
  - quantization is not the main reason for the current Track 1 collapse
  - stronger geometry-aware no-quant codec design is now the correct next experiment
  - Track 2 remains viable as a range-view-aware LiDAR compression direction, pending official metric extraction

## 2026-03-15: Pillar side-stream results and RangeDet raw decode audit

### Last Experiments Result

- `Track 1: pillar / BEV side-stream completed`
  - `25093` (`pillar_a`) finished with:
    - identity-domain reference `mAP3D(mod) = 54.7708`
    - reconstructed endpoint `mAP3D(mod) = 0.0185`
  - `25094` (`pillar_b`) finished with:
    - identity-domain reference `mAP3D(mod) = 52.7378`
    - reconstructed endpoint `mAP3D(mod) = 2.1922`
  - Interpretation:
    - explicit 3D side information **does help**
    - `pillar_b` is the first Track 1 run that materially lifts the reconstructed endpoint above the previous near-zero regime
    - but the endpoint is still far below the identity reference, so the codec is still not preserving enough geometry

- `Track 2: official raw/basic audit found a real decode bug`
  - The symbolic RangeDet fallback decode path in `rangedet/symbol/head/builder.py` was inconsistent with the original C++ `Decode3DBbox` operator.
  - This matters because the fallback path is used during training when `contrib_cxx.so` is not available, and our previous logs showed that fallback path was active.
  - We patched the symbolic decode to match the original `is_bin=False` semantics:
    - square-root coded `delta_x`, `delta_y`
    - `[log_width, log_length, cos_yaw, sin_yaw, z0, log_height]` ordering
    - direct `z0 -> z1` reconstruction
  - We also patched `tools/train.py` and `tools/test.py` to search for `contrib_cxx.so` via an absolute repo path when that library is available later.

- `Track 2: old raw/basic checkpoint is no longer trusted`
  - The old raw/basic RangeDet checkpoint was trained while the broken symbolic decode path was active.
  - Therefore the previous raw/basic numbers are not a valid final baseline.
  - Re-evaluating that old checkpoint is not enough; raw/basic must be retrained from scratch with the patched decode.

### Result Discussion & Problem Setting

- `Track 1`
  - The current evidence is now stronger than before:
    - the failure is not simply `uniform vs adaptive quantization`
    - the failure is also not fixed by adding a shallow 3D hint
  - However, the pillar/BEV side-stream result shows that the hypothesis about missing 3D structure was directionally correct.
  - The remaining bottleneck is likely the decoder:
    - current reconstructions still show strong stripe / banding artifacts
    - the previous decoder family relied mainly on transpose-conv upsampling
    - there was no real encoder-to-decoder skip path to preserve fine spatial geometry

- `Track 2`
  - The current Track 2 weakness is now split into two different issues:
    - raw/basic was compromised by a real RangeDet decode bug during training
    - reconstructed range images are still visibly distorted by the codec
  - So the correct order is:
    - first repair the raw/basic baseline with patched retraining
    - then compare Stage 0 / Stage 1 against that repaired detector

### Next Steps

- `Track 1 next run: stronger decoder on top of the pillar side-stream`
  - We implemented a new skip-connected decoder family:
    - bilinear upsampling instead of transpose-conv
    - explicit encoder skip features
    - optional coordinate-conditioned implicit prediction head
  - Two new main Track 1 runs are now submitted:
    - `25892`: `t1nq_pillar_skip_a`
    - `25894`: `t1nq_pillar_skip_b`
  - Goal:
    - keep the useful pillar/BEV side stream from `pillar_b`
    - directly attack the current stripe/banding failure mode in Stage 0

- `Track 2 next run: full raw/basic retraining with patched decode`
  - A new corrected raw/stage chain is submitted:
    - `25893`: patched raw/basic RangeDet retraining
    - `25895`-`25899`: dependent raw / Stage 0 / Stage 1 eval jobs
  - This new chain is the first one that combines:
    - correct `sampling_rate=1`
    - `2048` width fixes
    - patched symbolic decode path
  - Decision rule:
    - if raw/basic still collapses after this retrain, then a deeper KITTI conversion / box-convention audit is still required
    - if raw/basic recovers, then the remaining Track 2 gap can be attributed more cleanly to the codec

## 2026-03-16: Track 2 raw/basic repaired, codec bottleneck isolated

### Last Experiments Result

- `Track 2 corrected raw/basic chain completed`
  - Completed jobs:
    - `25893`: patched raw/basic retraining
    - `25895`: raw/basic eval
    - `25896`: Stage 0 baseline eval
    - `25897`: Stage 0 enhanced eval
    - `25898`: Stage 1 baseline eval
    - `25899`: Stage 1 enhanced eval
  - New archive summary:
    - `logs/260316_rangedet_archive_car_ap_summary_rddecodefixfull.csv`

- `Custom lidar-space AP summary`
  - `raw/basic`
    - `AP3D@0.3 = 0.5700`
    - `AP3D@0.5 = 0.4979`
    - `AP3D@0.7 = 0.2435`
    - `APBEV@0.3 = 0.5777`
    - `meanBestIoU3D = 0.6098`
  - `Stage 0 baseline`
    - `AP3D@0.3 = 0.0215`
    - `AP3D@0.5 = 0.0028`
    - `AP3D@0.7 = 0.0000`
    - `APBEV@0.3 = 0.0272`
    - `meanBestIoU3D = 0.1202`
  - `Stage 0 enhanced`
    - `AP3D@0.3 = 0.0099`
    - `AP3D@0.5 = 0.0010`
    - `AP3D@0.7 = 0.0000`
    - `APBEV@0.3 = 0.0134`
    - `meanBestIoU3D = 0.1040`
  - `Stage 1 baseline`
    - `AP3D@0.3 = 0.0083`
    - `AP3D@0.5 = 0.0007`
    - `AP3D@0.7 = 0.0000`
    - `APBEV@0.3 = 0.0107`
    - `meanBestIoU3D = 0.0874`
  - `Stage 1 enhanced`
    - `AP3D@0.3 = 0.0050`
    - `AP3D@0.5 = 0.0003`
    - `AP3D@0.7 = 0.0000`
    - `APBEV@0.3 = 0.0069`
    - `meanBestIoU3D = 0.0704`

### Result Discussion & Problem Setting

- `Track 2 raw/basic is now credible`
  - The patched raw/basic detector is no longer near zero.
  - This confirms that the previous raw/basic collapse was dominated by the RangeDet decode-path bug.
  - Therefore, Track 2 can now be interpreted as a real `raw/basic vs reconstructed RI` comparison.

- `The main Track 2 bottleneck is now the codec`
  - `Stage 0` already collapses relative to `raw/basic`.
  - `Stage 1` is even worse.
  - So the dominant Track 2 failure is not the detector baseline anymore.
  - The dominant failure is that the current encoder/decoder produces reconstructed range images that are too distorted for 3D detection.

- `Current enhanced codec is not helping Track 2`
  - On the repaired Track 2 detector baseline, `enhanced` is worse than `baseline`.
  - That means the previous “enhanced” changes do not improve detector-facing geometry preservation, even if they help reconstruction loss or internal feature richness.

- `Track 1 implication`
  - The new skip-decoder Track 1 runs are justified by this result.
  - The main hypothesis is now:
    - pillar/BEV side information was the correct direction
    - but the old decoder still destroyed fine geometry through transpose-conv upsampling and lack of explicit skip connections

### Next Steps

- `Track 1 running now`
  - `25892`: `t1nq_pillar_skip_a`
  - `25894`: `t1nq_pillar_skip_b`
  - These runs keep the pillar/BEV side stream and replace the decoder with a skip-connected bilinear upsampling variant.

- `Track 2 follow-up now submitted`
  - New dependency chain submitted to test whether the new Track 1 skip-decoder also improves the Track 2 Stage 0 endpoint:
    - evaluate `25892` reconstructed RI with RangeDet after completion
    - evaluate `25894` reconstructed RI with RangeDet after completion
    - automatically compare:
      - repaired `raw/basic`
      - old Stage 0 baseline / enhanced
      - new skip-decoder Stage 0 variants
  - Submission manifest:
    - `logs/260316_t2skip_stage0_track2_rangedet_skip_stage0_manifest.csv`

- `Decision rule`
  - If the skip-decoder variants improve Track 2 Stage 0 materially, then the main Stage 0 bottleneck is decoder structure / artifact pattern.
  - If they still stay near zero, then the next change should go beyond decoder structure and add stronger geometry-aware supervision or representation changes.
