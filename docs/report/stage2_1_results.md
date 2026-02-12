# Stage2.1 Results and Interpretation (for meeting)

## TL;DR
- `20988` job was a `1 epoch` sweep, so numbers were bad (undertrained).
- `21013` job is the valid overnight result: `20 epoch`, `seq08 full-val (4071 frames)`, all 3 runs completed.
- Current recommendation:
  - quality-priority default: `ld=1.0, li=0.2` (run `r0`)
  - compression-priority default: `ld=1.0, li=0.5` (run `r2`)

## 1) What Stage2.1 is doing
- Pipeline: input range image -> encoder -> latent -> importance head -> adaptive quantizer -> decoder.
- Distillation: frozen teacher (currently proxy backend) is used during training only.
- Inference path is deployable:
  - no labels,
  - no teacher execution,
  - only model-predicted importance map.

## 2) What `ld`, `li`, `be` mean (run-id notation)

Training objective in `src/train/train_stage2_1.py`:

`L_total = L_recon + be * L_rate + ld * L_distill + li * L_importance`

- `ld` = `lambda_distill`
  - teacher feature/logit matching loss weight.
  - bigger `ld` means stronger "follow the teacher".
- `li` = `lambda_importance`
  - importance map supervision loss (BCE) weight.
  - bigger `li` means stronger pressure to match teacher/ROI importance map.
- `be` = `beta_rate`
  - rate regularization weight.
  - bigger `be` means stronger bitrate penalty.

Run-id token example:
- `..._ld1.0_li0.5_be0.005` means
  - `lambda_distill=1.0`,
  - `lambda_importance=0.5`,
  - `beta_rate=0.005`.

## 3) Why there was a "1 epoch only" run
- `logs/stage2_1_overnight_20988.out` used sweep config with `--epochs 1`, so each run stopped after one epoch by design.
- That run was followed by `logs/stage2_1_overnight_21013.out` with `--epochs 20`, which completed normally (`Done time: 2026-02-11 00:45:12`).
- Default queue is already `gpuqm` in `src/scripts/stage2_1_overnight.slurm`.

## 4) Stage2.1 sweep summary

### 4.1 Chronology (teacher-score mean from CSV)

| Job / run | Frames | Epochs | ld | li | teacher_drop_mean |
|---|---:|---:|---:|---:|---:|
| `20988/r0` | 512 | 1 | 1.0 | 0.2 | `+0.557337` |
| `20988/r1` | 512 | 1 | 1.5 | 0.2 | `+0.546927` |
| `20988/r2` | 512 | 1 | 1.0 | 0.5 | `+0.559314` |
| `20986/r0` | 512 | 4 | 1.0 | 0.2 | `-0.093511` |
| `21013/r0` | 4071 | 20 | 1.0 | 0.2 | `-0.129634` |
| `21013/r1` | 4071 | 20 | 1.5 | 0.2 | `-0.108606` |
| `21013/r2` | 4071 | 20 | 1.0 | 0.5 | `-0.132197` |

Interpretation:
- 1 epoch is not enough (large positive drop).
- 4 epoch already flips to negative drop.
- 20 epoch full-val gives stable negative drop across all runs.

### 4.2 Final full-val numbers (job `21013`, 4071 frames)

| Run | qbits | teacher_drop | p_BPP | PSNR_Range | PSNR_Intensity | CD |
|---|---:|---:|---:|---:|---:|---:|
| `r0 (ld1.0/li0.2)` | 8 | `-0.129634` | `0.984644` | `33.928090` | `19.069175` | `0.430050` |
| `r1 (ld1.5/li0.2)` | 8 | `-0.108606` | `0.962357` | `33.803011` | `19.033299` | `0.466664` |
| `r2 (ld1.0/li0.5)` | 8 | `-0.132197` | `0.949950` | `33.673568` | `19.062396` | `0.484933` |
| `r0 (ld1.0/li0.2)` | 4 | `-0.146993` | `0.165417` | `21.824575` | `16.606993` | `1.971965` |
| `r1 (ld1.5/li0.2)` | 4 | `-0.142293` | `0.160799` | `21.427901` | `15.666332` | `1.940402` |
| `r2 (ld1.0/li0.5)` | 4 | `-0.198209` | `0.157368` | `20.984219` | `16.578871` | `1.822931` |

Trade-off:
- q8 quality best: `r0` (best PSNR_Range and CD).
- q8 compression/drop best: `r2` (lowest p_BPP, most negative drop).
- q4 best overall operating point: `r2`.

## 5) Stage1 vs Stage2.1 (same eval log example)

From `logs/stage2_1_eval_s2_1_s2_1_full_j21013_r0_ns0.02_lr0.0002_bs4_e20_sd42_bc64_lc64_st4_bp1_ld1.0_li0.2_be0.005.out`:

- q8:
  - Stage1: `teacher_drop=0.456418`, `p_BPP=1.383694`, `PSNR_Range=15.083255`, `CD=12.382332`
  - Stage2.1: `teacher_drop=-0.129634`, `p_BPP=0.984644`, `PSNR_Range=33.928090`, `CD=0.430050`
- q4:
  - Stage1: `teacher_drop=0.456478`, `p_BPP=0.536345`, `PSNR_Range=15.083250`, `CD=12.382327`
  - Stage2.1: `teacher_drop=-0.146993`, `p_BPP=0.165417`, `PSNR_Range=21.824575`, `CD=1.971965`

## 6) Caution on interpreting negative drop
- `teacher_drop < 0` means reconstructed input scored higher than original under the current proxy teacher.
- This can happen due to denoising/bias effects of the teacher proxy, so this is not enough to claim true detection gain.
- Required cross-check for paper-level claim:
  - PointPillars detection mAP / recall on reconstructed clouds,
  - ROI-region-specific accuracy,
  - qualitative frame checks for intermediate outputs.

## 7) Mapping to professor feedback (next actions)
- Detection accuracy first:
  - add PointPillars evaluation pipeline (mAP/recall, class-wise AP).
- Data cleanliness / preprocessing:
  - sample key frames and inspect artifacts before/after reconstruction.
- ROI estimator + encoder coupling:
  - sweep ROI size / moving window / step / threshold as explicit knobs.
- Multiple metrics and backups:
  - keep run-wise CSV + summary table keyed by hyperparameters.
- Intermediate result check:
  - save visualization panels (original vs recon vs importance map).
- Baseline-first strategy:
  - keep Stage1 as internal baseline and test one stronger baseline branch incrementally.

## 8) Reporting note
- `p_BPP` here is estimated entropy proxy on latent codes, not true arithmetic-coded bitstream BPP.
- True-bitstream baseline comparisons (GPCC/Draco) are evaluation-only follow-up work.
