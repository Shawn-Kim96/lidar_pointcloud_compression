# Stage 1 Baseline Results

## 1) Objective
Build and validate a baseline LiDAR compression model:
- Multi-channel projection -> encoder bottleneck -> quantization -> decoder reconstruction
- Evaluate compression quality using `BPP`, `PSNR (Range/Intensity)`, `Chamfer Distance`

## 2) Main Training Run (Baseline Checkpoint)
- Job: `20934` on `gpuqm`
- Script: `src/scripts/stage1_baseline_gpuqm.slurm`
- Train split: `00,01,02,03,04,05,06,07,09,10`
- Validation split: `08`
- Epochs: `20`
- Checkpoint: `data/results/checkpoints/stage1_baseline.pth`
- Train log: `logs/stage1_train.out`
- Eval log: `logs/stage1_eval_baseline.out`

## 3) Training Convergence (from `logs/stage1_train.out`)
- Epoch 1: `Total=0.838233`, `Range=0.762446`, `Intensity=0.075787`
- Epoch 20: `Total=0.311199`, `Range=0.257268`, `Intensity=0.053931`
- Trend: all losses decreased consistently over 20 epochs.

## 4) Final Validation Metrics (Seq 08, 4071 frames)
- `q8`: `BPP=1.755795`, `PSNR_Range=35.606316`, `PSNR_Intensity=19.662548`, `CD=10.533694`
- `q4`: `BPP=0.655665`, `PSNR_Range=34.767764`, `PSNR_Intensity=19.238122`, `CD=10.524710`

Interpretation:
- `q4` reduces bitrate significantly vs `q8` (about 62.7% lower BPP).
- PSNR drops moderately at `q4`.
- CD is almost unchanged in this run.

## 5) Overnight Ablation (Previous Run)
Source: `logs/stage1_eval_overnight.out` (Seq 08, 512 frames each)
- `n010` (`noise_std=0.10`): q8 `CD=11.636837`, q4 `CD=11.631089`
- `n005` (`noise_std=0.05`): q8 `CD=12.091840`, q4 `CD=12.087716`
- `n000` (`noise_std=0.00`): q8 `CD=11.646872`, q4 `CD=11.639210`

Observation:
- `noise_std=0.10` was a stable choice among tested noise settings.

## 6) Stage 1 Outputs
- `data/results/checkpoints/stage1_baseline.pth`
- `logs/stage1_train.out`
- `logs/stage1_eval_baseline.out`
