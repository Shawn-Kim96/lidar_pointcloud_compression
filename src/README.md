# `src/` Code Map

This directory contains the active in-repo code. If you only want to understand how the model and experiment pipeline work, this is the main directory to read.

## Layout

| Path | Purpose |
|---|---|
| [`main_train.py`](../src/main_train.py) | Main training entrypoint |
| [`dataset/`](../src/dataset) | Dataset loaders and dataset download helpers |
| [`models/`](../src/models) | Model definitions |
| [`loss/`](../src/loss) | Distillation and task losses |
| [`train/`](../src/train) | Trainer and evaluation code |
| [`scripts/`](../src/scripts) | Slurm launchers and experiment wrappers |
| [`utils/`](../src/utils) | Export, bookkeeping, helper utilities |

## Main Files

### Core training

- [`main_train.py`](../src/main_train.py)
  - Parses experiment flags
  - Builds datasets and model
  - Launches training through `Trainer`

- [`train/trainer.py`](../src/train/trainer.py)
  - Main training loop
  - Reconstruction / rate / auxiliary losses
  - Checkpoint writing

### Models

- [`models/compression.py`](../src/models/compression.py)
  - Top-level compression model assembly
  - Quantizer routing
  - Decoder selection
  - Optional position branch
  - Optional pillar / BEV side stream

- [`models/autoencoder.py`](../src/models/autoencoder.py)
  - Encoder
  - Deconv decoder
  - Coord-conditioned decoder
  - Skip/U-Net-style decoder variants

- [`models/pillar_side.py`](../src/models/pillar_side.py)
  - Dynamic pillar feature extraction
  - BEV backbone
  - RI-stage feature gathering

- [`models/quantization.py`](../src/models/quantization.py)
  - Adaptive quantizer

- [`models/stage3_necks.py`](../src/models/stage3_necks.py)
  - Multiscale fusion variants such as `rangeformer`, `frnet`, etc.

### Evaluation

- [`train/evaluate_kitti_map_vs_rate.py`](../src/train/evaluate_kitti_map_vs_rate.py)
  - PointPillars-based Track 1 evaluation

- [`scripts/eval_rangedet_archive_car_ap.py`](../src/scripts/eval_rangedet_archive_car_ap.py)
  - Relative car AP on archived RangeDet outputs

- [`scripts/eval_rangedet_archive_kitti_official.py`](../src/scripts/eval_rangedet_archive_kitti_official.py)
  - KITTI-style evaluator bridge

## Current High-Value Scripts

### Track 1

- [`scripts/run_track1_noquant_chain.sh`](../src/scripts/run_track1_noquant_chain.sh)
- [`scripts/submit_track1_pillar_main.sh`](../src/scripts/submit_track1_pillar_main.sh)
- [`scripts/submit_track1_pillar_skip_main.sh`](../src/scripts/submit_track1_pillar_skip_main.sh)

### Track 2

- [`scripts/run_rangedet_kitti_train.sh`](../src/scripts/run_rangedet_kitti_train.sh)
- [`scripts/run_rangedet_raw_eval.sh`](../src/scripts/run_rangedet_raw_eval.sh)
- [`scripts/run_rangedet_stage_eval.sh`](../src/scripts/run_rangedet_stage_eval.sh)
- [`scripts/submit_track2_rangedet_patched_chain.sh`](../src/scripts/submit_track2_rangedet_patched_chain.sh)
- [`scripts/submit_track2_rangedet_skip_stage0.sh`](../src/scripts/submit_track2_rangedet_skip_stage0.sh)

## Historical vs Current

This directory contains many historical experiment scripts. That is intentional; they preserve the exact launch configurations used during the project.

If you want the current path rather than the historical record:

1. Start with `main_train.py`
2. Read `models/compression.py`
3. Read `scripts/run_track1_noquant_chain.sh`
4. Read `scripts/run_rangedet_kitti_train.sh` and `scripts/run_rangedet_stage_eval.sh`
