# `src/` Code Map

This directory contains the active in-repo code. If you only want to understand how the model and experiment pipeline work, this is the main directory to read.

## Layout

| Path | Purpose |
|---|---|
| [`main_train.py`](/home/018219422/lidar_pointcloud_compression/src/main_train.py) | Main training entrypoint |
| [`dataset/`](/home/018219422/lidar_pointcloud_compression/src/dataset) | Dataset loaders and dataset download helpers |
| [`models/`](/home/018219422/lidar_pointcloud_compression/src/models) | Model definitions |
| [`loss/`](/home/018219422/lidar_pointcloud_compression/src/loss) | Distillation and task losses |
| [`train/`](/home/018219422/lidar_pointcloud_compression/src/train) | Trainer and evaluation code |
| [`scripts/`](/home/018219422/lidar_pointcloud_compression/src/scripts) | Slurm launchers and experiment wrappers |
| [`utils/`](/home/018219422/lidar_pointcloud_compression/src/utils) | Export, bookkeeping, helper utilities |

## Main Files

### Core training

- [`main_train.py`](/home/018219422/lidar_pointcloud_compression/src/main_train.py)
  - Parses experiment flags
  - Builds datasets and model
  - Launches training through `Trainer`

- [`train/trainer.py`](/home/018219422/lidar_pointcloud_compression/src/train/trainer.py)
  - Main training loop
  - Reconstruction / rate / auxiliary losses
  - Checkpoint writing

### Models

- [`models/compression.py`](/home/018219422/lidar_pointcloud_compression/src/models/compression.py)
  - Top-level compression model assembly
  - Quantizer routing
  - Decoder selection
  - Optional position branch
  - Optional pillar / BEV side stream

- [`models/autoencoder.py`](/home/018219422/lidar_pointcloud_compression/src/models/autoencoder.py)
  - Encoder
  - Deconv decoder
  - Coord-conditioned decoder
  - Skip/U-Net-style decoder variants

- [`models/pillar_side.py`](/home/018219422/lidar_pointcloud_compression/src/models/pillar_side.py)
  - Dynamic pillar feature extraction
  - BEV backbone
  - RI-stage feature gathering

- [`models/quantization.py`](/home/018219422/lidar_pointcloud_compression/src/models/quantization.py)
  - Adaptive quantizer

- [`models/stage3_necks.py`](/home/018219422/lidar_pointcloud_compression/src/models/stage3_necks.py)
  - Multiscale fusion variants such as `rangeformer`, `frnet`, etc.

### Evaluation

- [`train/evaluate_kitti_map_vs_rate.py`](/home/018219422/lidar_pointcloud_compression/src/train/evaluate_kitti_map_vs_rate.py)
  - PointPillars-based Track 1 evaluation

- [`scripts/eval_rangedet_archive_car_ap.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_car_ap.py)
  - Relative car AP on archived RangeDet outputs

- [`scripts/eval_rangedet_archive_kitti_official.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_kitti_official.py)
  - KITTI-style evaluator bridge

## Current High-Value Scripts

### Track 1

- [`scripts/run_track1_noquant_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_track1_noquant_chain.sh)
- [`scripts/submit_track1_pillar_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_main.sh)
- [`scripts/submit_track1_pillar_skip_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_skip_main.sh)

### Track 2

- [`scripts/run_rangedet_kitti_train.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_kitti_train.sh)
- [`scripts/run_rangedet_raw_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_raw_eval.sh)
- [`scripts/run_rangedet_stage_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_stage_eval.sh)
- [`scripts/submit_track2_rangedet_patched_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_patched_chain.sh)
- [`scripts/submit_track2_rangedet_skip_stage0.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_skip_stage0.sh)

## Historical vs Current

This directory contains many historical experiment scripts. That is intentional; they preserve the exact launch configurations used during the project.

If you want the current path rather than the historical record:

1. Start with `main_train.py`
2. Read `models/compression.py`
3. Read `scripts/run_track1_noquant_chain.sh`
4. Read `scripts/run_rangedet_kitti_train.sh` and `scripts/run_rangedet_stage_eval.sh`
