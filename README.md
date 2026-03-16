# LiDAR Point Cloud Compression

Research codebase for learned LiDAR compression with two downstream evaluation tracks:

- `Track 1`: `Point Cloud -> Range Image -> Point Cloud`, evaluated with `PointPillars`
- `Track 2`: `Point Cloud -> Range Image`, evaluated directly with `RangeDet`

The repository contains both the core compression models and the orchestration scripts used to run large Slurm experiments on KITTI / SemanticKITTI.

## Current Status

- `Track 1` is currently focused on geometry preservation for reconstructed point clouds.
- `Track 2` now has a repaired `RangeDet` raw/basic baseline; the current bottleneck is the codec, not the detector baseline.
- The most up-to-date running notes are in:
  - [research_journal_en.md](/home/018219422/lidar_pointcloud_compression/docs/notes/research_journal_en.md)

## Repository Map

| Path | Purpose |
|---|---|
| [`src/main_train.py`](/home/018219422/lidar_pointcloud_compression/src/main_train.py) | Main PyTorch training entrypoint for the in-repo compression model |
| [`src/models/`](/home/018219422/lidar_pointcloud_compression/src/models) | Compression model, autoencoder/decoder, quantization, side branches |
| [`src/dataset/`](/home/018219422/lidar_pointcloud_compression/src/dataset) | SemanticKITTI and KITTI range-image loaders |
| [`src/train/`](/home/018219422/lidar_pointcloud_compression/src/train) | Trainer and evaluation utilities |
| [`src/scripts/`](/home/018219422/lidar_pointcloud_compression/src/scripts) | Reproducible Slurm launch scripts, dataset export, external detector evaluation |
| [`src/utils/`](/home/018219422/lidar_pointcloud_compression/src/utils) | Identity export, reconstruction export, experiment bookkeeping |
| [`docs/notes/`](/home/018219422/lidar_pointcloud_compression/docs/notes) | Research journal and informal notes |
| [`docs/report/`](/home/018219422/lidar_pointcloud_compression/docs/report) | Experiment plans, ablation writeups, structured reports |
| [`docs/papers/`](/home/018219422/lidar_pointcloud_compression/docs/papers) | Local copies / text extracts of external papers used for analysis |
| [`notebooks/`](/home/018219422/lidar_pointcloud_compression/notebooks) | Visualization and post-hoc analysis notebooks |
| [`logs/`](/home/018219422/lidar_pointcloud_compression/logs) | Slurm logs, manifests, archive evaluation summaries |
| [`data/results/experiments/`](/home/018219422/lidar_pointcloud_compression/data/results/experiments) | Per-run checkpoints and `config.yaml` files |
| [`third_party/`](/home/018219422/lidar_pointcloud_compression/third_party) | External dependencies such as OpenPCDet, RangeDet, and RENO |

For a more detailed orientation guide, start here:

- [docs/repo_guide.md](/home/018219422/lidar_pointcloud_compression/docs/repo_guide.md)
- [src/README.md](/home/018219422/lidar_pointcloud_compression/src/README.md)
- [notebooks/README.md](/home/018219422/lidar_pointcloud_compression/notebooks/README.md)
- [logs/README.md](/home/018219422/lidar_pointcloud_compression/logs/README.md)

## Main Workflows

### 1. Train the in-repo compression model

Primary entrypoint:

- [`src/main_train.py`](/home/018219422/lidar_pointcloud_compression/src/main_train.py)

Typical usage:

```bash
PYTHONPATH=src python src/main_train.py \
  --backbone resnet \
  --dataset_type kitti3dobject \
  --data_root data/dataset/kitti3dobject \
  --kitti_split train \
  --img_h 64 \
  --img_w 2048 \
  --quantizer_mode none \
  --epochs 180
```

### 2. Track 1: full no-quant chain

This chain trains a codec, exports `KITTI_Identity`, fine-tunes PointPillars, and runs endpoint evaluation.

- [`src/scripts/run_track1_noquant_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_track1_noquant_chain.sh)

Recent submitters:

- [`submit_track1_pillar_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_main.sh)
- [`submit_track1_pillar_skip_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_skip_main.sh)

### 3. Track 2: RangeDet raw/basic and reconstructed RI evaluation

Core scripts:

- raw training:
  - [`run_rangedet_kitti_train.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_kitti_train.sh)
- raw eval:
  - [`run_rangedet_raw_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_raw_eval.sh)
- reconstructed Stage0/Stage1 eval:
  - [`run_rangedet_stage_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_stage_eval.sh)

Recent submitters:

- [`submit_track2_rangedet_patched_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_patched_chain.sh)
- [`submit_track2_rangedet_skip_stage0.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_skip_stage0.sh)

### 4. Evaluation and archive comparison

- custom lidar-space archive AP:
  - [`eval_rangedet_archive_car_ap.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_car_ap.py)
- KITTI-style archive evaluation bridge:
  - [`eval_rangedet_archive_kitti_official.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_kitti_official.py)
- combined compare wrapper:
  - [`run_rangedet_archive_compare.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_archive_compare.sh)

## Key Result Locations

### Experiment runs

- Compression runs:
  - [`data/results/experiments/`](/home/018219422/lidar_pointcloud_compression/data/results/experiments)
- High-level result snapshots:
  - [`src/results/experiments_result.md`](/home/018219422/lidar_pointcloud_compression/src/results/experiments_result.md)
  - [`src/results/experiments_result.csv`](/home/018219422/lidar_pointcloud_compression/src/results/experiments_result.csv)

### Logs and manifests

- Slurm `.out/.err` files:
  - [`logs/`](/home/018219422/lidar_pointcloud_compression/logs)
- Machine-readable submission manifests:
  - `logs/*manifest.csv`
- RangeDet archived prediction dumps:
  - [`logs/rangedet_eval_outputs/`](/home/018219422/lidar_pointcloud_compression/logs/rangedet_eval_outputs)

### Notebook outputs

- Track 1 point-cloud / range-image visualization:
  - [`track1_identity_vs_codec_visualization.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/track1_identity_vs_codec_visualization.ipynb)
- RangeDet diagnostic notebook:
  - [`rangedet_analysis.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/rangedet_analysis.ipynb)

## External Dependencies

This repository includes local copies of major external components:

| Path | Purpose |
|---|---|
| [`third_party/OpenPCDet/`](/home/018219422/lidar_pointcloud_compression/third_party/OpenPCDet) | PointPillars / KITTI 3D detection |
| [`third_party/external_range_det/RangeDet/`](/home/018219422/lidar_pointcloud_compression/third_party/external_range_det/RangeDet) | Range-image 3D detector used in Track 2 |
| [`third_party/external_codecs/RENO/`](/home/018219422/lidar_pointcloud_compression/third_party/external_codecs/RENO) | External reconstruction/compression baseline under study |
| [`third_party/range_view_checkpoints/`](/home/018219422/lidar_pointcloud_compression/third_party/range_view_checkpoints) | Downloaded range-view checkpoints |

These third-party repositories are not normalized to a single coding style. The repo-specific integration scripts in `src/scripts/` are the supported entrypoints.

## Guidance for External Users

If you are new to this repository, do not start by reading random Slurm logs.

Start in this order:

1. [docs/repo_guide.md](/home/018219422/lidar_pointcloud_compression/docs/repo_guide.md)
2. [src/README.md](/home/018219422/lidar_pointcloud_compression/src/README.md)
3. [research_journal_en.md](/home/018219422/lidar_pointcloud_compression/docs/notes/research_journal_en.md)
4. [notebooks/README.md](/home/018219422/lidar_pointcloud_compression/notebooks/README.md)
5. [logs/README.md](/home/018219422/lidar_pointcloud_compression/logs/README.md)

## Notes

- Many files in `docs/report/`, `logs/`, and `data/results/experiments/` are historical and preserved for auditability.
- The current codepath is concentrated in:
  - `src/main_train.py`
  - `src/models/`
  - `src/scripts/run_track1_noquant_chain.sh`
  - `src/scripts/run_rangedet_kitti_train.sh`
  - `src/scripts/run_rangedet_stage_eval.sh`
