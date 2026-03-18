# `notebooks/` Guide

This directory contains analysis notebooks and a curated set of summary CSV/MD files.

## What belongs here

### Interactive notebooks

- [`track1_identity_vs_codec_visualization.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/track1_identity_vs_codec_visualization.ipynb)
  - Visualizes raw point cloud, range image, reconstructed range image, reconstructed point cloud, and GT overlays for Track 1

- [`rangedet_analysis.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/rangedet_analysis.ipynb)
  - Visualizes raw vs reconstructed range images and RangeDet predictions

- [`track2_codec_root_cause.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/track2_codec_root_cause.ipynb)
  - Multi-frame Track 2 artifact analysis using repaired raw/basic, Stage 0, and Stage 1 RangeDet outputs

- [`stage0_stage1_kitti_pointpillar_visualization.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/stage0_stage1_kitti_pointpillar_visualization.ipynb)
  - Track 1 endpoint visualization

### Executed notebooks

- Files ending in `.executed.ipynb`
  - Saved executed outputs for reproducibility

### Summary files

- [`results_summary_current.md`](/home/018219422/lidar_pointcloud_compression/notebooks/results_summary_current.md)
  - Human-readable project summary across Track 1 and Track 2

- [`results_summary_current.csv`](/home/018219422/lidar_pointcloud_compression/notebooks/results_summary_current.csv)
  - Single-table machine-readable summary across the current high-value results

- `kitti_map_vs_rate_summary_*.csv`
  - High-level Track 1 result summaries kept for experiment provenance

- `pointpillar_finetune_kitti_summary.csv`
  - Fine-tune summary across PointPillars runs

## Reading order

If you are new to the repo:

1. Read [`README.md`](/home/018219422/lidar_pointcloud_compression/README.md)
2. Read [`docs/repo_guide.md`](/home/018219422/lidar_pointcloud_compression/docs/repo_guide.md)
3. Use the notebooks only after you know which experiment family you are looking at

## Important note

Only curated summary CSVs are kept in this directory.

Low-level `detail` and `pairs` CSV exports were removed to reduce clutter. If needed, they should be regenerated from the original experiment outputs rather than stored permanently in `notebooks/`.

The remaining file names still encode experiment identity:

- `summary`
- experiment tag
- variant
- timestamp

Example:

- `kitti_map_vs_rate_summary_t1nq_pillar_b_pillar_b_260309_201414.csv`
