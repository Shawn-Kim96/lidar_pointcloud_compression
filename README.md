# LiDAR Point Cloud Compression

Research codebase for learned LiDAR compression with two downstream evaluation tracks:

- `Track 1`: `Point Cloud -> Range Image -> Point Cloud`, evaluated with `PointPillars`
- `Track 2`: `Point Cloud -> Range Image`, evaluated directly with `RangeDet`

The repository contains both the core compression models and the orchestration scripts used to run large Slurm experiments on KITTI / SemanticKITTI.

## Current Status

- `Track 1` is currently focused on geometry preservation for reconstructed point clouds.
- `Track 2` now has a repaired `RangeDet` raw/basic baseline; the current bottleneck is the codec, not the detector baseline.
- The most up-to-date running notes are in:
  - [research_journal_en.md](docs/notes/research_journal_en.md)

## Project Page

- A GitHub Pages-ready project page now lives in:
  - [`docs/index.html`](docs/index.html)
- Supporting public-facing notes:
  - [`docs/gallery.html`](docs/gallery.html)
  - [`docs/evidence.html`](docs/evidence.html)
  - [`docs/asset-gaps.html`](docs/asset-gaps.html)
- Recommended publish setup:
  - GitHub Pages -> `Deploy from a branch`
  - Branch: `main`
  - Folder: `/docs`

## Repository Map

| Path | Purpose |
|---|---|
| [`src/main_train.py`](src/main_train.py) | Main PyTorch training entrypoint for the in-repo compression model |
| [`src/models/`](src/models) | Compression model, autoencoder/decoder, quantization, side branches |
| [`src/dataset/`](src/dataset) | SemanticKITTI and KITTI range-image loaders |
| [`src/train/`](src/train) | Trainer and evaluation utilities |
| [`src/scripts/`](src/scripts) | Reproducible Slurm launch scripts, dataset export, external detector evaluation |
| [`src/utils/`](src/utils) | Identity export, reconstruction export, experiment bookkeeping |
| [`docs/notes/`](docs/notes) | Research journal and informal notes |
| [`docs/report/`](docs/report) | Experiment plans, ablation writeups, structured reports |
| [`docs/papers/`](docs/papers) | Local copies / text extracts of external papers used for analysis |
| [`notebooks/`](notebooks) | Visualization and post-hoc analysis notebooks |
| [`logs/`](logs) | Slurm logs, manifests, archive evaluation summaries |
| [`data/results/experiments/`](data/results/experiments) | Per-run checkpoints and `config.yaml` files |
| `third_party/` | External dependencies such as OpenPCDet, RangeDet, and RENO when that local workspace is populated |

For a more detailed orientation guide, start here:

- [docs/repo_guide.md](docs/repo_guide.md)
- [src/README.md](src/README.md)
- [notebooks/README.md](notebooks/README.md)
- [logs/](logs)

## Main Workflows

### 1. Train the in-repo compression model

Primary entrypoint:

- [`src/main_train.py`](src/main_train.py)

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

- [`src/scripts/run_track1_noquant_chain.sh`](src/scripts/run_track1_noquant_chain.sh)

Recent submitters:

- [`submit_track1_pillar_main.sh`](src/scripts/submit_track1_pillar_main.sh)
- [`submit_track1_pillar_skip_main.sh`](src/scripts/submit_track1_pillar_skip_main.sh)

### 3. Track 2: RangeDet raw/basic and reconstructed RI evaluation

Core scripts:

- raw training:
  - [`run_rangedet_kitti_train.sh`](src/scripts/run_rangedet_kitti_train.sh)
- raw eval:
  - [`run_rangedet_raw_eval.sh`](src/scripts/run_rangedet_raw_eval.sh)
- reconstructed Stage0/Stage1 eval:
  - [`run_rangedet_stage_eval.sh`](src/scripts/run_rangedet_stage_eval.sh)

Recent submitters:

- [`submit_track2_rangedet_patched_chain.sh`](src/scripts/submit_track2_rangedet_patched_chain.sh)
- [`submit_track2_rangedet_skip_stage0.sh`](src/scripts/submit_track2_rangedet_skip_stage0.sh)

### 4. Evaluation and archive comparison

- custom lidar-space archive AP:
  - [`eval_rangedet_archive_car_ap.py`](src/scripts/eval_rangedet_archive_car_ap.py)
- KITTI-style archive evaluation bridge:
  - [`eval_rangedet_archive_kitti_official.py`](src/scripts/eval_rangedet_archive_kitti_official.py)
- combined compare wrapper:
  - [`run_rangedet_archive_compare.sh`](src/scripts/run_rangedet_archive_compare.sh)

## Key Result Locations

### Experiment runs

- Compression runs:
  - [`data/results/experiments/`](data/results/experiments)
- High-level result snapshots:
  - [`src/results/experiments_result.md`](src/results/experiments_result.md)
  - [`src/results/experiments_result.csv`](src/results/experiments_result.csv)

### Logs and manifests

- Slurm `.out/.err` files:
  - [`logs/`](logs)
- Machine-readable submission manifests:
  - `logs/*manifest.csv`
- RangeDet archived prediction dumps:
  - `logs/rangedet_eval_outputs/` when archived detector dumps are present

### Notebook outputs

- Track 1 point-cloud / range-image visualization:
  - [`track1_identity_vs_codec_visualization.ipynb`](notebooks/track1_identity_vs_codec_visualization.ipynb)
- RangeDet diagnostic notebook:
  - [`rangedet_analysis.ipynb`](notebooks/rangedet_analysis.ipynb)

## External Dependencies

This research workflow expects several local external components in fuller environments:

| Path | Purpose |
|---|---|
| `third_party/OpenPCDet/` | PointPillars / KITTI 3D detection |
| `third_party/external_range_det/RangeDet/` | Range-image 3D detector used in Track 2 |
| `third_party/external_codecs/RENO/` | External reconstruction/compression baseline under study |
| `third_party/range_view_checkpoints/` | Downloaded range-view checkpoints |

These third-party repositories are not normalized to a single coding style. The repo-specific integration scripts in `src/scripts/` are the supported entrypoints.

## Guidance for External Users

If you are new to this repository, do not start by reading random Slurm logs.

Start in this order:

1. [docs/repo_guide.md](docs/repo_guide.md)
2. [src/README.md](src/README.md)
3. [research_journal_en.md](docs/notes/research_journal_en.md)
4. [notebooks/README.md](notebooks/README.md)
5. [logs/](logs)

## Notes

- Many files in `docs/report/`, `logs/`, and `data/results/experiments/` are historical and preserved for auditability.
- The current codepath is concentrated in:
  - `src/main_train.py`
  - `src/models/`
  - `src/scripts/run_track1_noquant_chain.sh`
  - `src/scripts/run_rangedet_kitti_train.sh`
  - `src/scripts/run_rangedet_stage_eval.sh`
