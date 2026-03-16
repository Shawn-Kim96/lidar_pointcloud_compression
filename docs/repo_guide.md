# Repository Guide

This document is the shortest path to understanding where the active code lives, which files are historical, and where to find outputs.

## 1. Mental Model

The repository has three layers:

1. `Core model/training code`
   - In-repo PyTorch compression model
   - Dataset loaders
   - Trainer and evaluation logic

2. `Experiment orchestration`
   - Slurm submit scripts
   - Dataset export scripts
   - External detector wrappers

3. `Research artifacts`
   - Notes, reports, notebooks, manifests, logs, archived outputs

The project currently revolves around two downstream tracks:

- `Track 1`
  - `Point Cloud -> Range Image -> Point Cloud`
  - evaluate with `PointPillars`
- `Track 2`
  - `Point Cloud -> Range Image`
  - evaluate directly with `RangeDet`

## 2. Where the Active Code Is

### Training entrypoint

- [`src/main_train.py`](/home/018219422/lidar_pointcloud_compression/src/main_train.py)

This is the main in-repo entrypoint for codec training. It wires:

- dataset choice
- model config
- quantizer mode
- decoder type
- side branches such as pillar / BEV stream

### Model code

- [`src/models/compression.py`](/home/018219422/lidar_pointcloud_compression/src/models/compression.py)
  - top-level compression model assembly
- [`src/models/autoencoder.py`](/home/018219422/lidar_pointcloud_compression/src/models/autoencoder.py)
  - encoder / decoder variants
  - coordinate-conditioned decoder
  - skip-connected decoder variants
- [`src/models/pillar_side.py`](/home/018219422/lidar_pointcloud_compression/src/models/pillar_side.py)
  - dynamic pillar / BEV side stream
- [`src/models/quantization.py`](/home/018219422/lidar_pointcloud_compression/src/models/quantization.py)
  - adaptive/uniform quantization
- [`src/models/stage3_necks.py`](/home/018219422/lidar_pointcloud_compression/src/models/stage3_necks.py)
  - multiscale fusion blocks

### Dataset code

- [`src/dataset/kitti_object_loader.py`](/home/018219422/lidar_pointcloud_compression/src/dataset/kitti_object_loader.py)
  - KITTI range-image dataset
- [`src/dataset/semantickitti_loader.py`](/home/018219422/lidar_pointcloud_compression/src/dataset/semantickitti_loader.py)
  - SemanticKITTI dataset

### Training/evaluation utilities

- [`src/train/trainer.py`](/home/018219422/lidar_pointcloud_compression/src/train/trainer.py)
  - core training loop
- [`src/train/evaluate_kitti_map_vs_rate.py`](/home/018219422/lidar_pointcloud_compression/src/train/evaluate_kitti_map_vs_rate.py)
  - Track 1-style endpoint evaluation summaries

## 3. Which Scripts Matter Today

Many scripts in `src/scripts/` are historical. The ones below are the current important ones.

### Track 1

- [`run_track1_noquant_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_track1_noquant_chain.sh)
  - train codec
  - export `KITTI_Identity`
  - fine-tune PointPillars
  - run reconstructed endpoint evaluation

- [`submit_track1_pillar_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_main.sh)
  - older pillar/BEV side-stream submitter

- [`submit_track1_pillar_skip_main.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track1_pillar_skip_main.sh)
  - current submitter for pillar-side + skip-decoder Track 1 runs

### Track 2

- [`run_rangedet_kitti_train.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_kitti_train.sh)
  - raw/basic RangeDet training

- [`run_rangedet_raw_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_raw_eval.sh)
  - raw/basic evaluation and archive dump

- [`run_rangedet_stage_eval.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_stage_eval.sh)
  - reconstructed RI export + RangeDet evaluation

- [`submit_track2_rangedet_patched_chain.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_patched_chain.sh)
  - current repaired raw/basic + Stage0/Stage1 chain

- [`submit_track2_rangedet_skip_stage0.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/submit_track2_rangedet_skip_stage0.sh)
  - follow-up evaluation for the new skip-decoder Track 1 runs

### Evaluation utilities

- [`eval_rangedet_archive_car_ap.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_car_ap.py)
  - custom lidar-space car AP used for relative comparison

- [`eval_rangedet_archive_kitti_official.py`](/home/018219422/lidar_pointcloud_compression/src/scripts/eval_rangedet_archive_kitti_official.py)
  - bridge from archived RangeDet outputs to KITTI-style eval

- [`run_rangedet_archive_compare.sh`](/home/018219422/lidar_pointcloud_compression/src/scripts/run_rangedet_archive_compare.sh)
  - combined comparison wrapper

## 4. Where Outputs Go

### Training runs

- [`data/results/experiments/`](/home/018219422/lidar_pointcloud_compression/data/results/experiments)

Each run directory typically contains:

- `config.yaml`
- `model_epoch_*.pth`
- sometimes `model_final.pth`

### Notebook summaries

- [`notebooks/`](/home/018219422/lidar_pointcloud_compression/notebooks)

Typical files:

- `kitti_map_vs_rate_summary_*.csv`
- `kitti_map_vs_rate_detail_*.csv`
- `kitti_map_vs_rate_pairs_*.csv`
- visualization notebooks

### Logs and manifests

- [`logs/`](/home/018219422/lidar_pointcloud_compression/logs)

Typical files:

- `YYMMDD_<experiment>_<jobid>.out`
- `YYMMDD_<experiment>_<jobid>.err`
- `*_manifest.csv`
- summary `.md` / `.csv`

### RangeDet archived outputs

- [`logs/rangedet_eval_outputs/`](/home/018219422/lidar_pointcloud_compression/logs/rangedet_eval_outputs)

These `.pkl` files are archived detector outputs used for post-hoc evaluation.

## 5. Where to Read Results

### Chronological record

- [`docs/notes/research_journal_en.md`](/home/018219422/lidar_pointcloud_compression/docs/notes/research_journal_en.md)

Use this for:

- what was tried
- what failed
- why code changed
- what the next decision was

### Structured reports

- [`docs/report/`](/home/018219422/lidar_pointcloud_compression/docs/report)

Use this for:

- experiment plans
- formal summaries
- design notes

### Visual inspection

- [`notebooks/rangedet_analysis.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/rangedet_analysis.ipynb)
- [`notebooks/track1_identity_vs_codec_visualization.ipynb`](/home/018219422/lidar_pointcloud_compression/notebooks/track1_identity_vs_codec_visualization.ipynb)

## 6. Which Parts Are Historical

The following directories are valuable for auditability, but they are not the shortest path for a new external user:

- many older submit scripts in `src/scripts/`
- older result tables in `src/results/`
- old Slurm logs in `logs/slurm_*`
- exploratory notes in `docs/report/`

If you are trying to understand the current pipeline, prioritize:

1. `README.md`
2. this guide
3. `src/README.md`
4. `research_journal_en.md`

## 7. External Repositories

These directories contain external code that is integrated locally:

- [`third_party/OpenPCDet/`](/home/018219422/lidar_pointcloud_compression/third_party/OpenPCDet)
- [`third_party/external_range_det/RangeDet/`](/home/018219422/lidar_pointcloud_compression/third_party/external_range_det/RangeDet)
- [`third_party/external_codecs/RENO/`](/home/018219422/lidar_pointcloud_compression/third_party/external_codecs/RENO)

Do not assume these repositories follow the same conventions as the in-repo code. The supported integration layer is in `src/scripts/` and `src/utils/`.
