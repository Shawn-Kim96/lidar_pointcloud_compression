# Experiment Registry (How To Track Checkpoints)

## Problem
Historically, checkpoints like `stage1_baseline.pth` do not encode what model/training configuration produced them.
This makes it hard to:
- reproduce results,
- run parallel sweeps safely,
- compare ablations without confusion.

## Current Solution (Implemented)
Training scripts now emit a *run directory* under `data/results/runs/<run_id>/` that contains:
- `data/results/runs/<run_id>/checkpoint.pth`: model weights plus a copy of `config` and `model_info`
- `data/results/runs/<run_id>/manifest.json`: human/audit metadata (command, config, model_info, env, git rev)

This does not break the existing behavior of also writing a traditional checkpoint into `data/results/checkpoints/`.

## How To Use
### Stage1
- Single run:
  - `PYTHONPATH=src python -u src/train/train.py ... --run_name s1_ae --save_name auto`
- Sweep (parallel via Slurm array):
  - `sbatch src/scripts/sweep_stage1_array.slurm`

### Inspect runs
- `python src/utils/list_runs.py`

## Naming Policy
- Default: if `--save_name` is omitted or set to `auto`, checkpoints are saved as `data/results/checkpoints/<run_id>.pth`.
- If you want a stable alias (e.g., for downstream scripts), pass `--alias_name <alias>.pth` and keep that alias updated intentionally.

## Notes
- Rate is currently "estimated BPP" (entropy on latent codes), not "true bitstream BPP".
- For paper-quality reporting, keep a strict mapping between:
  - `data/results/runs/<run_id>/manifest.json`
  - plotted curves/tables in `documents/report/`
