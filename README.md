# lidar_pointcloud_compression

## Layout
- Code: `src/`
- Reports: `documents/report/`
- Papers: `documents/papers/`
- Dataset (SemanticKITTI): `data/dataset/semantickitti/`
- Outputs (checkpoints/runs/plots): `data/results/`

## Quick Commands
- Download + extract SemanticKITTI:
  - `bash src/scripts/download_semantickitti.sh --data-dir data/dataset/semantickitti`
- Stage1 train (debug):
  - `PYTHONPATH=src python -u src/train/train.py --debug`
- Stage1 eval:
  - `PYTHONPATH=src python -u src/train/evaluate.py --model data/results/checkpoints/stage1_baseline.pth --max_frames 64`
- Stage2.1 train (debug, deployable no-label path):
  - `PYTHONPATH=src python -u src/train/train_stage2_1.py --debug --no_labels --teacher_backend proxy --teacher_ckpt ""`
- Stage2.1 teacher score eval (orig vs recon CSV):
  - `PYTHONPATH=src python -u src/train/evaluate_teacher_scores.py --model_stage stage2_1 --model_ckpt <ckpt> --no_labels --teacher_backend proxy --teacher_ckpt ""`
- Stage2.1 RD(-T) smoke plot:
  - `PYTHONPATH=src python -u src/train/evaluate_stage2_1.py --stage2_1_ckpt <ckpt> --teacher_backend proxy --teacher_ckpt "" --max_frames 64`
- List experiment manifests:
  - `python src/utils/list_runs.py`
