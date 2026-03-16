# Track 2 Overnight Grid

- Base tag: `260302_track2grid`
- Run dir: `data/results/experiments/260301_resnet_uniform_q6_lr1e-4_bs4_j22769_r4`
- KITTI root: `data/dataset/kitti3dobject`
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_manifest.csv`
- Results CSV: `/home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_results.csv`
- Results MD: `/home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_results.md`
- Policy: full train/val split, frozen compression backbone

## Sweep

- Heads: `refine`, `deep`
- Hidden channels: `96`, `128`, `192`
- Learning rates: `1e-4`, `2e-4`
- Epochs:
  - `refine`: `120`, `180`
  - `deep`: `120`, `180`

Total jobs: `24`

## Result Collection

```bash
python3 src/scripts/collect_track2_range_roi_results.py \
  --manifest /home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_manifest.csv \
  --output_csv /home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_results.csv \
  --output_md /home/018219422/lidar_pointcloud_compression/logs/260302_track2grid_results.md
```
