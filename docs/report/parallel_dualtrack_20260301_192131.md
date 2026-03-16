# Parallel Dual-Track Recovery Launch (parallel_dualtrack_20260301_192131)

- Created: 2026-03-01 19:21:31 PST
- Git head: 5862d0b
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_192131.csv`

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22835 | trackb_eval_retry_p100_poolA | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;pool=g8,g10,g11,g12,g13;gres=gpu:p100:1;patch=ff9724f` | retry after p100-default + cuda-op smoke patch |
| 22836 | trackb_eval_retry_p100_poolB | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;pool=g14,g15,g1,g2,g3,g4,g5,g6,g7;gres=gpu:p100:1;patch=ff9724f` | parallel hedge retry across alternate p100 pool |

## Queue Snapshot

_Auto-updated: 2026-03-01 20:16:26 _

```text
     JOBID  PARTITION                         NAME     USER ST       TIME NO NODELIST(REASON)
```
