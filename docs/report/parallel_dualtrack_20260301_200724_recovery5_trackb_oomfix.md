# Parallel Dual-Track Recovery Launch (parallel_dualtrack_20260301_200724_recovery5_trackb_oomfix)

- Created: 2026-03-01 20:07:24 PST
- Git source: origin/main@d28f0ae
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_200724_recovery5_trackb_oomfix.csv`

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22957 | trackb_eval_recovery5_oomfix | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `partition=gpuqm;conda=lidarcomp311;dataset=kitti3dobject;run_dirs=1;gpu_spec=none;smoke_retry=on;gpu_autoselect=on` | retry after OOM smoke-test failure with patched GPU binding/autoselect/retry |

## Immediate Queue Snapshot

_Auto-updated: 2026-03-01 20:07:24 PST_

```text
             JOBID PARTITION                                 NAME    STATE       TIME     NODELIST(REASON)
             22957     gpuqm                       kitti_map_rate  PENDING       0:00           (Priority)
             22780     gpuqm                     stage2_fix2_post  PENDING       0:00         (Dependency)
             22778     gpuqm                     stage2_fix2_post  PENDING       0:00         (Dependency)
           22923_1     gpuqm                          lidar_train  RUNNING       6:02                  g18
          22925_11     gpuqm                    stage2_fix2_sweep  RUNNING       6:02                  g19
           22926_4     gpuqm                    stage2_fix2_sweep  RUNNING       6:02                  g19
           22927_3     gpuqm                    stage2_fix2_sweep  RUNNING       6:02                  g19
          22927_10     gpuqm                    stage2_fix2_sweep  RUNNING       6:02                  g19
           22928_2     gpuqm                         lidar_stage3  RUNNING       6:02                  g19
           22928_4     gpuqm                         lidar_stage3  RUNNING       6:02                  g19
           22897_0     gpuqm                    stage2_fix2_sweep  RUNNING       8:56                  g19
           22779_6     gpuqm                    stage2_fix2_sweep  RUNNING    1:34:07                   g7
           22777_0     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                cs002
           22777_1     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                cs003
           22777_2     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                cs001
           22777_3     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g1
           22777_4     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g2
           22777_5     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g4
           22777_6     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g5
           22777_7     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g3
           22777_8     gpuqm                    stage2_fix2_sweep  RUNNING    1:35:10                   g6
           22762_0     gpuqm                          lidar_train  RUNNING    1:35:12                  g15
           22763_0     gpuqm                        lidar_distill  RUNNING    1:35:12                cs002
           22764_0     gpuqm                        lidar_distill  RUNNING    1:35:12                cs002
           22761_1     gpuqm                        lidar_uniform  RUNNING    1:35:12                  g10
           22761_2     gpuqm                        lidar_uniform  RUNNING    1:35:12                  g11
           22761_3     gpuqm                        lidar_uniform  RUNNING    1:35:12                  g12
           22761_4     gpuqm                        lidar_uniform  RUNNING    1:35:12                  g13
           22761_5     gpuqm                        lidar_uniform  RUNNING    1:35:12                  g14
           22762_2     gpuqm                          lidar_train  RUNNING    1:35:12                cs003
           22763_1     gpuqm                        lidar_distill  RUNNING    1:35:12                cs003
           22763_2     gpuqm                        lidar_distill  RUNNING    1:35:12                cs001
           22764_1     gpuqm                        lidar_distill  RUNNING    1:35:12                cs003
           22761_0     gpuqm                        lidar_uniform  RUNNING    1:35:13                   g8
```
