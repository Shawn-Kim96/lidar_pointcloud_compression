# Overnight Stage3 Ultralight Retry (overnight_stage3_20260301_210017_ultralight_retry)

- Created: 2026-03-01 21:00:17 PST
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/overnight_stage3_20260301_210017_ultralight_retry.csv`
- Reason: previous stage3 lightmem run failed all indices due `insufficient GPU memory`.

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 23035 | overnight_stage3_ultralight_retry | `src/scripts/run_stage3_multiscale_heads.sh` | `partition=gpuqm;array=0-4;backbone=resnet;epochs=80;batch=1;workers=1;min_gpu_mem_gb=8;retry=6` | retry after insufficient_mem failures |

## Immediate Queue Snapshot

_Auto-updated: 2026-03-01 21:00:17 PST_

```text
             JOBID PARTITION                                 NAME    STATE       TIME     NODELIST(REASON)
       23035_[0-4]     gpuqm                         lidar_stage3  PENDING       0:00               (None)
           23011_0     gpuqm                        lidar_distill  RUNNING       1:33                  g18
           23012_0     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g18
           23010_1     gpuqm                          lidar_train  RUNNING       1:33                  g13
           23010_2     gpuqm                          lidar_train  RUNNING       1:33                  g14
           23010_3     gpuqm                          lidar_train  RUNNING       1:33                  g15
           23011_1     gpuqm                        lidar_distill  RUNNING       1:33                  g16
           23011_2     gpuqm                        lidar_distill  RUNNING       1:33                  g16
           23012_1     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_2     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                cs002
           23012_3     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_4     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_5     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g18
           23012_6     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_7     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_8     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
           23012_9     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g18
          23012_10     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                  g16
          23012_11     gpuqm                    stage2_fix2_sweep  RUNNING       1:33                   g7
           23010_0     gpuqm                          lidar_train  RUNNING       1:34                   g7
           22962_1     gpuqm                          lidar_train  RUNNING      43:12                   g8
           22963_2     gpuqm                        lidar_distill  RUNNING      43:12                  g11
           22966_0     gpuqm                    stage2_fix2_sweep  RUNNING      43:12                  g16
           22962_3     gpuqm                          lidar_train  RUNNING      43:12                  g10
           22965_3     gpuqm                    stage2_fix2_sweep  RUNNING      43:12                  g19
           22965_9     gpuqm                    stage2_fix2_sweep  RUNNING      43:12                cs003
          22965_11     gpuqm                    stage2_fix2_sweep  RUNNING      43:12                  g19
           22966_4     gpuqm                    stage2_fix2_sweep  RUNNING      43:12                  g19
           22967_4     gpuqm                         lidar_stage3  RUNNING      43:12                  g19
```
