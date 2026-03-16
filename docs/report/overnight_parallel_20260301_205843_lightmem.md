# Overnight Parallel Launch (overnight_parallel_20260301_205843_lightmem)

- Created: 2026-03-01 20:58:43 PST
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/overnight_parallel_20260301_205843_lightmem.csv`
- Goal: maximize overnight throughput under gpuqm without explicit GPU-type request.
- Strategy: memory-light reruns (batch/workers/min-mem/retry tuning) for OOM-prone stages.

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 23010 | overnight_stage1_full_lightmem | `src/scripts/run_stage1.sh` | `partition=gpuqm;array=0-3;dataset=kitti3dobject;epochs=80;batch=2;workers=2` | overnight parallel expansion |
| 23011 | overnight_stage2_full_lightmem | `src/scripts/run_stage2.sh` | `partition=gpuqm;array=0-2;backbone=resnet;dataset=kitti3dobject;epochs=80;batch=2;workers=2` | overnight parallel expansion |
| 23012 | overnight_stage2fix_w64_full_lightmem | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0-11;width=64;epochs=80;batch=2;workers=2;min_gpu_mem_gb=10;retry=5` | reduce OOM + maximize overnight parallelism |
| 23013 | overnight_stage3_full_lightmem | `src/scripts/run_stage3_multiscale_heads.sh` | `partition=gpuqm;array=0-4;backbone=resnet;epochs=80;batch=2;workers=2;min_gpu_mem_gb=14;retry=5` | reduce OOM + maximize overnight parallelism |

## Immediate Queue Snapshot

_Auto-updated: 2026-03-01 20:58:43 PST_

```text
             JOBID PARTITION                                 NAME    STATE       TIME     NODELIST(REASON)
       23013_[0-4]     gpuqm                         lidar_stage3  PENDING       0:00               (None)
      23012_[0-11]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
       23011_[0-2]     gpuqm                        lidar_distill  PENDING       0:00               (None)
       23010_[0-3]     gpuqm                          lidar_train  PENDING       0:00               (None)
           22962_1     gpuqm                          lidar_train  RUNNING      41:38                   g8
           22963_2     gpuqm                        lidar_distill  RUNNING      41:38                  g11
           22966_0     gpuqm                    stage2_fix2_sweep  RUNNING      41:38                  g16
           22962_3     gpuqm                          lidar_train  RUNNING      41:38                  g10
           22965_3     gpuqm                    stage2_fix2_sweep  RUNNING      41:38                  g19
           22965_9     gpuqm                    stage2_fix2_sweep  RUNNING      41:38                cs003
          22965_11     gpuqm                    stage2_fix2_sweep  RUNNING      41:38                  g19
           22966_4     gpuqm                    stage2_fix2_sweep  RUNNING      41:38                  g19
           22967_4     gpuqm                         lidar_stage3  RUNNING      41:38                  g19
```
