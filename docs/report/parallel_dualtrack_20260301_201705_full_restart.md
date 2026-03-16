# Parallel Dual-Track Restart Launch (parallel_dualtrack_20260301_201705_full_restart)

- Created: 2026-03-01 20:17:05 PST
- Git source: origin/main@d28f0ae
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_201705_full_restart.csv`
- Note: Requested plan path `.omx/plans/dual-trak` was not found on remote host.

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22961 | trackb_eval_restart | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `partition=gpuqm;conda=lidarcomp311;dataset=kitti3dobject;run_dirs=1;gpu_spec=none` | restart by user request |
| 22962 | stage1_retry_restart | `src/scripts/run_stage1.sh` | `partition=gpuqm;array=1+3;dataset=kitti3dobject;epochs=80;batch=4;gpu_spec=none` | restart by user request |
| 22963 | stage2_resnet_retry_restart | `src/scripts/run_stage2.sh` | `partition=gpuqm;array=2;backbone=resnet;dataset=kitti3dobject;epochs=80;batch=4;gpu_spec=none` | restart by user request |
| 22964 | stage2fix_w64_retry_restart | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=9+10+11;width=64;epochs=80;batch=4;gpu_spec=none` | restart by user request |
| 22965 | stage2fix_w96_retry_restart | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0+1+2+3+4+5+7+8+9+10+11;width=96;epochs=80;batch=4;gpu_spec=none` | restart by user request |
| 22966 | stage2fix_w128_retry_restart | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0+1+2+3+4+5+8+9+10+11;width=128;epochs=80;batch=4;gpu_spec=none` | restart by user request |
| 22967 | stage3_heads_retry_restart | `src/scripts/run_stage3_multiscale_heads.sh` | `partition=gpuqm;array=0-4;dataset=semantickitti;backbone=resnet;epochs=80;batch=4;gpu_spec=none` | restart by user request |

## Immediate Queue Snapshot

_Auto-updated: 2026-03-01 20:17:05 PST_

```text
             JOBID PARTITION                                 NAME    STATE       TIME     NODELIST(REASON)
       22967_[0-4]     gpuqm                         lidar_stage3  PENDING       0:00               (None)
  22966_[0-5,8-11]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
  22965_[0-5,7-11]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
      22964_[9-11]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
         22963_[2]     gpuqm                        lidar_distill  PENDING       0:00               (None)
       22962_[1,3]     gpuqm                          lidar_train  PENDING       0:00               (None)
             22961     gpuqm                       kitti_map_rate  PENDING       0:00           (Priority)
```
