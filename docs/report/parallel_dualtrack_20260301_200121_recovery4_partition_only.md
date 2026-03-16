# Parallel Dual-Track Recovery Launch (parallel_dualtrack_20260301_200121_recovery4_partition_only)

- Created: 2026-03-01 20:01:21 PST
- Git head: 5862d0b
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_200121_recovery4_partition_only.csv`

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22922 | trackb_eval_partition_only | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `partition=gpuqm;dataset=kitti3dobject;run_dirs=2;gpu_spec=none` | resubmit per cluster policy (partition-only) |
| 22923 | stage1_retry_partition_only | `src/scripts/run_stage1.sh` | `partition=gpuqm;array=1+3;dataset=kitti3dobject;epochs=80;batch=4;gpu_spec=none` | retry failed stage1 tasks with script-level retry logic |
| 22924 | stage2_resnet_retry_partition_only | `src/scripts/run_stage2.sh` | `partition=gpuqm;array=2;backbone=resnet;dataset=kitti3dobject;epochs=80;batch=4;gpu_spec=none` | retry failed stage2 resnet task with script-level retry logic |
| 22925 | stage2fix_w64_retry_partition_only | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=9+10+11;width=64;epochs=80;batch=4;gpu_spec=none` | retry failed width64 tasks with script-level retry logic |
| 22926 | stage2fix_w96_retry_partition_only | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0+1+2+3+4+5+7+8+9+10+11;width=96;epochs=80;batch=4;gpu_spec=none` | retry failed width96 tasks with script-level retry logic |
| 22927 | stage2fix_w128_retry_partition_only | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0+1+2+3+4+5+8+9+10+11;width=128;epochs=80;batch=4;gpu_spec=none` | retry failed width128 tasks with script-level retry logic |
| 22928 | stage3_heads_retry_partition_only | `src/scripts/run_stage3_multiscale_heads.sh` | `partition=gpuqm;array=0-4;dataset=semantickitti;backbone=resnet;epochs=80;batch=4;gpu_spec=none` | retry stage3 tasks with script-level retry logic + vram guard |

## Queue Snapshot

_Auto-updated: 2026-03-01 20:16:26 _

```text
     JOBID  PARTITION                         NAME     USER ST       TIME NO NODELIST(REASON)
```
