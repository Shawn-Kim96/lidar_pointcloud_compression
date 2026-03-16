# Remaining Tasks Launch (remaining_stage2fix_20260302_184202_ultralight)

- Created: 2026-03-02 18:42:02 PST
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/remaining_stage2fix_20260302_184202_ultralight.csv`
- Objective: execute remaining failed stage2-fix tasks from previous runs.

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 23214 | remaining_w64_idx2_ultralight | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=2;width=64;epochs=80;batch=1;workers=1;min_gpu_mem_gb=8;retry=6` | run remaining failed task |
| 23215 | remaining_w96_failed_idx_ultralight | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=0+1+2+4+5+7+8+10;width=96;epochs=80;batch=1;workers=1;min_gpu_mem_gb=8;retry=6` | run remaining failed tasks |
| 23216 | remaining_w128_failed_idx_ultralight | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `partition=gpuqm;array=1+2+3+5+8+9+10+11;width=128;epochs=80;batch=1;workers=1;min_gpu_mem_gb=8;retry=6` | run remaining failed tasks |

## Immediate Queue Snapshot

_Auto-updated: 2026-03-02 18:42:02 PST_

```text
             JOBID PARTITION                                 NAME    STATE       TIME     NODELIST(REASON)
23216_[1-3,5,8-11]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
23215_[0-2,4-5,7-8     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
         23214_[2]     gpuqm                    stage2_fix2_sweep  PENDING       0:00               (None)
           23012_1     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
           23012_3     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
           23012_4     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
           23012_6     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
           23012_7     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
           23012_8     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
          23012_10     gpuqm                    stage2_fix2_sweep  RUNNING   21:43:18                  g16
```
