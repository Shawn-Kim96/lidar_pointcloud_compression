# Parallel Dual-Track Recovery Launch (parallel_dualtrack_20260301_193502_recovery2)

- Created: 2026-03-01 19:35:02 PST
- Git head: 5862d0b
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_193502_recovery2.csv`

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22841 | stage1_retry_failed_idx | `src/scripts/run_stage1.sh` | `array=1+3;dataset=kitti3dobject;epochs=80;batch=4;gpu=a100` | retry failed stage1 tasks from 22762 on a100 |
| 22842 | stage2_resnet_retry_failed_idx | `src/scripts/run_stage2.sh` | `array=2;backbone=resnet;dataset=kitti3dobject;epochs=80;batch=4;gpu=a100` | retry failed stage2 resnet task from 22764 on a100 |
| 22843 | stage2fix_width64_retry_failed_idx | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=9+10+11;width=64;epochs=80;batch=4;gpu=a100` | retry failed width64 tail tasks from 22777 on a100 |
| 22844 | stage2fix_width96_retry_failed_idx | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0+1+2+3+4+5+7+8+9+10+11;width=96;epochs=80;batch=4;gpu=a100` | retry failed width96 tasks from 22779 on a100 |
| 22845 | stage2fix_width128_retry_all_idx | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0-11;width=128;epochs=80;batch=4;gpu=a100` | retry width128 full set from 22781 on a100 |
| 22846 | stage3_heads_retry_all_idx | `src/scripts/run_stage3_multiscale_heads.sh` | `array=0-4;dataset=semantickitti;backbone=resnet;epochs=80;batch=4;gpu=a100` | retry stage3 head sweep from 22783 on a100 with vram guard |
| 22863 | stage1_retry2_failed_idx_p100 | `src/scripts/run_stage1.sh` | `array=1+3;dataset=kitti3dobject;epochs=80;batch=4;gpu=p100` | retry failed stage1 tasks on p100 pool after a100 busy/oom |

## Queue Snapshot

_Auto-updated: 2026-03-01 20:16:26 _

```text
     JOBID  PARTITION                         NAME     USER ST       TIME NO NODELIST(REASON)
```
