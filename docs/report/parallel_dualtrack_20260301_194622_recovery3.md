# Parallel Dual-Track Recovery Launch (parallel_dualtrack_20260301_194622_recovery3)

- Created: 2026-03-01 19:46:22 PST
- Git head: 5862d0b
- Manifest: `/home/018219422/lidar_pointcloud_compression/logs/launches/parallel_dualtrack_20260301_194622_recovery3.csv`

## Submitted Jobs

| Job ID | Label | Script | Params | Notes |
|---:|---|---|---|---|
| 22878 | stage1_retry3_idx1_3 | `src/scripts/run_stage1.sh` | `array=1+3;dataset=kitti3dobject;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry after GPU binding patch |
| 22879 | stage2_resnet_retry3_idx2 | `src/scripts/run_stage2.sh` | `array=2;backbone=resnet;dataset=kitti3dobject;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry after GPU binding patch |
| 22880 | stage2fix_w64_retry3_idx9_11 | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=9+10+11;width=64;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry after GPU binding patch |
| 22881 | stage2fix_w96_retry3_failed_idx | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0+1+2+3+4+5+7+8+9+10+11;width=96;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry after GPU binding patch |
| 22882 | stage2fix_w128_retry3_failed_pending_idx | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0+1+2+3+4+5+8+9+10+11;width=128;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry failed+canceled-pending after GPU binding patch |
| 22884 | stage3_heads_retry3_all_idx | `src/scripts/run_stage3_multiscale_heads.sh` | `array=0-4;dataset=semantickitti;backbone=resnet;epochs=80;batch=4;gpu=a100;gpu_bind=patched` | retry after GPU binding patch + vram guard |

## Queue Snapshot

_Auto-updated: 2026-03-01 20:16:26 _

```text
     JOBID  PARTITION                         NAME     USER ST       TIME NO NODELIST(REASON)
```
