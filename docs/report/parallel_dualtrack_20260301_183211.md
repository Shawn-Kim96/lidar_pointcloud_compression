# Parallel Dual-Track Launch (20260301_183211)

- updated_at: `2026-03-01 19:49:23 PST`
- host: `coe-hpc2.hpc.coe`
- git_head: `5862d0b`
- launch_tag: `parallel_dualtrack_20260301_183211`
- manifest: `logs/launches/parallel_dualtrack_20260301_183211.csv`

## Submitted Jobs

| job_id | job_type | script | params | notes |
| --- | --- | --- | --- | --- |
| 22757 | trackb_eval_running | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2` | already running before this launch |
| 22761 | stage0_uniform_array | `src/scripts/run_uniform_baseline.sh` | `array=0-5;dataset=kitti3dobject;epochs=80;batch=4` | independent |
| 22762 | stage1_array | `src/scripts/run_stage1.sh` | `array=0-3;dataset=kitti3dobject;epochs=80;head=pp_lite64` | independent |
| 22763 | stage2_darknet_array | `src/scripts/run_stage2.sh` | `array=0-2;backbone=darknet;dataset=kitti3dobject;epochs=80` | independent |
| 22764 | stage2_resnet_array | `src/scripts/run_stage2.sh` | `array=0-2;backbone=resnet;dataset=kitti3dobject;epochs=80` | independent |
| 22777 | stage2fix_train_array | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0-11;backbone=resnet;width=64;epochs=80` | independent |
| 22778 | stage2fix_post | `src/scripts/submit_stage2_distill_fix_twoexp_sweep.sh` | `depends_on=22777;width=64` | auto-summary |
| 22779 | stage2fix_train_array | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0-11;backbone=resnet;width=96;epochs=80` | independent |
| 22780 | stage2fix_post | `src/scripts/submit_stage2_distill_fix_twoexp_sweep.sh` | `depends_on=22779;width=96` | auto-summary |
| 22781 | stage2fix_train_array | `src/scripts/run_stage2_distill_fix_twoexp_sweep.sh` | `array=0-11;backbone=resnet;width=128;epochs=80` | independent |
| 22782 | stage2fix_post | `src/scripts/submit_stage2_distill_fix_twoexp_sweep.sh` | `depends_on=22781;width=128` | auto-summary |
| 22783 | stage3_heads_array | `src/scripts/run_stage3_multiscale_heads.sh` | `array=0-4;dataset=semantickitti;backbone=resnet;epochs=80` | independent |
| 22824 | trackb_eval_retry | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;patch=9f8b085` | retry after tuple-loader patch |
| 22825 | trackb_eval_retry | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;patch=5862d0b` | retry after CUDA-busy sanity retry patch |
| 22826 | trackb_eval_retry | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;patch=5862d0b;exclude=cs001,cs002,cs003` | retry with cs-node exclusion for CUDA busy mitigation |
| 22830 | trackb_eval_retry | `src/scripts/run_kitti_map_vs_rate_sbatch.sh` | `KITTI_ROOT_OFFICIAL=/home/018219422/lidar_pointcloud_compression/data/dataset/kitti3dobject;RUN_DIRS=2;patch=5862d0b;nodelist=g7` | retry with pinned node g7 (known OpenPCDet CUDA-op compatible) |

## Queue Snapshot

_Auto-updated: 2026-03-01 20:16:26 _

```text
     JOBID  PARTITION                         NAME     USER ST       TIME NO NODELIST(REASON)
```
