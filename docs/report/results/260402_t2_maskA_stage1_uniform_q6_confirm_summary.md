# Result Summary

- experiment_id: EXP-T2-MASKA-S1-Q6-CONFIRM-260402
- experiment_name: t2_maskA_stage1_uniform_q6_confirm
- track: track2
- branch_id: maskA_stage1_uniform_q6_confirm
- owner_agent: eng2
- stage: stage1_confirm
- objective: Track 2 Stage 1 q6 confirmatory rerun on the winning maskA family
- baseline: maskA_stage1_uniform_q6_recovered_0p2565
- plan_path: /home/018219422/lidar_pointcloud_compression/docs/report/plans/260402_t2_maskA_stage1_uniform_q6_confirm_plan.md
- manifest_path: /home/018219422/lidar_pointcloud_compression/logs/manifests/EXP-T2-MASKA-S1-Q6-CONFIRM-260402_manifest.json
- raw_log_path: /home/018219422/lidar_pointcloud_compression/logs/260402_t2_maskA_stage1_uniform_q6_confirm_ep80_compare_fixed_archive_car_ap_summary.csv
- environment_name: lidarcomp311+rangedet39
- config_path: /home/018219422/lidar_pointcloud_compression/data/results/experiments/260402_t2_maskA_stage1_uniform_q6_confirm_train/config.yaml
- compare_archive_csv: /home/018219422/lidar_pointcloud_compression/logs/260402_t2_maskA_stage1_uniform_q6_confirm_ep80_compare_fixed_archive_car_ap_summary.csv
- compare_official_csv: /home/018219422/lidar_pointcloud_compression/logs/260402_t2_maskA_stage1_uniform_q6_confirm_ep80_compare_fixed_kitti_official_summary.csv

## Status

- status: completed_confirmed

## Metrics

- headline_metrics: AP3D@0.3=0.2492; AP3D@0.5=0.1266; AP3D@0.7=0.0127; meanBestIoU3D@0.3=0.3170
- baseline_comparison: vs recovered Stage1 AP3D@0.3 -0.0073 (0.2492 vs 0.2565); vs maskA_stage0_fullscale +0.0615 (0.2492 vs 0.1877), retention 132.8%

## Interpretation

- interpretation: confirm rerun cleared the thesis closure gate and stayed well within the +/-0.05 tolerance from the recovered anchor. The originally submitted compare job only ingested the first archive because comma-separated ARCHIVES_CSV was passed through sbatch --export; the corrected GPU compare rerun (job 29195) produced the auditable final metrics above.

## Next Action

- next_action: launch the bounded Stage 2 adaptive baseline A0 from the confirmed Stage 1 checkpoint and keep the same frozen evaluator bundle.
