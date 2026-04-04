# Result Summary

- experiment_id: EXP-T2-MASKA-S2-ADAPTIVE-A0-260403
- experiment_name: t2_maskA_stage2_adaptive_a0
- track: track2
- branch_id: maskA_stage2_adaptive_a0
- owner_agent: eng2
- stage: stage2_adaptive
- objective: First bounded Stage 2 adaptive baseline on the confirmed maskA family
- baseline: t2_maskA_stage1_uniform_q6_confirm
- plan_path: /home/018219422/lidar_pointcloud_compression/docs/report/plans/260403_t2_maskA_stage2_adaptive_a0_plan.md
- manifest_path: /home/018219422/lidar_pointcloud_compression/logs/manifests/EXP-T2-MASKA-S2-ADAPTIVE-A0-260403_manifest.json
- raw_log_path: /home/018219422/lidar_pointcloud_compression/logs/260403_t2_maskA_stage2_adaptive_a0_ep80_compare_archive_car_ap_summary.csv
- environment_name: lidarcomp311+rangedet39
- config_path: /home/018219422/lidar_pointcloud_compression/data/results/experiments/260403_t2_maskA_stage2_adaptive_a0_train/config.yaml
- compare_archive_csv: /home/018219422/lidar_pointcloud_compression/logs/260403_t2_maskA_stage2_adaptive_a0_ep80_compare_archive_car_ap_summary.csv
- compare_official_csv: /home/018219422/lidar_pointcloud_compression/logs/260403_t2_maskA_stage2_adaptive_a0_ep80_compare_kitti_official_summary.csv

## Status

- status: completed
- classification: success

## Metrics

- headline_metrics: AP3D@0.3=0.2717; AP3D@0.5=0.1539; AP3D@0.7=0.0257; meanBestIoU3D@0.3=0.3611; avg_det_per_frame=7.9414
- baseline_comparison: vs confirmed Stage1 AP3D@0.3 +0.0225 (0.2717 vs 0.2492); AP3D@0.5 +0.0273 (0.1539 vs 0.1266); meanBestIoU3D@0.3 +0.0441 (0.3611 vs 0.3170); avg_det_per_frame -0.2031 (7.9414 vs 8.1445); train rate_proxy 18.2034 vs 63.0000; eq_bits 4.1843 vs 6.0000

## Interpretation

- interpretation: A0 meets the Track 2 Stage 2 success gate. It lowered effective rate relative to the confirmed Stage 1 baseline while improving AP3D@0.3, AP3D@0.5, and meanBestIoU3D@0.3 under the same frozen evaluator bundle.

## Next Action

- next_action: stop Track 2 branch expansion at A0 and use this as the canonical Stage 2 adaptive-positive result; do not proceed to A1 or B1 remediation.
