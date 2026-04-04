# Result Summary

- experiment_id: EXP-T3-B0-PP16-260402
- experiment_name: t3_b0_pretrained_val256_pp16
- track: track3
- branch_id: t3_b0_pretrained_val256_pp16
- owner_agent: eng3
- stage: endpoint_b0
- objective: Track 3 B0 posQ16 bounded PointPillars endpoint on precomputed recon artifacts
- baseline: t3_b0_pretrained_val256_geometry_ready_recovered
- plan_path: /home/018219422/lidar_pointcloud_compression/docs/report/plans/260402_t3_b0_pp16_plan.md
- manifest_path: /home/018219422/lidar_pointcloud_compression/logs/manifests/EXP-T3-B0-PP16-260402_manifest.json
- raw_log_path: /home/018219422/lidar_pointcloud_compression/logs/260402_t3_b0_pp16_29152.out
- environment_name: lidarcomp311
- config_path: /home/018219422/lidar_pointcloud_compression/third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml

## Status

- status: completed

## Metrics

- headline_metrics: true_bpp=8.6341; AP3D_car_mod=71.8146; map3d_mod_mean=36.2923
- baseline_comparison: reference_map3d_mod_mean=50.2711; map_drop_vs_original=13.9788

## Interpretation

- interpretation: bounded B0 posQ16 endpoint completed cleanly on precomputed Track 3 reconstructions.

## Next Action

- next_action: compare this endpoint against the paired B0 operating point and decide whether B1 promotion is justified once smoke fine-tune checks return.
