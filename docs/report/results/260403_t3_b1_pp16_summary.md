# Result Summary

- experiment_id: EXP-T3-B1-PP16-260403
- experiment_name: t3_b1_pretrained_val256_pp16
- track: track3
- branch_id: t3_b1_pp16_260403
- owner_agent: eng3
- stage: endpoint_b1
- objective: Track 3 B1 posQ16 bounded PointPillars endpoint on precomputed recon artifacts
- baseline: t3_b1_resume_val256_p16_geometry_completed
- plan_path: /home/018219422/lidar_pointcloud_compression/docs/report/plans/260403_t3_b1_pp16_plan.md
- manifest_path: /home/018219422/lidar_pointcloud_compression/logs/manifests/EXP-T3-B1-PP16-260403_manifest.json
- raw_log_path: /home/018219422/lidar_pointcloud_compression/logs/260403_t3_b1_pp16_29208.out
- environment_name: lidarcomp311
- config_path: /home/018219422/lidar_pointcloud_compression/third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml

## Status

- status: completed

## Metrics

- headline_metrics: true_bpp=8.6323; AP3D_car_mod=72.2564; map3d_mod_mean=36.4479
- baseline_comparison: reference_map3d_mod_mean=50.2865; map_drop_vs_original=13.8386

## Interpretation

- interpretation: bounded B1 posQ16 endpoint completed cleanly on precomputed Track 3 reconstructions.

## Next Action

- next_action: recompute B_ref/drift across B0/B1 at posQ16 and posQ64 and decide whether the adaptive attempt gate remains open.
