# Current Results Summary

This file keeps the smallest set of high-value results that explain the current project state.

## Track 1

### Identity-domain PointPillars reference on `KITTI_Identity`

| Experiment | `mAP3D(mod)` |
|---|---:|
| `pp_ft_track1nq_a_baseline_260304_230724` | `73.5935` |
| `pp_ft_track1nq_b_enhanced_260304_230724` | `73.5501` |
| `pp_ft_t1nq_geo_rf_geo_rf_260305_215136` | `73.1277` |
| `pp_ft_t1nq_geo_fr_geo_fr_260305_215136` | `73.6671` |
| `pp_ft_t1nq_impl_a_impl_base_260305_222216` | `73.1809` |
| `pp_ft_t1nq_impl_b_impl_ms_260305_222216` | `73.1609` |
| `pp_ft_t1nq_pillar_a_pillar_a_260309_201414` | `72.8484` |
| `pp_ft_t1nq_pillar_b_pillar_b_260309_201414` | `73.4484` |

### Reconstructed point-cloud endpoint after no-quant codec

| Experiment | Identity reference `map3d_mod_mean` | Reconstructed `map3d_mod_mean` |
|---|---:|---:|
| `track1nq_a_baseline_260304_230724` | `51.5667` | `0.000251` |
| `track1nq_b_enhanced_260304_230724` | `54.3868` | `0.000000` |
| `t1nq_geo_rf_geo_rf_260305_215136` | `53.8357` | `0.021675` |
| `t1nq_geo_fr_geo_fr_260305_215136` | `54.4261` | `0.150146` |
| `t1nq_impl_a_impl_base_260305_222216` | `54.6734` | `0.811525` |
| `t1nq_impl_b_impl_ms_260305_222216` | `53.5699` | `0.019433` |
| `t1nq_pillar_a_pillar_a_260309_201414` | `54.7708` | `0.018513` |
| `t1nq_pillar_b_pillar_b_260309_201414` | `52.7378` | `2.192242` |

### Track 1 interpretation

- The identity-domain PointPillars detector itself is stable around `~73 mAP3D(mod)`.
- The reconstructed endpoint is still the bottleneck.
- The best completed reconstructed Track 1 run so far is `pillar_b`, but `2.19` is still far from the identity-domain reference.

## Track 2

### Repaired RangeDet baseline and reconstructed RI evaluation

Metric source:

- [`260316_rangedet_archive_car_ap_summary_rddecodefixfull.csv`](/home/018219422/lidar_pointcloud_compression/logs/260316_rangedet_archive_car_ap_summary_rddecodefixfull.csv)

| Setting | `AP3D@0.3` | `AP3D@0.5` | `AP3D@0.7` | `APBEV@0.3` | `meanBestIoU3D` |
|---|---:|---:|---:|---:|---:|
| `raw/basic` | `0.5700` | `0.4979` | `0.2435` | `0.5777` | `0.6098` |
| `Stage0 baseline` | `0.0215` | `0.0028` | `0.0000` | `0.0272` | `0.1202` |
| `Stage0 enhanced` | `0.0099` | `0.0010` | `0.0000` | `0.0134` | `0.1040` |
| `Stage1 baseline` | `0.0083` | `0.0007` | `0.0000` | `0.0107` | `0.0874` |
| `Stage1 enhanced` | `0.0050` | `0.0003` | `0.0000` | `0.0069` | `0.0704` |

### Track 2 interpretation

- The repaired `RangeDet` raw/basic baseline is now strong enough to use as a real reference.
- The main Track 2 bottleneck is now clearly the codec.
- `Stage0` already collapses relative to raw/basic.
- `Stage1` is worse than `Stage0`.
- The current `enhanced` codec is not helping Track 2.
