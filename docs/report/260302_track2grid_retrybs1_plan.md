# Track 2 Grid Retry (Batch Size 1)

- Source results: `logs/260302_track2grid_results.csv`
- Retry reason: initial jobs failed with CUDA OOM or device-unavailable errors.
- Policy: same hyperparameters, retry with `batch_size=1`.

## Retry rows

- original_job=23346 run_name=`260302_track2grid_refine_h128_lr2em4_e120_refine` head=`refine` hidden=`128` epochs=`120` lr=`2e-4`
- original_job=23349 run_name=`260302_track2grid_refine_h192_lr1em4_e180_refine` head=`refine` hidden=`192` epochs=`180` lr=`1e-4`
- original_job=23355 run_name=`260302_track2grid_deep_h96_lr2em4_e180_deep` head=`deep` hidden=`96` epochs=`180` lr=`2e-4`
- original_job=23359 run_name=`260302_track2grid_deep_h128_lr2em4_e180_deep` head=`deep` hidden=`128` epochs=`180` lr=`2e-4`
- original_job=23360 run_name=`260302_track2grid_deep_h192_lr1em4_e120_deep` head=`deep` hidden=`192` epochs=`120` lr=`1e-4`
- original_job=23361 run_name=`260302_track2grid_deep_h192_lr1em4_e180_deep` head=`deep` hidden=`192` epochs=`180` lr=`1e-4`
- original_job=23362 run_name=`260302_track2grid_deep_h192_lr2em4_e120_deep` head=`deep` hidden=`192` epochs=`120` lr=`2e-4`
- original_job=23363 run_name=`260302_track2grid_deep_h192_lr2em4_e180_deep` head=`deep` hidden=`192` epochs=`180` lr=`2e-4`
