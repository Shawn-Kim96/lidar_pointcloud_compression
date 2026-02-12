# Stage 2 Transition Notes

## 1) Why Stage 2
Stage 1 uses uniform compression over the whole frame.  
Stage 2 adds adaptive compression so important regions get higher fidelity at similar bitrate.

## 2) Stage 2 Goal (from spec)
- Adaptive quantization by ROI (fine for ROI, coarse for background)
- Task-aware loss integration (PointPillars-driven objective)
- Compare Stage 2 vs Stage 1 at similar BPP

## 3) Baseline Reference for Stage 2
- Baseline model: `data/results/checkpoints/stage1_baseline.pth`
- Baseline eval record: `logs/stage1_eval_baseline.out`
- Baseline split policy:
  - Train: `00,01,02,03,04,05,06,07,09,10`
  - Val: `08`
  - Test (future benchmark): `11-21`

## 4) Stage 2 Immediate Work Package
- Start with `S2-T1`: ROI mask data loading from `.label`
- Then `S2-T2`: adaptive quantizer module
- Then task loss integration and joint training (`S2-T3`, `S2-T4`)

## 5) Success Criteria for Stage 2
- At similar BPP, Stage 2 should improve detection-oriented quality (mAP) over Stage 1.
- Evaluation should report `mAP_3d` and `p_BPP`.
