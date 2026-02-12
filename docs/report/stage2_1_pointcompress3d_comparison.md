# Stage 2.1 Comparison: Our Pipeline vs PointCompress3D (Design Reference)

## Scope
Reference repository inspected: `/home/018219422/pointcompress3d`

This is a design comparison only. We are not replacing our baseline with PointCompress3D components wholesale.

## 1) Architecture depth/capacity comparison

### Our current model
- File: `src/models/autoencoder.py`
- Structure:
  - residual downsample encoder (`num_stages`, `blocks_per_stage`),
  - transpose-conv decoder,
  - latent on full-frame range image (`64x1024 -> 4x64` at stage=4).
- Typical audited capacity:
  - `bp1`: ~1.99M params,
  - `bp2`: ~3.61M params.

### 3DPCC
- Files:
  - `3dpcc/range_image_compression/architectures/Oneshot_LSTM.py`
  - `3dpcc/range_image_compression/utils.py`
- Structure:
  - Conv + ConvLSTM recurrent encoder/decoder,
  - iterative unrolling (`num_iters`),
  - explicit bottleneck setting (`bottleneck`) controlling rate.
- Representation/training style:
  - patch-based (`32x32`) training for range images,
  - progressive reconstruction dynamics.

### DePoCo
- Files:
  - `depoco/depoco/architectures/network_blocks.py`
  - `depoco/depoco/trainer.py`
  - `depoco/depoco/encode.py`, `depoco/depoco/decode.py`
- Structure:
  - point-native encoder/decoder blocks (KPConv-based path + upsampling/transforms),
  - compressed package stores latent/transform information.
- Not projection-range-image-first; stronger geometric handling but different data assumptions.

## 2) Quantization / rate modeling comparison

### Ours
- Per-sample affine quantization (`src/models/autoencoder.py`, `src/models/quantization.py`).
- Stage2/Stage2.1 use adaptive level maps (`bg_levels -> roi_levels`) with entropy-estimated BPP proxies.
- No arithmetic-coded production bitstream yet.

### 3DPCC
- Binarized iterative codes (sign/stochastic binarization training).
- Rate controlled by architecture knobs (`num_iters`, `bottleneck`) rather than a separate entropy model path.

### Draco path in PointCompress3D
- File: `/home/018219422/pointcompress3d/draco/compress.py`
- Produces actual compressed `.drc` bitstreams.
- Useful as external evaluation philosophy reference for true-bitstream rate reporting.

## 3) Training recipe comparison

### Ours
- Full-frame feed-forward training on SemanticKITTI range images.
- Stage2 originally used ROI from labels; Stage2.1 adds learned importance map to remove label dependency at inference.

### 3DPCC
- Patch crops + recurrent unrolling/progressive reconstruction.
- Heavier temporal recurrence in coding path than our feed-forward baseline.

### DePoCo
- Point-cloud map compression with point-native processing and Chamfer-driven training.
- Different target domain (dense maps) and data flow from our range-image codec.

## 4) Upgrade candidates derived from comparison

1. **Increase residual depth moderately (`bp1 -> bp2`)**
   - Benefit: stronger encoder capacity while preserving current training/eval pipeline.
   - Scope: low-medium (config + training sweeps).
   - Risk: increased memory/compute.

2. **Keep feed-forward path but improve rate modeling**
   - Benefit: closer to learned compression literature (hyperprior/context-ready design path).
   - Scope: medium-high.
   - Risk: implementation complexity and training stability.

3. **Introduce lightweight progressive residual refinement head**
   - Benefit: capture part of recurrent/progressive gains without full RNN complexity.
   - Scope: medium.
   - Risk: training-time overhead; potential instability if unrolled too deep.

4. **Deployable learned importance map (implemented in Stage2.1)**
   - Benefit: adaptive quantization at inference without GT labels.
   - Scope: medium.
   - Risk: importance collapse if teacher supervision is weak.

5. **Teacher-feature distillation (implemented scaffold in Stage2.1)**
   - Benefit: task-aware signal beyond reconstruction loss.
   - Scope: medium.
   - Risk: backend dependency (OpenPCDet integration complexity on HPC).

## 5) Decision output
- First upgrade path selected for implementation in Stage2.1:
  1) deployable importance head + adaptive quantization without labels,
  2) teacher distillation scaffolding via adapter + distill loss,
  3) preserve Stage1 as baseline control and use architecture-depth sweeps to scale capacity incrementally.

- Rationale:
  - This gives deployability and task-awareness immediately,
  - avoids a high-risk full architecture rewrite before validating the teacher-driven objective.
