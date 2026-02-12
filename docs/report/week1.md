# Week 1 Notes: Stage1-Stage2 Baseline / Codec / Task-Aware Direction

## 1) Baseline Concern: "From Scratch" Is It OK?
### What Stage1 baseline means in THIS repo
- Stage1 baseline is an internal reference point ("ours-uniform"), not an attempt to match a specific paper's reported numbers.
- Purpose: validate the pipeline (data -> model -> metrics) and provide a controlled comparison target for Stage2.

### What baseline means in the THESIS/PAPER
For the thesis, the baseline set should include:
- External baselines: classical codecs (e.g., MPEG-GPCC, Draco) and representative learned point-cloud codecs (as scope allows).
- Internal baselines: Stage1 uniform model ("ours-uniform") and Stage2 adaptive/task-aware ("ours-adaptive").

### Your worry (good) and the pragmatic answer
Your worry: training from scratch might never reach a good operating point.
Pragmatic answer:
- Stage1-from-scratch is acceptable for an internal baseline.
- For publication-grade strength, we either:
  1) start from a stronger pretrained backbone (transfer learning), or
  2) use a strong pretrained detector as a teacher (task signal) so Stage2 is anchored by a strong target.


## 2) Range Image vs "Direct Codec": Concrete Difference
This is the key conceptual split:

### Representation (what we feed into the model)
- Range image representation: project a point cloud into a fixed 2D grid (H=64, W=1024) with channels (range, intensity, xyz, etc.).
- Direct point-cloud representation: an unordered set of points (x,y,z,attributes) without a fixed grid.

### Codec (what "compression" actually means)
A codec implies:
- Encoder produces a bitstream (or symbols that can be entropy-coded).
- Decoder reconstructs a signal from that bitstream.
- Rate is measured in bits (file size) or a validated entropy estimate.
- Distortion is measured against the original (e.g., Chamfer, PSNR, task metrics).

### Classical baselines (GPCC/Draco)
- Work directly on point clouds and produce real bitstreams.
- Evaluation is straightforward:
  1) compress (bitstream size -> bits),
  2) decompress (reconstructed point cloud),
  3) compute distortion (CD, etc.),
  4) compute rate (bits per point).

### Our Stage1/2 baseline (range-image AE)
- Works on range images (a chosen representation).
- Does not yet produce a real arithmetic-coded bitstream; rate is estimated from latent-code entropy (BPP estimate).
- Still useful for research iteration, but we must be explicit in writing:
  - "estimated BPP" vs "true bitstream BPP".

### What "run external codecs in our evaluation script" means (concretely)
It means adding an evaluation path that:
- takes raw point clouds,
- runs GPCC/Draco to generate a compressed file,
- measures file size for rate,
- decodes back to a point cloud,
- computes the same distortion/task metrics as our method.

This gives an external baseline curve without changing our learning pipeline.

### What "reproduce a codec and add ROI" means (why it's non-trivial)
It is not "swap a loss".
It means changing the codec so it allocates more bits to ROI points and fewer to background:
- classical codecs: local quantization / segmentation-based coding / per-region QP control,
- learned codecs: importance map + conditional quantization and entropy models.


## 3) What Stage2 Does Today (In This Repo)
Current Stage2 implementation is closer to:
"ROI is given (from labels) -> adaptive quantization uses ROI -> train reconstruction + a proxy task loss".

### ROI source today
- ROI mask is derived from SemanticKITTI `.label` (GT semantics) and projected to the range image.
- This is a *training-time supervision signal*, not something available at deployment.

### "Task loss" today (important limitation)
- The current `TaskLossModule` uses a small proxy head that predicts ROI mask from the reconstructed range image.
- OpenPCDet import is detected, but the full PointPillars loss path is not wired yet.
- So the current "mAP" is a proxy AP over ROI-mask prediction, NOT true 3D detection mAP.

Why this was added:
- To keep Stage2 end-to-end differentiable and runnable on HPC without dependency explosions.
- To validate that task-weighting + ROI-weighted reconstruction can change training dynamics.

Limitation to communicate clearly:
- This is a scaffold; for thesis-grade Stage2, the task signal must become real detector-based.


## 4) Which Stage2 Direction Fits the Research (A/B/C)
You asked whether we should:
1) do end-to-end: compress -> decompress -> detector -> detector loss backprop, or
2) predefine ROI and only train compress/decompress with ROI weighting.

### Option A: Learned importance/ROI map (no GT at runtime)
- Train a small "importance head" to predict a per-pixel importance map from the input (or early features).
- Use it to allocate bits (adaptive quantization) during inference.
- Training signal can come from GT semantics OR a teacher detector.
Pros: deployable (no GT ROI needed).
Cons: you must define what "importance" is and how to supervise it.

### Option B: Teacher distillation (recommended core for thesis)
- Use a pretrained detector as a frozen teacher (OpenPCDet PointPillars, etc.).
- Define task loss as matching teacher outputs:
  - teacher logits / heatmaps / intermediate BEV features (more stable than NMS boxes).
- Backprop flows into the compressor/decoder; teacher stays frozen.
Pros: strong task signal without training detector; closer to "task-aware compression".
Cons: requires integrating teacher inference in training; heavier compute.

### Option C: Fully end-to-end detector training/loss
- Backprop true detection losses (cls/reg) through the detector.
Pros: conceptually clean.
Cons: hard in practice due to non-differentiable steps (NMS) and engineering load; also spec says "do not train detector from scratch".

### Recommendation for this thesis
- Use B as the main task-aware mechanism (frozen pretrained detector as teacher).
- Add A as the *deployment mechanism*: train an importance head that imitates teacher-derived importance, so inference does not require GT ROI.
- Avoid C unless we have a strong reason and enough time budget.


## 5) Proposed Changes (Near-Term)
To turn Stage2 from scaffold to thesis-grade:
1) Replace proxy task loss with a teacher-detector distillation loss (B).
2) Add an importance/ROI prediction head for inference-time bit allocation (A).
3) Keep "ours-uniform" (Stage1) as internal baseline; add external baselines in evaluation.

Minimal viable milestone:
- For validation sequence 08:
  - measure true detector performance on reconstructions using OpenPCDet inference (not training),
  - report mAP vs rate (bitrate from either estimated BPP or true bitstream where possible).


## 6) Talking Points for Weekly Meeting
- Stage1 baseline is an internal control ("ours-uniform"), not an external SOTA baseline.
- External baselines (GPCC/Draco + learned point-cloud codecs) should be added at paper-writing stage via evaluation pipelines.
- Current Stage2 uses GT-derived ROI masks and a proxy task loss; it validates plumbing but is not yet "detector-aware".
- Recommended direction: B (teacher distillation) + A (learned importance map for deployability).
