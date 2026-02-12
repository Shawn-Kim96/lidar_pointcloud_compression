# Stage 2.1 Related Work Scan (Range-Image / Projection Learned Compression)

## Scope and goal
This note focuses on prior work that is directly relevant to our Stage2.1 design choices:
- 2D projection/range-image LiDAR compression,
- neural image-compression backbones we can port to range images,
- point-cloud-native references that bound expectations for geometry fidelity and real bitstreams.

## Papers reviewed (8)

| Paper | Representation | Backbone / Codec Type | Quantization / Rate Modeling | Relevance to Stage2.1 |
|---|---|---|---|---|
| Beemelmanns et al., **IV 2022** ("3D Point Cloud Compression with Recurrent Neural Network and Image Compression Methods") | Range image | Conv + recurrent (LSTM/GRU) iterative codec | Binarized iterative codes; bitrate controlled by bottleneck + iteration count | Strong projection-based reference used in our T3 comparison |
| Lu et al., **CVPR 2022** ("RIDDLE: Learned Lossless Compression of LiDAR Point Cloud Data via Range Image Delta Prediction") | Range-image delta stream | Learned predictive lossless coder | Entropy coding over predicted residual symbols | Shows projection-based route can also target lossless coding |
| Bian et al., **arXiv 2023** ("BIRD-PCC: Bi-directional Range Image-based Deep Point Cloud Compression") | Bi-directional range images | Learned bidirectional projection codec | Variable-rate operating points via model/rate settings | Directly aligned with our projection strategy and adaptive-rate goal |
| Zaoui et al., **arXiv 2025** ("Range Image-based Implicit Neural Representation for Efficient Point Cloud Compression") | Range image | Implicit neural representation (INR) | Compression through compact learned representation parameters | Suggests alternate lightweight decoder path at low bitrates |
| Wiesmann et al., **RA-L 2021** ("Deep Compression for Dense Point Cloud Maps", DePoCo) | Point-cloud-native (not 2D projection) | Sparse point encoder/decoder + learned transforms | Stores compressed package (latent + transforms/scales), memory-aware evaluation | Useful contrast: stronger geometry handling but different representation assumptions |
| Toderici et al., **arXiv 2016** ("Full Resolution Image Compression with Recurrent Neural Networks") | 2D image patches | Recurrent encoder/decoder iterative refinement | Stochastic binarization at train, deterministic sign at inference | Architectural ancestor of 3DPCC recurrent range-image codecs |
| Ballé et al., **ICLR 2018** ("Variational Image Compression with a Scale Hyperprior") | 2D images | Analysis/synthesis transforms + hyperprior | Learned entropy model with hyperlatent side info | Baseline for modern learned rate modeling we can port |
| Minnen et al., **NeurIPS 2018** ("Joint Autoregressive and Hierarchical Priors for Learned Image Compression") | 2D images | Hyperprior + autoregressive context model | Stronger entropy model; improved R-D vs hyperprior-only | Guides a practical entropy-model upgrade path for Stage1/Stage2.1 |

## Comparison summary

### What projection-based LiDAR papers are doing that our Stage1 did not
- Better rate-control mechanisms than fixed per-sample affine quantization.
- Either iterative coding (RNN/binarization path) or stronger entropy models.
- More explicit framing of operating points (bitrate/quality sweeps).

### What our current Stage1 already does reasonably
- SemanticKITTI-compatible range-image pipeline.
- Multi-stage residual encoder/decoder with configurable capacity.
- Clean reproducibility path with run manifests/checkpoints.

### Gap we must close (and are now closing in Stage2.1)
- Stage1 had no learned importance map for inference-time adaptive allocation.
- Stage2 depended on GT ROI during inference, which is not deployable.
- Task awareness was proxy-head based, not detector/teacher distillation.

## Conclusion and decision
- **Decision:** keep Stage1 as internal control baseline ("ours-uniform"), but **upgrade the primary Stage2.1 path** with:
  1) teacher-guided distillation objective,
  2) learned importance head for deployment without labels,
  3) adaptive quantization driven by soft importance map.
- **Rationale:** this preserves comparability while moving from scaffold-level task awareness to a deployable task-aware method.

## Links
- 3DPCC (IV 2022): https://arxiv.org/abs/2205.10331
- RIDDLE (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Lu_RIDDLE_Learned_Lossless_Compression_of_LiDAR_Point_Cloud_Data_via_Range_CVPR_2022_paper.html
- BIRD-PCC (2023): https://arxiv.org/abs/2309.10801
- Range-Image INR (2025): https://arxiv.org/abs/2506.08689
- DePoCo (RA-L 2021): https://arxiv.org/abs/2102.00380
- Toderici et al. (2016): https://arxiv.org/abs/1608.05148
- Ballé et al. (2018): https://arxiv.org/abs/1802.01436
- Minnen et al. (2018): https://arxiv.org/abs/1809.02736
