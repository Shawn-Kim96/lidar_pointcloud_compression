## Questions
- [ ] baseline is an internal control (autoencoder, simple decoder format), not an external SOTA baseline.
- [ ] External baselines (GPCC/Draco + learned point-cloud codecs) should be added at paper-writing stage via evaluation pipelines.
- [ ] Current Stage2 uses GT-derived ROI masks and a proxy task loss; it validates plumbing but is not yet "detector-aware".
- [ ] Recommended direction: B (teacher distillation) + A (learned importance map for deployability).


## Feedback
- Use object detection model / segmentation can also be helpful
- Pointpillar를 이용해 detection 할 때, detection acc에 집중해라. -> Point cloud 데이터가 clean 하지 않아서, preprocessing이 필요할 수 있다. -> accuracy가 정확하지 않으면, preprocessing을 확인해야 될 수 있다.
  - frames of the point cloud 를 보고, key frames가 깔끔한지 좀 봐라.
  - k frames를 보고 괜찮은 데이터인지 한번 확인해보라.
- Stage 2 에서는 ROI estimator랑 encoder가 합쳐지는게 제일 중요할거 같고, ROI estimator에는 여러 hyperparameter가 있을거다. ROI 사이즈, moving window, step size 등이 parameterize 될 수 있고, threshold (특정 threshold를 넘으면 high importance를 줄 수 있다 이런식으로)
- Overleaf 쓰냐?
- Encoder, decoder를 쓸 떄 여러 accuracy metrics가 있는데, 다양한 accuracy를 확인할 수 있는 방법을 찾아봐라. Accuracy of ROI 도 중요할듯
- keep backup of metrics for results by hyperparameter
- Check the intermediate result -> ex. result 가 좋고 intermediate result가 좋지 않은 적이 있는데, intermediate result이 reasonable한지도 체크를 해야된다.
- 중간 결과를 좀 볼 수 있는 이미지를 생성하면 좋을거 같다.
- 다른 baseline 모델을 이용해서 시작하는게 좋을 수 도 있다.
- Adaptive Quantization에서 sample 별 정규화를 하는데, 이게 샘플 전체 기준으로 정규화를 하면, 채널마다 분포가 다른 경우엔 최적이 아닐 수 있어. (하지만 구현 단순/안정성 좋음)
  - z_min/z_max를 샘플 전체로 할지, 채널별로 할지 (채널별이 RD 성능은 좋아질 때가 많음)
  - M resize는 bilinear가 맞는지 (segmentation mask 성격이면 nearest를 선호하기도 함(경계 유지))
  - round 후 clamp(min=2) (필수 안전장치 OK)
  - level_map의 분포 (대부분이 bg로 쏠리면 효과 적음 → M 학습/정의 점검)
  - bitrate 추정 (지금은 level만 바꾸는 quantizer라서 실제 entropy coding에서 이득이 얼마나 나는지(분포가 실제로 더 peak해지는지) 확인 필요)