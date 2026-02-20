# Stage1/Stage2 UML and Flow Diagrams (PlantUML)

이 문서는 논문/교수님 공유용으로 Stage1, Stage2 실험 흐름을 PlantUML 문법으로 정리한 자료입니다.

## 1) End-to-End Experiment Flow

```plantuml
@startuml
title Stage1/Stage2 End-to-End Experiment Flow

start
:SemanticKITTI (.bin/.label);
:Range Projection\n5-channel input\n(range, intensity, x, y, z);

fork
  :Train Loader\nseq 00,01,02,03,04,05,06,07,09,10;
fork again
  :Val/Test Loader\nseq 08;
end fork

if (Training Stage?) then (Stage1)
  :Backbone Sweep\n(darknet/resnet, lr sweep);
else (Stage2)
  :Darknet Fixed\n(lambda_distill sweep);
endif

:Encoder Backbone;
:Feature Projection\nto latent 64;
:Importance Head;
:Adaptive Quantizer\nbg_levels=16, roi_levels=256;
:Decoder;
:Reconstructed 5-channel range image;

:Loss Computation;
:Optimizer Step;
:Epoch Logs + Checkpoints;

:Post Analysis\n- Convergence curves\n- Intermediate visuals\n- Proxy detection score;
stop
@enduml
```

## 2) Model Component Diagram

```plantuml
@startuml
title Stage1/Stage2 Model Components
skinparam classAttributeIconSize 0

class SemanticKittiDataset {
  +__getitem__()
  +do_range_projection()
  +returns data[5,H,W], valid_mask, roi_mask
}

class LidarCompressionModel {
  +backbone
  +feature_projection
  +importance_head
  +quantizer
  +decoder
  +forward(x, noise_std, quantize, importance_map)
}

class AdaptiveQuantizer {
  +roi_levels
  +bg_levels
  +forward(latent, importance_map)
  +returns dequant, codes, level_map
}

class Trainer {
  +train_epoch()
  +run()
  +loss = recon + rate + importance + distill
}

class TeacherAdapter {
  +backend: proxy/openpcdet
  +infer()
}

SemanticKittiDataset --> Trainer : provides batches
Trainer --> LidarCompressionModel : trains
LidarCompressionModel *-- AdaptiveQuantizer : uses
Trainer o-- TeacherAdapter : optional teacher
@enduml
```

## 3) Training Step Sequence (Stage1 vs Stage2)

```plantuml
@startuml
title Stage1 vs Stage2 Training Step Sequence

participant DataLoader as DL
participant Trainer as T
participant LidarCompressionModel as M
participant AdaptiveQuantizer as Q
participant TeacherAdapter as TA

DL -> T: batch(data, valid_mask, roi_mask)

alt Stage2 (teacher enabled)
  T -> TA: infer(data)
  TA --> T: teacher_out
else Stage1 (teacher disabled)
  note over T
    teacher_out = {}
  end note
end

T -> M: forward(data, quantize=True, importance_map=None)
M -> Q: quantize(latent, importance_map_pred)
Q --> M: latent_deq, codes, level_map
M --> T: recon, aux

T -> T: loss_recon (MSE)
T -> T: loss_rate (mean(level_map))
T -> T: loss_importance (BCE with roi_mask)

note over T
current code status:
distill loss branch exists,
but actual distill term is not applied (pass)
end note

T -> T: total_loss backward + optimizer step
@enduml
```

## 4) Experiment Sweep Structure

```plantuml
@startuml
title Stage0/Stage1/Stage2 Sweep Design
left to right direction

rectangle "Stage0 Sweep (Uniform Baseline)" as S0 {
  rectangle "backbone x qbits\n(ROI-unaware)" as S0A
}

rectangle "Stage1 Sweep" as S1 {
  rectangle "Run0\n darknet lr1e-4" as S1A
  rectangle "Run1\n resnet lr1e-4" as S1B
  rectangle "Run2\n darknet lr5e-5" as S1C
  rectangle "Run3\n resnet lr5e-5" as S1D
}

rectangle "Stage2 Sweep" as S2 {
  rectangle "Run0\n darknet ld=0.5" as S2A
  rectangle "Run1\n darknet ld=1.0" as S2B
  rectangle "Run2\n darknet ld=0.1" as S2C
}

rectangle "Compare convergence\nand final loss" as M
rectangle "Visualize\nintermediate outputs" as V
rectangle "Detection proxy\ncomparison" as D

S0 --> M
S1 --> M
S2 --> M
M --> V
M --> D
@enduml
```

## 5) Figure Mapping for Slides/Thesis

- Figure A: End-to-End Experiment Flow
- Figure B: Model Component Diagram
- Figure C: Stage1 vs Stage2 Training Sequence
- Figure D: Stage0/1/2 Sweep Design Matrix

위 4개를 먼저 넣고, 다음으로 노트북에서 생성한 정량 그래프(loss curve, final loss bar, intermediate panel)를 연결하면 발표 자료 구조가 깔끔합니다.
