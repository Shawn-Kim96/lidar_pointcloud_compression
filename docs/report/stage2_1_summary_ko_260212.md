# Stage2.1 결과 요약 (한국어)

## 한 줄 결론
- `1 epoch` 실험(20988)은 미학습 상태라 성능이 나빴고,
- `20 epoch` 실험(21013)은 정상 수렴해서 full-val(4071프레임)에서 일관된 결과를 보였습니다.

## `ld`, `li`, `be`가 무엇인가?
- 손실식:
  - `L_total = L_recon + be*L_rate + ld*L_distill + li*L_importance`
- `ld` (`lambda_distill`):
  - teacher를 얼마나 강하게 따라갈지 정하는 가중치
- `li` (`lambda_importance`):
  - importance map(ROI 중요도 맵) 학습을 얼마나 강하게 할지 정하는 가중치
- `be` (`beta_rate`):
  - bitrate(압축률) 패널티 가중치

즉,  
- `ld`를 키우면 task-aware(teacher 추종) 성향이 강해지고,  
- `li`를 키우면 ROI/중요영역 중심 학습이 강해지고,  
- `be`를 키우면 더 압축하려는 성향이 강해집니다.

## 왜 epoch 1번만 돌았나?
- `20988` 잡은 sweep 설정이 `epochs=1`이라서 의도대로 1 epoch만 수행되었습니다.
- 이후 `21013` 잡에서 `epochs=20`으로 다시 돌렸고, 정상 완료(`2026-02-11 00:45:12`)했습니다.

## 최종적으로 봐야 할 결과 (job 21013, full-val)

### q8
- `r0 (ld=1.0, li=0.2)`:
  - teacher_drop `-0.129634`, p_BPP `0.984644`, CD `0.430050` (품질 안정적)
- `r1 (ld=1.5, li=0.2)`:
  - teacher_drop `-0.108606`, p_BPP `0.962357`, CD `0.466664`
- `r2 (ld=1.0, li=0.5)`:
  - teacher_drop `-0.132197`, p_BPP `0.949950`, CD `0.484933` (압축/드롭 우세)

### q4
- `r0 (ld=1.0, li=0.2)`:
  - teacher_drop `-0.146993`, p_BPP `0.165417`, CD `1.971965`
- `r1 (ld=1.5, li=0.2)`:
  - teacher_drop `-0.142293`, p_BPP `0.160799`, CD `1.940402`
- `r2 (ld=1.0, li=0.5)`:
  - teacher_drop `-0.198209`, p_BPP `0.157368`, CD `1.822931` (현재 최적)

## 해석
- 품질(왜곡 최소화) 우선이면: `r0`
- 압축률/teacher_drop 우선이면: `r2`
- 현재 기준 `r1`은 명확한 장점이 적습니다.

## 중요한 주의점
- `teacher_drop < 0`는 "proxy teacher 기준 점수 향상"이지,
  곧바로 "실제 detection 성능 향상"을 의미하지는 않습니다.
- 따라서 교수님 피드백대로 반드시 추가 확인이 필요합니다:
  - PointPillars mAP/recall
  - 프레임 샘플 시각화(원본 vs 복원 vs 중요도맵)
  - ROI 영역 정확도(ROI 내부 품질 지표)

## 다음 액션(교수님 피드백 반영)
1. PointPillars 기반 detection metric 파이프라인 추가
2. key frame 시각화 자동 저장
3. ROI 관련 하이퍼파라미터 sweep
   - ROI size, moving window, step size, threshold
4. 하이퍼파라미터별 metric 백업 표준화(csv + summary table)
