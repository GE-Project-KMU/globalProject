# 포트홀 v5:10epoch 추가 학습·baseline 유지

추가 학습10epoch와 실제 optimizer update3,334회를 완료했지만 기존 VAL402장에서 개선되지 않았다. 사전 규칙에 따라 비교에 포함한 **epoch0의 v4 baseline을 최종 선택했다.** 마지막 epoch10 VAL AP50은58.0196%로 초기/선택 AP50 68.3948%보다 낮았다. 추가 정확도 개선을 달성했다고 주장하지 않는다. [공개 집계](reports/POTHOLE-REFINEMENT-V5-RESULTS.json)에 epoch별 VAL과 검증 증거를 기록했다.

## 동결한 후속 recipe

v4에서 예정된 no-augmentation/L1 단계가 시작되기 전에 plateau로 종료되었던 점을 조사하는 단일 후보였다. batch16/input640/LR0.00025, SGD momentum0.9·Nesterov·weight decay0.0005(BN/bias 제외), EMA, seed20261105로10epoch를 수행했다. augmentation은 없고 native L1 box regression은 매 update 활성화했다. 초기 v4 checkpoint를 epoch0으로 포함하고 같은 기존 VAL AP50 최고를 선택하며 동률은 이른 epoch를 우선했다. 운영 confidence는 같은 VAL F2 grid의0.1, NMS0.5다. TEST를 사용해 recipe·선택·threshold를 조정하지 않았다.

| 항목 | 결과 |
| --- | --- |
| 실제 추가 학습 | 10epoch,3,334 optimizer updates |
| 최고/초기 VAL AP50 | 68.3948% |
| 마지막 epoch10 VAL AP50 | 58.0196% |
| 선택 epoch | 0, 기존 baseline 유지 |
| 추가 개선 | 없음 |
| 새 독립 holdout | 확보하지 못함 |
| TEST 재추론 | 수행하지 않음 |

## 직렬화와 실제 추론 함수의 동일성

원 checkpoint SHA는`2fb2c082…`, 새 저장 파일은`7632b927…`로 서로 다르다. CPU PyTorch로 두 파일의 `model` state를 읽어 **642개 tensor 모두 key·shape·dtype·rawbytes가 정확히 같은지** 확인했다. shape/dtype/raw 불일치는 각각0이다. 파일 직렬화 hash가 달라도 저장된 모델 tensor가 같다는 사실을 구분한다. optimizer metadata 등 checkpoint의 모든 비모델 metadata가 동일하다는 주장은 아니다.

원/새 ONNX는 실제 바이트가 완전히 같으며 SHA는 모두`9b956d8ead74d1b11db0d1c43bac9ae5bdda2b0d6b843062319ae28de2f2f269`다. 같은 앱 adapter의 입력640, top-left letterbox114·BGR float32·255로 나누지 않는 전처리, pothole 클래스·decoded xywh·confidence0.1·NMS0.5가 동일하다. 최종 classwise NMS→source clip 순서도 같다. 따라서 선택된 실제 prediction function은 v4와 동일하다. 새 manifest의 provenance·epoch·저장 경로 차이를 새 정확도 개선으로 해석하지 않는다.

## 고정 VAL canonical 검증

기존 v4에서 이미 고정했던 positive VAL 사진 한 장을 바꾸지 않고 새 manifest에 대해 canonical verifier를 한 번 실행했다. 앱 ORT1.31.1 CPU, 독립 ORT1.20.1 CPU, PyTorch2.7.1 CPU의 기본 rtol0.001/atol0.0001 raw 비교와 최종 class·NMS 후 bbox/confidence 비교가 모두 통과했다. `numerical_parity_passed`, `postprocess_parity_passed`, `artifact_contract_verified`가 모두true다. 앱↔ORT raw max error0, 앱↔Torch max error0.00486755이며 원래 elementwise 절대·상대 허용오차를 변경하지 않았다.

검증 시점 adapter SHA는`bb5e3590e6d299b494fd9ebfbdd3d3374fdfa7f9b1284180ceaa5f3d3dadba74`, verifier는`a7a349179feb919f1515a9088618f0cf5e7e9904fc7547c4b9a8bb589c8f9794`였고 실행 전후 일치를 확인했다. 이후 runtime guard 수정까지 이 검증에 포함했다고 주장하지 않는다. 전체 source·계획·모델 SHA는 공개 JSON에 있다. 개인별 ID·사진 해시·경로·bbox는 공개하지 않는다. 이 수치/후처리 검사는 새 정확도나 현장 인수 시험을 대신하지 않는다.

## TEST를 재실행하지 않은 이유

모델 tensor·ONNX·추론 계약이 동일하므로 과거 Japan86/Czech231를 다시 추론할 필요가 없었다. 이번 작업에서 해당 TEST 사진·예측을 새로 열거나 평가하지 않았다. 이전 AP는 [v4 최종 기록](reports/POTHOLE-IMPROVEMENT-FINAL.json), [당시 Japan86 actual-app 비교](reports/POTHOLE-APP-FRESH-COMPARISON-RESULTS.json), [당시 Czech231 회귀](reports/POTHOLE-APP-REGRESSION-COMPARISON-RESULTS.json)의 과거 문맥으로만 남긴다. v5의 새 평가 수치로 재사용하지 않는다.

추가 공식 자료 취득은 HTTP403으로 막혀 새 독립 holdout을 확보하지 못했다. 같은 VAL을 반복 사용했으므로 최종 독립 정확도 증거가 아니다. CC-BY-NC-3.0 연구 조건·원소스 만료/철회 정책을 유지하며 기한을 연장하지 않았다. production 승인과 프로젝트/현장 인수는false다. 학습 가중치·원본 사진·개별 예측은 Git에 포함하지 않는다.

## 검증 재현 도구

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/verify_checkpoint_identity.py --old-checkpoint LOCAL_V4_CHECKPOINT --new-checkpoint LOCAL_V5_CHECKPOINT --old-manifest LOCAL_V4_MANIFEST --new-manifest LOCAL_V5_MANIFEST --output NEW_LOCAL_IDENTITY_REPORT
.\.venv\Scripts\python.exe scripts/verify_trained_models.py LOCAL_V5_MANIFEST --image SAME_FIXED_LOCAL_V4_VAL --checkpoint LOCAL_V5_CHECKPOINT --output NEW_LOCAL_CANONICAL_REPORT
.\.venv\Scripts\python.exe -m pytest tests/test_checkpoint_identity.py -q
```

실행 전에 원소스 권리·기한과 pinned SHA를 확인한다. 기존 보고서를 덮어쓰지 않고 새 로컬 출력에 기록하며 TEST를 명령 입력으로 대체하지 않는다.
