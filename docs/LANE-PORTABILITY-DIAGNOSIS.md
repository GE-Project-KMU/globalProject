# 차선 모델 CPU 변환 진단

53epoch의 같은 학습 checkpoint와 사전 고정 validation을 사용했다. 원형 ONNX와 엄격 검증 실패는 그대로 보존한다. 도막 마스크가 일치해도 `rtol=0.001, atol=0.0001` raw tensor 검증 실패를 합격으로 바꾸지 않는다. 현재 portable 모델도 numerical_parity_passed=false이며 운영 사용을 차단한다. 집계 증거는 [진단 보고서](reports/LANE-PORTABILITY-DIAGNOSIS.json)에 있다.

## 관측과 가설

CPU MKLDNN on/off 및 checkpoint로부터 유도한 Conv/BN53쌍 folding만으로 원래 차이는 해결되지 않았다. HardSwish·HardSigmoid를 각각 또는 함께 primitive 표현으로 바꾼 복사본은 native Torch와 비트가 같았으나 ONNX의 raw 차이는 유지됐다. 이 결과로 activation 표현을 주원인이라고 확정할 수 없다. 공식 문서는 수학상 동등하지만 다른 표현을 설명한다: [PyTorch HardSwish](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Hardswish.html), [ONNX HardSwish](https://onnx.ai/onnx/operators/onnx__HardSwish.html).

[PyTorch2.7.1 global1 pooling 소스](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/AdaptiveAveragePooling.cpp)는 1×1 출력에서 spatial mean을 사용한다. [ORT1.20.1 pooling 소스](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/providers/cpu/nn/pool.cc)는 별도 CPU pooling 경로를 포함한다. 소스 차이만으로 이번 오차 원인이 증명되지는 않는다. 실제 같은 checkpoint의 global1 pool9개를 explicit mean으로 바꾸면 Torch 결과는 그대로이고 ONNX의 불일치 원소가 같은 고정 표본에서1,938개→5개로 줄었다. 이 관측은 pooling 축약 경로가 오차에 크게 기여한다는 근거이며 잔여 차이 전체의 설명은 아니다.

모델·threshold·허용 오차를 바꿔 성공하는 표현을 고르지 않았다. 최소한의 확인된 mean export만 별도 portable 산출물로 만들었다. Conv/BN·activation 조합과 최적화 해제는 진단 기록이며 기본 앱 설정으로 승격하지 않는다.

## 재현 절차와 차단 조건

실제 경로와 checkpoint는 프로젝트 로컬에서만 보관한다. `scripts/export_lane_portable.py`에 원형 manifest SHA와 checkpoint SHA를 명시하여 CPU export한다. 이 helper는 strict checkpoint load 뒤 export 복사본의 global1 pool만 mean으로 치환하고, 학습 state tensor가 그대로인지 확인한다. 원형 artifact·실패 보고서는 바꾸지 않으며 provenance에는 모델 연결만 추가하고 assets·기한은 유지한다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/export_lane_portable.py --original-manifest LOCAL_ORIGINAL_MANIFEST --original-manifest-sha256 PINNED_ORIGINAL_SHA --checkpoint LOCAL_CHECKPOINT --checkpoint-sha256 PINNED_CHECKPOINT_SHA --output NEW_LOCAL_PORTABLE_DIRECTORY
.\.venv\Scripts\python.exe scripts/calibrate_lane_model.py LOCAL_DEVELOPMENT_MANIFEST LOCAL_PORTABLE_MANIFEST NEW_LOCAL_CALIBRATION_REPORT
```

같은 validation-only100장과 고정 grid에서 threshold0.4를 유지했다. source_id 순서의 첫 GT 양성3장을 미리 고정하고 모두 기본 canonical verifier로 기록했다. 세 경우 모두 앱↔독립 ORT는 엄격 raw 기준을 통과했고 세 엔진의 최종 양성 도막 마스크는100% 같았다. 반면 native Torch와의 raw 검증은3개 모두 실패했다. 실패한 표본을 버리거나 다른 영상·오차 범위로 대체하지 않았다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_trained_models.py LOCAL_PORTABLE_MANIFEST --image PREDECLARED_LOCAL_VALIDATION_IMAGE --checkpoint LOCAL_CHECKPOINT --output NEW_LOCAL_PROBE_REPORT
```

원형과 portable의 source 관계, 학습 checkpoint SHA, 변환 방식, unresolved 차단 사유를 manifest에 남긴다. 원형 export 소스도 로컬에 보존하며 gate field 이름을 보완한 metadata-only 수정은 별도 영수증으로 기록한다. 개별 이미지 해시·ID·좌표·예측은 공개 집계에 넣지 않는다.

기본 benchmark gate는 계속 엄격하다. 별도로 사전 고정한 진단용 actual-app 정확도 계획과 gate를 검토하기 전에는 새 시험·회귀 시험을 실행하지 않는다. 진단용 정확도 측정이 승인돼도 raw parity 실패나 실차·개인정보 인수 조건이 해결됐다는 뜻이 아니다.
