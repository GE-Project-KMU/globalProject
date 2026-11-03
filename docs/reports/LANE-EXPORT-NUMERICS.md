# 도막 내보내기의 수치 검증과 진단용 정확도 평가

원래 고정 양성 validation 영상에서 앱 ONNX Runtime1.31과 별도1.20의 출력은 기본 `rtol=.001 / atol=.0001` 검사를 통과했다. PyTorch2.7 CPU 원시 출력과의 검사는 실패했다. 최종 도막 마스크는 세 경로에서 100% 일치했지만, 이를 원시 출력 검사 통과로 바꾸지 않았다. 원래 모델과 `verification-final.json` 실패 기록, 처음 비교 계획을 보존한다. 새 TEST나 회귀 영상의 결과를 읽기 전 확인한 문제다.

같은 체크포인트와 같은 validation 입력으로 MKLDNN 사용 여부, 체크포인트에서 계산한 Conv/BN 결합, 활성화 연산의 명시적 표현을 나눠 검사했다. 이 변경만으로 문제를 해결하지 못했다. ONNX 가중치를 PyTorch에 복사하지 않았고, 학습·정답·임계값·검사 허용오차를 바꾸지 않았다.

global1 평균 연산 9개를 명시적 mean으로 내보내면 PyTorch 복사본의 출력은 원래 PyTorch와 비트 단위로 같았다. 해당 진단 영상에서 앱과 PyTorch의 허용오차 초과 원소는 2,097,152개 중 1,938개에서 5개로 줄었다. 최대 절대 차이는 약0.08710에서0.01010, 상대 L2 오차는 약7.59e-5에서6.24e-6으로 감소했다. 그래도 엄격한 원시 출력 검사는 **실패**다. 이 관측은 평균 연산 경로의 기여를 보여주며, 남은 차이의 원인까지 확정한 것은 아니다.

PyTorch2.7.1의 global1 adaptive pooling은 [mean 경로](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/AdaptiveAveragePooling.cpp)를 사용한다. ONNX Runtime1.20.1의 GlobalAveragePool은 [MLAS pooling 경로](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/providers/cpu/nn/pool.cc)를 사용한다. 실제 원인 판단은 코드의 수학적 동치만으로 대신하지 않고 위 개발용 측정과 구분한다.

최소 변경인 mean 내보내기만 별도 `portable` 연구 산출물로 재현한다. 활성화나 다른 표현을 성공할 때까지 선택하지 않는다. 원래 체크포인트·factory·출처 기한은 유지하고, 새 산출물에도 `checkpoint_raw_tensor_parity_passed=false`와 배포 차단 사유를 기록한다. 별도 validation 세 영상도 정답의 양성 여부와 source 순서로 미리 정하여 모든 성공·실패를 보존한다.

기존 비교기의 기본 모드는 원시 출력 검사를 계속 필수로 요구한다. 별도로 명시한 `*-DIAGNOSTIC-PLAN.json`만 연구용 실제 앱 정확도 측정을 허용한다. 이 경우에도 같은 manifest에 대한 실제 영상 검사, 체크포인트 해시, 앱/별도 ORT 원시 출력 일치, 사전 선택 양성 영상의 세 경로 마스크 일치, 배포 차단을 요구한다. 보고서는 원시 출력 실패와 진단용 평가 여부를 반드시 표시한다. 시험 점수를 보고 후보·임계값을 바꾸지 않는다. 정확도 평가가 수치 검증 통과나 운영 승인을 의미하지 않는다.
