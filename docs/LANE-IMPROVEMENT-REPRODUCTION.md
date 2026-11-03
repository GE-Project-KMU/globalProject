# 차선 모델 개선 실험 재현과 결과

새 recording100장의 actual-app 도막 IoU는 baseline28.42%→후보48.75%(+20.32%p), 과거에 확인한100장 회귀는39.44%→58.72%(+19.28%p)였다. 두 평가 모두 목표75%에 미달한다. 이 결과는 **진단용 연구 정확도**이며 운영 승인이나 엄격한 변환 검증 합격이 아니다. 전체 집계·해시는 [결과 보고서](reports/LANE-IMPROVEMENT-RESULTS.json)에 있다.

## 고정된 실험

기존 train400·validation100을 유지하고 새 train 후보400장을 추가했다. 연결 SHA/pHash 중복1장을 제외하여 train799, validation100이다. 과거 전체 자료와 중복을 검사하고 학습·검증과 recording/date를 분리한 새100장은 사전 계획으로 봉인했다. 과거 TEST100장은 회귀로만 구분했다. 실제 물리 도로의 독립성까지 확인된 것은 아니다.

후보 recipe는 한 개다. 같은 MobileNetV3 기반2class DeepLab, native640 patch/1024 full validation, frozen BN 통계, encoder/decoder별 LR, EMA, 사전 고정 validation grid와 종료 조건을 사용했다. 학습은53epoch에서 종료됐고 validation 최고 checkpoint는41epoch다. 실제 학습·factory 소스 해시는 결과 보고서에 고정한다. 시험 점수를 확인한 뒤 checkpoint·threshold·recipe를 다시 고르지 않았다.

## 수치 실패와 portable 관계

원형 ONNX의 native Torch CPU raw allclose 실패는 원본 보고서와 함께 보존했다. pooling 축약 경로의 오차 기여를 분리한 뒤 같은 checkpoint의 export 복사본에서 global1 pool9개만 explicit mean→ReduceMean으로 바꾸었다. [변환 진단](LANE-PORTABILITY-DIAGNOSIS.md)의 activation·BN·최적화 실험들은 진단 기록이며 임의 조합을 최종 모델로 골라 통과시키지 않았다.

portable 모델도 `rtol0.001/atol0.0001` raw 기준에서 사전순 양성 validation3장 모두 native Torch와 불일치했다. 앱↔독립 ORT는3장 모두 엄격 수치 기준을 통과하고 세 엔진의 최종 양성 도막 마스크는3장 모두100% 일치했다. 이로 raw 실패를 덮지 않는다. `checkpoint_raw_tensor_parity_unresolved` 차단 상태와 numerical_parity_passed=false를 유지한다.

기본 benchmark gate는 이 후보를 계속 거부한다. 별도로 사전 고정·검토한 진단용 actual-app 정확도 gate와 계획에서만 위 새100장·과거100장 비교를 각 한 번 허용했다. 원형·portable·학습 checkpoint 관계와 최종 threshold0.4는 시험 전에 고정했다. 원형 모델·기존 baseline·원래 실패 보고서·학습 가중치를 덮어쓰지 않았다.

## 실행 순서

원본·가중치·개별 데이터 경로는 로컬에만 보관한다. 아래 값은 개인별 파일명을 공개하지 않는 자리표시자다. 원본 자료·manifest·환경·코드 해시와 만료·사용조건이 맞는 경우에만 재현한다. 만료를 새 실행 시각으로 연장하지 않는다.

1. 공식 A2D2 목록에서 recording/date와 균등 프레임 선택을 먼저 고정한다. 원래 자료를 바꾸지 않는 `scripts/prepare_a2d2_improvement.py`로 크기·MD5·PNG SHA·분할 중복을 검사한다.
2. 고정된 `scripts/improve_lane.py`를 전용 학습 Python에서 실행한다. 학습 로더에는 development manifest만 전달하고 full release는 출처 SHA로만 연결한다. TEST 사진은 학습·선택에 쓰지 않는다.
3. 앱 CPU `scripts/calibrate_lane_model.py`로 development validation만 보정하고, 원래 `scripts/verify_trained_models.py`의 실패도 보존한다.
4. `scripts/export_lane_portable.py`로 같은 checkpoint의 최소 mean export를 별도 폴더에 만든다. 같은 development validation/grid로 threshold0.4를 한 번 고정하고, 사전순 첫3개 GT 양성 validation 모두 원래 strict verifier로 기록한다.
5. 진단 gate 사전 검사 후 `scripts/compare_lane_models.py`에 사전 고정 fresh diagnostic plan과 regression diagnostic plan을 각각 한 번 전달한다. 기존 출력이나 다른 선택으로 반복 시험하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_lane_model.py LOCAL_DEVELOPMENT_MANIFEST LOCAL_PORTABLE_MANIFEST NEW_LOCAL_CALIBRATION_REPORT
.\.venv\Scripts\python.exe scripts/verify_trained_models.py LOCAL_PORTABLE_MANIFEST --image PREDECLARED_LOCAL_POSITIVE_VALIDATION_IMAGE --checkpoint LOCAL_CHECKPOINT --output NEW_LOCAL_PROBE_REPORT
.\.venv\Scripts\python.exe scripts/compare_lane_models.py PREDECLARED_FRESH_DIAGNOSTIC_PLAN NEW_LOCAL_FRESH_REPORT
.\.venv\Scripts\python.exe scripts/compare_lane_models.py PREDECLARED_REGRESSION_DIAGNOSTIC_PLAN NEW_LOCAL_REGRESSION_REPORT
```

## 비용과 남은 조건

새100장 CPU predict 중앙값은 baseline약0.097초, 후보약0.535초였다. 입력 크기와 ReduceMean을 포함한 후보 비용이 더 크며, 공유 호스트 측정이다. capture·전송·검수·보고를 포함한 전체 FPS 인수 결과로 환산하지 않는다.

[공식 A2D2 자료](https://registry.opendata.aws/aev-a2d2/)의 solid/dashed 픽셀은 도막 proxy다. 독립적인 예상 외곽O·정상 간격G·유효 영역V 정답이 없어 잔존율 MAE를 만들지 않았다. 실제 ZED2i·차량·국내 도로·사람의 개인정보 검수와 독립 주행 인수도 별도다. 소스 만료·철회와 배포 권한 제한을 유지하며 가중치를 외부에 공개하거나 자동 적용하지 않았다. 공개 보고서에는 개별 이미지 해시·ID·경로·좌표·예측·원본 픽셀을 넣지 않는다.
