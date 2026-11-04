# Lane retry1의 별도 단회 평가 정책

`scripts/evaluate_lane_retry1.py`는 원 `evaluate_cycle2.py`·비교/수치 worker·학습 source를 수정하지 않고 새 평가 경로만 제공한다. 후보는 `data/training/lane-cycle2-v5-retry1/lane.json`, 출력은 별도 `data/training/lane-cycle2-retry1-evaluation`이다.

공개 `LANE-CYCLE2-RETRY1-EVALUATION-POLICY.json`은 원 fresh119/회귀180 계획, original/retry 학습 계획, supervision V2, 원 numeric/canonical case 계획의 SHA를 유지한다. 후보 학습이 미완료여도 이 정책은 metadata만으로 사전 고정할 수 있다. **후보 hash는 원 정책에 나중에 추가하지 않는다.** 완료 후 별도 `candidate-freeze.json`에 manifest/weights/checkpoint·calibration·실제 CPU proof·source hash를 결합한다.

인계 요청의 회귀280장은 착오였다. 원 `lane-regression-manifest.json`은 고정 SHA 아래180개 고유행이며, 이전100+80으로 구성된다. 정책 생성 preflight가 이 차이를 거부한 뒤 root가 원180 유지로 정정 승인했다. 새 정책의 `revision_reason`에 남겼으며 원 plan/cohort/SHA를 수정하거나100장을 추가·중복하지 않았다. 이 정정 전후 실제 TEST 추론은0이다.

## 평가 전 gate

- 순차 retry supervision의 정상 종료·exit0·own child 정리·학습 및 감독 source/plan binding
- 후보 manifest·weights·원 full release·DEV·provenance의 SHA와 원기한, 폐기/재검증 표시 거부
- 원 사전 고정 양성 VAL3개 각각의 실제 ORT CPU1.31.1/native Torch/default raw 및 post PASS, 동일 checkpoint/case binding
- 전체179 VAL의 완전·고유 index/image identity와 모든 raw 비교 기본 `rtol=.001/atol=.0001`·exact-mask PASS
- full179 원 plan/source/manifest/checkpoint/weights/DEV binding과 원행 순서 유지
- VAL에서 고정한 checkpoint epoch·paint threshold·calibration report hash 유지
- 원 baseline의 strict canonical·manifest/weights/provenance binding 유지
- 두 모델 모두 benchmark와 TRAIN/VAL의 SHA·leakage group 겹침 거부

새 helper는 canonical이나 full proof의 실패를 덮어쓰지 않는다. diagnostic/느슨한 tolerance 대체 경로가 없다. 원 fresh/reg benchmark나 원 cycle2 policy를 바꾸지 않으며 실제 학습·calibration·TEST를 policy/freeze/prepare 단계에서 실행하지 않는다.

```powershell
# 사전 정책 고정(metadata만); 기존 파일은 덮어쓰지 않는다.
.\.venv\Scripts\python.exe scripts/evaluate_lane_retry1.py policy
# 학습 및 CPU canonical3+full179 실제 완료 후 별도 artifact receipt 고정
.\.venv\Scripts\python.exe scripts/evaluate_lane_retry1.py freeze
.\.venv\Scripts\python.exe scripts/evaluate_lane_retry1.py prepare
# root가 실제 실행을 별도로 승인한 뒤에만:
.\.venv\Scripts\python.exe scripts/evaluate_lane_retry1.py run fresh --approved-test-inference
.\.venv\Scripts\python.exe scripts/evaluate_lane_retry1.py run regression --approved-test-inference
```

`run`은 승인 flag가 없으면 거부한다. 원 cycle2에서 해당 TEST 비교를 시작한 흔적이 있으면 새 정책도 거부한다. 새로운 attempt는 비교 호출 전에 exclusive-create하고 영구 보존한다. worker 실패 후에도 지우거나 자동 재시도하지 않는다. 동시에 같은 role을 호출해도 하나만 비교로 진입한다. 완료 receipt는 started 파일을 덮어쓰지 않고 별도 파일에 기록한다.

현재 검증은 합성 metadata/proof·가짜 비교 worker를 사용한 계약시험이다. 실제 TEST119/회귀180 추론·정확도 결과는 아직0이다. 이번 fresh는 publisher recording proxy 분리이며 물리 도로 독립성을 입증하지 않는다. 회귀180은 과거 확인된 자료이므로 새 최종 holdout으로 표시하지 않는다. paint proxy 평가이며 유효 마모율·실물 ZED 깊이·현장 정확도·전체10FPS 승인을 대체하지 않는다.

실제 개별 수치 출력은 로컬 private namespace에만 저장한다. 별도 보관 준비 scope를 추가했으며 lane 원기한(최대2026-11-10 17:07:26.069653 UTC)을 상속한다. 개별 sample 정보가 없는 exact artifact freeze/comparison control은 최소 provenance로 보존한다. 원8개 control pin과 inventory는 변경하지 않으며 실제 보관 등록·삭제는 root의 최종 수동 단계 전까지0이다.
