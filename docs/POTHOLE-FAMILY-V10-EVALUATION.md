# 포트홀 V10 동일 CPU 평가와 단회 비교

V10은 기존 V9의 학습·체크포인트 선택 기록을 보존하면서 Nano 동일 가중치의 960·1280 입력 옵션을 추가한다. 새 학습이나 threshold 변경은 없다. V9 native/GPU AP는 학습 시 선택한 checkpoint의 증거이며 V10 family 순위에는 사용하지 않는다.

사전 고정한 `POTHOLE-RESOLUTION-V10-PLAN.json`과 V9 정책 hash를 새 `POTHOLE-FAMILY-V10-ONE-SHOT-PLAN.json`이 연결한다. 실제 사진이나 TEST를 읽지 않고 정책을 생성했다. 실행 도구는 `scripts/evaluate_pothole_family_v10.py`이다.

후보 순서는 baseline640, Nano640, Small640, Nano960, Nano1280이다. Small의 순차 retry 완료와 원실패 receipt·동일 입력·초기화 재시작 계약을 확인한다. Small이 epoch0을 유지하여 export하지 않은 경우만 명시적으로 제외한다. retry 미완료를 제외 처리하여 먼저 선택할 수 없다.

선택에는 `scripts/evaluate_pothole_resolution.py`가 한 번 작성한 실제 앱 ONNX Runtime CPU 1.31.1 VAL652 결과만 사용한다. original402와 MWPD250의 동일 membership hash, runtime/provider, 입력 크기, checkpoint·manifest·ONNX hash를 모두 확인한다. original AP50가 같은 CPU baseline보다 0.03 이상 낮은 후보는 제외하고, 나머지의 두 domain AP50 평균을 최대화한다. 완전 동률은 위 후보 순서를 따른다. native/GPU 점수나 과거 TEST 점수로 보충하지 않는다.

```powershell
# 정책은 이미 생성됨. 기존 파일을 덮어쓰지 않는다.
# Small 완료 및 root의 실제 CPU VAL652 실행 이후:
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family_v10.py freeze
# 선택 artifact의 기본 허용오차 raw+post CPU canonical 검증 이후:
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family_v10.py prepare
# root가 실행 승인한 단회 비교:
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family_v10.py run fresh
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family_v10.py run regression
```

출력은 로컬 `data/training/pothole-family-v10`에 분리한다. freeze는 CPU report·attempt·exports와 family digest를 결합한다. prepare는 기존 공통 strict proof 검사와 고정 양성 VAL 사례를 확인한 후 US100 및 기존 MWPD97+RDD48 회귀 계획만 만든다. 기본 raw 허용오차나 post 조건을 완화하지 않는다. canonical 실패 시 다른 후보로 자동 전환하거나 TEST를 먼저 실행하지 않는다.

각 run은 실제 공통 비교 호출 전에 지워지지 않는 started receipt를 생성한다. 실패하더라도 자동 재시도하지 않는다. V9에서 같은 TEST 비교를 시작했으면 V10은 거부한다. US100은 source proxy 분리이며 실제 도로·지역 독립성을 입증하지 않는다. 과거 145장은 회귀 벤치이며 새 최종 holdout으로 표시하지 않는다. TEST 이후 recipe·threshold·family 재선택은 금지한다.

현재 구현 검증은 합성 메타데이터 계약 시험이다. 실제 CPU VAL652 선택·fresh/회귀 성능을 통과했다고 주장하지 않는다. 원본 사진·개별 ID·좌표·checkpoint·가중치는 Git에 포함하지 않는다. source TTL과 기존 inventory는 그대로 유지하며 실제 보관 등록은 모든 writer 종료 후 별도 수동 단계다.
