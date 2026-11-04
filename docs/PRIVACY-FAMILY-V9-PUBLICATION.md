# V9 개인정보 모델 가족 운영·공개 집계 초안

이 문서는 V8을 덮어쓰지 않는 별도 V9 계획이다. V8 queue는 정확한 PID·생성시각·자식 없음·phase0을 확인하고 중지했다. 원 V8의 VAL508·winner freeze·TEST는 모두0회이며 원 source·receipt·log를 보존했다. 신규 V9 queue·GPU 평가·TEST·공개 집계는 root의 소스·SHA 검토 승인 전까지 실행하지 않는다.

정책의 부모는 `PRIVACY-FAMILY-V9-PREDECLARATION.json` SHA `2717f0a737aa73d72afca8cc567519d0630c3ae2ebe77b980d608a9018228117`이다. **얼굴 개발 VAL 6epoch 관측 후** 추가 후보를 정한 탐색적 변경이다. 최종 epoch12 EMA 가중치 생성과 TEST 이전에 선언했으나, 모든 VAL 이전부터 사전등록했던 것으로 주장하지 않는다. source7bc·learning6bb·고정12epoch 학습은 변경하지 않는다.

세 후보는 다음과 같다.

- `legacy_v4`: 기존 기준 모델.
- `rescue_v8`: 원래 AP 선택을 그대로 보존한 모델. `rescue_v8_AP_selected`는 설명용 alias다.
- `fixed_final12`: 원 학습의 정확한 epoch12 EMA. epoch 탐색·추가 학습·TEST 선택은 하지 않는다.

원 confidence grid, 큰 얼굴 Recall gate, Recall95% 가능군의 macro Precision 우선 규칙, 미달군의 최소 Recall 우선 규칙을 유지한다. 정확한 동률은 baseline→원 AP 선택본→고정12 순서다. TEST는 기존 fresh200·과거 회귀1400을 최종 승자에 대해 한 번씩만 사용한다.

운영 큐는 `.cache/privacy_family_v9_authorized_queue.py`와 별도 `.cache/privacy-family-v9-operator` namespace를 사용한다. root가 승인한 고정 inputs JSON의 SHA가 필요하다. 입력은 신규 family/parallel/controller/export source 및 정책 SHA와 **실제 GPU 학습 자식 PID·생성시각**을 포함하며, 도구 세션번호와 프로세스 ID를 혼동하지 않는다.

학습 완료·12epoch·source closure·정확한 자식 종료 이후에도 root CPU exporter의 실행 receipt가 없거나 starting/running이면 메타데이터 대기만 한다. queue는 CPU export를 실행하거나 재시도하지 않는다. `supervise_final12_cpu_export.py`의 `completed_execution()`이 실패·해시·cleanup 불일치를 거부하고 완료를 검증해야 한다. 이후 GPU 여유 두 표본<2GiB를 확인한다.

그 다음 순서는 baseline GPU VAL508 → 원 AP 선택본의 완료된 VAL508 import(추가 추론 없음) → fixed12 GPU VAL508 → 세 가족 VAL freeze → baseline/winner 각각 face·plate의 기본 CPU strict 검증4개 → 통과 시 fresh200·reg1400 두 worker 비교다. 두 GPU 평가에는 동일 Torch/native profile 및 자기 자식9GiB 감독을 적용한다. CPU canonical과 TEST worker는 같은 직접 Python3.11.9/NumPy1.26.4/CV4.10/ORT1.31.1 환경을 사용하며, 기존 canonical CVthreads2와 parallel CVthreads1은 그대로 명시한다. 실패 증거를 보존하고 자동 재시도·사례 교체·허용오차 변경·TEST 재선택은 하지 않는다.

새 공개 도구는 `scripts/finalize_privacy_family_v9_results.py`다. 원 V8 publisher12d9는 수정하지 않으며, 그 검증된 lexical 경계·control reader·indexed AP101 수학·pair 검증의 순수 helper를 읽기 전용으로 재사용한다. 세 VAL의 동일 GT·전체508·manifest/checkpoint/source 연결과 원 정책의 세 가족 선택을 다시 확인한다. 새 fixed12 export receipt/상위 completion/candidate freeze 및 세 후보의 provenance·모델 해시도 닫는다.

공개 필드는 완료 comparison의 집계 복사본과 전체 증거 SHA, 실제 runtime, V9의 개발 관측 후 변경 사실뿐이다. 점수 재계산은 일치 검사 전용이다. 개별 image SHA·ID·경로·bbox·예측 배열·사진·가중치 bytes를 공개하지 않는다. source/data expiry·연구/비상업 및 재배포 제한을 유지하며 물리적 인물·차량·경로 독립성, 완전한 개인정보 보호, 현장10FPS를 주장하지 않는다.

현재 검증은 합성 fixture만 사용한다. 실제 `gather`·publication이나 TEST를 실행하지 않는다. 이후 root가 허가하면 직접 CPU 환경에서 고정 operator source SHA와 함께 집계한다.

```powershell
# 현재 실행하지 않는다. 전체 운영 완료와 별도 집계 승인 후에만 실행한다.
$env:CUDA_VISIBLE_DEVICES = ''
.\.cache\lane-eval-311-ort1311\python.exe scripts/finalize_privacy_family_v9_results.py --authorized --operator-source-sha256 <승인된-V9-operator-SHA>
```

출력은 별도 `PRIVACY-FAMILY-V9-RESULTS.json` 및 `.md`다. 동일 source·inputs 재실행은 동일 bytes/SHA를 유지하고 재작성하지 않는다. 원 V8 결과·미완료/실패 상태를 V9 성공으로 바꾸지 않는다.
