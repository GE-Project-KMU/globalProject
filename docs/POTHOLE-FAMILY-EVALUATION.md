# 포트홀 family 선택·one-shot 평가 준비

현재는 도구·사전 정책·합성 CPU 계약만 준비했다. 실제 family freeze, canonical 실행, US100/회귀 이미지 read·추론은 아직0이다. Small의 순차 retry 완료 후 root가 각 단계를 명시적으로 실행한다. 기존 공통 evaluator·strict proof 소스와 원실험은 수정하지 않는다.

## VAL-only family

고정 v6의 `selected_validation` old/new AP50를 기준으로 oldVAL이 v6보다3pp 넘게 하락하지 않은 완료 checkpoint만 비교한다. mean(oldVAL,newVAL)이 가장 큰 것을 고르고 정확한 동률은 baseline→Nano→Small 순이다. TRAIN loss·새 TEST·과거 회귀 점수로 고르지 않는다. 각 trainer의 선택 checkpoint도 원history의 동일한 VAL gate·최고 mean·최초 epoch와 일치해야 한다.

- TinyV7: selected0·rejected·artifact 없음·TEST0을 확인하고 제외한다.
- NanoV8: `pothole-localization-v8/model-development.json`을 사용한다. 원export/snapshot SHA를 확인하고 packaging4개 필드 외 모델 계약이 달라지면 거부한다.
- SmallV9: 공유 device memory 감독 중 중단된 원실험은 제외한다. 실패 receipt를 보존하고 `pothole-small-v9-retry1`만 후보로 삼는다. 원4개 입력과 retry 복사본의 실제 SHA·full restart/optimizer state 미복원·원부분 weights 미사용·동일 학습 소스·완료 supervisor를 확인한다. retry 선택 receipt가 없거나 실행 중이면 family를 고정하지 못한다. 감독 한도 초과는 실제 OOM 발생 증거와 구분한다.

가중치·checkpoint·manifest·selection/history·개발 release·부모 정책을 family digest로 묶는다. 후속 metadata 변경이나 TEST 이후 재선택을 허용하지 않는다. baseline의 기존 TEST 결과가 선택 근거라는 뜻이 아니며, 사용하는 점수는 고정된 VAL snapshot뿐이다.

## 고정 두 cohort

[사전 정책](reports/POTHOLE-FAMILY-V9-ONE-SHOT-PLAN.json)은 US100 release SHA와 원benchmark를 고정한다. US100은 단일 국가50양성+50음성이다. 실패한2개국200 목표가 달성됐다고 바꾸지 않는다. 초기 US100 계획·자료를 먼저 materialize하도록 수정한 prebuild revision·실제 protocol·수집 보고서의 TEST0 연결을 보존한다.

별도 회귀는 기존 MWPD97양성+RDD48 no-D40 음성, 총145장이다. 과거 사용한 TEST이므로 회귀 점수이며 새로운 최종 독립 검증이 아니다. 두 cohort 모두 물리적 도로/주행 세션 독립성이나 완전한 음성 GT를 입증하지 않는다.

family가 고정된 뒤 root가 선택 artifact의 원positive VAL3개 canonical을 실행한다. `prepare`는 공통 `strict_proof`를 재사용해 exact manifest/weights/checkpoint·실제 VAL SHA·raw 기본 rtol.001/atol.0001·독립 Torch/ORT/app·positive output·detection post 통과를 확인한다. 원baseline도 같은3개 gate가 필요하다. TEST 이미지를 열지 않고 비교 plan만 만든다.

```powershell
# 사전 policy는 이미 생성됨. 같은 policy를 덮어쓰지 않는다.
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family.py freeze
# 선택 artifact canonical은 root가 별도로 실행한 뒤:
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family.py prepare
# 아래 두 명령만 실제 TEST read/추론을 시작한다.
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family.py run fresh
.\.venv\Scripts\python.exe scripts/evaluate_pothole_family.py run regression
```

실행은 `compare_detector_models`의 실제 앱 CPU 경로를 재사용한다. AP floor.01·운영 confidence.1·고정.35 참고값·NMS.5를 사전에 고정하고 결과를 보고 threshold를 다시 고르지 않는다. 비교 plan에는 family digest·부모 plan·strict proof SHA·평가 소스를 닫아서 연결한다.

비교 함수 호출 전에 cohort별 exclusive·fsync된 영구 `attempt`를 생성한다. 중간 실패·프로세스 종료라도 표식을 지우거나 자동 재시도하지 않는다. 두 동시 호출도 실제 comparator를 한 번만 시작할 수 있다. 실패 후 TEST가 일부 사용됐을 수 있으므로 실패를 TEST0 성공으로 기록하지 않는다.

개별 GT·원파일명·경로·예측 좌표는 ignored `data/training/pothole-family-v9`에만 저장한다. 코드·최소 provenance·공개 집계는 구분하고, weights·raw source는 Git에 넣지 않는다. 실제 source 기한이 지나면 준비/실행을 거부하며 mtime·새 날짜로 연장하지 않는다. 단일 모델 CPU latency는 pipeline/10FPS/현장 인수가 아니다.
