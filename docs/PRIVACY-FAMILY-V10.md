# 개인정보 모델 family V10: provenance packaging 수정

V10은 원12 epoch 학습 완료와 V9 CPUexport 실패 이후의 packaging/provenance 수정이다. 고정 마지막 epoch12 EMA 후보 자체는 여섯 epoch의 개발 VAL 관측 이후 작성한 원 V9 공개 사전선언에서 최종12 weights/TEST 전에 이미 정했다. V10 정책이 가중치 생성 전 작성됐다고 주장하지 않는다. 원 V8 trainer·학습 계획·AP 최고 선택·V8/V9 도구와 실패 증거를 보존하며 새 학습·추가 gradient update·TEST 기반 후보 변경은 없다.

## 고정 후보와 선택

| 실제 role | 의미 |
|---|---|
| legacy_v4 / baseline | 기존 고정 tiled 모델 |
| rescue_v8 | 원 V8 AP 최고 checkpoint; 사전선언의 rescue_v8_AP_selected와 동일 |
| fixed_final12 | 같은 학습의 고정 마지막 epoch12 EMA; AP 최고 재선택이 아님 |

세 후보 모두 같은 native FP32 runtime으로 동일한 DEV VAL508(face400/plate108)을 평가한다. 원 AP 최고 후보의 원 native 전체508 예측은 완료훈련과 manifest/checkpoint/source hash를 확인해 metadata로 import한다. baseline/final12는 전체508을 별도로 native 평가한다. 큰 얼굴 GT의 sqrt(area)>=64, confidence .1 고정 recall이 baseline-.01 이상인 후보만 적격이다. confidence grid는 .01/.03/.05/.1/.2/.35/.5이며, 두 class recall95가 가능한 후보는 macroPrecision→minRecall, 없으면 minRecall→macroPrecision이다. 완전 동률은 baseline→원 AP 최고→고정12 순서다. confidence point 내부 동률 규칙은 원 calibrate 수학을 그대로 유지한다.

원 AP 최고 모델과 최종 EMA를 구분해 원 선택 결과를 보존한다. 최종 후보는 VAL 선택 결과가 나쁘더라도 숨기거나 다른 epoch로 교체하지 않는다. 목표 미달이나 정밀도 tradeoff를 개선 완료로 표시하지 않는다.

## 명령과 순서

모든 경로는 프로젝트 내부다. 준비는 metadata-only이며 실제 추론에는 단계별 root 승인이 필요하다. 학습 완료·정확 child 종료·export 정책의 승인 후 vision 담당의 별도 CPU exporter가 fixed12를 생성한다. 학습 중 recovery/실가중치를 읽어 후보를 생성하지 않는다.

```powershell
# 새 exporter 정책이 고정된 후 수행: 사진/VAL/TEST 추론 없음
.\.venv\Scripts\python.exe scripts/evaluate_privacy_family_v10.py prepare
.\.venv\Scripts\python.exe scripts/compare_privacy_family_v10_parallel.py prepare
# 아래 native GPU VAL 단계는 별도 GPU slot 및 학습/export 완료 이후에만
.\.venv-training\python.exe scripts/evaluate_privacy_family_v10.py evaluate --role baseline --authorized
.\.venv-training\python.exe scripts/evaluate_privacy_family_v10.py evaluate --role rescue_v8 --authorized
.\.venv-training\python.exe scripts/evaluate_privacy_family_v10.py evaluate --role fixed_final12 --authorized
.\.venv\Scripts\python.exe scripts/evaluate_privacy_family_v10.py freeze
```

운영 큐는 별도 새 V10 버전에서 각 helper/protocol SHA를 확인한다. 직접 Python3.11/ORT1.31.1 검증 interpreter로 baseline/winner 각각 같은 양성 face·plate VAL 두 장을 원 canonical verifier에 넣어 총4개 raw 기본 rtol .001/atol .0001 및 최종 postprocess strict PASS를 확인한다. 실패하면 허용오차/threshold/사진을 바꿔 성공으로 만들지 않는다.

그 후 별도 승인하에 NEW parallel helper의 run --role fresh/regression --authorized를 각 한 번 실행한다. 두 owned CPU worker가 같은 고정 cohort 전체를 baseline/winner로 각각 평가하며 CPU ORT1.31.1, OpenCV1, 기본 전처리·NMS·known-class 수학을 유지한다. 사용자 영상·새 TRAIN·TEST 반복 추론은 허용하지 않는다.

## 반복·변조 차단

원 namespace는 data/training/privacy-family-v8-evaluation 그대로 보존한다. V10는 data/training/privacy-family-v10-evaluation만 쓴다. V9 namespace도 보존하며 V9 TEST 시작 흔적이 있으면 V10을 거부한다. 원 V8 fresh/regression attempt·comparison·parallel 작업 디렉터리가 있으면 V9를 거부한다. root가 exact identity로 중지한 이전 대기 큐 receipt SHA와 phase/VAL/freeze/TEST0 증거를 고정한다. V10 TEST 시작 후에는 새 VAL/재선택을 거부한다. fresh200/regression1400은 원 manifest SHA 그대로이며 benchmark를 분할하거나 늘리지 않는다.

원 parent4개와 predeclaration, frozen trainer/학습 계획, exporter 계획/소스, 세 후보 manifest/checkpoint/weights/provenance, 모든 VAL 예측·canonical positivecase·최종 winner/threshold·실행 코드·benchmark SHA를 연결한다. 학습/선택 source나 원본 기한을 바꾸면 닫힌다. serial/parallel은 같은 독점 attempt 생성·flush/fsync를 사용한다. 시작/실패/부분 출력을 보존하고 자동 재시도하지 않는다. worker는 정확한 active direct parent·attempt·plan SHA가 필요하다.

## 공개 범위와 보관

공개 파일에는 집계·코드/모델/source hash·정책만 포함한다. 개별 source ID·사진/좌표·검증 image hash·예측은 ignored 로컬 자료다. 원 cap 2026-11-10T16:55:19.285523+00:00 이하를 상속하며 새 namespace 생성으로 기한을 연장하지 않는다. final12 모델·최소 provenance·집계와 개인별 VAL/TEST 자료는 구분한다. 보관 root 추가는 별도 준비 작업이며 이 문서/도구 준비만으로 scan/register/move/delete를 실행하지 않는다.

현장 개인정보 안전성·독립 physical identity·5명 사용자시험·장비 깊이 정확도·운영 승격을 이 연구의 성공 여부로 대체하지 않는다.


## V9 실패와 V10의 제한된 수정

원 exporter는 DEV 전체5501행과 원 training provenance의 일치를 요구해 실패했다. 원 trainer는 `load_development()`가 반환한 TRAIN 얼굴3000행과 모든 VAL508행, 총3508행만 `begin_training`/`register_asset`에 전달한다. 번호판 TRAIN1993행은 원 trainer의 학습에 쓰지 않았다. 새 exporter는 이 정확한 원 predicate와 membership를 검증하며 임의 subset·TEST·여분 행·누락·다른 source SHA를 허용하지 않는다. 이는 모델 수학이나 데이터 선택을 바꾸는 작업이 아니다.

V10은 원 학습 완료와 V9 export 실패 이후 만든 packaging/provenance revision이다. 고정12 EMA 후보 자체는 기존 공개 V9 사전선언으로 이미 미래 weights/TEST 전에 고정됐으며, V10에서 epoch·checkpoint·confidence·family/VAL 수학을 다시 선택하지 않는다. 원 AP 최고 모델·V9 source/protocol·exporter·실패 namespace와 receipt/log는 보존한다.

독립 증거는 V9 자연중지 phase/VAL/TEST0 receipt, 실패 실행 receipt, 원9입력 SHA 보존 감사다. 원 실패 wrapper의 `parent_artifacts_preserved_verified:false`는 검증 단계에 도달하지 못한 원값 그대로 남기고, 별도 감사의 `all_original_inputs_unchanged:true`와 실제 단계의9입력 SHA 재검증을 사용한다. 이를 원 실패가 성공했다는 표시로 바꾸지 않는다.

새 fixed모델 root는 `data/training/redaction-rescue-v8-final12-retry1`이며 기존 실패 root를 재사용하지 않는다. 새31scope 보관 연결은 별도 승인된 코드 준비 단계에서만 추가한다. 실제 자산 스캔·등록·이동·삭제는 모든 writer가 끝난 뒤 root가 따로 판단한다.
