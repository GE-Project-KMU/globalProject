# 개인정보 모델 가족 최종 CPU 평가

VAL508에서 모델 가족과 임계값을 먼저 동결했다. 아래 값은 완료된 비교 보고서의 집계를 그대로 옮긴 것이다.

선택 가족: `fixed_final12`, confidence 0.01.

|범위|클래스|모델|GT|AP50|TP|FP|FN|정밀도|재현율|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|fresh|face|baseline|950|84.4377%|875|3965|75|18.0785%|92.1053%|
|fresh|license_plate|baseline|100|95.9592%|96|3|4|96.9697%|96.0000%|
|fresh|face|candidate|950|84.4318%|878|4276|72|17.0353%|92.4211%|
|fresh|license_plate|candidate|100|95.9592%|96|3|4|96.9697%|96.0000%|
|regression|face|baseline|1789|82.9405%|1598|6945|191|18.7054%|89.3236%|
|regression|license_plate|baseline|300|97.9535%|294|18|6|94.2308%|98.0000%|
|regression|face|candidate|1789|82.9134%|1601|7376|188|17.8345%|89.4913%|
|regression|license_plate|candidate|300|97.9535%|294|18|6|94.2308%|98.0000%|

fresh: 200장, 후보 최소 클래스 Recall 92.4211%, 양 클래스 Recall95% 목표 미달. 시간이 공유 호스트 경합을 포함하므로10FPS 근거로 사용하지 않는다.

regression: 600장, 후보 최소 클래스 Recall 89.4913%, 양 클래스 Recall95% 목표 미달. 시간이 공유 호스트 경합을 포함하므로10FPS 근거로 사용하지 않는다.

## 해석 범위

- Fresh200 is the original completed once-only comparison; it was not rerun.
- The original plan incorrectly expected1400 for an unchanged600-row regression manifest; original failed operator/attempt remains failed.
- Corrected regression600 contains300face and300plate original images; no new rows, model or threshold selection.
- Known publisher class coverage only; unknown other class is not negative and full privacy protection is unverified.
- Model family/grid/large-face gate were frozen on VAL508 before TEST; all four default strict canonical proofs passed.
- Per-model/worker CPU latency includes shared-host contention and is not field/camera/full-pipeline10FPS proof.
- Physical person/vehicle/route independence remains unverified; research/non-commercial/redistribution limits and original expiries remain.

## V10 메타데이터 교정

이전 CPU export가 사용하지 않은 plateTRAIN까지 요구해 실패한 것을, 원 trainer가 실제 사용한 faceTRAIN3000+VAL508=3508의 동일 멤버십으로 교정했다. 같은 원 epoch12 EMA만 사용하며 재학습·추가 gradient·새 checkpoint 선택·TEST 재실행은 없다. 이전 실패와 parentflag false는 보존하고, 독립9입력 SHA 보존 audit를 별도로 연결했다.

## 원 V9 개발 관측 후 정책 변경

완료된 얼굴 VAL 6epoch 관측 후 original AP 선택본과 고정 epoch12 EMA를 별도 후보로 비교하도록 변경했다. 최종12 가중치 생성 및 TEST 전에 선언했으며, 모든 VAL 이전의 사전등록으로 주장하지 않는다. 원 V8 평가와 TEST는 실행하지 않았다.

세 후보는 legacy_v4, rescue_v8(설명용 alias rescue_v8_AP_selected), fixed_final12다. 학습 source·12epoch recipe·큰 얼굴 gate·confidence grid·TEST cohort를 바꾸지 않았다. TEST 뒤 모델·임계값을 재선정하지 않는다.

## 회귀 표본 수 계획 오류와 교정

원 계획의1400은 잘못된 count metadata였다. 원 manifest bytes는 그대로600행(얼굴300/번호판300)이다. 원 operator/attempt 실패는 그대로 보존하며, 이미지·모델·threshold를 바꾸지 않은 새 count-only 평가600을 구분한다. 완료된 fresh200은 재실행하지 않았다. 전체1400을 평가했다고 주장하지 않는다.
