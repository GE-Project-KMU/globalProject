# 회귀 표본 수 교정 평가의 별도 공개 집계

원 V10 operator는 fresh200을 한 번 완료한 뒤 회귀 admission에서 실패했다. 원 계획에는1400으로 잘못 적혀 있었지만, 고정된 manifest SHA `03dbb7a8e60b0316e4c06508bcb76175097f594a29d0a25d40a2129998ae4c8a`의 실제 행은600개다. 얼굴300장·번호판300장이고 원 bytes·멤버십은 바꾸지 않는다. 원 실패 단계는 count 검사에서 중단되어 image hash/디코딩·모델 worker·추론은0회였다.

이 교정은 새 사진이나 모델을 선택하는 작업이 아니다. 별도 root 승인 runner가 원 계획의 `expected_samples`만1400→600으로 바꾼 새 계획/attempt에서 같은 모델·threshold·전처리·NMS·순서·점수 함수를 사용한다. 원1400 계획, failed attempt, failed operator, 완료 fresh 결과는 모두 그대로 보존한다. fresh200은 재실행하지 않는다.

새 공개 도구 `scripts/finalize_privacy_regression_count_retry1.py`는 원 publisher6ea를 수정하지 않는다. 기존 lexical 경계·control reader·indexed AP101 수학을 읽기 전용으로 재사용한다. pair 검증 함수의 AST는 원형과 같고, 이 새 모듈의 명시적 cohort 상수만 fresh200/regression600이다. 기존 모듈의1400 상수는 변경하지 않는다.

집계 gate는 원 failed operator의9개 완료 phase와 failed regression phase, 원 fresh200 report/worker/per-image/attempt SHA, count 실패 경계, 새 protocol/derived plan, 새 completed reg600 attempt·worker·execution receipt, 동결 승자·모델·provenance·4개 기본 strict proof를 함께 검증한다. 원 실패를 completed로 고치거나 실패를 숨긴 보고서를 만들지 않는다. score 재계산은 원 comparison 값과 일치하는지 검사할 뿐이며 공개 수치는 완료된 두 report의 값을 복사한다.

출력은 별도 `PRIVACY-FAMILY-V10-COUNT-RETRY1-RESULTS.json`과 `.md`다. 여기에는 원 계획1400 오류, 실제 동일 manifest600, 원 operator/attempt 실패 유지, 새 교정 평가 완료를 구분한다. 전체1400장을 평가했다고 주장하지 않는다. 사진 장수300/300과 클래스별 GT box support도 구분한다.

공개 whitelist는 cohort·클래스별 AP50/TP/FP/FN/Precision/Recall·95% 목표 충족 여부·전체 모델/control SHA·runtime·한계뿐이다. 개별 image SHA·ID·경로·bbox·예측 배열·사진·weights bytes는 공개하지 않는다. 동일 source/입력 재집계는 동일 bytes/SHA이고 기존 다른 결과·부분 파일은 덮어쓰지 않는다.

모든 모델·confidence는 원 VAL freeze 그대로다. 물리적 인물·차량·경로 독립성, 완전한 개인정보 주석, GPU 정합성, 카메라/전체 pipeline10FPS, 기관 인수를 주장하지 않는다. CPU 수치 gate는 CPU 근거이고 GPU 진단은 별도 증거다. 연구/비상업·재배포 제한과 원 만료일은 유지한다.

준비 검증은 합성 fixture만 사용하며 실제 publication은 새 reg600 완료와 root의 helper 검토·승인 후다. 이 helper는 모델/session을 만들거나 사진/NPY를 읽지 않는다.

```powershell
# 아직 실행하지 않는다. 완료된 두 report와 별도 집계 승인 후에만 사용한다.
$env:CUDA_VISIBLE_DEVICES = ''
.\.cache\lane-eval-311-ort1311\python.exe scripts/finalize_privacy_regression_count_retry1.py --authorized --operator-source-sha256 d02daeedf75a16706601fcacee76fc67f380d7d8a528346e2848a2b17a265728
```
