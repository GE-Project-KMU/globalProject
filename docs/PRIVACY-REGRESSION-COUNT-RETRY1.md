# 개인정보 회귀 count-only retry1

원 V10 fresh200은 완료됐지만 회귀는 원 고정 manifest SHA03dbb7…의 실제600행(face300/plate300)을 원 정책의 expected1400과 비교하다 중단했다. 회귀 attempt는 생성됐으나 `len(rows)!=expected or not validate_research_release(rows)`의 첫 항에서 short-circuit되어 데이터 validator의 사진 SHA 읽기·픽셀 decode·모델 session·worker 생성은 모두0이다. 원 failed attempt·로그·source·정책·fresh 결과는 보존한다. `.cache/privacy-regression-count-failure-boundary.json`은 이 경계의 source/plan/failed/log SHA와 순수 AST 조건 검증을 고정한다.

새 도구는 원600행 그대로이며 corrected plan의 expected_samples 한 필드만1400→600으로 바꾼다. cohort·모델·가중치·threshold·canonical cases·runtime·수학·source TTL·family 선택을 바꾸지 않는다. 원 V10 admission/4 CPU proof·완료 fresh200·실패 전launch0·모델/holdout metadata 바인딩을 확인한다. 새 helper source와 원 공통 math/source를 각각 실제SHA로 기록하고 원 worker가 실행된 것처럼 표시하지 않는다. 원 source에 monkeypatch/overwrite하지 않는다.

```powershell
# metadata-only: root source검토 후 공개protocol/별도 corrected plan 생성
.\.venv\Scripts\python.exe scripts/compare_privacy_regression_count_retry1.py prepare
# root가 별도 승인한 뒤에만, 기존과 같은 직접 Python3.11/ORT1.31.1 환경
.\.cache\lane-eval-311-ort1311\python.exe scripts/compare_privacy_regression_count_retry1.py run --authorized
```

Private paths는 `data/training/privacy-family-v10-evaluation/` 아래다. 새 plan=`regression-count-retry1-plan.json`, attempt=`regression-count-retry1-attempt.json`; 새 폴더=`parallel-regression-count-retry1/`; 그 안에 execution-plan.json, baseline/candidate.json 및 로그, comparison.json, comparison-per-image.json, execution-receipt.json을 저장한다. 공개 protocol은 `PRIVACY-REGRESSION-COUNT-RETRY1-PROTOCOL.json`이다. 이 protocol 준비로 TEST/모델 추론은 실행하지 않는다.

실행은 CPU2개 owned worker의 ORT1.31.1 CPU2/inter1/CV1, 원 예측 정렬·known-class·NMS·confidence floor .01 및 원 metric assembly를 그대로 쓴다. 직접 active parent와 별도 독점 attempt를 검증하고 실패·부분 출력을 보존한다. 자동 재시도·fresh 재실행·새 cohort 구성·성과에 따른 재선택은 없다. 원1400을 평가했다고 보고하지 않고 원 동일600회귀로 명확히 기록한다. 메모리/timeout/원TTL과 원 raw/post proof gate는 완화하지 않는다.
