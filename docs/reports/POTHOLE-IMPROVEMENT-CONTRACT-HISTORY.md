# 포트홀 계약 검사 이력

`POTHOLE-IMPROVEMENT-CONTRACT.json`은 중단한 v2 준비 단계의 검사 기록이다. 당시 `fresh_test_no_prior_image_sha`는 기존 공식 RDD2020 릴리스만 비교했으며, 과거 격리된 제3자 실험까지 포함하지 않았다. 이후 추가 감사에서 28장의 과거 학습 이력 중복을 발견하여 해당 실험을 중단했다. 이 파일의 `passed=true`를 최종 시험군의 독립성 또는 현재 후보의 합격 근거로 사용하지 않는다. 원래 검사 기록은 보존한다.

현재 후보의 소프트웨어 계약 근거는 `POTHOLE-IMPROVEMENT-CONTRACT-v4.json`이다. v4는 기존 공식 자료, 격리된 과거 실험, 이번 중단 실험의 실제 사용 이력까지 제외한다. 준비 과정과 원본 해시는 `POTHOLE-IMPROVEMENT.md` 및 `POTHOLE-HISTORY-EXCLUSIONS.json`에 기록했다. 이 계약 검사 역시 모델 정확도나 독립 물리 경로 검증을 의미하지 않는다.
