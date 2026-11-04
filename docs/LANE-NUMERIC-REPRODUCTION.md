# 차선 수치 정합성 진단

이 작업은 기존 A2D2 v2 체크포인트의 내보내기 수치 차이를 검사한다. 학습,
임계값 재선택, TEST를 이용한 선택은 수행하지 않는다. 정확도 개선을 증명하는
실험도 아니다. 모든 모델은 연구 전용이며 배포 승인은 없다.

## 보존한 실패

원래 FP32 Torch 구현을 기준으로 비교한다. 9개 전역 평균만 FP64로 누산하고
FP32로 복귀한 ONNX는 기본 rtol=0.001/atol=0.0001에서 전체 VAL100의 raw
비교를 통과했다. ORT1.20.1은 한 영상에서 최종 마스크 4픽셀이 달랐다.
마지막 1×1 분류기의 누산까지 FP64 MatMul로 바꾼 추가 후보도 같은 실패를
보였다. 두 후보의 원본 증거는 각각 `data/training/lane-a2d2-fp64-mean-v3`
및 `data/training/lane-a2d2-fp64-head-mean-v3`에 보존한다. 초기 양성 VAL3
통과를 전체 정합성 통과로 해석하지 않는다.

부동소수점 누산 순서에 따라 수치 차이가 생길 수 있다.
[PyTorch 수치 정확도 설명](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html).
본 실험은 logits 반올림/클램프/상수 오프셋이나 마스크 허용오차를 추가하지 않는다.

## 제한된 지원 환경 검증

단순 FP64 평균 후보의 동일 그래프를 새 폴더에 복사하고, Windows AMD64의
공식 ORT1.31.1 CPU 세션으로 지원 범위를 제한한다. 구형 환경 실패를 해결했다고
주장하지 않는다. `required_runtime_profile`은 실제 로드한 버전과 세션 provider를
검사하며, 다른 버전/OS/아키텍처/provider 및 잘못된 프로파일을 거부한다.
이 필드가 없는 기존 manifest의 정책은 바뀌지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_lane_eval.py prepare
.\.venv\Scripts\python.exe scripts/create_lane_supported_profile.py data/training/lane-a2d2-supported-ort131-v3
.\.venv\Scripts\python.exe scripts/check_lane_precision_runtime.py validate-all data/training/lane-a2d2-supported-ort131-v3 --training-python .cache/lane-eval-311-ort1311/python.exe
.\.venv\Scripts\python.exe -m pytest tests/test_lane_runtime_profile.py tests/test_model_verification.py tests/test_vision.py -q
```

새 출력 폴더가 이미 있으면 create/validate-all은 덮어쓰기를 거부한다. 재현 시
새 이름을 사용한다. `setup_lane_eval.py`는 기존 공식 embedded Python3.11.9
실행 파일을 프로젝트 내부로 복사하고 ORT만 별도 target에 설치한다. 기존
Torch2.7.1과 NumPy1.26.4는 읽기 전용으로 공유하고 기존 ORT1.20.1을 변경하지
않는다. 캐시/임시 파일/로그는 프로젝트 내부다. 핵심 실행 파일과 패키지 진입점
해시의 변경 여부는 `environment-proof.json`에 기록한다. 전체 패키지 트리의
포렌식 감사라고 주장하지 않는다.

개별 양성 VAL 검증은 기존에 고정한 `validation-plan.json`의 세 영상을 각각
`scripts/verify_trained_models.py`에 `--checkpoint`와
`--training-python .cache/lane-eval-311-ort1311/python.exe`로 전달한다.
기본 허용오차를 유지한다. aggregate 최종 결과는 별도 보고서에 기록한다.

실제 완료 결과는 `docs/reports/LANE-SUPPORTED-RUNTIME-NUMERIC.json`이다.
고정 양성 VAL3과 전체 VAL100에서 세 raw 비교 쌍 모두 기본 허용오차를
통과했고, 원 Torch/앱/참조 ORT의 최종 마스크는 정확히 일치했다. 런타임
제약 계약 20개를 포함한 관련 회귀 109개가 통과했다. 실제 구형 ORT1.20.1
실행에서도 새 manifest가 `model_not_ready`로 거부되었다. 의미는 제한된
환경의 수치 정합성 확인이며, 목표 정확도 75% 충족이나 배포 승인이 아니다.
`scripts/finalize_lane_numeric_report.py`로 완료 증거를 점검하고 집계 보고서를
다시 만들 수 있다. 원 manifest 및 원 증거는 덮어쓰지 않는다.

개별 영상 해시/경로/마스크/입력 배열/개별 진단 결과는 비공개 보관 대상이며
원본 source 만료를 상속한다. 고정된 이전 inventory를 수정하거나 기한을
연장하지 않고 새 진단 inventory에 등록해야 한다. 공개 보고서는 집계와
코드/모델/데이터 manifest 해시만 포함한다.

## 독립 정확도 검증의 제한

공식 [A2D2 AWS 공개 목록](https://registry.opendata.aws/aev-a2d2/)을 메타데이터만
조회했을 때 기존 미사용 recording은 11개지만 과거 모든 split과 다른 날짜는
없었다. 따라서 새 날짜와 물리적 도로 독립성을 갖춘 시험이 이미 준비되어
있다고 주장할 수 없다. 새 recording을 사용하려면 경로/지리 정보로 독립성을
추가 입증하거나 별도 합법적 출처를 확보해야 한다. 동일 날짜의 미사용 recording
시험은 그 한계를 명시해야 한다.

이후 시험은 선택된 checkpoint/export/threshold를 먼저 고정하고, 과거 모든
train/val/TEST와 SHA 및 연결된 pHash 유사군을 제외한 표본을 결과를 보지 않고
동결해야 한다. 종전 TEST 두 집합은 회귀 벤치마크다. Solid/Dashed 픽셀 GT는
독립적인 마모율/기대 윤곽/잔존율 GT가 아니므로 이 성능을 검증하려면 별도
전문가 라벨이 필요하다. 공개 데이터 수치는 한국 도로/ZED2i 실물 검증을
대체하지 않는다.
