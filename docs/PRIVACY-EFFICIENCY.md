# 개인정보 추론 비용·오탐 연구 v5

기존 학습 checkpoint를 그대로 두고 VAL508장(face400/plate108)에서 표준960·전체 tiled1920·사전 고정한 연구 후보3개를 각각 한 번 실행했다. 선택 타일은 비용을 줄였지만 얼굴 Recall도 낮아졌다. **어떤 옵션도 두 클래스 최소 Recall95% 목표를 달성하지 못했으며 자동 선정·배포·기본 runtime 변경은 없다.** 이미 반복 사용한 VAL이므로 새 독립 최종 정확도 근거가 아니다. TEST·GPU·추가 gradient 학습은 사용하지 않았다.

수치는 [허용 필드만 추린 공개 집계](reports/PRIVACY-EFFICIENCY-V5-RESULTS.json)에 있다. 모든 옵션 confidence0.01/NMS0.5를 유지했다.

| 옵션 | 얼굴 Recall | 얼굴 Precision | 얼굴 FP | CPU median | CPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 표준960 | 86.755% | 36.010% | 4,295 | 108.3ms | 168.2ms |
| 전체 tiled1920 | 92.498% | 22.415% | 8,920 | 439.0ms | 632.4ms |
| 작은 얼굴 증거 상위2타일 | 91.816% | 25.654% | 7,413 | 257.3ms | 446.8ms |
| 증거 밀도 상위2타일 | 91.852% | 24.625% | 7,833 | 320.1ms | 490.7ms |
| 전체 타일+합의 필터 | 91.350% | 28.924% | 6,254 | 572.8ms | 807.8ms |

번호판은 모든 옵션에서 VAL108장에 대해 Recall/Precision100%였지만, 이 작은 알려진 클래스 표본의 결과를 현장 보장으로 해석하지 않는다. 얼굴 GT는2,786개다. 다른 클래스가 완전히 주석되지 않은 사진에서는 알려지지 않은 클래스 FP를 계산하지 않는 조건부 평가다.

첫 선택 타일은 **같은 실행의** 전체 tiled 대비 median439.0→257.3ms(약41.4% 단축), 얼굴 Recall 약0.68%p 감소, Precision 약3.24%p 증가였다. 합의 필터는 FP를 줄였지만 더 느렸으므로 CPU 효율 개선이라고 부르지 않는다. 모든 속도·재현율·정밀도 교환관계를 보존한다. 이전 실험의349ms와 이번439ms는 다른 실행이므로 두 수치로 개선량을 계산하지 않는다. 현재 비용은 실제 전처리·타일 선택·ORT·후처리를 포함하고 이미지 decode·차량 전체 pipeline은 제외한다.

## 사전 고정한 세 규칙

1. 작은 얼굴 증거: global face confidence0.01 이상,1920 canvas의 최소 변 길이64이하인 decoded proposal 중심의 사분면을 고른다. count→confidence 합 내림차순, 동률은 좌상·우상·좌하·우하 순으로 최대2타일을 실행한다. 증거가 없으면0타일이다.
2. 증거 밀도: global face confidence0.03 이상 전체 proposal에 같은 순서를 적용해 최대2타일을 실행한다. 증거가 없으면0타일이다.
3. 합의: 전체4타일을 실행하되 confidence0.1 미만 tile face는 global 또는 다른 타일 face와 IoU0.3 이상 지지가 있을 때만 유지한다. global·plate·0.1 이상 tile face는 보존한다. 동일 타일의 중복만으로 지지하지 않는다.

count는 NMS 이전 proposal 수이며 독립 얼굴 수가 아니다. padding 영역의 proposal도 gate에 영향을 줄 수 있다. global이 놓친 작은 얼굴은 selective gate에서도 놓칠 수 있다. 이번 개발 결과를 본 뒤 규칙이나 threshold를 다시 조정하여 같은 시험의 성능으로 제시하지 않았다.

## 기하·수치와 재현 범위

원 selected tiled ONNX의 최종 concat6개 dependency subgraph를 추출해 기존 weights·1920 letterbox·Resize·좌표 offset을 그대로 재사용했다. 고정 합성 입력에서 원 전체 graph와 추출 concat의 strict rtol0.001/atol0.0001 비교는 **max absolute error0**으로 통과했다. study 후처리와 변경하지 않은 앱 adapter의 동일 raw 출력도 비정방 원본·경계·threshold tie를 포함한 합성 차분검사에서 정확히 일치했다. 최종 순서는 classwise NMS→source clip이다.

이는 모든 실제 VAL의 독립 PyTorch raw tensor 비교를 새로 수행했다는 뜻이 아니다. 전용 합성6tests, 기존 후처리 회귀와 합친12tests/189subtests가 통과했다. 실행 전후 모델 manifest·weights·VAL·study source·adapter source 해시가 일치했다. 소스·모델·VAL·사전 계획 SHA는 공개 JSON의 `evidence`에 있으며 개인별 이미지 해시는 포함하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/privacy_efficiency_study.py freeze
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/privacy_efficiency_study.py extract
.\.venv\Scripts\python.exe scripts/privacy_efficiency_study.py run --plan-sha256 4a18c61245ffbc1fb79b1aeac847183dac5036f23063c406ce4da013d65d0933
.\.venv\Scripts\python.exe -m pytest tests/test_privacy_efficiency_study.py tests/test_detection_postprocess.py -q
```

기존 출력이 있으면 재사용을 거부한다. 실제 학습 환경·원본 기한·연구 사용조건은 [옵션 선택 절차](REDACTION-OPTION-SELECTION.md)와 [모델 이식 정책](MODEL-ARTIFACTS.md)을 따른다. 파생 관측·개별 예측은 로컬에만 두고 원VAL의 가장 이른 기한을 상속하며 새 hash-bound inventory에 등록한다. 기존 inventory나 원기한을 수정하지 않는다. weights·원본 사진을 공개 문서에 첨부하지 않는다.

## 다음 독립 시험 조건

기존에 본 모든 TEST는 회귀용이다. 새 최종 holdout은 과거 v4 TRAIN/VAL/TEST/중단 자료 전체의 source identity·SHA·near duplicate와 알려진 사람·번호판·recording을 배제하고 추론 전에 사전 동결해야 한다. 실제 인물·도로 독립성이 확인되지 않으면 unknown으로 기록한다. 새 취득 기록과 원기한을 보존하고, 개인정보 전체 GT와 음성 장면을 별도 검수하기 전에는 개인정보 보호 인수를 주장하지 않는다. 현재 새 자료 취득·최종시험은 수행하지 않았다.
