# 포트홀 Nano 개선 실험

## 원인 진단과 사용 범위

기존 RDD2020 Nano는 India 1,800장으로 학습하고 Japan 375장으로 검증했다. 검증 AP50은 7 epoch에서 7.98%로 최고였고, 30 epoch에는 3.33%였다. 학습 loss는 9.27에서 2.65로 감소했다. 도로 외형의 국가 차이와 과적합이 의심되지만, 데이터와 학습 방법이 동시에 달라지므로 인과를 단정하지 않는다. 일본 검증 GT의 박스 면적 제곱근 중앙값은 42.3px로 India 학습의 68.1px보다 작았다. 원본 D40 라벨은 모델 예측을 근거로 수정하지 않았다.

기존 AdamW·단순 밝기/flip 방식 대신 [공식 Nano 설정](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/exps/default/yolox_nano.py) 및 [공식 학습 설정](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/exp/yolox_base.py)을 참고하여 native Mosaic/HSV, SGD parameter groups, warmup/cosine, EMA를 적용한다. Nano 구조와 SimOTA/native loss를 그대로 사용한다. 외부 모델의 성능을 이 프로젝트 정확도로 대체하지 않는다.

원본 자료는 [저자 RDD2020, DOI10.17632/5ty2wb6gvg.1](https://data.mendeley.com/datasets/5ty2wb6gvg/1), CC-BY-NC-3.0이다. 원본 tar SHA256 `e06f74463b2d7410fa7e02f478818025be9b051087f80e0c51631abb32c0af55`를 확인하고 XML의 literal `D40`만 사용했다. 다른 손상 코드를 포트홀로 치환하지 않는다. 연구전용이고, 개인정보 수동 검토와 운영 승인을 받았다는 의미가 아니다.

## 시험군 동결과 중단 이력

첫 개선 준비 v2는 기존 공식 RDD2020 릴리스와만 중복을 차단했다. 추가 감사에서 이미 폐기한 제3자 매핑 실험의 Japan TRAIN과 의도한 시험 후보 28장이 SHA로 일치했다. epoch1 step400에서 학습을 중단했고, 검증·최종 시험을 수행하거나 ONNX를 만들지 않았다. v2 가중치는 최종 후보가 아니다. v3 준비도 중단하여 역사적 사용 범위를 보완했다.

유효 v4는 다음 모두의 사진 SHA·dHash Hamming≤3 및 원본 파일번호 20개 블록을 새 최종 시험에서 제외했다.

- 기존 공식 RDD2020 전체 train/validation/test.
- 격리된 제3자 릴리스 전체: 라벨을 사용하지 않고 역사적 사진 식별을 제외하는 용도로만 읽었다.
- 이번 중단 v2의 TRAIN/validation. 평가하지 않은 v2 TEST는 그 사실을 기록하고 다른 모든 제외 조건을 통과한 경우만 남겼다.

v4는 train 5,352 / validation 402 / 최종 시험 86장으로 학습 전에 고정했다. 최종 시험에는 양성 43장, 음성 43장이 있다. India+Japan을 학습하며 Japan 파일번호 블록으로 validation/test를 분리했다. 실제 주행 recording/route ID가 없어 **물리 경로 독립은 검증하지 못했다**. 시험 표본이 작으므로 파일번호 블록 bootstrap과 보조 Wilson 구간을 보고하며, 좁은 통계 구간으로 현장 수용 기준을 주장하지 않는다.

v4 release SHA256은 `6c5618898d96dd032ae5d9579c8c79aa676a063d6557c1c2f361a37fe5a8388c`다. split seed는 시험 `20261104`, 검증 `20261105`, 음성 추출 `20261106`이다. 원본 만료 `2026-11-10T17:14:46.862846+00:00`을 그대로 상속했고 연장하지 않았다. 기존 Czech TEST는 어떤 새 학습 분할에도 넣지 않고 **이미 살펴본 회귀 벤치마크**로만 평가한다.

## 사전 고정한 후보

완료 모델 후보는 1개다. COCO Nano 공식 체크포인트에서 새로 전이하며, 중단 실험 가중치를 재사용하지 않는다.

| 항목 | 고정값 |
|---|---|
| 구조/입력 | YOLOX-Nano depth .33 / width .25 / depthwise, 640×640 |
| 학습 seed | 20261104 |
| epoch | 최대 60, 최소 15 이후 validation AP50 delta .001 / patience 12 epochs 정체 시 종료 |
| batch | 16; 실제 최대 GT13 사진을 Mosaic 100%로 구성하여 AMP forward/backward 스모크 통과 |
| optimizer | 공식 SGD momentum .9, Nesterov, 가중치 decay 5e-4, BN/bias decay 제외 |
| LR | 이미지당 .01/64, 3 epoch warmup, native YOLOX cosine, min ratio .05 |
| augmentation | 공식 Mosaic 확률 .5 / scale .5–1.5 / rotation10° / translate.1 / shear2°, HSV/flip, mixup 없음 |
| smoothing | native ModelEMA |
| 최종 noaug | 최대 일정의 마지막 10 epoch에서 Mosaic off + native L1; 조기 종료하면 이 단계에 도달하지 않을 수 있음 |
| checkpoint | 새 validation AP50 최대 EMA, 2 epoch마다 및 noaug 기간 매 epoch 평가 |
| 임계값 | 선정 checkpoint의 validation F2 최대, grid .01/.03/.05/.1/.15/.2/.25/.35/.5; 동률 recall→precision |
| AP 평가 | confidence floor .01 / NMS IoU .5 / 101-point AP50 |
| 최종 비교 | 같은 새 시험군에서 기존 모델과 개선 후보 1회 비교; 두 모델 모두 고정 .35 운영 지표도 별도 보고 |

실제 실행한 훈련 파일 SHA256은 `88cd0e6ffd80e1f3d3268537cf1daf5167817fab9d46e9657a860f4ee9a122c0`이며 실행 중 변경하지 않았다. 원본과 당시 helper 및 35개 계약 검증 결과를 실험 폴더에 바이트 그대로 보존했다. **훈련 종료 후** 향후 CLI에 quarantine/literal-D40/과거 사용 사진·블록/과거 Czech TEST 차단 guard만 추가했다. 현재 향후 CLI SHA256 `337be3debe9423677151ec72a20b68b123163c7de0f967b1f3dd5db1e5b7b4a4`는 실행 원본과 다르며, 이 변경으로 기존 가중치나 선택 임계값을 바꾸지 않았다.

OpenCV CPU thread 1, 원본 디코딩 RAM 캐시 최대 4GiB를 썼다. 캐시와 비캐시 이미지·라벨 동일성 및 캐시의 복사 보호를 검사했다. 공유 GPU의 실행 시간은 격리된 추론 속도 벤치마크가 아니다.

## 재현 명령

명령은 프로젝트 루트에서 실행한다. 원본 RDD2020 자료·기존 사용 릴리스는 기존 다운로드 및 보관 규칙을 따른다. 과거 격리 자료를 다시 확보하거나 그 잘못된 라벨을 재사용하지 않도록, 최소 이미지 해시/dHash/파일번호 블록만 담은 `POTHOLE-HISTORY-EXCLUSIONS.json`을 고정했다(SHA256 `3fc9525e28585b194574103f7b3d6d57b18ad42df17bdee3e5fbec0962348b3d`). 현재 준비 CLI는 이 snapshot을 검증하고 사용한다. 사진·라벨·인물 UUID·개별 사진 경로는 넣지 않았다. 원본 역사 릴리스와 snapshot의 정확한 동일성도 별도 검사한다. 이 준비 도구 변경은 실행 중인 훈련 소스와 동결 v4 릴리스를 바꾸지 않는다. 출력 디렉터리는 새 경로만 허용하며 동결 릴리스를 덮어쓰지 않는다.

clean checkout에서는 먼저 `scripts/research_detection_data.py fetch-yolox`, `fetch-pretrained`, `fetch-rdd2020`, `prepare-rdd2020`으로 공식 코드/초기 가중치/원본 자료 및 Czech 회귀 참조를 준비한다. 격리·중단 실험의 원본 메타데이터가 모두 있으면 snapshot 동일성을 반드시 직접 감사한다. 일부가 없으면 남은 식별자를 snapshot 범위와 대조하고 **원본 직접 감사 미실행/고정 snapshot 대체**라고 기록하며, 같은 전체 exclusion scope로 freshness를 검사한다. 누락을 원본 직접 감사 통과로 표시하거나 기존 snapshot을 축소하지 않는다.

```powershell
# 새 clone의 공식 자료 준비(기존 자료/만료일을 덮어쓰지 않는다):
.\.venv\Scripts\python.exe scripts/research_detection_data.py fetch-yolox
.\.venv\Scripts\python.exe scripts/research_detection_data.py fetch-pretrained
.\.venv\Scripts\python.exe scripts/research_detection_data.py fetch-rdd2020
.\.venv\Scripts\python.exe scripts/research_detection_data.py prepare-rdd2020
# diagnose는 당시 baseline 학습/예측 artifact가 있을 때만 실행한다.
.\.venv\Scripts\python.exe scripts/improve_pothole_data.py data/training/research-pothole-rdd2020-v4
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_check.py data/training/research-pothole-rdd2020-v4/release.json
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_smoke.py data/training/research-pothole-rdd2020-v4/release.json
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_train.py data/training/research-pothole-rdd2020-v4/release.json data/training/pothole-nano-rdd2020-v4 --epochs 60 --batch-size 16 --input-size 640 --cache-gib 4
# model.json 및 최종 candidate 선택/임계값 동결이 끝난 뒤만 실행:
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_compare.py data/training/pothole-nano-rdd2020-v4
# 완료된 실제 모델의 검증과 사전 동결한 비교 plan:
$potholeValidation = Get-Content data/training/pothole-nano-rdd2020-v4/verification-case.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe scripts/verify_trained_models.py data/training/pothole-nano-rdd2020-v4/model.json --image $potholeValidation.image --checkpoint data/training/pothole-nano-rdd2020-v4/best.pt --output data/training/pothole-nano-rdd2020-v4/verification-final.json
.\.venv\Scripts\python.exe scripts/compare_detector_models.py docs/reports/POTHOLE-APP-FRESH-COMPARISON-PLAN.json data/training/pothole-nano-rdd2020-v4/app-fresh-comparison.json
.\.venv\Scripts\python.exe scripts/compare_detector_models.py docs/reports/POTHOLE-APP-REGRESSION-COMPARISON-PLAN.json data/training/pothole-nano-rdd2020-v4/app-regression-comparison.json
.\.venv\Scripts\python.exe scripts/improve_pothole_publish.py
```

`POTHOLE-IMPROVEMENT-DIAGNOSIS.json`, `POTHOLE-IMPROVEMENT-SPLIT.json`, `POTHOLE-IMPROVEMENT-CONTRACT-v4.json`, `POTHOLE-IMPROVEMENT-BATCH16-SMOKE.json`에 원인·분리·소프트웨어 검증 근거를 남겼다. 마지막 비교의 집계 결과는 `POTHOLE-IMPROVEMENT-RESULTS.json`에 저장한다. 합성 bootstrap/F2 검사는 평가 코드 계약 검사이며 정확도 증거가 아니다.

당시 baseline 가중치와 frozen comparison plan은 완료 실험의 실제 SHA를 참조한다. raw/가중치는 Git에 배포하지 않는다. baseline을 새로 학습하려면 `docs/TRAINING-DETECTION.md`의 절차를 사용하고, 새 실험에서는 새 계획을 평가 전에 동결해야 한다. **현재 86장 시험은 이 실행이 끝난 뒤 후속 실험의 새로운 최종 시험으로 다시 부를 수 없다.** 후속 모델 개선에는 완료 v4를 포함한 전체 사용 이력과 별도 새 holdout을 준비하고 새 버전의 snapshot/plan을 만들어야 하며, 기존 snapshot을 덮어쓰거나 축소하지 않는다.

## 실제 완료 결과

사전 등록한 plateau 조건으로 50 epoch에서 종료했고, validation AP50 최고 68.39%의 epoch38 EMA를 선택했다. Mosaic-off/L1은 epoch51부터 예정되어 있었으므로 **실제 이번 실행에서는 도달하지 않았다**. validation F2로 선택한 confidence는 .1, NMS는 .5다. 새 시험 결과를 보고 가중치·confidence·NMS를 재선택하지 않았다.

| 실제 앱 CPU 평가 | 기존 Nano | 개선 Nano |
|---|---:|---:|
| 새 Japan 86장 AP50, floor .01 | 13.29% | **61.81%** |
| 같은 새 시험, 고정 .35 recall | 12.90% (TP8/FN54) | **61.29% (TP38/FN24)** |
| 같은 새 시험, 고정 .35 precision | 30.77% (FP18) | **69.09% (FP17)** |
| 기존 Czech 회귀 231장 AP50 | **13.11%** | 10.70% |
| 같은 Czech, 고정 .35 recall | **11.68%** | 9.14% |

새 시험 GT는 포트홀 62개이고, validation에서 선택한 .1 운영점에서는 recall72.58% / precision59.21%(TP45/FP31/FN17)다. 새 시험 21개 파일번호 블록의 paired bootstrap500회 AP50 차이 95% 구간은 **+34.28~+64.84%p**다. 실제 route 독립이 확인된 구간이라는 뜻은 아니다. Czech에서는 오히려 악화되었으므로 **국가 전반/현장 정확도가 일괄 향상됐다고 주장하거나 운영 모델로 자동 승격하지 않는다**.

실제 앱/ONNX Runtime/PyTorch CPU 양성 validation 입력 검증에서 raw 및 후처리를 모두 통과했다. 기본 rtol .001 / atol .0001을 바꾸지 않았다. 앱↔ORT raw는 완전 일치했고, PyTorch와도 3개 검출이 모두 일치했다(최대 bbox차 0.0000763px, confidence차 0.000000238). 앱 CPU의 새 시험·Czech 회귀에서 baseline/candidate **4개 AP 및 TP/FP/FN 집계가 native 평가와 정확히 일치**했다. 새 시험과 회귀 비교 plan은 각각 한 번 실행했다.

기존 실행 35개 계약 검증 원본을 보존했고, 사후 입력 guard13개 및 향후 CLI 계약35개가 통과했다. 공개 집계에서 사진·개별 ID·좌표·영상 해시·가중치 경로가 차단되는 9개 계약도 통과했다. 이 소프트웨어 검증 숫자는 모델 정확도 증거가 아니다.

최종 공개 근거는 `POTHOLE-IMPROVEMENT-FINAL.json`, `POTHOLE-APP-FRESH-COMPARISON-RESULTS.json`, `POTHOLE-APP-REGRESSION-COMPARISON-RESULTS.json`이다. 사진·개별 ID/path/bbox·가중치 원본은 로컬 ignored 폴더에 두었다. manifest SHA256은 `28efe86e7c9eacdf90d11caf0eb942dbb19b0a4d6830229069b3277cc8a45a6a`, ONNX SHA256은 `9b956d8ead74d1b11db0d1c43bac9ae5bdda2b0d6b843062319ae28de2f2f269`다.

양성/음성 균형 시험은 실제 주행의 포트홀 발생 빈도를 재현하지 않으므로 precision을 현장 오탐/km 지표로 해석하지 않는다. 데이터와 학습 방법을 동시에 개선하여 개별 요소의 인과적 기여를 분리한 ablation 실험도 아니다. 연구용 최종 가중치도 데이터의 비상업 라이선스, 작은 시험 표본, 한국 도로/실물 ZED 2i 검증 부재를 해소하지 않는다.
