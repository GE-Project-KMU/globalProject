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

훈련 파일 SHA256은 `88cd0e6ffd80e1f3d3268537cf1daf5167817fab9d46e9657a860f4ee9a122c0`로 실행 중 변경하지 않는다. OpenCV CPU thread 1, 원본 디코딩 RAM 캐시 최대 4GiB를 쓴다. 캐시와 비캐시 이미지·라벨 동일성 및 캐시의 복사 보호를 검사했다. 공유 GPU의 실행 시간은 격리된 추론 속도 벤치마크가 아니다.

## 재현 명령

명령은 프로젝트 루트에서 실행한다. 원본 RDD2020 자료·기존 사용 릴리스는 기존 다운로드 및 보관 규칙을 따른다. 과거 격리 자료를 다시 확보하거나 그 잘못된 라벨을 재사용하지 않도록, 최소 이미지 해시/dHash/파일번호 블록만 담은 `POTHOLE-HISTORY-EXCLUSIONS.json`을 고정했다(SHA256 `3fc9525e28585b194574103f7b3d6d57b18ad42df17bdee3e5fbec0962348b3d`). 현재 준비 CLI는 이 snapshot을 검증하고 사용한다. 사진·라벨·인물 UUID·개별 사진 경로는 넣지 않았다. 원본 역사 릴리스와 snapshot의 정확한 동일성도 별도 검사한다. 이 준비 도구 변경은 실행 중인 훈련 소스와 동결 v4 릴리스를 바꾸지 않는다. 출력 디렉터리는 새 경로만 허용하며 동결 릴리스를 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/improve_pothole_diagnose.py
.\.venv\Scripts\python.exe scripts/improve_pothole_data.py data/training/research-pothole-rdd2020-v4
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_check.py data/training/research-pothole-rdd2020-v4/release.json
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_smoke.py data/training/research-pothole-rdd2020-v4/release.json
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_train.py data/training/research-pothole-rdd2020-v4/release.json data/training/pothole-nano-rdd2020-v4 --epochs 60 --batch-size 16 --input-size 640 --cache-gib 4
# model.json 및 최종 candidate 선택/임계값 동결이 끝난 뒤만 실행:
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_pothole_compare.py data/training/pothole-nano-rdd2020-v4
```

`POTHOLE-IMPROVEMENT-DIAGNOSIS.json`, `POTHOLE-IMPROVEMENT-SPLIT.json`, `POTHOLE-IMPROVEMENT-CONTRACT-v4.json`, `POTHOLE-IMPROVEMENT-BATCH16-SMOKE.json`에 원인·분리·소프트웨어 검증 근거를 남겼다. 마지막 비교의 집계 결과는 `POTHOLE-IMPROVEMENT-RESULTS.json`에 저장한다. 합성 bootstrap/F2 검사는 평가 코드 계약 검사이며 정확도 증거가 아니다.

실제 학습·최종 평가가 완료되기 전 수치를 채우지 않는다. 연구용 최종 가중치도 데이터의 비상업 라이선스, 작은 시험 표본, 한국 도로/실물 ZED 2i 검증 부재를 해소하지 않는다.
