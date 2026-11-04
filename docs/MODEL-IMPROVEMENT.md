# 모델 정확도 개선 통합 runbook

이 문서는 첫 개선 실험의 보존 기록이다. 이후 시연 분리·차선 지원 런타임 검증·포트홀 추가 학습·개인정보 효율 연구는 [후속 구현·검증](FOLLOWUP-IMPLEMENTATION.md)에 별도로 기록한다.

2026-11-04 기준. 완료된 차선·포트홀·얼굴·번호판의 같은 시험군 비교와 재현 조건을 기록한다. 얼굴·번호판은 VAL에서 옵션·임계값을 먼저 고정한 뒤 수행한 selected tiled actual-app 평가를 기록한다. 원본·가중치·개별 예측은 공개하지 않는다.

## 1. 현재 판정과 기록 범위

| 분야 | 이번 개선 단계 | 새 동일 시험 baseline→후보 | 이전 시험 회귀 | 현재 판정 |
| --- | --- | --- | --- | --- |
| 차선 도막 A2D2 개선v2 portable | 학습·사전 고정 진단 정확도 비교 완료 | 새100장 paint IoU28.42%→48.75%(+20.32%p) | 과거100장39.44%→58.72%(+19.28%p) | 목표75% 미달. 원형/portable Torch raw parity FAIL 유지, 진단용 정확도만 측정, 운영 차단 |
| 포트홀 개선 | 학습·엄격 수치/후처리·actual-app 동일 시험 비교 완료 | 새86장 AP50 13.29%→61.81%(+48.52%p) | 과거 Czech231장 AP50 13.11%→10.70%(-2.42%p) | 목표 AP50 80% 미달. 고정 양성 VAL의 strict3엔진 raw+후처리 통과; Czech 회귀 하락·연구 전용/운영 미승인 |
| 얼굴·번호판 개선 | VAL 선택·strict 검증·같은 시험 actual-app 비교 완료 | 새400장 face AP50 65.55%→79.98%, plate 95.92%→97.94% | 과거400장 face AP50 50.07%→80.70%, plate97.03%→97.95% | 선택점0.01에서 face Recall86.49%로 두 클래스95% 목표 미달, plate98%. face Precision24.78%→17.35% 하락; 연구 전용/운영 미승인 |

차선의 개선 수치는 데이터 확대·해상도·학습 recipe를 함께 바꾼 후보 하나의 비교다. 어느 단일 변경의 정확도 기여를 분리한 ablation이라고 해석하지 않는다. pooling 수치 진단은 별도의 같은 checkpoint/같은 VAL 연산 표현 비교다.

완료 여부를 네 층으로 구분한다: 파일 실행 가능(ready), 출처/수치 변환 계약 검증, 고정 공개 시험의 목표 달성, 독립 실차/개인정보/전체 시스템 인수. 앞 단계의 성공으로 뒷 단계를 통과 처리하지 않는다. 연구 모델은 production_approved=false를 유지한다.

## 2. 초기 baseline와 후속 개선 기록의 범위

[초기 모델 학습 기록](MODEL-TRAINING.md)은 A2D2 v1·포트홀 v1·redaction v3와 당시 고정 검증 사례를 보존하는 문서다. 초기 table과 strict 통과 기록을 새 후보의 검증 성공으로 해석하지 않는다. 문서의 초기 범위 표기와 README의 후속 도막 개선 연결은 이미 반영돼 있다.

[모델 이식 안내](MODEL-ARTIFACTS.md)의 strict build/install 및 권리 guard는 그대로다. lane의 별도 진단 정확도 gate가 raw 검증 실패 모델의 verified package 생성을 허용하지 않는다. 연구용 파일 실행, 수치 검증, 공개 시험 목표, 실차·개인정보 인수는 서로 다른 상태다. 새 결과는 초기 수치나 실패 기록을 덮어쓰지 않고 모델/manifest SHA로 구분한다.

## 3. 재현 환경과 변경 동결

실제 사용 환경은 Windows 프로젝트 내부 Python3.11.9/PyTorch2.7.1+cu126/torchvision0.22.1+cu126, NumPy1.26.4/OpenCV4.10/ONNX1.17/독립 ORT1.20.1이다. 앱 Python3.14의 canonical inference는 ORT1.31.1 CPU이며 OpenCV5는 영상 처리에만 쓴다. 초기 OpenCV DNN engine 실패를 대체하지 않고 보존한 이유는 [변환 안내](MODEL-ARTIFACTS.md) 참조.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py bootstrap
.\.venv\Scripts\python.exe scripts/setup_training.py install
.\.venv\Scripts\python.exe scripts/setup_training.py run -m pip check
```

[학습 환경](TRAINING-ENVIRONMENT.md)의 고정 requirements/Windows lock을 따른다. Linux는 Windows venv를 복사하지 않고 별도 프로젝트 내부 환경을 구성한다. 현재 개선 run의 실제 학습·평가 증거는 Windows이며 전체 Linux/Docker 학습 실행 완료로 표시하지 않는다.

TEMP/TMP/TMPDIR, pip/Torch/CUDA/HF/XDG/matplotlib/pycache는 launcher를 통해 프로젝트 `.cache` 안으로 지정한다. GPU 학습이 진행 중일 때 의존성·trainer·factory·metric 코드를 변경하지 않는다. 전역 드라이버/SDK/OS Python을 설치하지 않는다. CPU export helper와 verifier worker가 자기 프로세스에서 CUDA를 숨기므로 학습을 이어갈 셸에 CUDA_VISIBLE_DEVICES=''를 전역으로 남기지 않는다.

실제 실행 기록과 현재 재현·검증 도구를 구분하여 대조했다. 현재 파일 SHA가 다르다고 과거 학습/시험을 새 소스로 실행한 것으로 소급하지 않는다. 포트홀의 실행 원본88cd와 future CLI337be는 명확히 구분하며 checkpoint SHA와도 종류가 다르다. portable export 당시 소스도 별도 로컬 snapshot으로 보존한다.

| 구분 | 코드 | 실행/동결 기록 SHA256(있는 경우) | 현재 파일 SHA256 |
| --- | --- | --- | --- |
| 차선 실행 동결 | `scripts/improve_lane.py` | `6e15f29a5a98156d531711b232d9e791f96ddd799d5ac98d388a3023ed8cf377` | `6e15f29a5a98156d531711b232d9e791f96ddd799d5ac98d388a3023ed8cf377` |
| 차선 portable export 당시 소스 | `scripts/export_lane_portable.py` | `23d7f040bb850c06087f74c3cd05ee670b130020848a158866db1710217dd7cc` | `6f4f8a6664ad4933d6680112bbb5439979a44c62343f0c5c4b71a9f8545ee6d8` |
| 차선 factory | `agent/lane_network.py` | `648961d31d6b6bd77f6f5b54f447e5f08a6d7c3928d133b15ed246ca1fd07f72` | `648961d31d6b6bd77f6f5b54f447e5f08a6d7c3928d133b15ed246ca1fd07f72` |
| 차선 metric | `agent/lane_metrics.py` | `76e08ebcb35fd3bec34e81f1d1de9799e888344fc384769c71e3294a869a305c` | `76e08ebcb35fd3bec34e81f1d1de9799e888344fc384769c71e3294a869a305c` |
| 포트홀 실제 실행 원본 / 현재 future CLI | `scripts/improve_pothole_train.py` | `88cd0e6ffd80e1f3d3268537cf1daf5167817fab9d46e9657a860f4ee9a122c0` | `337be3debe9423677151ec72a20b68b123163c7de0f967b1f3dd5db1e5b7b4a4` |
| 비식별화 학습 동결(최종 결과 확인 전) | `scripts/improve_redaction_train.py` | `43ae1e92d49d51f2e52c3e22939fa3ccba8f18831557ea3e0832e3eb91c1dc26` | `43ae1e92d49d51f2e52c3e22939fa3ccba8f18831557ea3e0832e3eb91c1dc26` |
| 비식별화 표준 factory | `agent/redaction_network.py` | `9d26b063b22b54b1606cb33be1291b6e71f57945849aab21119764e1c847ca5d` | `9d26b063b22b54b1606cb33be1291b6e71f57945849aab21119764e1c847ca5d` |
| 비식별화 조건부 타일 factory | `agent/redaction_tiled_network.py` | `f8a1fe9121ad36cbab1e14a1f2477dd821ec6f04ee7bd6fd862996967dbd376f` | `f8a1fe9121ad36cbab1e14a1f2477dd821ec6f04ee7bd6fd862996967dbd376f` |
| VAL 옵션 도구(현재 코드 SHA; 실행본과 구분) | `scripts/select_redaction_inference.py` | 실행 SHA로 주장하지 않음 | `c880ce60b510c8f042220c0b9fa46312bb6c6b8542f86ecd3a616ff2d24de73a` |
| 차선 증거 작성 당시 verifier / 현재 gate 보완본 | `scripts/verify_trained_models.py` | `3e42b4eda3275ccf09a6ed07844ff5cdbde163f06f967e61df765852fd030bd3` | `a7a349179feb919f1515a9088618f0cf5e7e9904fc7547c4b9a8bb589c8f9794` |
| 차선 비교 당시 소스 / 현재 gate 보완본 | `scripts/compare_lane_models.py` | `3b879336dbe5b84925507a593f0f54f3bee6191fec25dce0ce32907576da3748` | `01eab04f519fe002683bcae15992c646df5db2fc94455f83753b764eaa61ee5e` |
| 앱 runtime 현재 snapshot | `agent/models.py` | 실행 SHA로 주장하지 않음 | `bb5e3590e6d299b494fd9ebfbdd3d3374fdfa7f9b1284180ceaa5f3d3dadba74` |

차선 portable의 현재 helper는 metadata gate 이름을 보완한 버전이며 당시 export 소스와 learned state/mean9 변환은 같다. 현재 verifier/comparison의 추가 일관성 guard를 과거 benchmark 재실행으로 표시하지 않는다. 모델별 영수증에는 release/checkpoint/ONNX/manifest/calibration/verification SHA가 구분되어 있다. 모델의 개별 이미지 지문이나 원본 경로는 이 코드 표에 넣지 않는다.

코드 snapshot과 CI 실행은 각 기록 시점의 revision에 해당한다. 이후 정리 커밋이 이미 완료된 CI 실행에 포함됐다고 소급하지 않는다.

## 4. 모든 모델에 공통인 데이터·선택 프로토콜

1. 기존 train/validation/TEST·중단/격리 시도의 학습 노출을 누출 제외에 포함한다. 이미 본 TEST는 회귀 benchmark로 표시하고 새 시험으로 재명명하지 않는다.
2. 새 source/recording/identity 분할·uniform/고정 seed 선택과 제외 정책을 학습 전에 기록한다. SHA와 유사 이미지 연결, 원본/파생 관계, 알려진 recording/plate identity 등 그룹을 분리한다. 실제 물리 도로/인물 독립성을 모르면 false/unknown으로 남긴다.
3. 원본 크기·게시자 MD5/해시·라벨 class mapping·bbox/mask·사용조건·기한을 확인한다. 게시자 라벨과 기계 검수가 사람의 개인정보 검수나 독립 마모 GT를 대신하지 않는다.
4. checkpoint·confidence·NMS·입력/타일 옵션은 train/VAL에서만 선택한다. 사용한 후보 수와 grid/선택 규칙을 먼저 고정한다. fresh TEST 출력/summary/원model.json의 점수까지 봉인한다.
5. 실제 앱 CPU 경로와 독립 ORT/Torch를 검증한다. raw FAIL을 마스크/박스 일치로 덮거나 허용 오차를 늘리지 않는다. 같은 raw tensor에 실제 class별 NMS→source clip 후처리를 적용한다.
6. 최종 artifact·threshold·옵션을 고정한 후 baseline과 후보를 같은 새 시험에서 각각 한 번 비교한다. 과거 TEST 회귀는 별도로 보고한다. 시험 후 recipe/threshold를 다시 고르지 않는다.
7. 원본·sample ID·이미지 해시·전체 파일명·개별 bbox/마스크/경로/예측은 로컬에서만 보관한다. 공개 집계에는 대상 수·지표·관계 없는 모델/데이터셋 SHA·한계만 남긴다.

## 5. 차선: 완료된 diagnostic-only run

- 데이터: 기존 train400+추가400 후보 중 연결 중복1 제외→train799, VAL100, 새 recording2개100장, 과거100장 회귀.
- 학습: frozen recipe 한 개, native640 patches/full1024, BN 통계 고정, 차등 LR, EMA, max60/min15/patience12. 실제53epoch 종료·best41.
- VAL: 고정14 threshold에서 portable도0.4. 도막 IoU약0.663703. source_id순 첫3 GT 양성은 모두 app↔독립 ORT strict PASS·세 엔진 최종 마스크100%이나 Torch raw FAIL.
- 변환: 원형과 원FAIL 불변. 학습 checkpoint 그대로 global1 pool9개만 explicit mean/ReduceMean으로 내보낸 portable. source/weights/threshold 고정, no_new_gradient_steps=true, checkpoint_raw_tensor_parity_unresolved 차단.
- 평가: default strict gate는 계속 거부. 별도 사전 고정 diagnostic gate/plan과 root preflight를 통해서만 새100/회귀100을 각 한 번 측정했다. 결과는 운영/변환 증명서가 아니다.
- 새 시험 precision/recall: baseline44.20%/44.33%→후보66.41%/64.69%. 회귀 precision/recall:55.88%/57.28%→77.65%/70.66%.
- 비용: 새 시험 CPU predict median 약0.097→0.535초. 데이터 I/O·capture·검수·보고 전체 FPS가 아니다. 출처별 true O/G/V가 없어 R-MAE는 null이다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_a2d2_improvement.py --plan-only
.\.venv\Scripts\python.exe scripts/prepare_a2d2_improvement.py
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_lane.py LOCAL_DEV_MANIFEST LOCAL_FULL_RELEASE NEW_LOCAL_TRAINING_OUTPUT --epochs 60 --min-epochs 15 --patience 12 --batch-size 4 --patch-size 640 --input-size 1024 --seed 20261104
.\.venv\Scripts\python.exe scripts/calibrate_lane_model.py LOCAL_DEV_MANIFEST LOCAL_ORIGINAL_MANIFEST NEW_LOCAL_CALIBRATION_REPORT
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/export_lane_portable.py --original-manifest LOCAL_ORIGINAL_MANIFEST --original-manifest-sha256 PINNED_ORIGINAL_SHA --checkpoint LOCAL_CHECKPOINT --checkpoint-sha256 PINNED_CHECKPOINT_SHA --output NEW_LOCAL_PORTABLE_DIRECTORY
.\.venv\Scripts\python.exe scripts/calibrate_lane_model.py LOCAL_DEV_MANIFEST LOCAL_PORTABLE_MANIFEST NEW_LOCAL_PORTABLE_CALIBRATION_REPORT
.\.venv\Scripts\python.exe scripts/verify_trained_models.py LOCAL_PORTABLE_MANIFEST --image PREDECLARED_LOCAL_POSITIVE_VAL --checkpoint LOCAL_CHECKPOINT --output NEW_LOCAL_CANONICAL_PROBE_REPORT
.\.venv\Scripts\python.exe scripts/compare_lane_models.py PREDECLARED_FRESH_DIAGNOSTIC_PLAN NEW_LOCAL_FRESH_REPORT
.\.venv\Scripts\python.exe scripts/compare_lane_models.py PREDECLARED_REGRESSION_DIAGNOSTIC_PLAN NEW_LOCAL_REGRESSION_REPORT
```

명령은 새 checkout/새 출력과 동일한 출처·만료·사전 계획을 전제로 한다. 이미 완료한 기존 폴더에 다시 실행하지 않는다. [lane 결과](reports/LANE-IMPROVEMENT-RESULTS.json), [재현](LANE-IMPROVEMENT-REPRODUCTION.md), [수치 진단](LANE-PORTABILITY-DIAGNOSIS.md)을 함께 읽는다.

## 6. 포트홀: 최종 공개 집계 기반 완료 기록

이 항목은 공개 최종 보고서와 actual-app fresh/regression 집계, 모델 점수를 포함하지 않는 frozen training-config의 실행 구성에 근거한다. 새 시험 개선과 Czech 회귀 하락을 함께 보존하며 범용/현장 정확도 향상을 주장하지 않는다.

- release와 누출 분리 근거: 공식 RDD2020 literal D40, 새 Japan filename20-block86장(양성43장·GT62개·proxy21블록). release SHA `6c5618898d96dd032ae5d9579c8c79aa676a063d6557c1c2f361a37fe5a8388c`. 과거 시험·격리/중단 학습 노출을 제외한 동결 snapshot 근거를 유지하며 실제 route/recording 독립성은 미확인. 원래 metadata 직접 audit도 snapshot 제외범위 축소 없이 검증됨
- 학습 recipe/완료 epoch/선택 checkpoint: 실제50epoch에서 사전 고정 validation plateau로 종료, best38/VAL AP50 68.3948%. noaug/L1 단계는 설정돼 있었지만51epoch 시작 전에 종료되어 미도달. 실제 실행 trainer SHA `88cd0e6ffd80e1f3d3268537cf1daf5167817fab9d46e9657a860f4ee9a122c0`, 실행 helper SHA `ff33c7cfcec15c4038f476127354f6e35a4052e8c03f4e4ec91a33c792572242`. 현재/future CLI SHA `337be3debe9423677151ec72a20b68b123163c7de0f967b1f3dd5db1e5b7b4a4`는 사후 input guard·선택적 direct-original audit 보완본이며 이 학습을 실행한 소스라고 쓰지 않는다. 학습 checkpoint SHA는 별도의 `2fb2c082964e81b918339a2a26f1a37fc2cd655aa78e9d98a2a38ed04fd89375`다
- raw 및 앱 후처리 strict 검증: 고정 실제 양성 validation에서 앱 ORT1.31.1/독립 ORT/PyTorch CPU raw(strict rtol0.001/atol0.0001)·class별 NMS 이후 bbox/confidence 후처리 모두 통과. numerical_parity_passed/postprocess_parity_passed/artifact_contract_verified=true. Native와 actual-app의 새/회귀 baseline·후보 AP/TP/FP/FN도 동일. 이는 전체 TEST의 모든 raw tensor나 실차 인수 검증을 뜻하지 않는다. 모델 manifest SHA `28efe86e7c9eacdf90d11caf0eb942dbb19b0a4d6830229069b3277cc8a45a6a`, ONNX SHA `9b956d8ead74d1b11db0d1c43bac9ae5bdda2b0d6b843062319ae28de2f2f269`
- 같은 새 시험 baseline/후보 AP50·Recall·Precision·지원 수·CI: 동일 새86장 full AP50(floor0.01) 13.2941%→61.8149%(+48.5207%p). 후보 VAL F2 선택 confidence0.1에서 Recall72.5806%/Precision59.2105%; 사전 고정 confidence0.35 비교는 baseline Recall12.9032%/Precision30.7692%→후보61.2903%/69.0909%. 후보 AP50의 filename-block bootstrap95% 구간은50.08~74.94%, paired 개선 구간은+34.28~+64.84%p(500반복, proxy21그룹). 작은 balanced cohort의 기술적 구간으로 실제 route 독립성·현장 분포·FP/km를 보장하지 않는다
- 과거 시험 회귀와 인수 목표 미달: 과거 Czech231장 full AP50 13.1119%→10.6962%(-2.4157%p) 하락을 보존한다. 같은 fixed0.35에서 Recall11.6751%/Precision31.9444%→9.1371%/30.5085%도 하락한다. 후보 선택점0.1에서는 Recall15.2284%/Precision24.0%이며 서로 다른 운영점을 같은 조건의 개선으로 포장하지 않는다. AP50 80% 목표 미달, CC-BY-NC-3.0 연구/국내 ZED2i 실차·상업 사용·사람의 개인정보 검수 미확립. 자료/recipe 동시 변경이라 개별 개선 원인 분리 실험이 아님

원본 RDD literal D40만 pothole로 매핑하고 잘못 변환된 초기 mirror 라벨은 학습 성공으로 되살리지 않는다. filename block/국가/해시 분리로 실제 route 독립성이 입증됐다고 쓰지 않는다. 학습 노출이 있었던 중단 시도의 TRAIN/VAL도 새 TEST 제외에 포함한다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run LOCAL_FROZEN_EXECUTED_TRAINER_88CD PINNED_LOCAL_RELEASE NEW_LOCAL_OUTPUT --epochs 60 --batch-size 16 --input-size 640 --cache-gib 4
```

이 값은 frozen executed_argv와 대조한 실제 최대60epoch/batch16/input640/cache4GiB 설정이다. 실제로는 사전 고정 early stop에 의해50epoch에서 끝났고51epoch부터의 noaug/L1에 도달하지 않았다. 정확 재현에는 SHA88cd의 실행 소스 snapshot을 사용한다. 현재 `scripts/improve_pothole_train.py`는 SHA337be의 후속 input guard·audit CLI이므로 위 실행 원본과 구분한다. [초기 검출기 기록](TRAINING-DETECTION.md)은 baseline 범위다.

공개 근거: [포트홀 최종](reports/POTHOLE-IMPROVEMENT-FINAL.json), [새86장 actual-app 비교](reports/POTHOLE-APP-FRESH-COMPARISON-RESULTS.json), [Czech231장 회귀](reports/POTHOLE-APP-REGRESSION-COMPARISON-RESULTS.json). 게시 필터 검사는 image_sha/pixelcoords 차단 보강을 포함한9개이며 모두 통과했다. 이9개는 출판 계약 검사 수로, 모델 정확도 증거와 구분한다. 실행 원본 계약35개/사후 CLI 계약35개/input guard13개와도 서로 다른 집계다.

## 7. 얼굴·번호판: VAL 선택 후 완료된 actual-app 비교

두 genuine YOLOX-Nano specialist는 공통 960 입력에서 face960·내부 bilinear plate640을 사용한다. 타일1920 옵션은 같은 face/plate state를 공유하며 전역 얼굴1회+4quadrant 얼굴+번호판1회다. 두 옵션은 VAL에서 선택·검증과 artifact 동결을 마친 뒤 사전 계획에 따라 새400장과 과거400장을 각각 평가했다. 표준/타일이 자동 운영 default라는 뜻이 아니다.

- 학습/실제 source SHA·완료 상태: batch4로 face60epoch·plate20epoch 완료. 두 genuine Nano의 표준960 학습 결과와 selected tiled1920 actual-app 결과를 구분한다. 실행 trainer SHA43ae1e92d49d51f2e52c3e22939fa3ccba8f18831557ea3e0832e3eb91c1dc26, 표준 factory SHA9d26b063b22b54b1606cb33be1291b6e71f57945849aab21119764e1c847ca5d는 동결했다. 선택된 checkpoint SHA6c985f7e3afb0fa061cbb82d7ec7b304eb5f9fbfcd0cf9f775dbab28873c82ad를 두 옵션이 공유하며 추가 gradient 학습은 없다. native 표준 training 점수는 이 절의 최종 비교 수치로 대체하지 않는다
- VAL508(face400/plate108)의 두 옵션 결과/threshold/타이밍과 고정 선택: 같은 VAL508장만 앱 CPU로 평가했다. 고정7개 grid에서 표준960과 tiled1920 모두 confidence0.01을 선택했다. 표준 min-class Recall86.76%/macro Precision68.01%, tiled92.50%/61.21%로 둘 다95% 미달이므로 사전 규칙의 min Recall 우선에 따라 tiled를 선택했다. VAL predict median은87.9→349.2ms, p95는96.1→370.4ms였다. option-selection-freeze SHA d2bcdf37495c5d1cb5be29ca36ba95a33196547ea1483692635daf1a81c0414c. 이 선택과 두 임계값·모델/ONNX SHA를 고정한 뒤 TEST를 열었으며 TEST 점수로 재선택하지 않았다
- strict raw/최종class·bbox·confidence parity: 고정 face·plate 양성 VAL 사례에서 앱 ORT1.31.1/독립 ORT/PyTorch CPU의 기본 rtol0.001/atol0.0001 raw 및 class·NMS 후 bbox/confidence 비교가 모두 통과했다. 사례는 고정 confidence에서 GT양성과 같은 클래스 예측이 있는 사전식 순서의 첫 사례이며 정답 여부로 고르지 않았다. 이후 독립 IoU0.5 GT match 검사도 교체 없이 수행했다. 전체 TEST의 모든 raw tensor 검증으로 확대하지 않는다. selected manifest SHA971dd630295aaffce09760b73ba07c080e92a65a76daad5d4a91bf0a14642edc, ONNX SHA bf54a5ad4211e945e71c84502a75a2d32e03de397883d70df8c64881c515aa3a
- 동일 새 시험의 클래스별 baseline→선택 후보 Recall·Precision·AP50 및95% 목표 판정: 새400장 동일 known-class conditional 시험에서 기존 v3 Nano640→선택 tiled1920의 face AP50 65.55%→79.98%(+14.42%p), Recall71.05%→86.49%, Precision24.78%→17.35%; plate AP50 95.92%→97.94%(+2.03%p), Recall96%→98%, Precision94.12%→92.89%. face 미검출은 줄었으나 오탐 증가와 정밀도 하락을 함께 보존한다. face Recall95% 목표는 미달이다. CPU predict median26.7→350.9ms, p95 32.2→381.3ms로 느려졌다. 전처리·NN·후처리를 포함하되 decode/차량 전체 pipeline은 제외한 측정이며 현장 FPS가 아니다
- 과거 시험 회귀·FP/누락·계산 비용·한계: 과거400장 별도 회귀에서 face AP50 50.07%→80.70%, Recall58.82%→89.08%, Precision16.49%→16.77%; plate AP50 97.03%→97.95%, Recall97.5%→98.5%, Precision96.53%→95.17%. 이미 본 과거 시험이므로 새 독립 인수 근거가 아니다. 같은 인물·실제 도로 독립성, 음성만 있는 주행 FPR, 누락 없는 개인정보 GT는 확인되지 않았다. WIDER CC-BY-NC-ND4.0/CCPD MIT 제약과 비상업 연구 전용·production_approved=false·project_acceptance_passed=false를 유지한다

원래 model.json에도 TEST AP가 추가되므로 옵션 선택 전 봉인한다. 안전 입력은 training-config/candidate-freeze/VAL confidence-calibration/provenance/best.pt/model.onnx와 validation-only manifest/source proof다. 전체 release는 opaque SHA로만 검증하고 새 TEST GT를 JSON으로 읽어 선택하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_redaction_train.py train --batch-size 4
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/select_redaction_inference.py export --training-dir LOCAL_TRAINING_DIRECTORY --release PINNED_LOCAL_RELEASE --release-sha256 PINNED_RELEASE_SHA --validation-manifest LOCAL_VALIDATION_ONLY_MANIFEST --validation-sha256 PINNED_VAL_SHA --output NEW_LOCAL_OPTIONS_DIRECTORY
.\.venv\Scripts\python.exe scripts/select_redaction_inference.py validate --options-root LOCAL_OPTIONS_DIRECTORY --export-receipt-sha256 PINNED_EXPORT_RECEIPT_SHA --public-report NEW_AGGREGATE_REPORT
```

같은7confidence grid/.5 NMS에서 최소 클래스 Recall95%를 만족한 옵션이 있으면 macro precision→min recall, 둘 다 미달이면 min recall→macro precision, 완전 동률이면 표준을 선택한다. 학습 checkpoint는 같지만 원본→1920→960/640 리샘플링과 원본→960→640은 달라 실제 VAL 경로로 비교한다. WIDER의 다른 클래스·CCPD의 얼굴 등 미주석 출력을 FP로 평가하지 않는 known-class conditional 범위를 유지한다. 이 수치를 전체 개인정보 제거 보증으로 바꾸지 않는다.

[옵션 선택 절차](REDACTION-OPTION-SELECTION.md), [타일 연구 계약](REDACTION-TILED-RESEARCH.md), [수치 검증·이식](MODEL-ARTIFACTS.md)을 연결한다. 최종 TEST 평가는 동결·실제 gate·사전 고정 계획 이후 수행했다. [얼굴·번호판 최종 집계](reports/REDACTION-IMPROVEMENT-RESULTS.json)와 [결과 설명](reports/REDACTION-IMPROVEMENT-RESULTS.md)을 근거로 하며 개별 사진·bbox·이미지 해시는 공개하지 않는다. 표준/tiled 비교에는 입력 재표본화 경로 차이도 포함되며 기존 baseline과의 향상을 타일만의 인과 효과로 주장하지 않는다.

## 8. 소프트웨어·모델 검증과 보관 마무리

최종 로컬 회귀는383tests와189subtests가57.21초에 통과했다. 실제 GET/POST/PATCH/PUT/DELETE 응답을 검사했고 새 HTTP5xx와 서버 오류 로그는0건이었다. Chromium UI20개와 TypeScript/build도 통과했으며 이번 모델 개선 작업에서 UI는 변경하지 않았다. [GitHub Actions](https://github.com/GE-Project-KMU/globalProject/actions/runs/41742146998)는 Python/live HTTP/vision, PostgreSQL/PostGIS migrations·HTTP, TypeScript/build/Chromium UI의3개 job 모두 통과했다. 검증한 소스 revision은 `9670abe22c5a368901c28552a94338cb56bb44c8`이며 이후 문서 정리 커밋을 이 실행에 소급 포함하지 않는다. 로그·실행 증거의 공개 범위는 [테스트 기록](TESTING.md)에 따른다. 원본 로그나 개별 payload는 공개하지 않으며 테스트 성공을 실제 모델 accuracy·실차 인수 성공으로 바꾸지 않는다. [소프트웨어 집계](reports/IMPROVEMENT-SOFTWARE-REGRESSION.json)를 함께 읽는다.

새 원본·파생물은 기존 수집 기한을 그대로 상속한다. 얼굴/번호판 관련 원본은 원래7일, RTK는 원래90일이며 파일 mtime·재학습/다운로드 영수증으로 기한을 연장하지 않는다. 원본·학습/시험 crop·preview·개별 예측·검증보고·이미지 해시·개인별 이름이 든 metadata는 정확 파일 SHA/근거로 등록한다. 가중치/최소 provenance/공개 aggregate는 별도로 보존하되 소스 만료·철회 시 모델 재검증 차단을 유지한다.

기존 inventory는 수정하지 않고, 모든 모델의 평가·옵션 동결·개별 파일 쓰기가 끝난 뒤 새 plan/inventory를 별도로 만든다. old-active 전체 존재/SHA/evidence 확인, old+new union coverage, changed-file 거부를 검증한다. 자동 삭제 탐색·재귀 삭제가 아니며 manual purge의 기본은 dry-run이다. 실제 --apply는 별도 명시와 원래 만료 조건이 있어야 한다. [연구 보관 절차](RESEARCH-RETENTION.md)를 따른다. 보관 작업은 정확 파일과 원래 기한을 기준으로 수행하며 학습/평가 완료 기록과 분리해 남긴다.

[`MODEL-TRAINING`](MODEL-TRAINING.md)의 초기 범위·[`MODEL-ARTIFACTS`](MODEL-ARTIFACTS.md)의 strict gate·[`README`](../README.md)의 운영/실차 미검증 상태를 함께 읽는다. 새 가중치는 프로젝트 안에만 보관하며 Git/외부 Release/운영 bundle에 자동 등록하지 않는다.

최종 보관 검증에서 기존 15,970개 등록 파일과 기존 inventory의 SHA는 변하지 않았다. 새 23,083개를 별도 plan/inventory에 등록했으며 두 목록의 합집합에서 미등록 후보는0개였다. 원래 만료 기한을 연장하지 않았고 raw 파일·가중치의 실제 삭제는0건이다. dry-run만 수행했으며 자동 예약은 설치하지 않았다. 상세 집계는 [보관 검증](reports/IMPROVEMENT-RETENTION-VERIFICATION.json)에 있다.
