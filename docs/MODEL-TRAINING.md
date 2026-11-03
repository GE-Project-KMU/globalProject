# 초기 공개 데이터 모델 학습 기록

작성일: 2026-11-04. 실물 ZED 2i 없이 현재 호스트 RTX 2080 Ti에서 실제 역전파·가중치 갱신·checkpoint 선택·ONNX 내보내기를 수행한다. 공개 자료의 연구 결과이며 운영 승인 또는 실차 인수 결과가 아니다. 최종 수치는 `reports/MODEL-TRAINING-SUMMARY.json`에 기록한다.

이 문서는 개선 전 초기 가중치와 baseline 시험 기록이다. 후속 도막 개선의 같은 시험군 비교와 남은 수치 검증 문제는 [개선 실험 기록](LANE-IMPROVEMENT-REPRODUCTION.md)에 구분하여 기록한다. 아래 초기 결과를 후속 후보의 검증 통과로 해석하지 않는다.

## 초기 실험 결과

| 연구 가중치 | 실제 학습 | 고정 공개 시험 결과 | 판정 |
| --- | --- | --- | --- |
| 포트홀 YOLOX-Nano640 | 30epoch, 선택7 | AP50 13.11%; confidence.35에서 Recall11.68%/Precision31.94% | 목표 AP50 80% 미달 |
| RTK 도막 초기 실험 | 60epoch | 도막 IoU48.65% | 목표75% 미달 |
| A2D2 도막 | 50epoch | 도막 IoU39.44% | 목표75% 미달 |
| 얼굴·번호판 Nano640 원본 화면 | 30epoch, 선택18 | face Recall58.82%/Precision16.49%; plate Recall97.50%/Precision96.53% | 두 클래스 Recall95% 조건 미달 |

도막 두 시험은 서로 다른 데이터이므로 같은 benchmark의 순위로 비교하지 않는다. 얼굴·번호판은 validation에서 confidence.01을 고정했으며 게시자가 알려준 클래스에 한정한 수치다. 개인정보가 모두 제거된다는 보증이 아니다. 포트홀·비식별화의 실제 앱 ONNX Runtime CPU 시험은 native GPU 평가의 AP·TP/FP/FN과 일치했다. 공개 보고서 `reports/DETECTOR-APPLICATION-EVALUATION.json`에 기록한다.

이 초기 실험의 A2D2 v1·포트홀 v1·얼굴/번호판 v3 후보는 엄격한 앱/별도 ORT/PyTorch 수치 비교를 통과했다. 얼굴과 번호판은 각각 양성 validation 표본을 사용해 NMS 후 클래스·박스·신뢰도까지 확인했다. RTK 초기 모델은 한 test 표본에서 Torch raw allclose가 미달한 사실을 보존한다. 해당 표본도 앱↔reference ORT와 세 엔진의 도막 마스크는 일치했으나 전체 수치 검증 합격으로 바꾸지 않는다.

가중치 위치는 `data/training/lane-rtk-v1/lane.onnx`, `lane-a2d2-v1/lane.onnx`, `pothole-nano-rdd2020-v1/model.onnx`, `redaction-nano-v3/model.onnx`다(두 번째 이후 경로도 `data/training/` 아래). 해당 폴더의 manifest와 provenance를 함께 사용한다. 이 파일들은 Git에서 제외하고 로컬 프로젝트에 보관한다.

## 재현 환경

[학습 환경](TRAINING-ENVIRONMENT.md)의 별도 Python3.11/CUDA 환경을 구성한다. 앱의 Python3.14는 영상 처리와 ONNX Runtime CPU 추론에 사용한다. 원본·가중치·캐시·임시 파일은 프로젝트 안에만 보관하며 일반 Git에서 제외한다. 학습 중에는 의존성 또는 실행 중인 학습 코드를 변경하지 않는다.

```powershell
.venv\Scripts\python.exe scripts/setup_training.py bootstrap
.venv\Scripts\python.exe scripts/setup_training.py install
.venv\Scripts\python.exe scripts/research_detection_data.py fetch-yolox
.venv\Scripts\python.exe scripts/research_detection_data.py fetch-pretrained
```

YOLOX는 공식 pinned commit의 실제 Nano 구조와 SimOTA 손실을 사용한다. 일반 COCO checkpoint는 초기화에만 쓰며 포트홀·얼굴·번호판을 이미 학습한 모델로 표시하지 않는다. 현재 데이터 release는 원본 다운로드 후 source SHA·라이선스 증거·분할 누출·만료를 검증한다. 게시자 라벨과 기계 검수를 사람의 개인정보 검수나 운영 승인으로 바꾸지 않는다.

아래 명령은 새 checkout 또는 아직 산출물이 없는 새 모델 출력 폴더를 전제로 한다. 완료된 모델 폴더에 다시 학습해 checkpoint·manifest·검증 결과를 덮어쓰지 않는다. 재학습은 새 버전의 출력 폴더를 지정하고 출처·분할·코드·환경 해시를 다시 기록한다. 이미 고정한 차선 threshold의 재보정을 같은 manifest에 요청하면 거부되는 것이 정상이며, 재실행으로 데이터 보관기한이 연장되지 않는다. Linux에서는 [학습 환경의 Linux 절차](TRAINING-ENVIRONMENT.md#linux에서-다시-구성)를 따르고 `.venv/bin/python`과 프로젝트 내부 `.venv-training/bin/python`을 사용한다.

## 도막 분할

DeepLabV3-MobileNetV3를 background/paint 두 클래스로 전이 학습한다. RGB/255, ImageNet mean/std, 좌상단 letterbox512와 ignore255 마스크 규칙은 학습·앱·평가에서 일치시킨다. A2D2는 증강·가중 CE/Dice·cosine LR를 사용한다. checkpoint는 validation IoU로 선택하고 paint threshold도 validation에서 고정한 다음 test를 평가한다.

```powershell
.venv\Scripts\python.exe scripts/prep_a2d2.py
.venv\Scripts\python.exe scripts/setup_training.py run -m agent.train_lane data/public/a2d2/manifest.json data/training/lane-a2d2-v1 --research --pretrained --epochs 50 --batch-size 8 --learning-rate .0001 --augment --dice-weight .5 --cache-pairs
.venv\Scripts\python.exe scripts/calibrate_lane_model.py data/public/a2d2/manifest.json data/training/lane-a2d2-v1/lane.json data/training/lane-a2d2-v1/calibration.json
.venv\Scripts\python.exe scripts/evaluate_lane_model.py data/public/a2d2/manifest.json data/training/lane-a2d2-v1/lane.json data/training/lane-a2d2-v1/heldout.json
```

RTK 원본701장 중 guard/유사 영상 제외 후 train422/validation70/test71을 사용한 첫 연구 모델도 보존한다. RTK는 영상 순서가 확인되지 않은 파일 번호 block 분할이다. A2D2는 train4개 기록400장/validation1개100장/test1개100장이며 날짜·기록은 분리됐지만 같은 실제 도로가 다시 촬영됐는지는 확인하지 못했다. 두 데이터 모두 한국 도로의 마모율 GT가 아니다. 예상 도막 외곽 O·정상 차선 간격 G·유효 영역 V의 독립 GT가 없으므로 잔존율 R MAE를 산출하지 않는다.

RTK의 첫 OpenCV5 기본 DNN 시험에서 수치 폭주가 확인됐다. 동일 ONNX를 앱의 ONNX Runtime CPU로 재실행한 시험 결과는 IoU48.6464%, precision57.1566%, recall76.5655%다. 목표 IoU75%에 미달한다. engine 오류가 포함된 이전 IoU0 결과는 모델 성능 증거로 사용하지 않고 진단 기록으로 보존한다. 상세 비교는 [모델 산출물 안내](MODEL-ARTIFACTS.md)를 참고한다.

## 포트홀·개인정보 검출

전체 취득·전처리·30회 학습 명령과 평가 방법은 [검출기 학습 안내](TRAINING-DETECTION.md)를 참고한다. 원본 전체 화면 v3 전처리 전에는 crop release와 이전 fullscene 진단 release를 먼저 고정한다. crop 가중치를 재학습 없이 전체 화면 모델로 바꾸어 부르지 않는다.

RDD2020 저자 원본 VOC XML의 literal D40만 pothole로 매핑한다. India 학습/Japan 검증/Czech 시험으로 나라를 분리하되 실제 route ID는 확인되지 않았다고 기록한다. 초기 RDD2022 제3자 변환본은 사진과 클래스 대응 불일치가 확인되어 해당 학습을 중단·격리했으며 최종 가중치로 사용하지 않는다.

얼굴은 WIDER FACE, 번호판은 CCPD2020의 게시자 bbox를 사용한다. crop 기반 redaction v2는 crop 시험 AP가 높아도 원본 전체 화면에서 검출이 무너졌으므로 운영 후보로 삼지 않는다. v3는 원본 전체 화면, 입력640, 기존 자료와 원본·유사 이미지·번호판 식별 그룹이 겹치지 않는 새 test400장으로 다시 학습한다. 기존 시험을 본 사실을 숨기거나 새 시험으로 재명명하지 않는다.

WIDER에는 번호판의 완전한 라벨이 없고 CCPD에는 얼굴의 완전한 라벨이 없다. 각 자료에서 알려진 클래스만 평가한다. YOLOX의 공통 objectness 학습에서는 알려지지 않은 다른 클래스의 누락 라벨 문제가 남는다. 따라서 이 시험의 얼굴/번호판 수치를 전체 영상의 개인정보 제거 보증으로 해석하지 않는다. 잘못된 게시자 얼굴 GT가 있는 원본은 이미지 전체를 제외하고, 학습 padding 한도를 넘는 정답을 조용히 잘라내지 않는다.

## 사용·보관

최종 ONNX와 manifest는 `data/training/`에 저장한다. 해시·전처리·클래스 계약·provenance·만료·철회 상태를 함께 확인해야 한다. `model-status`의 ready는 파일 실행 가능 여부이며 deployment_ready와 별도다. 연구 모델은 production_approved=false를 유지한다. 수치 검증 실패도 보고서에 남기고 허용 오차를 임의로 넓혀 합격시키지 않는다.

```powershell
.venv\Scripts\python.exe -m agent model-status data/training/lane-a2d2-v1/lane.json
.venv\Scripts\python.exe scripts/verify_trained_models.py data/training/lane-a2d2-v1/lane.json --checkpoint data/training/lane-a2d2-v1/lane_best.pt --output data/training/lane-a2d2-v1/verification.json
```

사용조건은 RTK CC BY4.0, A2D2 CC BY-ND4.0, RDD2020 CC BY-NC3.0, WIDER FACE CC BY-NC-ND4.0, CCPD MIT로 구분한다. 자료의 공개 여부만으로 B2G 상업 운영 또는 파생 가중치 배포 권한을 부여하지 않는다. 현재 가중치는 외부 Release에 게시하지 않는다. 원본 얼굴/번호판과 그 파생물은 수집 시 고정된 짧은 만료를 유지하고, 출처 만료·철회 시 연결 모델의 재검증 차단도 유지한다. 재실행으로 기한을 자동 연장하지 않는다.

개인정보 승인·국내 마모 GT·독립 주행 성능·ZED 캡처/깊이/보정·전체 처리량 검증은 연구 가중치 생성과 별도의 인수 조건이다. 현재 수치가 기준 미달인 모델을 실차에 자동 적용하지 않는다.

원본 archive·시험 이미지·개인별 예측·미리보기의 실제 만료 정리는 [연구 자료 보관 정리](RESEARCH-RETENTION.md)를 따른다. 고정 인벤토리의 해시·만료를 확인하는 수동 도구이며 자동 예약 삭제를 설치하지 않는다. 모델 가중치는 보존하되 출처 만료·철회 시 준비 상태의 재검증 차단을 유지한다.
