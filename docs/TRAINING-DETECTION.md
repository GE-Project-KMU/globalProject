# 포트홀·비식별화 탐지기 실제 연구 학습

대상은 공식 YOLOX-Nano(depth=.33, width=.25, depthwise=True)다. 공개 COCO 모델을 목표 모델이라고 바꾸어 부르지 않고, 공식 COCO-Nano는 초기 파라미터 전이에만 사용한다. 별도의 실제 포트홀/얼굴/번호판 라벨로 native YOLOX SimOTA·IoU/objectness/classification loss를 역전파해 목표 파라미터를 학습한다.

이번 산출물은 **비공개·비상업 연구용**이다. 공개 자료의 저작권 이용조건을 확인했지만 피촬영자의 동의·개인정보 전수 검수·실차 운용 승인을 발명하지 않는다. 데이터는 restricted_research/publisher_annotations_machine_checked/human_privacy_reviewed=false, 모델은 research_only=true/production_approved=false/project_acceptance_eligible=false/independent_route_verified=false로 구분한다. 이미지나 모델을 외부 저장소에 자동 게시하지 않는다.

## 고정한 소스와 실제 데이터

| 대상 | 취득 근거와 조건 | 처리 |
| --- | --- | --- |
| YOLOX 코드 | [공식 저장소](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0, commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` | 공식 Nano 구조·loss를 그대로 사용하고 Windows용 좁은 학습 루프만 별도로 작성 |
| COCO 초기 가중치 | [공식 0.1.1rc0 Nano release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) | 크기/이름만 믿지 않고 SHA-256 확인, 목표 클래스와 크기가 맞는 파라미터만 전이 |
| 포트홀 | 저자 직접 배포 [RDD2020 DOI 10.17632/5ty2wb6gvg.1](https://data.mendeley.com/datasets/5ty2wb6gvg/1), CC-BY-NC-3.0 | 저자 `train.tar.gz` 전체 SHA-256 대조 후 원본 Pascal-VOC XML의 **literal D40**만 선택. D43/D44 등 다른 라벨은 포트홀로 바꾸지 않음 |
| 얼굴 | [CUHK-CSE 공식 WIDER FACE 배포](https://huggingface.co/datasets/CUHK-CSE/wider_face), CC-BY-NC-ND-4.0 | 저자 bounding box와 invalid 속성 사용. 연구용 내부 처리이며 이미지 변형/모델의 외부 재배포 권리를 인정받았다고 주장하지 않음 |
| 번호판 | [CCPD 저자 저장소](https://github.com/detectRecog/CCPD), 저자가 데이터셋에도 MIT를 명시 | 저자가 연결한 공개 Google Drive CCPD2020 ZIP과 파일명 annotation 사용. 실제 번호판 문자열은 학습 목표로 사용하지 않음 |

고정 해시:

- YOLOX codeload ZIP: `647bba7d16cc7e13feb5f2beca0296f86186c944f5a8fa9ccefc2629758bccba`
- COCO-Nano PTH: `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b`
- 저자 RDD2020 train.tar.gz: `e06f74463b2d7410fa7e02f478818025be9b051087f80e0c51631abb32c0af55` — Mendeley 공개 파일 metadata의 SHA와 전체 다운로드가 일치한다.

RDD2022의 일부 공개 YOLO 변환본은 class3 이름이 pothole여도 실제 차선 문자·균열 박스를 포함했다. 훈련/검증 자료 spot check에서 발견해 해당 학습을 시험 평가 전에 중단하고, 출처 release에 quarantine sidecar와 모델 provenance의 data_revoked/needs_revalidation을 기록했다. 그 변환본 준비 명령도 거부하도록 바꿨다. 이 실패 자료/가중치를 최종 포트홀 모델로 사용하지 않는다. 원본 literal 라벨과 임의의 정수 클래스 번호는 같은 근거가 아니다.

## 분할과 동결

포트홀은 India=훈련1800장, Japan=검증375장, Czech=시험231장이다. 원본 공개 train 자료에서 고정 seed로 추출하며 정답 없는 공식 challenge test는 사용하지 않는다. 국가를 통째로 분리하고 정확 SHA 중복 및 서로 다른 분할 간 64bit dHash Hamming≤3 표본을 제외한다. 이 검사는 영상이 같은 자료의 누수를 줄이는 것이며 실제 노면 객체 ID/주행 경로의 전수 독립성을 확인했다는 뜻은 아니다.

비식별화 초기 v2는 crop 학습에서 높은 수치를 보였지만 동일 모델·동결 임계값의 원영상 challenge에서 거의 동작하지 않았다. 해당 crop 수치를 실사용 성능으로 내세우지 않고 진단 baseline으로 남긴다. v3는 원본 전체 프레임으로 새로 학습하며, 원래 동결한 훈련/검증 출처를 유지하고 이전 crop·원영상 시험에서 보지 않은 별도 원본 시험 자료를 학습 전에 동결한다. 얼굴 event 그룹·번호판 identity hash를 분리하고 원본/crop SHA와 연결 dHash 중복을 배제한다. 구체적 건수/제외 규칙은 `docs/reports/REDACTION-FULLFRAME-V3-QC.json`에 기록한다.

WIDER에는 얼굴만, CCPD에는 번호판만 알려진 정답이 있다. 다른 클래스가 영상에 전혀 없다는 가짜 음성 라벨을 만들지 않는다. 평가에서는 해당 영상의 알려진 클래스만 채점하며 다른 클래스 예측을 오검출로 세지 않는다. native YOLOX의 class-agnostic objectness에는 미표기 다른 객체가 학습 잡음으로 작용할 수 있다. 이 한계와 개인정보 검수 미완료 때문에 운영 배포 합격으로 승격하지 않는다. invalid 얼굴이 있는 훈련/검증 원본은 통째로 제외하며, 유효 정답이 max_labels를 넘는 훈련 이미지는 잘라 버리지 않고 거부한다.

## 재현 명령

모든 다운로드·Python·캐시는 프로젝트 내부에 둔다. Windows에는 프로젝트 내부 Python3.11.9/PyTorch2.7.1+cu126/torchvision0.22.1 환경을 사용하고, Ubuntu는 같은 고정 패키지로 별도 환경을 만든다. RTX2080Ti sm75의 실제 CUDA forward/backward는 환경 보고서에 기록됐다. OS 전역 CUDA/드라이버를 설치하지 않는다.

```powershell
# 초기 학습 환경 준비: 프로젝트 문서의 requirements-training과 setup_training 절차
.\.venv\Scripts\python.exe scripts\setup_training.py bootstrap
.\.venv\Scripts\python.exe scripts\setup_training.py install

# 고정 공식 source/초기 가중치: clean checkout에서도 다시 취득 가능
.\.venv\Scripts\python.exe scripts\research_detection_data.py fetch-yolox
.\.venv\Scripts\python.exe scripts\research_detection_data.py fetch-pretrained

# 포트홀 저자 원본: 큰 본문 전송이 막히면 8MiB Range 병렬 수신 후 전체 SHA 검증
.\.venv\Scripts\python.exe scripts\research_detection_data.py fetch-rdd2020
.\.venv\Scripts\python.exe scripts\research_detection_data.py prepare-rdd2020

# 얼굴/번호판 공개 배포와 동결 fullframe 전처리
.\.venv\Scripts\python.exe scripts\research_detection_data.py fetch-redaction
.\.venv\Scripts\python.exe scripts\prepare_redaction_research.py
.\.venv\Scripts\python.exe scripts\prepare_redaction_fullscene.py
.\.venv\Scripts\python.exe scripts\prepare_redaction_fullframe_v3.py

# 실제 목표 모델 학습: 가중치/사진/중간 결과는 Git 제외 data 하위
.\.venv\Scripts\python.exe scripts\setup_training.py run scripts\train_yolox_research.py data\training\research-pothole-rdd2020\release.json data\training\pothole-nano-rdd2020-v1 --task pothole --epochs 30 --batch-size 8 --learning-rate 0.0005 --input-size 640
.\.venv\Scripts\python.exe scripts\setup_training.py run scripts\train_yolox_research.py data\training\research-redaction-fullframe\release.json data\training\redaction-nano-v3 --task redaction --epochs 30 --batch-size 8 --learning-rate 0.0005 --input-size 640 --max-labels 512 --known-class-evaluation
```

훈련 seed=2026, AdamW, cosine learning rate, 좌우 반전/광도 증강, BGR float32/255분할 없음, 좌상단 letterbox다. GPU 학습은 다른 실험과 공유했으며 총 메모리9GiB 이하를 감시한다. 공유 학습 시간으로 독립 추론 FPS나 60분 실차 성능을 주장하지 않는다.

checkpoint는 검증 AP50이 가장 높은 epoch로 선택한다. 비식별화 confidence는 선택된 checkpoint의 검증 정답만 사용해 `.01/.03/.05/.1/.2/.35/.5` 중 결정하고 JSON으로 동결한 뒤 새 시험을 평가한다. 두 클래스 Recall≥.95가 가능한 후보가 있으면 그 안에서 precision을 우선하고, 없으면 최소 클래스 Recall과 precision 순으로 선택한다. 포트홀 confidence는 SRS 기본 .35로 고정한다. AP50(confidence floor .01)과 실제 운영 confidence의 TP/FP/FN·Recall/Precision은 따로 기록한다. 잘 안 나온 시험을 보고 임계값을 다시 조정하지 않는다.

학습 폴더에는 원본 release SHA·Nano source/환경·seed·초기 가중치 해시, epoch loss/검증 지표, 최종 validation calibration, 시험 예측/지표, `best.pt`, ONNX와 manifest, source/TTL provenance를 남긴다. 실행한 trainer의 SHA·실제 명령과 소스 Git commit은 최종 `docs/reports/DETECTION-TRAINING-RESULTS.json`에 별도 기록한다(training-config가 trainer SHA를 자동 기록한다고 주장하지 않는다). 앱 검증은 OpenCV5 새 DNN 엔진 대신 canonical ONNX Runtime에서 수행하고 PyTorch/reference ORT와 함께 대조한다. 코드·실행 결과의 존재와 현장 SRS 정확도 합격은 구별한다.
