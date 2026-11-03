# 로컬 ZED 2i·영상 분석 에이전트

`agent/`는 원본 영상을 서버에 보내지 않는 로컬 수집·분석·검수 전송 도구다. 기준은 SRS v0.4, 라벨 가이드 v1.0, API v1.0이다. 명령은 프로젝트 루트 `C:\Users\user\proj\globalProject`에서 실행한다. 입력·출력·키·큐·모델 경로는 프로젝트 안으로 제한하며 경로 이탈을 거부한다.

## 현재 구현과 검증 범위

| 기능 | 구현 상태 | 확인한 범위 |
| --- | --- | --- |
| UVC/좌우 결합 영상 취득 | 구현 | 실제 생성한 합성 MP4를 열어 좌우 MP4로 나누고, 조각 단위 AES-GCM 암호화·복호화·재생 시험 |
| 시각·GNSS | 구현 | NMEA 체크섬·RMC UTC/좌표·GGA HDOP 연결·유효/저하/2초 초과 위치 없음 시험 |
| 차선 면적 계산 | 구현 | O/G/P/V의 분모·정상 공백·가림, 60/80% 경계, 70/80% 품질 경계, 고유 패치 중복 제거·면적 가중 합산 |
| 라벨/데이터 검증 | 구현 | bbox·RLE·마스크 크기·기한/해시·독립 검수·출처 필수값, 동일 실물/주행/원본/크롭의 연결 그룹 분할·누수 검출 |
| 모델 어댑터 | 구현 | 실제 OpenCV DNN ONNX 로딩 경로. 대상 클래스·학습 출처·파일 SHA-256·입력/출력 규약 검사 |
| 로컬 검수 전송 | 구현 | 사람이 확인한 해시만 큐 등록, 검토자 승인→기기 이미지→관측 전송, 암호화 SQLite, lease 복구·재시도·멱등 ID·TTL |
| 실제 ZED 하드웨어 | 미검증 | 이 작업에서 실제 카메라를 연결해 촬영한 것으로 주장하지 않음 |
| 포트홀 외부 참조 모델 | 실제 추론 확인 | 대상 학습 YOLOX-S ONNX를 해시 고정 다운로드하고 합성 입력에서 CPU 추론 확인. 선정된 Nano 및 운영 승인 모델과 구별 |
| 실노면 포트홀/도막 인식 정확도 | 미검증 | 도막/비식별화 대상 가중치 미제공. 필요한 ONNX가 없으면 `model_not_ready`, exit 2. 독립 현장 정확도 시험 필요 |
| GPU 추론·10 FPS·60분 실차 시험 | 미검증 | 현재 어댑터의 실행 대상은 OpenCV DNN CPU. GPU가 있다는 사실만으로 GPU 추론을 통과 처리하지 않음 |
| 완전한 10 m 마모 구간 자동 판정 | 부분 구현 | 계산·중복 제거 함수는 구현. 단일 프레임 분석은 구간 거리 근거가 없어 `unassessable`을 유지함 |

위 합성 영상·마스크 시험은 코드와 데이터 처리 오류를 확인하는 기능 시험이다. mAP·IoU·실물 Recall·개인정보 검출 재현율에 대한 현장 성능 증거가 아니다.

## 환경과 실행

현재 개발 PC에서 NVIDIA GeForce RTX 2080 Ti 11 GB/드라이버 610.88이 조회됐다. GPU 추론은 아직 검증하지 않았다. Python 패키지 목록은 `requirements-vision.txt`에 기록돼 있다. GPU 학습/ONNX export용 PyTorch 환경은 수집 PC 런타임과 분리해 호환 버전을 고정한다.

```powershell
.\.venv\Scripts\python.exe -m agent --help
.\.venv\Scripts\python.exe -m agent capture --source 0 --seconds 60 --chunk-seconds 30
.\.venv\Scripts\python.exe -m agent capture --source data\authorized-stereo.mp4 --seconds 60
```

Linux/Ubuntu에서는 같은 소스에 대해 프로젝트 내부 가상환경의 `python -m agent ...`를 사용한다. 카메라 정수 인덱스는 운영체제별로 확인한다. 네트워크 영상 URL·프로젝트 밖 파일 경로는 입력하지 않는다.

### 수집 저장 구조·암호화

한 run UUID 아래에 좌·우 MP4 조각, 각 프레임의 원본 번호·PTS·UTC·단조 시각, 수신 NMEA를 기록한다. 조각을 닫을 때 MP4/PTS JSON을 AES-256-GCM으로 암호화하고 평문 조각을 제거한다. manifest에는 암호문 파일 해시와 raw 만료 시각을 기록한다. 키는 `data/agent/local.key`에 생성되며 소스 저장소에 넣지 않는다.

**중요:** OpenCV의 활성 MP4 encoder는 조각을 닫기 전 평문 임시 파일을 쓴다. 따라서 원본 보호를 충족하려면 OS 암호화 볼륨·계정 접근권한도 필요하다. 같은 디스크의 키 파일과 암호문을 함께 탈취하는 공격을 이 파일 암호화만으로 막는다고 주장하지 않는다. 다른 PC로 인계할 때 키는 원본 묶음과 별도의 승인된 경로로 관리한다. 파일 unlink가 SSD의 물리적 완전 삭제를 뜻하지도 않는다.

저장 잔량이 10 GB 미만이면 새 취득을 중단한다. 조각은 최대 300초로 제한한다. 장치 연결·영상 유무·MP4 encoder 오류는 명시적으로 실패한다. 녹화 노트북의 시각 보정/실제 수집시각은 별도로 검증해야 한다. 이미 촬영한 영상의 replay 시각을 새 실차 촬영시각으로 제시하지 않는다.

```powershell
# 필요한 조각만 제한 영역으로 복호화한다. 복호화 사본에도 raw 7일 정책 적용.
.\.venv\Scripts\python.exe -m agent decrypt data\agent\runs\RUN\000000000.left.mp4.aesgcm data\agent\work\left.mp4

# 삭제 기본값은 dry-run이다. --apply로 실제 만료 raw run을 삭제한다.
.\.venv\Scripts\python.exe -m agent purge-raw
.\.venv\Scripts\python.exe -m agent purge-raw --apply
```

실제 재귀 삭제 전 resolved 경로가 해당 raw run 디렉터리와 프로젝트 내부인지 확인한다. 원본은 촬영 후 7일, 마스킹 점검 자료는 촬영 후 90일 정책을 따른다. 에이전트의 raw purge는 이 로컬 run 디렉터리만 담당하며 서버·공유 사본의 삭제는 서버 보관 작업과 함께 확인한다.

### GNSS

`capture --serial-port COM3`로 USB NMEA 115200bps를 수신할 수 있다. 현재 물리 수신기 미연결 상태다. 지원 문장은 RMC/GGA이며 체크섬·유한 수치·좌표 범위를 검사한다. `build_fixes()`는 수신시각 1초 이내 GGA의 HDOP를 RMC UTC 표본에 연결한다. `associate_location()`은 가장 가까운 유효 UTC 표본을 사용하고, 2초 초과 공백은 위치 없음, 시각 차이 0.5초 초과/HDOP 불명·3 초과는 품질 저하로 둔다. HDOP를 미터 오차로 환산하지 않는다.

## 모델 준비·실행

필수 모델은 포트홀 1클래스, 도막 2클래스(background/paint), 비식별화 2클래스(face/license_plate)다. 공개 COCO 모델에는 이 프로젝트의 목표 클래스/의미가 그대로 있지 않으므로 자동으로 대체하지 않는다. 승인된 학습 자료와 독립 시험으로 모델을 학습·평가한 뒤 ONNX·학습 릴리스·해시를 등록한다.

포트홀/비식별화 YOLOX는 `--decode_in_inference`가 켜진 출력 `[N, cx,cy,w,h,objectness,class scores...]`와 기본 raw head를 명시적으로 구별한다. manifest의 `yolox_raw_xywh`는 strides=[8,16,32] grid를 검증해 중심·크기를 복원하고, `yolox_decoded_xywh`는 이미 복원된 좌표를 받는다. 좌상단 letterbox·BGR float32(255 분할 없음)를 사용하고 원영상 좌표로 되돌려 class별 NMS를 적용한다. [공식 YOLOX exporter](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/tools/export_onnx.py)

DeepLabV3-MobileNetV3는 **남은 도막으로 추가 학습한 2클래스** ONNX를 사용한다. RGB/255, mean=[.485,.456,.406], std=[.229,.224,.225], 좌상단 letterbox, output=`N×2×H×W` logits다. padding을 잘라 원본 크기로 복원한다. 학습/export 전처리를 이 규약에 맞춰야 한다. Torchvision의 기본 COCO/VOC 가중치를 도막 정답 모델로 사용하지 않는다. [공식 모델 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html)

예시 manifest(실제 모델 파일의 해시/릴리스로 교체):

```json
{
  "format": "onnx",
  "task": "pothole",
  "classes": ["pothole"],
  "weights": "data/models/pothole.onnx",
  "sha256": "실제 파일의 SHA-256 64자리",
  "target_trained": true,
  "dataset_release": "승인된 실제 데이터 릴리스 ID",
  "input_size": [416, 416],
  "output_layout": "yolox_decoded_xywh",
  "confidence": 0.35,
  "nms_iou": 0.5
}
```

```powershell
.\.venv\Scripts\python.exe -m agent model-status data\models\pothole.json
.\.venv\Scripts\python.exe -m agent analyze data\agent\work\left.mp4 --output data\agent\analysis\run-01 --pothole-manifest data\models\pothole.json --lane-manifest data\models\lane.json --redaction-manifest data\models\redaction.json --profile data\profiles\vehicle-a.json
```

`model-status`는 파일·해시·목표 클래스·ONNX 로딩을 확인하며 정확도 합격을 인증하지 않는다. `analyze`는 세 모델이 모두 준비된 후에만 처리한다. 결과 사진은 자동 마스킹됐더라도 `reviewed_for_sharing=false`이며 자동 업로드되지 않는다.

### 선택적 외부 포트홀 모델

[AMD-PAVS-AI 모델 카드](https://huggingface.co/AMD-PAVS-AI/pothole-detection)의 **YOLOX-S** 1클래스 ONNX를 선택적 비교 기준으로 제공한다. 프로젝트의 선정 모델 YOLOX-Nano를 바꾸거나 운영 모델로 자동 등록하지 않는다. 카드 표기는 Apache-2.0이며, 학습 출처로 연결된 [Jay Gala 데이터 저장소](https://github.com/jaygala24/pothole-detection)는 MIT로 공개됐다. 원본 이미지가 웹에서 수집됐다는 설명이 있으므로 해당 저장소의 라이선스 표기가 이미지별 권리 검증을 대신하지 않는다. 데이터셋 이미지는 다운로드하지 않았다.

다운로더는 revision `cd5dd7a24988ac3cb8e58825a9ad477323787ed9`, 파일 크기 35,774,753바이트, SHA-256 `9d55b8cc134f034b1885bd97aded14dcb17d887b8d5b7fbb3940e820d24c7660`를 고정한다. `data/models/external`에 ONNX·manifest·모델 카드·라이선스·provenance를 저장한다. publisher가 보고한 검증 186장 mAP@50=72.1%, mAP@50:95=34.4%는 이 프로젝트의 측정값이 아니며 목표 mAP@50 80% 합격 근거가 아니다. `production_approved=false`를 유지한다.

```powershell
.\.venv\Scripts\python.exe -m agent download-baseline
.\.venv\Scripts\python.exe -m agent infer-potholes data\authorized-image.png --manifest data\models\external\pothole_yolox_s.json --output data\agent\analysis\pothole-result.json
```

실제 다운로드된 ONNX를 직접 만든 640×360 합성 영상에 실행했고 CPU 추론 1회 0.091초, 검출 빈 목록을 확인했다. 결과는 `.cache/vision-tests/external_baseline_smoke.json`에 기록했다. 이는 모델 로딩·raw head 해석·추론 코드의 실행 증거이며 포트홀 정확도나 지속 10 FPS 증거가 아니다. 이 명령은 로컬 결과 JSON만 생성하며 관측 등록·사진 승인·업로드를 수행하지 않는다.

### 학습·export 재현 경로와 남은 자료

검색한 공개 자료 중 본 프로젝트의 **남은 도막 2클래스** 정의와 가중치·이용조건을 함께 확인할 수 있는 호환 ONNX는 확보하지 못했다. 기본 VOC/Cityscapes 분할을 도막 모델로 취급하지 않는다. 실제 차량 높이·시야·노면·날씨를 포함한 승인된 포트홀 라벨, 도막 마스크, O/G/P/V, 실물/주행 ID, 얼굴/번호판 라벨과 독립 시험 자료가 필요하다.

현재 런타임 Python 3.14.7에서 오래된 YOLOX 학습 의존성을 그대로 설치하지 않았다. [YOLOX requirements](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/requirements.txt)의 `onnx-simplifier==0.4.10`은 [PyPI 배포 파일](https://pypi.org/project/onnx-simplifier/0.4.10/#files)에 cp314 wheel이 없으며 소스 빌드가 필요할 수 있다. 설치 실패를 실제 재현했다고 주장하지 않는다. 학습은 프로젝트 안의 별도 Python 3.10 환경에서 진행하고 런타임에는 ONNX만 반입한다. 아래 CPU 환경은 [PyTorch 공식 이전 버전 조합](https://pytorch.org/get-started/previous-versions/)을 사용한 준비 예시이며, 이 작업에서 대용량 PyTorch 설치·학습·export는 실행하지 않았다.

```powershell
# Python 3.10이 설치된 학습 PC에서, 모든 환경·캐시를 프로젝트 내부에 둔다.
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
$env:TEMP = "$PWD\.cache\tmp"; $env:TMP = $env:TEMP
$env:TORCH_HOME = "$PWD\.cache\torch"
py -3.10 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-training\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.10.0.84 onnx==1.17.0
.\.venv-training\Scripts\python.exe -m agent.train_lane data\manifests\release.json data\models\lane-training --epochs 30 --batch-size 2
```

`agent.train_lane`는 승인 release 검사와 누수 검사를 먼저 실행하고 train/validation만 사용한다. 각 항목은 `path`, `mask_path`, 두 파일 SHA-256과 검수 메타데이터를 가져야 한다. 마스크는 0=background, 1=paint, 255=ignore다. 런타임과 동일한 letterbox/정규화를 적용하고 validation paint IoU가 가장 좋은 체크포인트를 ONNX(opset 17)와 manifest로 export한다. test 분할은 학습/선택에 쓰지 않으며 별도 독립 평가가 필요하다. 이 trainer의 전처리만 현재 환경에서 시험했고 전체 학습은 미검증이다. 포트홀 Nano는 공식 YOLOX Nano 실험 설정을 1클래스 COCO 라벨에 맞춰 학습한 뒤 공식 exporter와 위 manifest 규약을 적용해야 한다. GPU 환경 설치 및 배포 승인은 실제 모델 평가 결과와 함께 진행한다.

현재 분석기는 포트홀의 IoU/1초 window 추적·다중 프레임/고신뢰 단발/보류 상태를 제공한다. 차선은 보정 homography·검증된 nominal_lane_width_px·solid 설정이 있으면 보이는 직선 도막의 지지 정도를 검사해 patch 면적을 계산한다. 불명확·곡선/복수 차선·완전 소실은 보류하고, 점선의 정상 공백은 임의로 추정하지 않는다. **이 patch 측정은 고유 10 m 노면 길이·패치 추적이 완성된 구간 판정이 아니며, 구간 지원 근거가 없으면 API grade는 unassessable로 남는다.** O/G/P/V와 고유 길이 근거가 있는 자료는 `measure_patch()`/`aggregate_segment()`로 검산하고 `api_lane()`으로 strict API 타입에 매핑할 수 있다.

## 데이터 검수·분할

```powershell
.\.venv\Scripts\python.exe -m agent validate-data data\manifests\annotation.json --annotation
.\.venv\Scripts\python.exe -m agent split-data data\manifests\samples.json data\manifests\split.json --seed 2026
.\.venv\Scripts\python.exe -m agent validate-data data\manifests\release.json
```

sample_id/parent_sample_id/source_sample_id, 같은 실물 포트홀·차선 구간/패치·원본 해시·run·leakage group을 union하여 연결 요소 전체를 한 분할로 배정한다. 독립 그룹 3개 미만은 분할 실패이며 70/15/15 수치보다 독립성을 우선한다. 공개 자료의 공식 이용조건/분할을 바꿀 권한을 자동 부여하는 도구는 아니다.

릴리스 검수는 샘플 UUID·출처/해시·분할·라벨 버전·경로·마스킹/QC 승인·이용조건·만료·실제 파일 해시를 확인한다. RLE는 열 우선 무압축 counts와 size=[H,W]를 지원하며 합계가 픽셀 수와 맞아야 한다. 압축 COCO 문자열은 별도 디코더/변환 후 입력해야 한다.

## 사람 검수 후 전송

1. 실제 비식별 최종 사진을 사람이 확인하고 필요 영역을 로컬에서 추가 가림한다.
2. API 스키마에 맞는 observation JSON과 최종 사진 해시를 준비한다. 이미지/관측 UUID와 원래 촬영시각을 고정한다.
3. 아래 명령으로 암호화 SQLite 큐에 등록한다. `--confirm-human-review`는 자동 추론/개인정보 검사를 통과했다는 대체 증명이 아니라 작업자의 명시적인 확인이다.

```powershell
.\.venv\Scripts\python.exe -m agent enqueue data\review\observation.json data\review\final.png --reviewed-by reviewer-id --confirm-human-review
.\.venv\Scripts\python.exe -m agent queue-status
# 토큰은 별도 로그인/기기 등록 결과를 현재 프로세스 환경변수에 설정한다.
# ROAD_DEVICE_TOKEN, ROAD_REVIEWER_TOKEN; 로그/파일/버전 관리에 넣지 않는다.
.\.venv\Scripts\python.exe -m agent upload --gateway http://127.0.0.1:8000/api/v1
```

큐는 관측/위치/검수 메타데이터를 AES-GCM으로 암호화해 SQLite WAL+FULL 동기화로 저장한다. 키는 위 로컬 키 정책을 따른다. 검토자 토큰은 image-approvals 요청, device 토큰은 images/observations 요청에만 사용한다. 토큰은 큐에 저장하지 않는다. HTTP 비암호화는 loopback gateway만 허용한다.

임시 오류는 5/15/60초 후 5분 간격, 429는 Retry-After를 반영한다. 영구 오류·승인 대기/만료·인증 실패·ID 충돌은 blocked다. `retry-blocked OBSERVATION_UUID`는 사람이 원인을 해결한 후 명시적으로 실행한다. 승인 뒤 사진 해시가 바뀌면 업로드하지 않는다. 90일 만료는 큐에 들어간 뒤에도 다시 확인한다. uploading lease가 만료되면 재시작 후 pending으로 복구하고 같은 ID로 재전송한다.

## 시험 결과

```powershell
$env:TEMP = "$PWD\.cache\tmp"
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = "$PWD\.cache\pycache"
.\.venv\Scripts\python.exe -m pytest tests\test_vision.py -q --basetemp=.cache\pytest-vision
```

현재 결과: **34 passed**. 실제 코덱 생성/분리/암복호화/재생, NMEA·위치 타입, RLE·면적/경계/null/패치·UUID와 서버 strict 스키마, 원본/증강 누수와 해시·기한, 잘못된 ONNX/클래스·해시, CLI 모델 없음 exit 2, outbox HTTP 요청 형식·승인/ACK·retry/lease·만료/파일 변경 차단·raw dry-run/삭제, YOLOX raw grid 복원·실제 외부 ONNX 추론·손상 다운로드 거부/임시 파일 정리·학습 전처리 일치를 확인했다. 외부 모델이 없는 새 checkout에서는 다운로드 없이 해당 선택적 시험 1개를 skip한다. outbox 요청 형식 시험은 HTTP MockTransport를 사용한 단위 시험이며 실제 gateway의 HTTP 연동 검수는 별도의 전체 통합 시험을 함께 확인해야 한다.
