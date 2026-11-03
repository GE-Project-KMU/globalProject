# 로컬 ZED 2i·영상 분석 에이전트

`agent/`는 원본 영상을 서버에 보내지 않는 로컬 수집·분석·검수 전송 도구다. 기준은 SRS v0.4, 라벨 가이드 v1.0, API v1.0이다. 명령은 프로젝트 루트 `C:\Users\user\proj\globalProject`에서 실행한다. 입력·출력·키·큐·모델 경로는 프로젝트 안으로 제한하며 경로 이탈을 거부한다.

## 현재 구현과 검증 범위

| 기능 | 구현 상태 | 확인한 범위 |
| --- | --- | --- |
| UVC/좌우 결합 영상 취득 | 구현 | 실제 생성한 합성 MP4를 열어 좌우 MP4로 나누고, 조각 단위 AES-GCM 암호화·복호화·재생 시험 |
| 시각·GNSS | 구현 | NMEA 체크섬·RMC UTC/좌표·GGA HDOP 연결·유효/저하/2초 초과 위치 없음 시험 |
| 차선 면적 계산 | 구현 | O/G/P/V의 분모·정상 공백·가림, 60/80% 경계, 70/80% 품질 경계, 고유 패치 중복 제거·면적 가중 합산 |
| 라벨/데이터 검증 | 구현 | bbox·RLE·마스크 크기·기한/해시·독립 검수·출처 필수값, 동일 실물/주행/원본/크롭의 연결 그룹 분할·누수 검출 |
| 모델 어댑터 | 구현 | ONNX Runtime1.31.1 CPU가 기본. 대상 클래스·학습 출처·파일 SHA-256·입력/출력 규약 검사. 실제 backend/version을 결과에 기록 |
| 로컬 검수 전송 | 구현 | 사람이 확인한 해시만 큐 등록, 검토자 승인→기기 이미지→관측 전송, 암호화 SQLite, lease 복구·재시도·멱등 ID·TTL |
| 실제 ZED 하드웨어 | 미검증 | 이 작업에서 실제 카메라를 연결해 촬영한 것으로 주장하지 않음 |
| 포트홀 외부 참조 모델 | 실제 추론 확인 | 대상 학습 YOLOX-S ONNX를 해시 고정 다운로드하고 합성 입력에서 CPU 추론 확인. 선정된 Nano 및 운영 승인 모델과 구별 |
| 독립 실차 인수 성능 | 미검증 | 공개 연구 데이터의 실제 학습·오프라인 평가와 구분한다. 후보 모델의 결과는 reports/학습 보고서에 기록하며 필요한 ONNX가 없거나 만료·철회되면 `model_not_ready`, exit2 |
| GPU 추론·10 FPS·60분 실차 시험 | 미검증 | 현재 앱 어댑터의 실행 대상은 ONNX Runtime CPU. CUDA 학습 검증을 GPU 추론 또는 실차 성능 통과로 표시하지 않음 |
| 10 m 고유 패치 구간 처리 | 구현·실차 미검증 | `workflow`에서 검증된 profile, GNSS/BEV 이동 일치, 0.1 m 노면 셀의 대표 선택·10 m 구간 면적 가중·점선 공백 제외. 근거 부족은 판정 불가 |

위 합성 영상·마스크 시험은 코드와 데이터 처리 오류를 확인하는 기능 시험이다. mAP·IoU·실물 Recall·개인정보 검출 재현율에 대한 현장 성능 증거가 아니다.

## 환경과 실행

현재 개발 PC에서 NVIDIA GeForce RTX 2080 Ti 11 GB/드라이버 610.88이 조회됐다. GPU 추론은 아직 검증하지 않았다. Python 패키지 목록은 `requirements-vision.txt`에 기록돼 있다. GPU 학습/ONNX export용 PyTorch 환경은 수집 PC 런타임과 분리해 호환 버전을 고정한다.

학습 환경은 프로젝트 내부 Python3.11/PyTorch2.7.1+cu126으로 별도 구성했고 실제 CUDA 학습 연산을 확인했다. 앱 Python3.14에는 ONNX Runtime1.31.1을 고정한다. 설치·캐시·버전 증거는 [학습 환경](TRAINING-ENVIRONMENT.md), 이식·엔진 비교는 [모델 산출물 검증](MODEL-ARTIFACTS.md)을 따른다. OpenCV5는 영상 처리에 사용한다. 실제 RTK ONNX에서 기본 새 DNN engine의 잘못된 큰 logit/빈 마스크가 재현됐으므로 앱의 추론에 자동 선택하지 않는다. `OnnxModel(path,backend="opencv-classic")`는 명시적인 진단용이며 기본 ORT 실패 시 자동 fallback하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent --help
.\.venv\Scripts\python.exe -m agent capture --source 0 --seconds 60 --chunk-seconds 30 --collection-plan data\profiles\collection-plan.json --confirm-stationary
.\.venv\Scripts\python.exe -m agent capture --source data\authorized-stereo.mp4 --seconds 60
```

Linux/Ubuntu에서는 같은 소스에 대해 프로젝트 내부 가상환경의 `python -m agent ...`를 사용한다. 카메라 정수 인덱스는 운영체제별로 확인한다. 네트워크 영상 URL·프로젝트 밖 파일 경로는 입력하지 않는다.

### 수집 저장 구조·암호화

한 run UUID 아래에 좌·우 MP4 조각, 각 프레임의 원본 번호·PTS·UTC·단조 시각, 수신 NMEA를 기록한다. **PyAV로 MP4를 RAM의 BytesIO에 mux하고 AES-256-GCM 암호문만 디스크에 기록한다.** PTS/NMEA JSON도 메모리에서 직렬화·암호화하므로 평문 raw 임시 파일을 쓰지 않는다. 분석은 인증 복호화한 메모리 bytes를 PyAV로 직접 decode한다. manifest에는 암호문 파일 해시와 raw 만료 시각을 기록한다. 키는 `data/agent/local.key`에 생성되며 소스 저장소에 넣지 않는다. [PyAV 파일 객체 API](https://pyav.org/docs/stable/api/_globals.html), [설치한 PyAV 19.0.2](https://pypi.org/project/av/19.0.2/)

파일 암호화를 위해 OS 암호화 볼륨에 의존하지 않는다. 키 파일은 계정 접근권한으로 보호하며 다른 PC로 인계할 때 키는 원본 묶음과 별도의 승인된 경로로 관리한다. 같은 디스크의 키와 암호문을 함께 탈취하거나 프로세스 RAM/OS 메모리 덤프에 접근하는 공격까지 이 파일 암호화가 막는다는 주장은 하지 않는다.

저장 잔량이 10 GB 미만이면 새 취득을 중단한다. 조각은 최대 30초, 각 eye의 압축 메모리 버퍼는128 MiB로 제한한다. 초과 시 수집을 실패 처리하며 이전 암호화 조각은 보존한다. 장치 연결·영상 유무·MP4 encoder 오류는 명시적으로 실패한다. 녹화 노트북의 시각 보정/실제 수집시각은 별도로 검증해야 한다. 이미 촬영한 영상의 replay 시각을 새 실차 촬영시각으로 제시하지 않는다.

```powershell
# 삭제 기본값은 dry-run이다. --apply로 실제 만료 raw run을 삭제한다.
.\.venv\Scripts\python.exe -m agent purge-raw
.\.venv\Scripts\python.exe -m agent purge-raw --apply
```

실제 재귀 삭제 전 resolved 경로가 해당 raw run 디렉터리와 프로젝트 내부인지 확인한다. 원본은 촬영 후 7일, 마스킹 점검 자료는 촬영 후 90일 정책을 따른다. 에이전트의 raw purge는 이 로컬 run 디렉터리만 담당하며 서버·공유 사본의 삭제는 서버 보관 작업과 함께 확인한다.

### GNSS

`capture --serial-port COM3`로 USB NMEA(기본115200bps, --baudrate로 수신기 설정에 맞춰 변경)를 수신할 수 있다. 현재 물리 수신기 미연결 상태다. 지원 문장은 RMC/GGA이며 체크섬·유한 수치·좌표 범위를 검사한다. `build_fixes()`는 수신시각 1초 이내 GGA의 HDOP를 RMC UTC 표본에 연결한다. `associate_location()`은 가장 가까운 유효 UTC 표본을 사용하고, 2초 초과 공백은 위치 없음, 시각 차이 0.5초 초과/HDOP 불명·3 초과는 품질 저하로 둔다. HDOP를 미터 오차로 환산하지 않는다.

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
.\.venv-training\Scripts\python.exe -m agent.train_lane data\manifests\release.json data\training\lane-training --epochs 30 --batch-size 2
```

`agent.train_lane`는 승인 release 검사와 누수 검사를 먼저 실행하고 train/validation만 사용한다. 각 항목은 `path`, `mask_path`, 두 파일 SHA-256과 검수 메타데이터를 가져야 한다. 마스크는 0=background, 1=paint, 255=ignore다. 런타임과 동일한 letterbox/정규화를 적용하고 validation paint IoU가 가장 좋은 체크포인트를 ONNX(opset 17)와 manifest로 export한다. test 분할은 학습/선택에 쓰지 않으며 별도 독립 평가가 필요하다. 이 trainer의 전처리만 현재 환경에서 시험했고 전체 학습은 미검증이다. 포트홀 Nano는 공식 YOLOX Nano 실험 설정을 1클래스 COCO 라벨에 맞춰 학습한 뒤 공식 exporter와 위 manifest 규약을 적용해야 한다. GPU 환경 설치 및 배포 승인은 실제 모델 평가 결과와 함께 진행한다.

`analyze`는 이전 프레임/패치 진단 명령이다. 아래 **`workflow`가 수집 메타데이터·GNSS·프로파일·10 m 고유 패치·검수·서버 전송을 연결하는 실행 경로**다. 포트홀은 IoU/1초 window 추적, 3프레임 이상/고신뢰 단발/보류 상태를 제공한다. `analyze`의 단일 패치 값을 구간 판정으로 사용하지 않는다.

## 장비 연결 후 전체 실행 경로

운영체제에 SDK/GPU 패키지를 자동 설치하지 않는다. 기본 수집은 ZED 2i의 UVC 좌우 결합 영상과 USB NMEA(기본115200bps, --baudrate로 수신기 설정에 맞춰 변경)다. 모든 명령은 프로젝트 내부 경로만 받는다. 실장비·보정·학습 가중치가 없는 상태를 합성 결과로 메우지 않는다.

### 1. 설치 프로파일·운행 전후 점검

로컬 프로파일은 서버 등록 결과의 `profile_id`, `calibration`과 `calibrated_by`, `measurement`를 가진다. calibration은 서버 필수 필드(image_size, height_m, pitch_deg, camera_matrix, distortion, homography, roi_polygon_px, visible_distance_m, calibrated_at)를 그대로 사용한다. ROI/homography의 원영상 좌표는 왜곡 보정된 좌영상 좌표다. 실제 해상도와 일치해야 한다. 카메라/렌즈/장착 위치를 바꾸면 새 프로파일로 보정한다.

`measurement` 예시의 수치는 형식 설명용이며 장비 측정으로 대체해야 한다:

```json
{
  "bev_size": [256, 200],
  "meters_per_pixel": 0.02,
  "lane_tracks": [{
    "lane_track_id": "실제 UUID",
    "color": "white",
    "line_type": "dashed",
    "bev_x_range": [100, 108],
    "outline_verified": true,
    "dash_length_m": 3.0,
    "gap_length_m": 5.0,
    "phase_m": 0.0,
    "phase_verified": true
  }]
}
```

BEV row 0은 먼 노면, 마지막 row는 가까운 노면이다. `visible_distance_m=[near,far]`와 BEV 높이×m/pixel이 일치해야 한다. 해상도는 16~4096, 길이 격자는 0.1 m보다 조밀해야 한다. 각 차선의 좌우 경계는 독립된 검증 띠이며 완전히 사라진 도막을 보고 임의 복원하지 않는다. 실선은 dash/gap/phase 필드를 사용하지 않는다. 점선은 주행 원점 기준 실측 phase와 설계 공백 근거가 필수다. 영상만으로 phase를 알 수 없으면 그 프로파일로 점선 판정을 진행하지 않는다. 곡선·경사·진동으로 고정 평면/띠 가정이 성립하지 않는 설치는 별도의 검증 프로파일이나 재촬영이 필요하다.

4개 실측 대응점의 `image_points`, `bev_points` JSON으로 homography를 계산할 수 있다. 이는 카메라 내부 파라미터·실측 거리·검수자를 대신하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent calibrate data\profiles\points.json data\profiles\homography.json
.\.venv\Scripts\python.exe -m agent validate-profile data\profiles\vehicle-a.json
# checklist.json: mount_secure/lens_clear/notice_present/daylight/dry_asphalt 전부 실제 확인한 true
.\.venv\Scripts\python.exe -m agent collect --source 0 --profile data\profiles\vehicle-a.json --device-id DEVICE_UUID --checklist data\profiles\checklist.json --serial-port COM3 --seconds 3600 --chunk-seconds 30 --collection-plan data\profiles\collection-plan.json --confirm-stationary
```

`collect`는 정차한 상태에서 사용한다. GNSS 포트를 3초 검사하고 UTC 2초 이내 유효 NMEA, 좌우 프리뷰·프로파일 해상도, 저장 10 GB와 작업자의 장착/렌즈/고지/주간/건조 확인을 요구한다. 녹화 중 카메라 read 실패는 `interrupted`와 원인으로 기록하고, 완료 조각을 보존한다. 초기 `capture` 명령은 파일/카메라 취득 진단용이며 서버 주행으로 사용하려면 동일한 점검·장치/프로파일 식별 근거가 필요하다.

실카메라 수집에는 `--collection-plan`과 `--confirm-stationary`가 필수다. 구조화 계획 JSON은 plan_id(UUID), organization, controller_contact, operator, legal_basis, purpose, collection_scope, notice_message(문자열), raw_retention_days=7, masked_retention_days=90, audit_retention_days=180을 모두 가진다. 선언된 실제 수집 근거·목적·담당자/연락처·범위·안내문을 기록하며 법적 근거를 프로그램이 임의로 선택하지 않는다. 수집 전에 notice_message와 담당/연락처를 콘솔에 표시하고 계획 파일 경로·SHA-256·내용을 run manifest에 고정한다. 실제 외부 고지 부착은 작업자가 확인해야 한다. 계획 내용이 바뀌면 기존 촬영 계획을 덮어쓰지 않고 별도 파일/plan_id를 사용한다. workflow는 카메라 수집의 계획 해시/내용·정차 시작 확인을 재검사한다.

운행 후 checks JSON `{ "mount_secure": true, "lens_clear": true, "defect_from": null }`을 작성한다. 결함이면 false와 마지막 정상 확인 시각 이후의 UTC `defect_from`을 기록한다. workflow는 해당 시각 이후를 품질 불가로 처리한다.

```powershell
.\.venv\Scripts\python.exe -m agent postcheck data\agent\runs\RUN_UUID\manifest.json data\profiles\postcheck.json
# 웹 분석 작업 생성 전에 주행 메타데이터만 등록한다. 이후 manifest는 변경하지 않는다.
.\.venv\Scripts\python.exe -m agent register-capture data\agent\runs\RUN_UUID\manifest.json --gateway http://127.0.0.1:8000/api/v1
```

### 2. 분석 작업 설정·실행

bundle JSON은 서버 등록한 `model_bundle_id`, `version`, `config_hash`, `weights_hashes`(pothole/lane/redaction)와 세 모델 manifest 경로 `pothole`, `lane`, `redaction`을 가진다. 서버 config_hash를 그대로 관측에 넣고 실제 ONNX 해시와 번들 해시가 같은지 검사한다. config 파일은 다음 필수8개와 선택 `ca_bundle`(프로젝트 내부 CA PEM 경로)을 받는다:

```json
{
  "run_id": "RUN_UUID",
  "device_id": "DEVICE_UUID",
  "profile_id": "PROFILE_UUID",
  "model_bundle_id": "BUNDLE_UUID",
  "run_manifest": "data/agent/runs/RUN_UUID/manifest.json",
  "profile_path": "data/profiles/vehicle-a.json",
  "bundle_path": "data/models/bundle.json",
  "output": "data/agent-jobs/results/JOB_UUID"
}
```

```powershell
.\.venv\Scripts\python.exe -m agent workflow data\agent-jobs\prepared\JOB_UUID.json
```

입력 ID를 서로 대조하고 원본 7일 기한·암호문 해시를 검사한다. 원래 PTS/UTC/NMEA를 사용하며 분석 시각을 촬영 시각으로 만들지 않는다. 암호화 좌영상 조각은 메모리로만 복호화·decode한다. 재생 수집 자료는 `original_capture_verified=true`와 실제 촬영시각 근거가 있어야 하며 새 실차 주행처럼 취급하지 않는다.

세 실제 모델을 먼저 로드하고 ROI 밖 포트홀 후보를 제외한다. 비식별화 검출 결과는 여백을 두고 가리며, 차선 유효 영역은 ROI·프레임 품질·과노출/암부·개인정보/포트홀 가림을 반영한다. 세 모델 중 하나라도 없으면 exit 2 `model_not_ready`다. 가짜 모델 주입은 단위/계약 시험 함수에만 있으며 CLI에 우회 플래그가 없다.

거리 계산은 단순 GPS 좌표 흔들림 적분을 하지 않는다. 유효 GNSS·0.5초 이내 연속 표본·20 km/h 이하 속도와 BEV 영상 phase correlation의 이동 방향/거리·응답도를 대조한다. 정지 구간은 0거리로 두고 불일치/공백은 거리 epoch를 끊는다. 불명확한 이동을 이어 붙여 10 m 구간을 만들지 않는다. 각 epoch에서 0.1 m 고유 도색 예정 셀을 UUID로 연결하고 유효면적/선명도 기준 대표 하나만 고른다. 점선의 실측 공백은 분자·분모·지원 길이에서 제외한다. 10 m 구간의 판정 가능한 고유 도색 길이≥80%, 유효면적≥70%일 때만 면적 가중 R/등급을 계산한다. 부분 구간/모션 근거 부족은 판정 불가다. 생성 영상으로 알고리즘 계약을 시험했으나 실차 모션·곡선 대응 성능은 미검증이다.

결과 `review.json`은 strict observation 초안·원 촬영 위치·품질/거리 사유·모델 승인 상태·실측 처리 FPS·대표 비식별 사진·patch ID를 포함한다. overlay는 사진에 대응하는 원영상 좌표 외곽과 최대512 RLE paint mask다. API overlay는 **행 우선**이고 COCO 라벨 RLE(열 우선)와 다르다. RLE counts가8192를 넘으면 overlay를 생략하고 로컬 사유를 남긴다. 개인정보 미검출 가능성이 있으므로 업로드 전 사람 검수가 필수다.

### 3. 검수·암호화 큐·HTTP 요청

담당자가 모든 공유 사진을 직접 검수한 후 decisions JSON 목록에 `observation_id`, `approve:true`, **실제로 검수한 최종 이미지의** `image_sha256`을 기록한다. 사진을 수정했다면 초안의 해시도 새 최종 파일로 갱신하고 다시 검수한다. 자동으로 true를 채우는 명령은 없다.

```powershell
.\.venv\Scripts\python.exe -m agent approve-review data\agent-jobs\results\JOB_UUID\review.json data\review\decisions.json --reviewed-by REVIEWER_UUID
# ROAD_DEVICE_TOKEN은 해당 기기의 토큰, ROAD_REVIEWER_TOKEN은 담당자 로그인 토큰
.\.venv\Scripts\python.exe -m agent sync-run data\agent-jobs\results\JOB_UUID\review.json --gateway http://127.0.0.1:8000/api/v1
.\.venv\Scripts\python.exe -m agent upload --gateway http://127.0.0.1:8000/api/v1
```

검수 전체를 먼저 strict schema/해시 검사하고 승인한 항목만 암호화 큐에 넣는다. `register-capture`는 종료/postcheck 완료 manifest의 고정 메타데이터/해시만 POST하여 웹 분석 작업의 주행 참조를 먼저 준비한다. 서버 상태는 아직 recording이며 원본은 전송하지 않는다. `sync-run`은 같은 POST body/해시로 멱등 등록 확인→PUT runs/{id}/path→If-Match PATCH 완료 순서로 수행한다. upload는 승인→이미지 multipart→관측의 내구성 단계/재시도를 수행한다. 서버에 같은 기관의 기기·프로파일·모델 번들이 먼저 등록돼 있어야 한다. 원본 MP4/NMEA/라벨은 업로드하지 않는다.

경로는 원 촬영 UTC와 실제 GNSS 좌표쌍만 사용한다. 좌표 없음/2초 초과 공백·GNSS 품질 변화·프레임 품질/차선 근거 판정 상태 변화마다 선을 끊고 profile_id/roi_id=main/assessed 또는 unassessable 및 사유를 기록한다. 각 구간의 양 끝과 끊김을 보존하며 최대1000좌표/500구간으로 내부 점만 줄인다. 500개를 넘는 분리 구간은 처음/끝을 포함해 최대500개를 선택하고 omitted_segments를 반환하여 생략을 숨기지 않는다. 이때 서로 떨어진 구간을 이어서 새 경로를 만들지 않는다. 인증된 주행 지도용 경로이며 공개 보고서용 이동 경로를 추가하지 않는다.

분석이 끝나 대표 관측/구간 사진을 선정하면 현재 작업이 생성한 파일 중 참조되지 않는 비식별 프레임 사진만 삭제한다. 결과에는 retained_photos/removed_unreferenced_photos 건수를 기록한다. 기존의 관계없는 파일을 glob으로 삭제하지 않는다.

내부 HTTPS CA를 쓰면 `sync-run`/`upload`에 `--ca-bundle .runtime/tls/ca.crt` 또는 `ROAD_CA_BUNDLE` 환경변수를 설정한다. 경로는 프로젝트 내부만 허용하며 SSLContext의 인증서/호스트명 검증을 유지한다. `verify=False`를 사용하지 않고 `trust_env=False`로 환경 프록시·임의 SSL_CERT_FILE을 신뢰하지 않는다. workflow config의 CA 경로는 review package에 기록돼 sync-run에서 재사용된다.

### 선택적 스테레오·SDK

`agent.equipment.stereo_depth`는 실제 좌우 보정행렬·회전·미터 baseline으로 rectification→SGBM→Q 역투영을 수행한다. 무효 disparity는 NaN이며 `depth_summary`는 지지 픽셀이 부족하면 null/사유를 반환한다. 합성 수평 시차를 통한 실제 SGBM 시험만 통과했으며 ZED 현장 깊이 정확도는 검증하지 않았다. depth를 포트홀 심각도·측량 치수로 사용하지 않는다. `object_coordinate`는 검증된 차량 좌표·heading·camera_to_vehicle 변환이 있을 때만 선택 추정값을 만들며, API location의 vehicle_observation 위치를 바꾸지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent stereo-depth data\stereo\left.png data\stereo\right.png data\profiles\stereo.json data\agent\depth.npz
.\.venv\Scripts\python.exe -m agent sdk-probe
```

depth 출력은 AES-GCM 암호화된다. `ZedSDK`는 pyzed를 지연 import하고 좌/우 영상·미터 depth·장치 timestamp를 반환하는 선택 어댑터다. SDK 미설치/카메라 미연결은 명시적인 오류다. UVC RGB 필수 흐름은 SDK에 의존하지 않는다. SDK API는 [Stereolabs 공식 depth 예제](https://www.stereolabs.com/docs/tutorials/depth-sensing)를 기준으로 작성했으며 실제 SDK 연동은 미검증이다.

SDK 어댑터는 신경망 가중치를 자동 다운로드하지 않도록 classical ULTRA 모드를 명시한다. `road_relative_step`은 주변 스테레오 지지점의 평면 대비 잔차를 선택적으로 계산하며 지지가 없으면 null이다. 이 실험값을 실제 깊이 인증이나 심각도 등급으로 공개하지 않는다. 가시 차선 색상은 도막 모델이 찾은 영역에서 HSV 백색/황색 규칙으로 프로파일과 대조하며 불명확/불일치/경계 정렬 불일치는 유효 영역에서 제외한다. 이 색상/기하 규칙을 별도 학습 AI의 성능으로 주장하지 않는다.

`purge-local`은 기본 dry-run이며 `--apply`로 raw run 7일, 중단 작업의 복호화 사본 기한, review 사진/목록 90일, 만료 큐 payload를 정리한다. 작업 스케줄러/cron에서 매일 실행할 운영 작업으로 등록해야 한다. `data/agent-jobs/results`의 생성 사진/목록만 지우며 같은 디렉터리의 관계없는 파일은 보존한다. 임의 위치로 복사한 자료나 별도 depth 진단 사본의 보관은 작업자가 동일 기한으로 관리해야 한다.

## 탐지 학습·독립 평가 명령

`agent.train_detector`는 포트홀 1클래스와 얼굴/번호판 2클래스 모두 지원한다. 승인 release의 train/validation만 COCO로 내보내며 test는 복사하지 않는다. `detection_labels_verified=true`, 명시적 objects 목록과 certainty=confirmed가 필수다. 불확실 객체를 빈 정상 이미지로 바꾸지 않고 안전한 crop/보류를 요구한다. 각 객체는 class, bbox_xywh_px, certainty를 가진다. 실제 학습은 프로젝트 내부의 별도 Python3.10 CUDA 환경·공식 YOLOX 소스를 사용한다. 생성 실험은 [공식 Nano 설정](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/exps/default/yolox_nano.py)을 상속해 depthwise 모델을 유지하고 공식 train/export를 호출한다. CPU 모드에서는 데이터 내보내기 이후 CUDA 환경 필요를 명시하고 성공으로 표시하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent.train_detector data\manifests\release.json data\training\pothole --task pothole --export-data-only
.\.venv\Scripts\python.exe -m agent.train_detector data\manifests\redaction-release.json data\training\redaction --task redaction --repository vendor\YOLOX --interpreter .venv-training\Scripts\python.exe --device gpu --epochs 100 --batch-size 8
.\.venv\Scripts\python.exe -m agent.evaluate data\evaluation\locked-input.json data\evaluation\metrics.json
```

평가 입력은 `independent_locked_truth=true`와 detection/lane/segmentation 중 하나 이상을 가진다. detection={classes,truth,predictions}에서 행은 sample_id,class,bbox_xywh_px와 예측 confidence다. IoU0.5 일대일 매칭·confidence 정렬·101점 보간 AP와 중복 FP를 계산한다. lane={truth,predictions}는 segment_id,retention_pct(null 허용 예측)를 사용하며 판정 불가 실제 마모를 FN으로 계산하고 R MAE·판정 가능 비율·Precision/Recall을 낸다. segment_id 매칭표는 독립 실물/구간 정답에서 고정해야 한다. segmentation은 truth/predicted 마스크 경로 목록이며 GT255 ignore를 제외한 paint IoU를 계산한다. 실제 자료 없이 수치를 생성하지 않고 `acceptance_approved=false`를 유지한다. 학습/추론/평가 프로그램의 존재와 독립 현장 목표 달성은 별도다.

### 학습 복사본 기한·철회 전파

학습 출력은 프로젝트 내부의 별도 디렉터리를 사용한다(프로젝트 루트 자체는 거부). `data/training`을 권장하며 `data/models` 등 프로젝트 내부 다른 출력도 registry로 추적한다. 두 trainer는 복사한 이미지/마스크를 sample_id·source_id·선택 observation_id, 원본/복사본 SHA-256, 원본 릴리스 경로/해시와 연결해 `provenance.json`에 등록한다. 포트홀/비식별화 COCO에는 이미지 ID별 라벨 경로/해시도 연결한다. 유효 기한은 expires_at/source_expires_at/collection_expires_at/mask_expires_at/license_expires_at 중 제공된 가장 이른 시각이다. 도막 trainer도 원본 경로에서 직접 장기 학습하지 않고 추적되는 staging 복사본을 사용한다. 시험 분할은 복사하지 않는다.

`data/training/registry`가 등록된 출력을 추적하므로 `purge-local --apply`가 만료된 복사 이미지/마스크와 COCO 해당 이미지/박스 항목을 정리한다. 원본 파일은 관리 training 루트 안의 등록된 이미지/마스크이고 SHA가 일치할 때만 함께 제거한다. 그 밖의 원본 파일은 자동 삭제 범위에 넣지 않는다. 릴리스 목록은 불변 근거로 유지하며 재사용 차단용 해시 tombstone과 시각/요청 해시/건수만의 감사 로그를 별도로 남긴다.

```powershell
# request.json: {"source_ids":["SOURCE_UUID"],"reason":"승인된 철회 요청 사유"}
# sample_ids 또는 observation_ids도 지원한다. 기본값은 dry-run이다.
.\.venv\Scripts\python.exe -m agent revoke-data data\privacy\request.json
.\.venv\Scripts\python.exe -m agent revoke-data data\privacy\request.json --apply
.\.venv\Scripts\python.exe -m agent purge-local --apply
```

철회는 등록 provenance의 정확한 소유 파일만 개별 삭제하며 알 수 없는 디렉터리를 재귀 삭제하지 않는다. 동일 사진을 참조하는 검수 초안/로컬 전송 큐도 함께 제거하고, 선택적 sample/source lineage가 있으면 해당 출처까지 연결한다. 기관 서버의 개인정보 요청/삭제 절차는 별도의 서버 기능으로 수행해야 한다. 로컬 CLI가 원격 서버 삭제를 대신했다고 표시하지 않는다.

관련 모델 manifest와 학습 provenance는 `data_revoked=true`, `needs_revalidation=true`, `production_approved=false`가 된다. 런타임 readiness는 이 상태뿐 아니라 학습 자료의 현재 만료/철회 여부를 검사하므로 purge 전이나 manifest를 다른 곳으로 복사한 뒤에도 차단한다. 철회된 출처는 재학습 내보내기·재승인 큐 유입도 거부한다. 새 승인 자료·새 릴리스·새 출력 디렉터리에서 재학습/독립 평가한 모델로 교체해야 한다. 파일 삭제로 모델이 자동으로 unlearn됐다고 주장하지 않는다.

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

현재 결과: **54 passed**. 코덱 생성/분리/암복호화/재생부터 원 촬영 메타데이터 분석, 실제 strict observation 생성, 명시적 사람 승인, 암호화 큐, 주행 POST/PATCH와 사진 승인/업로드/관측 ACK까지 계약 시험했다. 프로파일·homography/ROI·색상/경계, GNSS/영상 이동 합치/공백, 점선 정상 공백·10 m 고유 패치 중복 제거, 합성 시차의 실제 SGBM와 주변 평면 잔차·객체 좌표, 독립 AP/IoU/판정 불가 FN, 학습 export의 불확실 객체 보류와 시험 분리, 작업/사진 보관 기한도 검사했다. 실제 외부 포트홀 ONNX 추론 시험이 포함되며 외부 모델 없는 새 checkout에서는 다운로드 없이 선택적 시험 1개를 skip한다.

전체 파이프라인의 탐지기 계약 시험은 명시적인 테스트 전용 fake model을 주입하고 HTTP MockTransport를 사용한다. 실제 ONNX 실행 시험과 별개이며 실노면 인식 정확도·장비 가용성·SDK/GPU·10 FPS/60분 실차 성능을 증명하지 않는다. 실제 gateway HTTP 통합 검수는 별도의 통합 시험 결과와 함께 확인해야 한다. 모든 모델 학습·export 경로가 준비돼 있지만, 승인된 자료·가중치와 격리 학습 환경 없이 학습 완료로 표시하지 않는다.

추가 보안/장비 오류 시험은 디스크 MP4 writer·평문 MP4/PTS 파일 쓰기·파일 기반 복호화를 금지한 상태에서 실제 메모리 MP4 인코딩→AES-GCM→메모리 재생을 수행한다. 구조화 수집계획/정차 확인 누락·보관 기준 불일치, 프로젝트 CA의 인증서/호스트 검증 유지, Ctrl+C 정상 종료와 카메라 단절의 암호화 조각/초기 장치·프로파일 메타데이터 보존도 확인했다. Ctrl+C는 마지막 조각을 마무리하고 stop_reason=operator로 반환하므로 collect가 바인딩 정보를 잃지 않는다. 실제 read/encoder 오류는 interrupted와 원인을 남기며 메타데이터는 녹화 시작부터 manifest에 들어 있다.
