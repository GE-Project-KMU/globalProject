# 별도 GPU 학습 환경

작성일: 2026-11-04. 웹·수집 런타임 `.venv`와 학습 런타임 `.venv-training`을 분리한다. 현재 Windows 학습 인터프리터는 프로젝트 내부 `.venv-training/python.exe`이며 시스템 Python·드라이버·CUDA Toolkit을 설치하거나 변경하지 않았다.

## 고정 버전과 실행 증거

Python 3.11.9 공식 Windows embeddable package, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, NumPy 1.26.4, OpenCV 4.10.0, ONNX 1.17.0, ONNX Runtime 1.20.1을 사용한다. 직접 의존성은 `requirements-training.txt`, 실제 설치된 Windows CPython3.11 의존성 58개는 `requirements-training-lock-windows311.txt`로 고정했다. `pip check`는 의존성 충돌 없이 통과했다.

선정 근거: [PyTorch 공식 이전 버전 설치 조합](https://pytorch.org/get-started/previous-versions/)의 torch2.7.1/torchvision0.22.1/CUDA12.6 Windows wheel, [NVIDIA의 Turing 호환 안내](https://docs.nvidia.com/cuda/archive/12.6.0/turing-compatibility-guide/index.html), [Python 공식 3.11.9 배포](https://www.python.org/downloads/release/python-3119/). 애플리케이션 Python3.14와 구형 YOLOX 소스의 학습 의존성을 같은 환경에 혼합하지 않는다.

RTX 2080 Ti 11GB, capability7.5, torch compiled architecture `sm_75`, CUDA runtime12.6을 실제로 확인했다. 합성 텐서로 GPU 행렬 곱과 torchvision CUDA NMS, 다음 모델의 forward/backward/SGD1회를 실행했다.

| 모델/입력 | 최대 할당 | 최대 예약 | 학습 연산 시간 |
|---|---:|---:|---:|
| DeepLabV3-MobileNetV3, batch2,512×512 | 685.05MiB | 834MiB | 0.260초 |
| YOLOX-Nano, batch2,416×416 | 219.05MiB | 234MiB | 0.231초 |

이는 워밍업·데이터 로딩·전체 epoch를 포함한 학습 속도나 정확도 측정이 아니다. 유한한 loss와 실제 가중치 갱신, 메모리 여유를 확인한 환경 검사다. 별도 합성 Conv 모델의 ONNX 내보내기/checker, ONNX Runtime과 OpenCV 출력 비교도 통과했다. 실제 학습 모델은 각각 다시 내보내기·추론·holdout 평가를 수행한다. 보고서는 `data/training/environment/environment-report.json`에 저장한다.

## 재현과 캐시 범위

Windows 최초 구성:

```powershell
.\.venv\Scripts\python.exe scripts\setup_training.py bootstrap
.\.venv\Scripts\python.exe scripts\setup_training.py install
.\.venv\Scripts\python.exe scripts\setup_training.py run -m pip check
.\.venv\Scripts\python.exe scripts\setup_training.py run scripts\training_smoke.py
```

실제 고정된 Windows 패키지를 재현할 때는 bootstrap 후 `run -m pip install -r requirements-training-lock-windows311.txt`를 사용할 수 있다. Linux에서는 프로젝트 안의 Python3.11 venv에 직접 의존성 파일을 설치한다. Windows 전용 전이 의존성을 담은 lock은 Linux 환경에 그대로 적용하지 않는다.

`scripts.setup_training.environment()` 또는 `setup_training.py run ...`은 TEMP/TMP/TMPDIR, pip, Torch, CUDA, Hugging Face, XDG, matplotlib, Python bytecode, W&B 캐시를 모두 프로젝트 `.cache/training` 아래에 둔다. 사용자 site-packages와 전역 pip 설정을 사용하지 않는다. 학습 시작 후에는 다른 작업이 같은 환경에서 pip 설치·업그레이드를 동시에 수행하지 않는다. GPU 학습은 차선·포트홀·비식별 모델이 순차적으로 사용한다.

공식 YOLOX 소스는 `.cache/vendor/YOLOX`에 있으며 commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`를 기록했다. [공식 소스 의존성](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/requirements.txt)을 바탕으로 필요한 wheel을 설치했다. 임의 CUDA 확장을 컴파일하지 않고 공식 모델·손실 함수를 가져오는 단일 GPU 학습 루프를 사용한다. 소스 체크아웃에서는 `get_exp(exp_file=.../exps/default/yolox_nano.py)`를 사용한다. 기본 이름으로 찾는 패키지 복사본은 setup.py 설치 전에는 존재하지 않는다. 단순 ONNX 내보내기는 onnx-simplifier를 요구하지 않는다.

학습 CLI 예시(실제 승인/연구 데이터 release를 지정):

```powershell
.\.venv\Scripts\python.exe scripts\setup_training.py run -m agent.train_lane data\datasets\release.json data\training\lane --epochs 30 --batch-size 4 --pretrained --research
```

애플리케이션 전체 수집 CLI는 PyAV·암호화 모듈이 있는 기존 `.venv`에서 실행한다. 모델 상태 확인 예시: `.venv\Scripts\python.exe -m agent model-status data\training\lane\lane.json`.

## 학습 산출물 보관·전달

`.venv-training`, `.cache`, `data`, `.pt/.pth/.onnx`는 일반 Git과 Docker build context에서 제외한다. 코드·의존성 lock·출처·평가 요약만 코드 저장소로 관리하고 대형 가중치를 일반 Git 커밋에 넣지 않는다. 현재 작업은 외부 모델 저장소나 공개 Release에 가중치를 게시하지 않는다.

다른 실행 환경으로 전달할 때는 실제 ONNX, task/classes/input/output 계약을 담은 모델 manifest, SHA256, 코드·환경 lock, 학습 출처/만료·철회 메타데이터, 평가 요약을 함께 전달한다. 원본 영상·얼굴/번호판 원본·학습 이미지·`.env`·토큰·암호화 키를 모델 묶음에 포함하지 않는다. manifest의 weights/provenance 경로는 수신 환경 프로젝트 안의 위치로 조정하고 해시를 확인한다. 원본 데이터 철회/만료 시 재검증 차단 상태도 함께 전파해야 한다.

연구용 데이터의 사용조건과 split 출처 표시는 가중치 생성만으로 사라지지 않는다. `research`/프로젝트 인수 적격 여부, 외부 benchmark와 독립 주행 검증을 구분한다. 공개 데이터 평가 결과를 ZED 2i 실차 성능으로 표시하지 않는다. 수신 후 ONNX 로드·출력 비교와 독립 holdout 평가를 확인하고 그 결과와 배포용 기준을 별도로 기록한다.
