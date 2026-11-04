# 새 모델3개의 GPU 승인 절차

기존 세 모델의160건 통과나6.015FPS를 새 가중치에 소급하지 않는다. 새 모델은 선택·manifest freeze·CPU canonical을 끝낸 뒤 아래 단계로 GPU를 따로 검사한다. 기본 앱 backend는 계속 CPU다. 이 절차의 승인은 연구용 hardware 경로에 한정되며 정확도·현장·개인정보 보호 인수를 뜻하지 않는다.

## 입력과 freeze

프로젝트 내부의 비공개 specification JSON에 세 항목을 지정한다. 각 경로와 SHA는 현재 파일의 실제 값이어야 한다.

```json
{
  "models": {
    "pothole": {
      "manifest": {"path": "LOCAL_MODEL_JSON", "sha256": "PINNED_SHA256"},
      "cpu_verifications": [{"path": "LOCAL_CPU_CANONICAL_JSON", "sha256": "PINNED_SHA256"}],
      "validation_manifest": {"path": "LOCAL_PUBLIC_VALIDATION_RELEASE", "sha256": "PINNED_SHA256"}
    },
    "lane": {"manifest": {}, "cpu_verifications": [], "validation_manifest": {}},
    "redaction": {"manifest": {}, "cpu_verifications": [], "validation_manifest": {}}
  }
}
```

위 빈 항목은 같은 구조로 채우는 예시이며 실행 가능한 증거가 아니다. lane은 양성 canonical3개가 있으면 세 pointer를 모두 지정한다. 개인정보 모델도 고정 face/plate canonical이 있으면 모두 넣는다. manifest metadata가 canonical 이후 바뀌었다면 SHA 불일치를 무시하지 않고 최종 manifest에 맞는 canonical을 다시 확보한다.

CPU 증거는 정확한 manifest/weights, 실제 공개 VAL의 양성 GT·양성 출력, 앱 ORT1.31.1, 독립 ORT/Torch 비교, raw 기본 rtol0.001/atol0.0001 및 최종 post 통과를 요구한다. `artifact_verified` 문구만 보지 않고 실제 수치/mask 조건을 재검사한다. GPU 전에는 세 원CPU 모델과 provenance가 모두 로드되어야 한다.

freeze는 canonical 사례와 source-id 순서로 lane VAL100 전체, pothole 양성20장, privacy face20+plate20장을 결정한다. TRAIN/TEST를 선택하지 않고 각 case의 SHA·원래 만료일을 고정한다. case는 모델이 가리키는 원소스 release의 validation에 속해야 하며 별도 VAL 파일로 기한을 연장할 수 없다. 사용자 녹화 namespace는 이 공개 연구 VAL 입력으로 허용하지 않는다. 이미 사용한 VAL이므로 독립 최종 정확도 증거는 아니다.

여러 공개 소스의 대표성을 사전에 지정할 때는 각 모델 항목에 `case_sha256s` 배열을 추가할 수 있다. GPU 실행 전에 순서까지 고정하고 canonical 사례 전부를 포함해야 한다. lane은 VAL100 전체, pothole은 GT 양성20장, 개인정보 모델은 GT 양성 face20장과 plate20장을 유지한다. TEST·중복·누락 canonical·소스 기한 연장은 거부한다. GPU 결과를 본 뒤 사례를 바꾸는 용도로 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/gpu_model_set.py freeze LOCAL_SPECIFICATION.json --output data/gpu-model-sets/NEW_SET/plan.json
```

freeze는 CPU metadata·해시 확인이며 GPU session을 실행하지 않는다. 이후 코드·원모델·canonical·release·case 순서·기한이 달라지면 같은 plan을 사용하지 않는다. source freeze는 retention 등 다른 writer의 수정이 끝난 후 수행한다.

## 두 번의 엄격 검사와 등록

```powershell
# 승인된 독점 GPU slot에서만 실행: 아직 앱 지원 profile이 아닌 진단
.\.venv\Scripts\python.exe scripts/gpu_model_set.py diagnose data/gpu-model-sets/NEW_SET/plan.json --output data/gpu-model-sets/NEW_SET/diagnostic

# 위160건 모두 통과할 때만 staging child 생성. weights 복사 없음, GPU 실행 없음
.\.venv\Scripts\python.exe scripts/gpu_model_set.py stage data/gpu-model-sets/NEW_SET/plan.json --diagnostic data/gpu-model-sets/NEW_SET/diagnostic/aggregate.json --output data/gpu-model-sets/NEW_SET/stage

# 실제 앱 전처리·shared-memory backend·앱 postprocess로 같은160건 재검사
.\.venv\Scripts\python.exe scripts/gpu_model_set.py native data/gpu-model-sets/NEW_SET/stage/staging.json --output data/gpu-model-sets/NEW_SET/native

# native160/node 증거까지 통과한 exact descriptor core만 workflow registry 등록
.\.venv\Scripts\python.exe scripts/gpu_model_set.py register data/gpu-model-sets/NEW_SET/stage/staging.json --native data/gpu-model-sets/NEW_SET/native/aggregate.json --output data/gpu-runtime-profiles/NEW_SET.json
```

진단은 frozen CPU 승인을 확인한 전용 client로만 수행하며 일반 `OnnxModel`의 원CPU profile을 해제하지 않는다. 진단160건 중 하나라도 raw 또는 mask/bbox/count/class 조건에 실패하면 child를 생성하지 않는다. 허용오차를 올리거나 TEST로 threshold를 다시 고르지 않는다. staging descriptor는 workflow registry 밖에 있으며 등록된 지원으로 취급하지 않는다. native도 같은160건을 검증하고 실제 CUDA node event를 확인한 뒤에만 등록한다. 일부 shape/host 작업까지 전부 GPU였다고 확대하지 않는다.

각 단계는 부모manifest·weights·CPU증거·case set·platform·source SHA를 닫아서 연결한다. registry의 descriptor core는 세 child의 정확한 SHA를 포함하고, native proof가 그 core를 가리켜야 한다. 다른 모델의 기존 proof, 바뀐 case, 빈/실패 proof, source 변경은 등록 근거로 재사용하지 못한다. 원CPU manifest·연구/보관/철회 조건·가중치 파일은 그대로 둔다.

현재 hardware 경로는 Windows/AMD64, Python3.11.9 worker, ORT-GPU1.20.2, 프로젝트 CUDA12.6/cuDNN9.7.1 DLL, 검증한 RTX2080Ti/SM75, TF32off와 제한된 provider 옵션이다. shared tensor는 float32·고정 shape·64MiB 이내다. 다른 GPU·라이브러리·artifact는 확인 없이 지원한다고 표시하지 않는다.

## 같은 모델 집합의 성능 측정

```powershell
.\.venv\Scripts\python.exe scripts/gpu_model_set.py benchmark data/gpu-runtime-profiles/NEW_SET.json --output data/gpu-model-sets/NEW_SET/pipeline-benchmark
```

등록된 exact plan/native proof/source를 확인한 후 CPU-default/CUDA-fast의 ABBA 측정을 수행한다. 새 CPU-fast 수치 증거는 자동 부여하지 않으며 필요하면 별도 차분 검증한다. model load·warmup·teardown과 warmed 처리 시간을 구분하고, forward/transport·quality·tracking·redaction·PNG/JSON의 포함 범위를 기록한다. capture/decode/GNSS/보정/API·10분 지속/현장 시험은 이 짧은 합성 측정으로 대체하지 않는다.10FPS에 미달하면 그대로 기록한다.

개별 case ID·이미지 SHA·경로·수치 오류 좌표는 ignored 로컬 결과에만 두고 원기한을 상속한다. 공개 보고서는 model/code/case-plan의 집계 SHA, 통과 건수·latency·제약만 허용한다. 사용자 녹화를 새 학습·평가 자료로 추가하지 않는다. 일반 CPU CI의 합성 admission/protocol 테스트는 GPU를 실행하지 않으며, 실제 GPU 경로는 명시된 slot에서만 검증한다.
