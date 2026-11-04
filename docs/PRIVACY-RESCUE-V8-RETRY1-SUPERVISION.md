# Rescue V8 의존성 복구 후 retry1 감독

원 GPU preflight는 `ModuleNotFoundError`로 완료하지 못했다. 원 감독 source·c5e 감독 계획·failed receipt/log/device samples/observer log는 모두 보존한다. 원 실패는 완료된 preflight나 실제12epoch 학습의 성공 증거가 아니다. 원 preflight 임시 state를 재사용하거나 실패 기록을 덮어쓰지 않는다.

새 감독은 `scripts/supervise_redaction_rescue_v8_retry1.py`이며, 원 컨트롤러의 fixed trainer command·단일 GPU 자원 감시·own identity cleanup 구조를 복사한 별도 파일이다. 학습 OUT는 `data/training/redaction-rescue-v8` 그대로다. 학습 source SHA `7bc192db04ff79ff0cad9c3b440aa8357c99f387eecf4a7093ce3f59d9475f65`, 학습 계획 SHA `6bbd59191a2c352a6a2bf7dc0fbd15f25ced0f500be6010fa3011a4e0f94164d`와 학습 수학·데이터·초기화·optimizer는 바꾸지 않는다.

## 새 계획과 환경 증거

`scripts/prepare_rescue_v8_retry1_supervision.py`가 새 `PRIVACY-RESCUE-V8-RETRY1-SUPERVISION-PLAN.json`을 exclusive-create한다. 원 계획 SHA `c5e1dcc65cdd48d11d46b79f0fe6caf0f045bf5628c8597f5356a1ca0a29b1ae`를 부모로 유지한다. 원 failed preflight의 정확한 SHA·nonzero exit·source unchanged·own child gone·observer 종료·own OOM 없음·device budget 초과 없음·import 실패 로그를 확인한다. 원 PID가 다른 프로세스에 재사용된 경우 creation time을 대조하며 다른 프로세스를 종료하지 않는다.

환경 증거는 별도 `redaction-rescue-v8-env-repair`의 before/after/import-success/requirements snapshot와 실패·호환 설치 로그·공식 wheel metadata다. 프로젝트 Python3.11에서 av19 설치가 거부된 로그를 보존하고, CPU import에 성공한 **av18.1.0/cryptography50.0.2/cffi2.1.1/pycparser3.12**만 추가했다. before에 있던 package version은 모두 그대로이며 after map·snapshot·import 진단이 정확히 일치해야 한다. 필수 import는 `agent.evaluate`, `agent.pipeline`, `yolox.utils`, `torch`다. 실제 CUDA 초기화·source pixel 접근·학습은 이 진단에서0이다.

`validate_pins(spec, check_deadline=...)`는 metadata/SHA·기존 실패 child 정리만 확인한다. 큐가 이 함수를30초마다 호출해도 Torch import subprocess를 반복 생성하지 않는다. 별도 `verify_current_environment(spec)`는 **supervised phase의 exclusive attempt claim 후, observer/GPU Popen 직전에 한 번** 실행한다. 같은 training Python에서 실제 `importlib.metadata` package map을 다시 측정하고 필수 import와 빈 `CUDA_VISIBLE_DEVICES`, `torch.cuda.is_initialized()==False`를 확인한다. timeout60초이며 종료/timeout은 해당 CPU subprocess handle에만 적용한다. embedded Python의 `_pth`에 의존하지 않고 고정 project/vendor 경로를 명시한다.

실제 CPU 재측정도 전체 package map 일치·필수4import·CUDA 미초기화로 통과했다. 이는 실제 GPU preflight/학습 성공을 뜻하지 않는다.

## 별도 감독 namespace

| 종류 | retry1 경로 |
| --- | --- |
| receipt | `external-{preflight,train}-supervisor-retry1.json` |
| device samples | `external-{preflight,train}-device-samples-retry1.json` |
| log | `.cache/cycle2-rescue-v8-{preflight,train}-retry1.log` |
| observer log | `.cache/cycle2-rescue-v8-{preflight,train}-observer-retry1.log` |
| 공개 감독 집계 | `PRIVACY-RESCUE-V8-{PREFLIGHT,TRAIN}-RETRY1-SUPERVISION.json` |

원 trainer가 생성하는 `gpu-preflight.json`, `training-status.json`, 실제 모델 artifact 이름은 그대로다. 새 preflight는 기존 `gpu-preflight.json`이나 학습 artifact가 있으면 거부한다. train은 새 retry1 preflight receipt의 completed/exit0/child gone·정확한 report hash·8개 임시 step 보고·finite GPU 자원 증거·현재환경 CPU 검사 성공을 확인한 뒤 별도 fresh construct를 실행한다. preflight model/optimizer state를 resume하지 않는다.

9GiB device-reported budget, 요청100ms 감시, 초기2개 sample 각각2GiB 미만, 원기한 **2026-11-10 16:55:19.285523 UTC**, own identity cleanup을 유지한다. CPU 검사 후와 GPU Popen 전에도 만료를 재확인한다. WDDM sampling은 hard VRAM quota나 sample 사이 순간초과 부재 증거가 아니며, OOM 검사는 own child log 범위다. global OOM0을 주장하지 않는다.

```powershell
# metadata-only 새 계획 고정; 기존 파일 덮어쓰기 없음
.\.venv\Scripts\python.exe scripts/prepare_rescue_v8_retry1_supervision.py
# root 검토·별도 독점 GPU phase 승인 후에만:
.\.venv\Scripts\python.exe scripts/supervise_redaction_rescue_v8_retry1.py preflight --gpu-phase-approved
# preflight 완료/정리 및 root의 다음 phase 승인 후:
.\.venv\Scripts\python.exe scripts/supervise_redaction_rescue_v8_retry1.py train --gpu-phase-approved
```

승인 flag가 없으면 device query 전에 거부한다. phase receipt는 CPU probe보다 먼저 exclusive-create하여 동시 실행·실패 후 자동 재시도·다른 소유자의 덮어쓰기를 막는다. 실패 기록은 보존하며 새로운 수동 retry revision 없이 같은 phase를 다시 실행하지 않는다.

이번 준비에서 실제 GPU preflight·실제 학습·TEST는0이다. 계약시험은 synthetic package/failure evidence와 mocked child/device만 사용한다. 원8개 control pin·기존 inventory를 수정하거나 실제 보관 등록·삭제를 수행하지 않았다. 새 receipt/plan의 최소 provenance 보존은 정확 format·basename·알려진 필드만 허용한다. 개별 image ID·좌표·source filename·알 수 없는 private 추가값을 가진 같은 format을 일반 aggregate로 영구 보존하지 않는다. 실제 보관 처리와 cache log 이관은 모든 writer 종료 후 별도 수동 단계다.
