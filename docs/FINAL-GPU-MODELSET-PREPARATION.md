# 최종 GPU 모델셋 메타데이터 준비

`scripts/prepare_final_gpu_model_set.py`는 모든 CPU 증거가 완성된 뒤 운영자가 고정한 세 parent manifest를 받아 GPU 시험용 input specification과 subset JSON만 생성한다. GPU session·이미지 바이트 읽기·실제 GPU plan freeze·운영 승격은 수행하지 않는다. 기존 `gpu_model_set.py` 및 `RUNTIME_SOURCES`는 수정하지 않는다.

## 사전 고정 선택

| Task | 원 VAL | GPU subset | 선택 |
| --- | --- | --- | --- |
| pothole | original402+MWPD250=652 | 20 | 각 domain 양성10, 필수 canonical3 모두 포함 |
| lane | 기존 recording100+추가 recording79=179 | 100 | recording별50, 양성 canonical3 모두 포함 |
| redaction | face400+plate108=508 | 40 | 알려진 평가 class별 양성20, face·plate canonical 각1 포함 |

각 그룹에서 필수 canonical을 먼저 `(source_id, sample_id)` 정렬로 배치하고, 같은 정렬의 나머지 원행으로 채운다. 원 full release·DEV·179 전체 검증은 수정하지 않는다. subset의 split·GT·원경로·SHA·기한은 해당 원행을 그대로 복사한다. lane subset100은 GPU 시험용이며 CPU full179 proof의 대체물이 아니다. 일반 lane VAL은 음성도 포함할 수 있으나 canonical3은 GT paint와 실제 앱 출력이 모두 양성이어야 한다.

## 입력 계약

로컬 request JSON의 format은 `final-cycle2-gpu-input-preparation-v1`, `TEST_images=0`, `GPU_used=false`이며 `models`에 `pothole/lane/redaction` 세 항목을 둔다. `source_sha256`는 helper의 `source_hashes()`가 반환하는 실행 소스 hash map을 그대로 고정한다. `preparation_protocol`에는 공개 `FINAL-GPU-MODELSET-PREPARATION-PROTOCOL-V2.json`의 정확한 pointer를 제공한다. 공개 protocol은 source SHA와 위 선택 규칙을 결합하며 변경되면 준비를 거부한다. 원 V1 protocol SHA `214267a3f3dd991604a5e54cf50ed24c08c2e6d6d1f62a8f3f81ba28f02573be`와 파일은 보존하고 V2가 부모 pointer와 변경 사유를 기록한다. 각 pointer는 `{ "path": "프로젝트 상대 경로", "sha256": "고정 SHA256" }`이다.

각 모델 항목에 다음 pointer를 제공한다.

- `manifest`, `checkpoint`, `full_release`, `development_manifest`, `training_provenance`
- `cpu_verifications`: pothole3·lane3·redaction2개 기본 허용오차 canonical proof
- lane에만 `full_validation`, `full_validation_plan`, `full_validation_source_plan`

full source와 DEV는 서로 다른 파일이어도 VAL 행 전체가 정확히 같아야 한다. 원 full release가 DEV 역할을 겸하는 경우만 그 안의 TEST 메타데이터 존재를 허용하며, TEST 영상/픽셀은 읽지 않는다. 모델의 parent release hash·DEV hash·provenance release hash·weights hash·checkpoint hash를 확인한다. 폐기/재검증 필요 표시와 만료 source/asset은 거부한다. source TTL 최소는 원 release의 모든 행과 provenance asset 중 최소값이며 연장하지 않는다.

lane full proof는 기존 `evaluate_cycle2.full_gate`의 179개 raw/default-tolerance 및 exact-mask 조건을 모두 통과해야 한다. 여기에 source plan의 `validation_count=179`·DEV/full release hash, worker source hash, manifest/checkpoint/weights, 실제 full plan의 모든 원행과 proof case index/image SHA를 확인한다. manifest에 checkpoint hash가 없는 기존 lane export 형식도 세 canonical과 full179 plan/report의 checkpoint hash가 모두 실제 checkpoint와 같아야 한다.

## V2의 Rescue export 호환 증거

동결된 Rescue trainer는 manifest에 `trained_checkpoint_sha256`를 기록하지 않는다. V1은 이 manifest를 일괄 거부했으며 아직 실제 준비를 실행하지 않았다. V2는 architecture가 정확히 `frozen legacy YOLOX-Nano specialists plus four tile P2 rescue`인 redaction에만 별도 증거를 허용한다. 이 경로의 entry에는 다음 두 pointer를 추가한다.

- `origin_manifest`: 학습이 export한 원 `model.json`
- `training_selection_freeze`: 그 학습의 원 `candidate-freeze.json`

원 manifest와 child manifest는 `confidence`, `training_provenance` 두 필드를 제외한 모든 값이 정확히 같아야 한다. 원 manifest의 `training_complete=true`도 확인한다. freeze는 `training_complete=true`, 정수 `TEST_images=0`, `manifest_sha256=origin_manifest.sha256`, `checkpoint_sha256=실제 entry checkpoint SHA`, `onnx_sha256=실제 ONNX SHA`, `confidence=원 manifest confidence`를 모두 만족해야 한다.

child에 연결된 두 strict canonical proof는 child manifest·같은 실제 checkpoint를 확인하며 기존 기본 raw 허용오차·post 일치·양성 face/plate 조건을 그대로 통과해야 한다. 표준 provenance의 release hash·폐기/재검증/만료 검사도 유지한다. 다른 detector architecture의 누락 필드는 계속 거부한다. 학습 source·원 export·가중치·threshold를 수정하거나 기존 FP32/GPU 증거를 소급하지 않는다.

```powershell
# 모든 CPU proof가 준비되고 root가 세 parent와 request를 확정한 후에만 실행
.\.venv\Scripts\python.exe scripts/prepare_final_gpu_model_set.py LOCAL_REQUEST_JSON data/gpu-model-sets/cycle2-final-v7/FRESH_CHILD
# 별도 root 승인 절차에서 기존 도구 실행: freeze → diagnose → stage → native → register
```

출력은 새 하위 디렉터리의 `input-specification.json`과 task별 `*-GPU-VAL-subset.json`이다. 기존 경로나 parent 자체를 출력으로 쓰지 못한다. 모든 증거를 검사하고 재확인한 뒤에만 디렉터리를 생성한다. 생성 도중 I/O 오류가 나면 불완전 출력을 성공으로 표시하지 않으며 해당 경로로 자동 재시도하지 않는다. 개별 row·이미지 SHA·경로는 로컬 ignored data에만 둔다. 기존 8개 보관 pin/inventory와 원학습 artifact는 변경하지 않는다.

이 도구의 합성 시험은 메타데이터 계약 검증이다. GPU 수치 지원·성능·10FPS·현장 정확도를 입증하지 않는다. 실제 GPU 승인에는 새 exact 세 모델·source·case-set에 대한 별도 strict 검증과 root 승인 slot이 필요하다.
