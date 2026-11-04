# 비식별화 fixed final12 EMA 후보

최종12epoch EMA를 별도 연구 후보로 추가하는 사전 정책입니다. 원 Rescue 학습 소스·12epoch 조건·데이터·AP50 checkpoint 선택·원 export는 그대로 보존합니다. 개발 VAL의6epoch까지 관측한 뒤 추가한 후보라는 이력을 기록했으며, 아직 생성되지 않은 최종12epoch 상태나 TEST 성적을 보고 epoch를 정한 후보가 아닙니다.

가족 선택의 목적은 비식별화 recall입니다. 원 AP50 최대 checkpoint 후보와 고정된 최종12epoch EMA 후보를 구분하고, 기존 모델을 포함한 가족 선택·confidence 보정은 별도 V9 정책에 따라 VAL에서 수행합니다. 이 후보가 더 우수하다는 주장이나 운영 승격은 없습니다.

정책은 [export plan](reports/PRIVACY-RESCUE-V8-FINAL12-EXPORT-PLAN.json)에 고정했습니다. root가 미리 공개한 [V9 사전 선언](reports/PRIVACY-FAMILY-V9-PREDECLARATION.json)을 부모로 참조합니다. 선언 시에는 recovery receipt의 완료epoch가12 미만이고 원 학습 감독이 실행 중임을 확인했습니다. 선언 과정에서 recovery `.pt`, 원 모델·가중치·사진·기존 NPY를 읽거나 GPU를 사용하지 않았습니다. 사전 metadata snapshot은 새 비공개 namespace에만 보관합니다.

`scripts/export_redaction_final12.py`의 실제 export는 별도 root 승인이 있어야 합니다. 다음 조건이 모두 충족되어야 recovery를 읽습니다.

- 원 학습 소스·학습 계획·factory·release·development·사전 정책 SHA가 유지됩니다.
- 원 감독과 training status가12epoch 정상완료이며 정확한 자식 PID와 생성시각의 프로세스가 종료됐습니다.
- recovery receipt가 최종12epoch를 가리키며 recovery 파일 SHA가 일치합니다.
- 원 보관기한이 유효하고 V8/V9 TEST attempt·결과 경로가 없습니다.
- 신규 출력에 이미 export attempt나 모델 파일이 없으며 명시적 승인 플래그와 고정된 정책 SHA가 제공됐습니다.

승인 이후에만 다음 명령을 실행합니다. 프로젝트 전용 Python3.11 학습 환경을 사용하며 exporter는 `CUDA_VISIBLE_DEVICES=''`를 강제합니다.

```powershell
.venv/Scripts/python.exe scripts/setup_training.py run scripts/export_redaction_final12.py export --cpu-export-approved --expected-plan-sha256 bca93d5d7163f4746c436ba2da4c277fe8b0356ac7cdfc8d0e4b91d857ad3daf
```

최종 recovery의 `EMA` 전체 state를 엄격하게 로드하고, optimizer update 수와 원 training history를 연결합니다. `model` live state나 AP 최대 checkpoint를 대신 사용하지 않습니다. 새 `best.pt`는 `{'model': EMA, 'epoch': 12}`입니다. 새 학습이나 optimizer 실행은 없습니다.

얼굴·번호판 legacy parameter와 BN buffer의 이름·dtype·shape·전체 bytes를 기존 v4와 대조합니다. 별도 합성 CPU 입력으로102,900개 legacy raw prefix의 bytes가 그대로인지 확인한 뒤 동일 factory에서333,300×7 출력을 CPU ONNX로 내보냅니다. 이 합성 prefix 검사는 실제 영상 성능이나 ONNX 수치 검증을 대신하지 않습니다. 새 모델은 별도 엄격한 CPU canonical 검증과 VAL 평가가 필요합니다.

비공개 출력은 `data/training/redaction-rescue-v8-final12/`에만 작성합니다.

| 파일 | 용도 |
|---|---|
| `plan.json` | 공개 정책과 동일한 사전 정책 |
| `best.pt`, `model.onnx`, `model.json` | 고정epoch12 EMA 후보와 명시적 checkpoint SHA |
| `provenance.json` | 부모 provenance SHA·원 TTL·source identity 상속 |
| `candidate-freeze.json` | manifest/checkpoint/ONNX/source/정책 SHA 고정 |
| `export-receipt.json` | 완료epoch12·EMA·원 학습/복구 SHA·CPU prefix 검사·TEST0 |

confidence.01은 native VAL 평가 floor의 임시값이며 운영 threshold로 승인된 값이 아닙니다. 가족 VAL 선택 이전 후보 생성용으로만 사용합니다. manifest에는 fixed final12 정책, checkpoint SHA, 원 학습 update 수와 연구용 상태를 명시합니다.

새 provenance는 원본을 복사하되 부모 모델 참조를 제거하고 새 모델만 참조합니다. `files`, `source_files`, `label_refs`, `delete_refs`를 비워 자식 후보의 보관·철회 작업이 부모 자료를 삭제하지 못하게 합니다. identity와 원 `expires_at`는 유지하고 부모 provenance SHA를 남깁니다. 새 provenance의 철회가 새 manifest를 차단할 수 있도록 registry에 별도로 등록합니다. 원 학습 provenance·모델·recovery는 수정하지 않습니다.

현재 합성 계약 검사는 작은 메모리 state와 metadata·AST만 사용합니다. 완료epoch11/불완전 EMA/변경된 update history/살아 있는 자식 상태를 거부하고, signed zero·BN buffer의 legacy digest 변경과 provenance 삭제 참조 분리를 검사합니다. 실제 export 완료나 정확도 개선을 의미하지 않습니다. 실패하면 해당 namespace를 보존하고 자동 재시도하지 않습니다.
