# 대표 공개 VAL32 성능 계획

현재는 **metadata 선택 도구와 합성 계약만 준비**했으며 실제 case 선택·이미지 read·resize·GPU 실행은0이다. 최종 새 모델 세 개의 CPU canonical·GPU160 strict/native 증거와 `cycle2-final-v7`의 exact registered modelset을 먼저 고정한다. 이전 v6 모델의 수치 증거를 새 모델에 재사용하지 않는다.

`scripts/prepare_representative_performance.py`는 고정 VAL160 안에서 포트홀8·차선8·face8·plate8을 선택한다. canonical 사례를 먼저 포함하고 나머지는 `source_id/sample_id` 순으로 정렬한다. canonical이8개를 넘거나 group이 부족하거나 중복·누락이 있으면 실패한다. frame 순서는 네 group round-robin이다. TEST·사용자 영상이나 성능 결과를 보고 사례를 바꾸지 않는다.

private 계획에는 원32개 SHA·경로·기한과 원160 plan/native/profile SHA를 고정한다. 이미지는 아직 읽지 않는다. 후속 실제 성능 실행에서 원바이트 hash를 확인하고 BGR uint8·종횡비 유지·왼쪽 위 정렬·114 padding으로1280×720 letterbox를 만든다. 이는 속도 입력용 파생이며 정확도 release·GT·원영상을 수정하거나 새 정확도 수치로 사용하지 않는다.

원촬영일을 모르는 공개 JPEG에 현재 시각을 `source_captured_at`으로 넣지 않는다. pipeline에는 **고정 원기한 `source_expires_at`만** 전달하고, frame index/12의 replay clock은 합성 timeline이라고 기록한다. 원 capture UTC·GNSS·연속 주행 정보라고 주장하지 않는다. 모든 파생·출력은 로컬 미검토 상태이며 원기한/더 짧은 cap을 상속하고 Git에 올리지 않는다.

후속 실험은 동일 FP32 조합의 GPU 직렬/overlap/overlap/직렬 ABBA 각32장, 한 번으로 계획한다. source read·decode·letterbox 준비 시간은 별도 기록하고 warmed 모델 pipeline에서 제외한다. 모델 전처리·추론·후처리·quality·tracking·마스킹·PNG/SHA·JSON/최종 flush는 포함한다. PNG byte·관측 순서·quality 일치를 별도로 검사한다. 서로 다른 장면의 round-robin은 실제 연속 도로 주행이나 추적 정확도 증거가 아니다.

```powershell
# 새 모델 strict/native 증거가 모두 고정된 이후에만 metadata plan 생성.
.\.venv\Scripts\python.exe scripts/prepare_representative_performance.py --profile LOCAL_NEW_REGISTERED_PROFILE --output data/gpu-model-sets/cycle2-final-v7/representative-val32-plan.json
```

위 명령은 metadata 선택만 한다. 별도 `benchmark_representative_pipeline.py`가 아래 두 단계로 실제 실행을 연결한다. **현재 실제 공개사진 read·GPU 실행은 하지 않았으며 합성 CPU 계약만 확인한 준비 상태**다.

```powershell
# metadata only: exact profile/native/VAL32를 재검사하고 실행 소스·환경·출력경로 SHA를 고정
.\.venv\Scripts\python.exe scripts/benchmark_representative_pipeline.py freeze --representative-plan data/gpu-model-sets/cycle2-final-v7/representative-val32-plan.json --execution-plan data/gpu-model-sets/cycle2-final-v7/representative-execution.json --output data/gpu-model-sets/cycle2-final-v7/representative-abba
# 실제 독점 GPU slot 승인 뒤에만 실행. 앞 명령의 execution SHA를 그대로 전달
.\.venv\Scripts\python.exe scripts/benchmark_representative_pipeline.py run --execution-plan data/gpu-model-sets/cycle2-final-v7/representative-execution.json --execution-sha256 PINNED_EXECUTION_SHA --approved-exclusive-gpu-slot
```

승인 flag가 없으면 metadata·이미지를 읽지 않고 중단한다. flag가 있어도 execution SHA·현재 코드/환경·원profile/native proof·기한·지원 Windows/AMD64 host를 먼저 검사한다. 이미지마다 byte SHA·32MiB 파일·24M pixel 상한을 확인하고 준비 buffer를128MiB로 제한한다. 출력 budget은1GiB이며 추가512MiB disk reserve가 필요하다. 실제 GPU 실행은 main RSS2GiB 상한을 load 전·warm 후·처리 후에 확인한다. 이는 연속 peak 측정이나 메모리 누수 부재 증명이 아니다. 같은 binding 출력이 있으면 재실행하지 않는다.

새 촬영일을 만들지 않고 `source_captured_at=None`을 전달한다. 실제 원32개와 세 모델 provenance의 최소 원기한을 함께 적용한다. 모델 load·5회 warmup·teardown·준비시간을 따로 기록하며, PNG/SHA·JSON·마지막 flush는 warmed processing에 포함한다. 종료 후 전체 PNG hash 감사 시간은 별도다. PNG byte·품질·track 상태·lane·frame 순서를 엄격 비교하되 작업마다 생성되는 run/event UUID와 출력 경로는 동일성 대상에서 제외한다. 불일치는 aggregate에 실패로 저장하고 성공 exit로 끝내지 않는다. 예외가 나도 모든 모델을 정리하고 원래 예외를 보존한다.

private preparation에는 원이미지 SHA·기한·준비시간만 기록한다. 원이미지·prepped raw buffer는 disk에 복제하지 않으며 미검토 출력은 해당 원기한으로 관리한다. 공개할 결과는 source/model/plan 집계 SHA·건수·latency·제약만 허용한다.

현재 CPU 계약은 합성 PNG·custom test model만 사용한다. 실제 pipeline의32장×ABBA 저장/flush·원본 cap·SHA/크기·지원 host·RSS guard·source binding·출력 재사용 거부·실패 시 모든 모델 종료를 검사했다. 전용16개를 포함한 확대 회귀171개가 통과했고 실제 GPU opt-in1개는 skip했다(21.47초). 테스트 double의 처리율은 GPU 성능 증거로 사용하지 않는다. 실제 공개사진 read·GPU ABBA 결과는 아직 없다.

대표 VAL32의 짧은 FPS와 비개인 합성 실제 decoder10분 soak는 별도 근거이며, 둘 모두 실물 카메라·stereo 깊이·GNSS·API·현장 전체10FPS 인수를 대신하지 않는다.
