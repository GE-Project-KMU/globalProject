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

이 명령은 실제 성능 측정 runner가 아니다. 실제 실행은 승인된 독점 GPU slot에서 후속 runner를 연결해 진행한다. 대표 VAL32의 짧은 FPS와 비개인 합성 실제 decoder10분 soak는 별도 근거이며, 둘 모두 실물 카메라·stereo 깊이·GNSS·API·현장 전체10FPS 인수를 대신하지 않는다.
