# GPU 10분 소프트웨어 soak 준비

`scripts/probe_gpu_pipeline_soak.py`는 **아직 실제10분 GPU 실행을 하지 않은 준비 도구**다. 최종 새 세 모델의 CPU/GPU 수치·후처리 증거와 명시적 registered modelset을 확보하고, 다른 GPU 학습/추론 및 CPU 모델 평가를 중지한 승인 시간대에서만 실행한다. 기존 v6의 짧은 IO 측정이나 다른 모델의160건 증거를 새 모델에 소급하지 않는다.

입력은 도구가 새로 생성하는 비개인1280×720 MP4다. 실제 decoder를 사용해 짧은 합성 clip을 반복하며 source 파일·profile·runtime/helper source를 hash로 고정한다. 처리시간600초 동안 프레임을 생략하거나 저장 파일을 ring으로 덮어쓰지 않는다. 입력은12FPS로 pacing하여 무제한 출력 증가를 제한한다. 이는 uncapped 최고 처리율 시험이 아니며 느린 처리를 pacing으로 성공 처리하지 않는다. 실제 완료한 frame 수를 decode·품질·추론·후처리·추적·PNG/SHA·JSON·마지막 flush를 포함한 wall time으로 나눈다.

모델 load·5회 워밍·종료 후 전체 PNG SHA 감사는 별도 범위다. 첫30초 이후 main/세 worker의 RSS와 전체 GPU의 memory/utilization을5초 간격으로 기록한다. GPU 관측은 per-worker VRAM이 아니며 메모리 누수 부재 증명으로 확대하지 않는다. host RSS12GiB·GPU device10GiB budget을 검사한다. source의 원기한을 상속하고10분 완료 전에 만료할 모델은 거부한다. 생성한 합성 자료의 별도 기한은2시간 이내이며 기존 소스 기한을 늘리지 않는다.

출력 budget은40GiB, free disk reserve는5GiB다. 시작 전에 최소45GiB의 가용 공간을 요구한다. 최악의 PNG 크기를 보수적으로 계산한7,202장 상한 및 예약 metadata 범위가 budget 안에 있어야 한다. 실제 사용량·free reserve·저장 개수·순서·SHA·source/profile 불변·모든 GPU worker 종료를 확인한다. budget/decoder/worker/memory/기한 오류는 성공으로 바꾸지 않고 failure receipt를 남긴다.

```powershell
# 아래 flag 자체가 승인 요청을 대신하지 않는다. 실제 독점 slot을 먼저 확보한다.
.\.venv\Scripts\python.exe scripts/probe_gpu_pipeline_soak.py --profile LOCAL_FRESH_REGISTERED_MODEL_SET --output data/gpu-model-sets/NEW_FINAL_SET/soak --approved-exclusive-gpu-slot
```

현재 검증은 CPU 설정·예산·승인 guard·실제 합성1280×720 decode·순서·메모리 집계 계약에 한정된다. GPU나600초 성공을 주장하지 않는다. 실행 후에도 반복 합성 clip은60분 실제 주행·카메라·GNSS·stereo 깊이·현장 정확도·개인정보 recall·API 통합 시험을 대체하지 않는다.
