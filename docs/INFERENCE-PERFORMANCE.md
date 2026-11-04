# 추론 속도 개선과 검증 범위

## 최신 추가 학습 모델의 검증 상태

2026-11-05 기준, 포트홀 Nano V8·차선 retry1·고정12회차 비식별화 모델의 CPU 합성16장 ABBA 결과는 기본 median1.0516FPS, fast median2.0322FPS다. 기본/fast의 원시 출력 차이는0이며 PNG·품질·추적·차선 결과가 같았다. 각 run에서 세 모델 모두16번 실제 추론했다. 전처리·추론·후처리·품질·추적·비식별·PNG/JSON·최종 flush를 포함하며 load/warmup/별도 동등성 검사/teardown은 제외한다. 영상 decode·카메라·GNSS·API·10분 지속 검증은 포함하지 않는다. [현재 시험과 재현 절차](FINAL-CPU-PIPELINE-BENCHMARK.md)를 참조한다.

새 모델 구성의 GPU160건 진단은 포트홀20·차선100건에서 raw/post가 통과했지만 비식별화40건 중1건의 raw가 기본 rtol.001/atol.0001에 실패했다. post 검출 결과는 모두 통과했으나 strict GPU profile은 등록하지 않았다. 아래 기존 모델의 GPU 처리량과 프로파일은 과거 artifact에 한정되며 최신 세 모델의10FPS 증거가 아니다. 정확도 목표 미달은 [포트홀](POTHOLE-CYCLE2-RESULTS.md)·[차선](LANE-CYCLE2-RESULTS.md)·[비식별화](reports/privacy-family-v10-count-retry1-results.md)에 별도로 기록했다.

## 기존 모델의 개선 이력

기존 가중치·입력 크기·threshold를 바꾸지 않고 CPU 실행 설정과 모델별 병렬 실행을 추가했다. **기본은 기존 CPU2threads/spinning/직렬**이다. 명시적 `fast`는8threads/no-spinning/서로 다른 모델3worker를 사용한다. CPU2D 분석 pipeline의 짧은 warmed 합성 측정은 median0.977→2.036FPS였으며 **10FPS 목표에는 미달**했다. [공개 집계](reports/INFERENCE-PERFORMANCE-RESULTS.json)에 실패·범위를 함께 기록했다.

## CPU 실제 병목과 개선

동일1280×720 비개인 합성 입력에서 기존 lane은 ORT/유한성 검사 약405ms, 전처리 약30ms, postprocess 약5ms였다. tiled privacy는 약318/40/6ms였다. 후처리를 생략하거나 입력·정확도를 낮추는 대신, ORT 스레드와 독립 모델 session의 실행 방식을 조정했다. 공식 문서도 session별 thread pool과 경합을 구분한다. [ORT thread 관리](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

세 모델의 predict만 측정한 median은 기본 직렬803ms, 기본2threads 동시490ms, no-spin4threads 동시414ms, no-spin8threads 동시391ms였다. 이는 capture·추적·마스킹·저장·API를 포함하는 전체 FPS가 아니다. 일부 모델 단독 stage는4threads가8threads보다 빨라 무조건 스레드가 많을수록 좋다고 해석하지 않는다. 이8물리/16논리 core 호스트의 pipeline 관측을 기준으로8/no-spin 후보를 고정했다.

그 후 속도 선택에 쓰지 않은 고정 공개 VAL을 차분 검증했다. lane100장 전체, pothole20장, privacy face20+plate20장에 기존 canonical 사례를 포함했다. 실제 native `fast` 경로에서 baseline 대비160건 모두 **raw max error0**, 기본 rtol0.001/atol0.0001 및 최종 bbox/count/class·lane mask 비교를 통과했다. 이 검사로 새로운 모델 정확도나 GT 목표를 달성했다고 주장하지 않는다. 동일성 증거는 공개 집계에 SHA를 고정한 기존 pothole v4·lane supported-ORT131-v3·selected tiled privacy 세 artifact에만 해당한다. CPU fast 설정 자체는 일반 옵션이지만 새 학습 artifact까지 동일성이 입증된 것은 아니며 default canonical·해당 VAL 검사 후 별도 fast 차분 검증이 필요하다.

실모델을 워밍한 뒤 ABBA 순서로 각8프레임을 처리한 CPU2D pipeline에는 품질 검사·포트홀 추적·opaque 마스킹·PNG/JSON 저장을 포함했다. 저장된 마스킹 사진 바이트도 같았다. 모델 load/capture/decode/GNSS/실제 보정/API는 제외했다. host에서 GPU 학습에 따른 CPU 경합 가능성도 있었다. 짧은 합성 반복이므로10분 지속 처리율이나 현장 수용량 검증으로 확대하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent workflow LOCAL_PREPARED_CONFIG --cpu-profile fast
.\.venv\Scripts\python.exe -m agent analyze LOCAL_SOURCE --output NEW_LOCAL_OUTPUT --pothole-manifest LOCAL_POTHOLE --lane-manifest LOCAL_LANE --redaction-manifest LOCAL_PRIVACY --cpu-profile fast --source-expires-at CONFIRMED_ORIGINAL_DEADLINE_UTC
```

실제 analysis job의 운영자 prepared config에도 `"cpu_profile":"fast"`를 선택할 수 있다. 생략하면 기존 default다. runtime metadata는 실제 provider의 thread/spinning 설정과 실행 profile을 남긴다. custom model factory는 기존 한 경로 인자 계약을 유지하며, custom factory가 실제8threads를 사용했다고 추정하지 않는다. 독립 model session을 사용하고 batch 간 mutable input 경합을 막으며, 예외 시 실행 중인 다른 session을 기다린 뒤 pool과 worker를 정리한다.

## CUDA: 기존 세 artifact의 명시적 연구 경로

별도 `.venv-performance`의 Python3.11.9/ORT-GPU1.20.2는 기존 앱 Python3.14/ORT-CPU1.31.1 환경을 바꾸지 않는다. CUDA12.6/cuDNN9.7.1 DLL은 프로젝트 Torch2.7.1 환경에서 process-local로 준비하며 전역 PATH·driver·SDK는 변경하지 않는다. [공식 CUDA 호환표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)의1.20.x/CUDA12.x/cuDNN9 경로를 사용했다. 더 최신 GPU wheel을 같은 DLL에 임의 혼용하지 않는다.

첫 독점 GPU 진단에서는 앱이 만든 동일 blob을 별도 process로 전달하여 lane100/pothole20/privacy40건의 raw/post strict가 모두 통과했다. CUDA provider와 `use_tf32=0`도 실제 설정으로 확인했다. 그러나 이때는 디스크를 경유한 진단이고 CPU reference와 번갈아 실행해 GPU 워밍이 일정하지 않았다. 네트워크-only 시간이나 provider 목록을 native workflow 성능 또는 전노드 CUDA 실행 증거로 취급하지 않는다.

현재 구현된 persistent shared-memory bridge는 앱의 OpenCV5/NumPy2.5 전처리와 후처리를 유지한다. float32 tensor만 공유메모리로 왕복하며 raw pixel/blob 디스크 파일을 만들지 않는다. 원CPU manifest와 엄격한 CPU131 profile은 그대로 두고, 원manifest·weights·strict proof hash에 연결한 **별도 local GPU child/profile**만 선택한다. 새 가중치에는 과거160건 proof를 재사용할 수 없다. 원소스 만료·철회·연구 전용 조건도 상속한다.

정확한 Windows/AMD64·ORT1.20.2·Python3.11.9·CUDA12.6/cuDNN9.7.1·SM75·provider/TF32 설정을 handshake로 확인한다. tensor는64MiB까지며 shape/dtype·요청 순서를 검사한다. CPU-only fallback, worker 종료·timeout·잘못된 응답은 성공으로 바꾸지 않고 실패 후 process/공유메모리를 정리한다. CUDA 목록이 존재해도 shape 등 일부 작은 연산이 CPU일 수 있으므로 실제 node placement는 별도 profiling 결과로 확인해야 한다.

[실행 소스 afb734a](https://github.com/GE-Project-KMU/globalProject/commit/afb734a)에서 native shared-memory GPU160건 검증을 완료했다. lane100/pothole20/privacy40 모두 기본 raw·post strict 통과, 실패0이었다. 실제 ORT profiling에서 CUDA node event90,260개를 관측했다. 이는 host 전처리·shape 관리 등 모든 작업이 GPU였다는 주장이 아니다.

같은 실행 환경에서 비개인 합성16프레임씩 CPU-default/CUDA-fast를 ABBA로 처리하고 CPU-fast 두 회를 별도로 측정했다.

| 실행 | warmed 처리율 | 범위 |
| --- | ---: | --- |
| CPU default A/A | 0.938 / 1.151FPS | median1.044FPS |
| CUDA fast B/B | 5.990 / 6.039FPS | median6.015FPS |
| CPU fast 보조 C/C | 2.340 / 2.216FPS | median2.278FPS |

전처리·공유메모리 forward·post·품질·포트홀 추적·마스킹·PNG/JSON 저장을 포함했고 마스킹 PNG 바이트도 같았다. CUDA load 약7.7초,5회 warmup 약1.1초, worker teardown 약1.5초는 따로 측정해 위 warmed FPS에서 제외했다. benchmark의 load/warmup/teardown까지 포함한16장 출력 batch 처리율은 CUDA1.230/1.238FPS였다. 이 값은 별도 warmup5장을 처리하는 benchmark 시나리오이며 일반 작업의 cold FPS를 새로 측정한 값이 아니다. capture/decode/GNSS/보정/API와10분 지속 시험은 제외했다. **warmed6.015FPS도10FPS에 미달**한다.

이후 수학·IPC를 바꾸지 않는 metadata-only admission 보강을 별도 적용했다. 원proof는 변경하지 않고 실제 native receipt→실행 profile→실행 child→원manifest/weights를 결합하는 binding receipt를 추가했다. 현재 local registry는 v2이며 과거 unbound v1을 거부한다. transport class·models.py·GPU worker의 보강 전후 동일 hash를 확인했다. 이 후속 source에 과거 측정값을 소급하지 않으며, 현재 source로 원시 proof의 benchmark를 재생하려 하면 source mismatch로 GPU 시작 전에 거부한다. 원실행을 재현하려면 afb734a 소스/환경/원profile을 별도로 보존해야 한다.

CPU-only CI의 protocol double·fake worker 시험은 하드웨어 증거를 대신하지 않는다. 실제 GPU 시험은 명시적 독점 slot과 로컬 model/proof가 있을 때만 수행하며 일반 CI에서는 이유를 표시하고 건너뛴다. 기본 backend는 계속 CPU이고 배포·정확도·privacy 승인은 부여하지 않는다.

```powershell
# 별도 환경 준비: GPU session 실행이나 시스템 설치를 뜻하지 않는다.
.\.venv\Scripts\python.exe scripts/setup_performance.py prepare
# 기존 strict 증거와 정확히 같은 모델만 local child/profile 생성, weights 복사 없음
.\.venv\Scripts\python.exe scripts/create_gpu_runtime_profiles.py
# 아래는 native GPU 검증된 기존 세 artifact의 bound v2 profile이며 승인된 GPU slot에서만 사용한다.
.\.venv\Scripts\python.exe -m agent workflow LOCAL_PREPARED_CONFIG --inference-backend cuda --gpu-profile data/gpu-runtime-profiles/ort1202-v2.json --cpu-profile fast
```

prepared config의 `inference_backend`/`gpu_profile`로도 명시할 수 있다. GPU에서 `fast`는 모델별3worker를 선택하지만 worker 내부 CPU fallback thread 등은 검증된 GPU 설정으로 고정한다. 새 실행의 환경·profile을 기록하고 CPU bundle의 환경을 GPU 환경이라고 보고하지 않는다.

## 다음 독점 slot의 재현 명령

```powershell
.\.venv\Scripts\python.exe scripts/verify_gpu_runtime.py NEW_NATIVE_PROOF_DIRECTORY --gpu-profile data/gpu-runtime-profiles/ort1202-v2.json
.\.venv\Scripts\python.exe scripts/benchmark_verified_pipeline.py NEW_PIPELINE_REPORT_DIRECTORY --gpu-profile data/gpu-runtime-profiles/ort1202-v2.json --native-proof NEW_NATIVE_PROOF_DIRECTORY/aggregate.json
```

첫 명령은 같은160개 공개 VAL에 native shared-memory raw/post와 blob 동일성을 검사하고 실제 node event provider를 수집한다. 둘째는 통과한 native proof와 같은 profile/source일 때만 실행한다. 모델 load·warmup·forward/transport·quality/tracking/redaction·PNG/JSON·worker teardown을 분리하며 warmed 처리율의 제외 항목도 기록한다. capture/decode/GNSS/보정/API와10분 지속 시험은 포함하지 않으므로10FPS 현장 합격을 주장하지 않는다.
