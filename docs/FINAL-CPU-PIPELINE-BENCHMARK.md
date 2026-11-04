# 최종 고정3모델 CPU fallback 벤치마크

이 도구는 GPU raw FAIL을 CPU 성공으로 바꾸거나 GPU 지원을 승인하지 않는다. 정확 fda534… top-level model-set의 원CPU 부모3모델/가중치/CPU canonical proof/source/TTL만 사용한다. 기존 benchmark_cpu_pipeline.py의 oldMANIFESTS나 앱 코어는 수정·재사용하지 않는다.

prepare는 metadata/모델 SHA 검증 후 공개protocol과 별도 execution plan을 만든다. 모델/session·사진·GPU를 열지 않는다. run은 root의 독점 CPU 측정 승인과 고정 execution SHA가 있어야 하며 다른 모델 평가/학습/다운로드 경합과 분리한다.

```powershell
# 준비만; 기존 파일이 있으면 반복하지 않는다
.\.venv\Scripts\python.exe scripts/benchmark_final_cpu_pipeline.py prepare --execution-plan data/gpu-model-sets/cycle2-final-v7/cpu-fallback-execution-v1.json --output data/gpu-model-sets/cycle2-final-v7/cpu-fallback-v1
# root 검토와 독점 측정 승인 후에만
.\.venv\Scripts\python.exe scripts/benchmark_final_cpu_pipeline.py run --execution-plan data/gpu-model-sets/cycle2-final-v7/cpu-fallback-execution-v1.json --execution-sha256 EXACT_PREPARED_SHA --approved-exclusive-cpu-slot
```

입력은 새로 생성한 비개인1280×720 uint8 합성16장이며 모든 run에 같은 배열·순서·i/12 합성 timeline을 사용한다. 촬영일을 추정하지 않는다. 생성시각+2h와 원 model source 기한 최소를 적용하고 미검토 PNG/JSON도 그 cap을 상속한다. 원 공개VAL/TEST/사용자 사진은 읽지 않는다.

ABBA는 default(CPU2/spinningTrue/1modelworker), fast(CPU8/spinningFalse/3modelworkers), fast, default다. IO는 모두 overlap·OpenCV1로 고정한다. run마다 독립3session만 load하고 각5회 warmup한다. raw/post proof는 첫 default/fast run의16장을 **별도 untimed 단계**로 비교하며 같은 input blob SHA·기본 raw rtol .001/atol .0001·post exact가 필요하다. baseline raw는 최대512MiB RAM에서만 유지하고 fast 비교 후 해제한다. raw tensor는 디스크에 쓰지 않는다. 두 번째fast/default에는 raw proof를 반복하지 않지만 모든 timed PNG·관측결과를 비교한다.

timed pipeline은 매run16frame에 대해3모델 각각 실제predict16회를 assert/기록한다. quality가 모두 bad라도 모델 호출을 생략하면 실패한다. 전처리/추론/post/quality/추적/불투명비식별/PNG·JSON 저장/최종 flush를 포함하며 load/warmup/rawproof/teardown은 별도 기록한다. 추적 UUID만 첫 등장 순번으로 정규화해 frame간 연결을 유지하고, image path/runUUID/시작시간·CPU설정 차이 이외의 quality/lane/추적·PNG를 같게 비교한다. JSON byte 자체는 runtime과UUID 때문에 같다고 주장하지 않는다.

각run3session의 정리·GC가 끝난 뒤 다음profile을 load한다. 최대RSS8GiB·output1GiB+reserve512MiB·단회attempt/source변경/기한/예외 cleanup을 검사하고 실패·부분 output을 보존한다. 실패한fast를 동등성 통과로 표시하지 않는다. profile별medianFPS와 각stage 시간을 기록하되 concurrent stage 시간을 합산하지 않는다.

이 동등성 증거는 합성16장 한정이다. 긍정VAL 전체 정확도·현장10FPS·10분지속·카메라/decode/GNSS/물리깊이/API처리량·운영 승격을 입증하지 않는다. 실제FPS 미달과 CPU/GPU 제한을 그대로 보고한다. 코드 준비·합성 unit tests만으로 실제 모델 벤치마크를 완료했다고 표시하지 않는다.


## 실제 고정3모델 측정 결과

Root의 실제 단회 ABBA 실행이 정상 종료됐다. 같은 실행 안에서 default CPU median **1.0516FPS**, fast CPU median **2.0322FPS**로 약 **1.93배**였으나 **10FPS 목표는 미달**했다. 이 비교를 다른 모델 조합·과거 GPU 측정의 FPS와 직접 비교하지 않는다. 앱 기본 프로필을 자동 변경하거나 운영 승격하지 않는다.

| 실행 | CPU 프로필 | 처리시간(16장) | 처리 FPS | load | warmup5 | teardown |
|---|---|---:|---:|---:|---:|---:|
| A1 | default | 15.0641s | 1.0621 | 1.7108s | 4.6064s | .0813s |
| B1 | fast | 7.8351s | 2.0421 | 1.6921s | 2.3491s | .0877s |
| B2 | fast | 7.9121s | 2.0222 | 1.7079s | 2.3741s | .0891s |
| A2 | default | 15.3684s | 1.0411 | 1.8132s | 4.9345s | .0830s |

입력은 비개인 합성1280×720 동일16장, IO overlap/OpenCV1 고정이다. 각 실행에서3모델이 **각16회 실제 predict**했고16개 관측을 저장했다. 첫 default/fast의 별도 untimed16장×3모델 비교는 input blob 동일, 기본 raw rtol .001/atol .0001에서 최대 절대오차0, post exact였다. 네 실행 PNG byte와 quality·차선·추적 연결 관계도 같았다. 품질이 bad일 때도 모델을 건너뛰지 않는 합성 회귀가 별도로 있으며, 실제 실행의 호출수를 직접 기록·검사했다.

측정한 processing FPS는 PNG/JSON 저장·최종 flush를 포함한다. 모델 load, 각session5회 warmup, raw/post proof(첫 A1 약16.9885s/B1 약8.9036s), teardown은 별도다. fast의 개별 privacy predict median은 약483~495ms로 실제 병목이며, 프로필 개선은 세 모델 병렬화의 전체 이득이다. 겹치는 stage 시간을 더해 pipeline 시간으로 주장하지 않는다.

모든 frame-boundary RSS guard는8GiB 이하였고, 기록된 run 경계 최대RSS는456,740,864바이트였다. 이는 연속 순간 peak 측정이 아니다. raw reference는 RAM만 사용했고 디스크에 tensor를 저장하지 않았다. 합성 생성시각은2026-11-04T22:31:18.424353+00:00, 미검토 PNG/JSON cap은 생성+2h와 원기한 최소인2026-11-05T00:31:18.424353+00:00이다. 기한 연장·실제 보관 등록·삭제는 하지 않았다.

읽기 감사로 source/protocol/execution/model-set·완료attempt 연결, 시간/FPS/median 재계산,4개 runtime snapshot,64개 관측의순서·cap·의미 동일성과64개 저장PNG의현재SHA를 확인했다. 공개 집계에는 모델·코드·protocol·execution·aggregate 해시와 숫자만 보존하고 개별 프레임 hash/path/예측은 공개하지 않는다. 상세 집계는 [FINAL-CPU-PIPELINE-BENCHMARK-RESULTS.json](reports/FINAL-CPU-PIPELINE-BENCHMARK-RESULTS.json)에 있다.

이 결과는 CPU 합성16장 범위의 수치·출력 동일성과 처리속도다. 실제 독립 정확도·물리깊이·카메라/decode/GNSS/API·10분 지속·현장10FPS를 입증하지 않는다. GPU raw 검증 실패를 통과로 바꾸거나 GPU 지원을 승인하는 증거도 아니다. GPU pipeline/VAL32/600초 soak는 해당 strict gate를 통과하기 전 실행 완료로 표시하지 않는다.
