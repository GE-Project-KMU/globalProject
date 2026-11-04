# 현재 cycle2 모델: GPU 수치 검증 차단

현재 cycle2의 포트홀·차선·개인정보 조합은 **GPU raw 수치 검증 실패로 등록하지 않았다**. 기본 `rtol=0.001, atol=0.0001`과 후처리 기준을 유지했다. [현재 공개 집계](reports/FINAL-CYCLE2-GPU-NUMERIC-RESULTS.json)에 실제 plan·모델·가중치·실행 소스와 부모 결과 SHA를 기록했다. 아래의 `final-v6` 통과·7.793FPS 기록은 이전 모델 조합의 이력이다.

| 현재 전체 GPU 진단 | VAL 수 | raw 실패 | 후처리 실패 |
| --- | ---: | ---: | ---: |
| 포트홀 | 20 | 0 | 0 |
| 차선 | 100 | 0 | 0 |
| 개인정보 | 40 | 1 | 0 |

Windows/RTX2080Ti·SM75·TF32off, 앱 CPU ORT1.31.1과 GPU ORT1.20.2/CUDA12.6/cuDNN9.7.1을 비교했다. 실제 CUDA node event 98,120개를 확인했지만 모든 노드가 GPU였다는 뜻은 아니다. 개인정보 최대 raw 절대오차는 0.436584이며, 최대 절대오차 위치 자체는 허용오차 위반 위치가 아니다.

원래 실패한 개발 VAL 사례 하나만 고정하여 추가 진단했다. ORT1.20.2의 `DEFAULT` convolution search는 59개, `HEURISTIC`+graph optimization 비활성화는 58개 원소가 기본 허용오차를 넘었다. 완전히 분리한 ORT-GPU1.31.1/CUDA13 환경의 `HEURISTIC`+기본 graph도 59개 위반으로 실패했다. 세 진단 모두 위반은 rescue P2의 같은 tile width 출력에 모였고, 최종 후처리는 일치했다. 후처리 일치를 raw 통과로 대체하지 않았다. 상위 convolution과 width decode의 오차 기여는 분리 입증하지 못했다.

CUDA13 패키지 runtime은 13.0.96, 실제 Windows `cudaRuntimeGetVersion` 값은 13030, cuDNN API는 91301이었다. Windows CUDA13의 해당 API는 디스플레이 드라이버 runtime을 보고하므로 패키지 버전과 구분한다. 기존 세 환경의 패키지 스냅샷은 그대로이며 Torch/CUDA12 DLL을 새 환경에 혼합하지 않았다.

새 runtime의 수치 비교 전에 발생한 세 DLL 경계 검사 오류는 별도 보존했다. NVIDIA 시스템 드라이버 두 모듈과 설치된 ORT `.pyd`의 경로 분류 오류였으며 **GPU forward 전 실패이므로 수치 실패 횟수에 합산하지 않았다**. 수정한 검사에는 실제 preload 모듈 14개 전체 회귀를 적용했다. 이후 실제 수치 진단은 정상 종료했으나 위 기준에 실패했다. 새 retry worker의 PID/생성시각에 연결된 종료 증거를 보존했고, 최초 시도에는 과거 exact PID receipt가 없다는 한계도 집계에 구분했다.

추가 runtime·설정 탐색은 종료했다. 현재 조합의 stage/native 검증·등록·GPU FPS·600초 지속 검사는 수행하지 않았다. 현재 10FPS 또는 GPU 지원 완료를 주장하지 않는다. 모델·threshold·TEST 선택, 원소스 보관기한은 변경하지 않았으며 사용자 녹화는 사용하지 않았다. 가중치·사진·개별 결과는 로컬에만 둔다. cache 진단 코드는 채택하지 않은 로컬 진단이며 추적 runtime 수정이 아니다.

원래 전체 진단의 재현 명령은 다음과 같다. 원래 고정 소스·환경·가중치·유효한 VAL 자료가 필요하며, 이 명령의 기록은 새 실행이나 profile 승인을 뜻하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/gpu_model_set.py diagnose data/gpu-model-sets/cycle2-final-v7/plan.json --output <새로운-프로젝트-내-출력경로>
```

---

## 이전 모델 조합 검증 이력

# 새 연구 후보 세 모델의 GPU 구현 검증

새 포트홀·차선·개인정보 모델을 각 CPU canonical 통과 후 별도로 고정하여 GPU를 검사했다. 이전 세 모델의160건 통과나6.015FPS를 새 가중치에 소급하지 않았다. [공개 집계](reports/FINAL-MODEL-GPU-VALIDATION.json)에 정확한 부모 manifest·weights·실행 source·plan·profile·proof SHA를 기록했다. 개별 이미지 ID·SHA·경로·좌표는 공개하지 않는다.

이 조합은 사전에 고정한 연구 후보의 수치 호환성·성능 실험이며 모델 승격이나 운영 모델 선정이 아니다. 개인정보 후보의 정확도 평가와 기존 baseline 유지 권고는 별도의 판단이다. TEST 결과를 본 뒤 이 실험의 후보 조합·가중치·threshold를 바꾸지 않았다.

## 고정과 검사

포트홀은 기존 공개 RDD10장과 새 공개 MWPD10장의 양성 VAL 및 canonical3개를 포함하도록 GPU 실행 전에 지정했다. 차선은 VAL100장 전체, 개인정보는 양성 face20장과 plate20장을 사용했다. 이미 개발에 사용한 VAL이므로 독립 최종 정확도 시험으로 취급하지 않는다. TEST·사용자 녹화 접근은0이었다.

| 구분 | 포트홀 | 차선 | 개인정보 |
| --- | ---: | ---: | ---: |
| CPU 양성 canonical | 3 | 3 | 2 |
| GPU 진단 / 실제 앱 사례 | 20 / 20 | 100 / 100 | 40 / 40 |
| 두 단계 각각 raw 실패 / post 실패 | 0 / 0 | 0 / 0 | 0 / 0 |
| 두 단계 각각 최대 raw 절대오차 | 0.003173828125 | 0.00115966796875 | 0.01654052734375 |
| 두 단계 각각 CUDA node event | 5,380 | 20,500 | 65,680 |

raw는 기본 `rtol=0.001, atol=0.0001`을 유지하여 각 원소의 절대·상대 기준을 함께 검사했다. 위 최대 절대오차만으로 허용오차를 변경하지 않았다. 최종 bbox/class/count/confidence 또는 mask도 기존 엄격한 후처리 기준을 통과했다. 두 단계는 동일 blob·동일 후처리를 사용한다. node event 합계는 각91,560이며 host 전처리·shape 연산 등 모든 작업이 CUDA였다는 주장은 하지 않는다.

Windows/AMD64, RTX2080Ti/SM75, Python3.11.9 worker, ORT-GPU1.20.2, 프로젝트 CUDA12.6/cuDNN9.7.1 DLL, TF32off에서 실행했다. 앱 CPU 환경·기본 profile은 유지한다. 별도 shared-memory worker와 새 명시적 registry만 사용하며, 다른 가중치·환경은 재검증해야 한다. 소스 기준은 `e201a8e`이며 실제 smoke 선택 경로를 고친 테스트 후속 commit은 `1f1b615`다. 실제 실행 bytes의 SHA가 집계에 별도로 있다.

등록된 로컬 profile은 `data/gpu-runtime-profiles/final-v6.json`이다. 등록 후 명시적 opt-in 실제 GPU smoke도1개 통과했다(23.48초). 일반 CPU CI의 skip을 GPU 성공으로 바꿔 표시하지 않는다.

```powershell
$env:ROAD_RUN_VERIFIED_CUDA='1'
$env:ROAD_VERIFIED_CUDA_PROFILE='data/gpu-runtime-profiles/final-v6.json'
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_worker_protocol.py::test_actual_opt_in_cuda_profile_smoke -q
```

## 같은 후보 조합의 처리율

개인정보 fresh/regression CPU 평가와 전체 pytest가 끝난 뒤 다른 모델 실행이 없는 상태에서 같은 frozen profile을 ABBA 순서로 한 번 측정했다. 모델별5회 워밍 후 비개인 합성1280×720 영상16프레임을 각 구간에서 처리했다. 같은 조합·같은 입력이며 모델·threshold·사례를 재선택하지 않았다.

| 구간 | warmed 처리율 |
| --- | ---: |
| CPU default A / A | 1.10860 / 1.15680FPS |
| GPU fast B / B | 7.83582 / 7.75029FPS |
| 두 구간 median | CPU1.13270 / GPU7.79306FPS |

같은 실행의 GPU/CPU median 비율은6.880배이며, **GPU7.793FPS도10FPS 목표에 미달**한다. 이전 모델셋의6.015FPS와는 모델·소스가 다르므로 속도 개선율을 계산하지 않았다. 마스킹 PNG 바이트는 네 구간 모두 같았다.

전처리·shared-memory forward·postprocess·품질 검사·추적·마스킹·PNG/JSON 저장을 포함했다. GPU load12.807/12.636초,5회 warmup0.929/0.940초, teardown1.157/1.179초는 따로 측정해 warmed FPS에서 제외했다. 이 세 항목까지 포함한 benchmark16장 batch 처리율은0.945/0.951FPS다. 별도 워밍5장을 수행하는 benchmark의 값이며 일반 작업의 cold FPS로 확대하지 않는다. forward/transport·전처리·후처리·저장 단계별 시간은 공개 집계에 있다.

capture·실제 video decode·GNSS·보정·API·10분 지속 처리는 포함하지 않았다. 이 짧은 합성 시험은 전체 장비 처리량·현장 인수·지속10FPS를 입증하지 않는다. 실제 native 검증·등록·smoke·ABBA 종료 후 모든 worker를 정리했다.

이 결과는 연구용 구현 경로 검증이다. 목표 정확도·개인정보 recall·현장·물리 깊이·GNSS·10분 지속 처리·운영 배포 인수를 충족했다는 뜻이 아니다. 원소스 만료·철회·연구 전용 제한을 유지하며 가중치·threshold·사례·기한을 검사 후 변경하지 않았다. [명시적 새 모델셋 절차](GPU-MODEL-SET-VALIDATION.md)와 [이전 모델의 성능 기록](INFERENCE-PERFORMANCE.md)을 구분해 읽는다.
