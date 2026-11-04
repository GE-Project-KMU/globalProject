# 공유 GPU 메모리 검증 범위

첫 optimizer0 검사는3개 배치의 유한 loss/output, 자체 최대 예약1.746GiB,
자기 프로세스 OOM counter0, 원자료 체크섬 보존을 확인했다. 원 실행 소스0726d268…와
원 결과는 바꾸지 않는다. 다만 `cudaMemGetInfo` 기반 값만으로 장치 전체9GiB
예산을 통과했다고 해석하지 않는다. 사후 별도 `nvidia-smi`값과도 시간 범위가
다르므로 두 수치의 차이 자체를 동시 사용량 차이로 단정하지 않는다.

NVIDIA는 WDDM에서 GPU 메모리를 Windows가 관리하고 CUDA가 OS에서 받은 정보를
보고한다고 [설명한다](https://forums.developer.nvidia.com/t/cudamemgetinfo-no-longer-returns-free-memory-for-full-system-only-looks-at-current-process/66941).
[NVML 문서](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)는
장치 메모리 정보와 WDDM 관리 특성을 구분하며, 사용량이 모든 활성 channel의 합과
같다는 설명은 Linux/Windows TCC에 한정한다. 기존 결과의 pass flag는 실행 이력으로
보존하고 전역 물리 메모리 합격으로 승격하지 않는다. 범위 정정은
`PRIVACY-P2-CYCLE2-GPU-PREFLIGHT-SCOPE.json`에 남긴다.

probe2는 동일 TRAIN AMP 역전파2배치+VAL FP32 1배치(optimizer0)를 쓰고,
`nvidia-smi` 단일 장치 observer를 추가한다. 원 source와 TRAIN view/GT 및
FP32 VAL forward의 AST가 같다는 별도 source audit를 남겼다. 처음 CUDA 초기화
전에2개 장치 sample을 받고, 종료 뒤 모델이 아직 resident인 상태에서도2개를 더
받는다. 요청 loop100ms와 실제 monotonic 수신 시각 간격을 별도로 기록한다.
드라이버 query 시각 자체나 sample 사이 순간 초과가 없다는 보장은 하지 않는다.
장치가 보고한9GiB 초과·측정 실패·stale sample·자기 OOM이면 probe2만 실패시킨다.

실제 동시 학습에는 별도 supervisor를 쓴다. 고정 workspace Python3.11/P2 명령만
실행하며 새 직접 자식의 PID·생성 시각·부모 PID를 확인한다. 예산 위반/OOM이면
그 자식만 종료한다. 최초 identity 확보 자체가 실패하면 Windows Popen의
자기 프로세스 handle로 새 자식을 정리하며, 임의 PID/name 검색으로 종료하지
않는다. Tiny나 다른 프로세스를 제어하지 않는다. 자기 log의 native OOM fallback도
거부하되 전역 OOM0은 주장하지 않는다. 학습 source92fe1649…/V2 plan31503ca3…의
수학과 조건은 변경하지 않는다. 실패/중단 receipt와 마지막 checkpoint를 보존하고
자동 재학습·이어학습·TEST 재시도를 하지 않는다.

CPU 합성 장치 query와 실제 CPU dummy child로 생성 시각 불일치 시 종료 거부,
과예산·N/A 측정 fail closed, observer 자식 정리, OOM fallback 탐지를 검사했다.
기존 계약과 함께67개가 통과했다. 이 검사는 실제 GPU 처리 속도나 목표 정확도
증거가 아니다. 프로토콜/source를 root가 commit한 뒤에만 probe2를 실행하고,
실제 학습은 그 결과를 확인한 별도 승인을 기다린다.
