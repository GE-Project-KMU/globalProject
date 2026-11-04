# Rescue V8 외부 감독

이 도구는 기존 학습 source7bc192…/공개·비공개 plan6bbd591…/CPU contractf078368…와
factory61876d…/자료 helperff8315…를 바꾸지 않는다. 부모 파일과 원 실패 기록을
덮어쓰지 않고 별도 preflight/train receipt·log를 만든다.

두 단계 모두 Small retry1과 lane retry1의 plan에 결합된 completed/exit0/
sources_unchanged/cleanupFalse receipt와 기존 PID+생성 시각의 부재를 먼저
확인한다. 이후 장치 observer의 최근2개 sample이2GiB 미만이어야 한다. 조건
미충족 시 학습 Popen0이며,9GiB 초과·측정 N/A/stale·원기한 도달·자기 OOM
서명이면 자기 자식만 종료한다. 학습기는 CUDA를 직접 사용하며 GPU를 사용할 수
없으면 실패한다. 일반적인 CPU fallback 로그 탐지 기능을 구현했다는 주장은 하지 않는다. prior 작업은 조회만 하고 제어하지
않는다. Device 수치는 WDDM의 sampled FB accounting이며 순간 hard quota나
전역 OOM0을 뜻하지 않는다.

preflight는 별도 새 프로세스에서8개 TRAIN tile batch의 임시 optimizer step을
시도하고8개 VAL의 시간만 측정한다. 학습 실험의 가중치로 이어 쓰지 않는다.
원 trainer는 실제 skipped AMP update 수를 별도로 노출하지 않으므로 외부 감독도
8회를 실제 성공 update라고 덧붙이지 않고8개 step 시도로 기록한다. 모델·optimizer는
프로세스 종료와 함께 폐기된다. 실제 train은 다시 construct하는 별도 자식이다.

train은 완료 preflight의 자식 종료·동일 plan/source·정확한 resource report SHA·
legacy 보존·TEST0·임시 실험 상태를 검사한 뒤에만 시작한다. 실제 실행은
`setup_training.environment()`와 workspace Python3.11을 직접 사용해 기존
`setup_training.py run ...`과 같은 환경을 적용한다. 중간 launcher를 감독하는
대신 GPU를 쓰는 직접 자식의 PID/생성 시각/PPID를 소유하고 종료하기 위한 방식이다.
기존 trainer CLI action과 `--gpu-authorized`는 그대로 전달한다.

phase별 신규 경로는 `data/training/redaction-rescue-v8/external-*-supervisor.json`,
`external-*-device-samples.json`, `.cache/cycle2-rescue-v8-*.log`다. 기존 파일이 있으면
자동 재시도나 resume하지 않는다. training 완료는 원 trainer의 completed12epoch와
실제 update>0/가중치 SHA를 확인하되, 이를95% 비식별화 또는 실제 현장 개인정보
합격으로 해석하지 않는다. 공개 집계에는 자식 PID/생성 시각을 제외한다.

CPU 정책 계약은 선행 실패/미완료/자식 생존, plan 불일치, GPU 비유휴, 미완료
preflight, 잘못된8batch report, 중복 로그/receipt에서 학습이 시작되지 않는지
검사한다. GPU 작업은 root의 단계별 별도 허가 전에는 실행하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_rescue_v8_supervision.py
.\.venv\Scripts\python.exe -m pytest tests/test_rescue_supervision.py -q
# 선행조건 충족 및 root preflight slot 승인 후에만
.\.venv\Scripts\python.exe scripts/supervise_redaction_rescue_v8.py preflight
# 자식 종료/결과 binding 검토와 root train slot 승인 후에만
.\.venv\Scripts\python.exe scripts/supervise_redaction_rescue_v8.py train
```
