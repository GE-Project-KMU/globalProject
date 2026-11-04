# 공유 GPU 중단과 차선 retry1

동시 실행 중 장치가 보고한 최대 메모리는11,081,351,168bytes(약10.32GiB)로
설정9GiB를 넘었다. P2와 차선 감독은 자기 자식의 PID/생성 시각을 확인한 뒤
자기 프로세스만 종료했고, 두 자식이 남지 않은 상태를 기록했다. Root의 Small
작업도 같은 이유로 중단됐다는 별도 보고가 있다. P2/차선 자기 log의 OOM 서명은
관측되지 않았다. 이것을 전역 OOM0이라고 확대하지 않는다.

장치 수치는 WDDM의 sampled FB accounting이며 정확한 프로세스 합이나 순간
하드 쿼터가 아니다. 소수 배치 검사와 각 작업의 자체 예약 메모리 합만으로
동시 실행 안전성을 판단하기에는 부족했다. 실제로 차선 학습은
`cudnn.benchmark=True`, 사전 검사는 기본False였던 실행 환경 차이가 있었다.
이 차이를 검사에 반영하지 못했다. [PyTorch 2.7 문서](https://docs.pytorch.org/docs/2.7/notes/randomness.html#cuda-convolution-benchmarking)는
새 convolution 크기에 여러 알고리즘을 시험할 수 있다고 설명한다. 초기 차선
forward의 자동 탐색/workspace가 피크에 기여했을 가능성이 있지만, 인과관계를
분리 실험으로 확정한 것은 아니다.

실패한 원 소스401fab…/92fe…와 계획·로그·receipt·체크포인트는 보존한다.
`CYCLE2-SHARED-GPU-ABORT.json`은 원 자료를 고치지 않는 집계다. P2는 계획12epoch
중9epoch를 완료하고10epoch 도중 중단됐다. 완료 epoch의 실제 optimizer update는
6,731회, 마지막 로그의 하한은7,132회이며 정확한 최종 합계를 추정해 적지 않는다.
완료된 개발 VAL 최고 AP50은70.33%, 큰 얼굴 recall은93.71%로 기존 기준 gate에
미달했다. 최종 ONNX/model manifest는 만들지 않았고 이 후보를 승격하지 않는다.
차선은 epoch0 VAL도 끝나기 전에 중단됐으며 학습 update0이다. 두 작업 모두
blind TEST 예측0이다. 계획12epoch를 마친 것으로 표현하지 않는다.

이후 GPU 학습은 한 작업씩 실행한다. CPU 자료 검증·설계·수치 비교는 병렬로
할 수 있다. 차선 retry1은 별도 source/output/parent plan으로 등록하며 기존
중단 실행을 이어서 돌리지 않는다. optimizer state를 복원하지 않고 원래 v4
warmstart에서 full restart한다. 새 출력은 `data/training/lane-cycle2-v5-retry1`이다.

새 retry는 `cudnn.benchmark=False`를 명시한다. 데이터·손실·optimizer·schedule·
augmentation·checkpoint/threshold 선택 규칙은 같고, 같은 bitwise 학습 결과를
보장한다고 주장하지 않는다. loss/context sampling/VAL metric/export의 AST가
원본과 같은지 CPU 계약에서 검사한다. 실제 수량은 TRAIN1,397/VAL179(기존100+
새79)/새 TEST119이며, source/manifest SHA는 원 계획과 같다. 원 계획의 QC 이전
80/120 표기는 원 파일에 남기고 새 retry 설명을 실제79/119로 바로잡았다.
기한은 원2026-11-10T17:07:26.069653Z를 그대로 상속한다.

Root가 새 source/plan을 검토·commit한 뒤 GPU 단일 작업 조건으로만 시작한다.
완료 후 고정 양성 VAL3+전체 VAL179의 CPU1.31.1 raw 기본 rtol.001/atol.0001 및
최종 mask 정확 일치를 먼저 확인한다. 새 TEST/회귀는 Root의 최종 binding 확인과
별도 실행 승인 전에는 돌리지 않는다. 중단 결과나 CPU loss smoke를 독립 정확도,
75% IoU,95% 비식별화 또는10FPS 합격으로 표기하지 않는다.


재시작 감독 V2는 exclusive 요구를 보고 문구로만 남기지 않는다. Small retry1의
고정 plan에 결합된 completed/exit0/sources_unchanged/cleanupFalse receipt와
기존 PID+생성 시각의 자식 부재를 먼저 확인한다. 이후 장치 observer의 최근2개
sample이 모두2GiB 미만이어야 하며, 자식을 만들기 직전 완료 receipt SHA도 다시
확인한다. 조건이 충족되지 않으면 학습 Popen은0이다. V1 준비 protocol과 source
snapshot을 보존하고 V2에 부모 SHA와 변경 이유를 기록했다. 조건부 자동 시작의
별도 허가 전에는 실행하지 않는다. CPU 정책/실패/누출/numeric 관련98개가
통과했으며, 실제1024 TRAIN loss/backward도 유한했다. 이 CPU 검증은 GPU 메모리나
독립 정확도 합격 증거가 아니다.
