# 포트홀 용량 후보 V9 사전 계획

Tiny V7은 기존 도메인 VAL AP50 하락이 사전 제한 3%p를 넘어서 채택하지 않았다. 최종 TEST 예측을 수행하지 않은 상태에서 공식 COCO 초기화 YOLOX-S를 독립 후보로 추가한다. 공식 모델 설명과 초기화 출처는 https://github.com/Megvii-BaseDetection/YOLOX 이다. 이 자료의 COCO 점수를 프로젝트 포트홀 정확도로 사용하지 않는다.

TRAIN 7,540장과 VAL 652장만 사용한다. 기존 Nano V6와 위치 보정 Nano V8 및 이 후보의 사전 규칙으로 선택한 VAL 평균을 비교한다. 기존 도메인은 V6 대비 3%p 이내 하락만 허용하고 동률은 기존/먼저 선택한 모델을 유지한다. Tiny 탈락 기준을 변경하지 않는다. 최종 가족 후보와 임계값을 확정한 후에만 US100과 기존 회귀 TEST를 평가한다. TEST로 학습, 중단, 후보 선택 또는 임계값을 조정하지 않는다.

640 입력, 실제 8장씩 두 번 누적하는 SGD, 본체/헤드 학습률 분리, 초기 2epoch 본체 고정, BN 통계 고정, 6epoch 이후 native L1을 사용한다. 2개 마이크로배치의 foreground 정규화 손실 평균이며 16장 전체 손실과 정확히 같다는 주장은 하지 않는다. 각 epoch의 TRAIN 추출은 4,096회이며 도메인별 수량을 고정하고 hardness 가중치는 쓰지 않는다. 최대 28/minimum 20epoch와 patience 8을 사전에 고정한다. 자세한 수치는 `reports/POTHOLE-SMALL-V9-PLAN.json`에 있다.

CPU 사전 확인에서 두 도메인의 최대 객체 수 샘플을 사용하여 정답 보존, 유한한 손실과 역전파, 1×8400×6 출력을 확인했다. 실제 optimizer 갱신과 GPU 사용 및 TEST 사용은 모두 0이다. 공식 초기화 456개 텐서를 전이하고 클래스 예측기 6개 텐서만 재초기화했다. 이 결과는 학습 후 정확도 증거가 아니다.

재현 순서는 `scripts/acquire_yolox_small.py`, `scripts/setup_training.py run scripts/train_pothole_small.py prepare`, `preflight`, `train`이다. 원본 및 가중치는 프로젝트 내부에만 둔다. 실행 중 학습 소스와 의존 파일을 수정하지 않는다. 채택 기준을 통과하지 못한 학습도 진단 체크포인트와 집계 결과를 남기되 실행 모델로 승격하지 않는다.
