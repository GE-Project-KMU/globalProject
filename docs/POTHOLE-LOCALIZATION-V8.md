# 위치 정밀도와 학습 난도 기반 포트홀 개선 후보

기존 v6 모델의 개발 VAL652장만 분석했다. 운영 confidence0.1의 누락259개 중 입력640 기준32픽셀 미만은34개였고, 예측과 GT가 어느 정도 겹치지만 IoU0.5에 못 미친 누락도120개였다. 이에 입력 해상도를 올리는 대신 기존640 Nano를 warmstart하고 native L1 위치 손실과 TRAIN 난도 표집을 사용하는 별도 후보를 고정한다. [크기·위치 진단](reports/POTHOLE-V6-VALIDATION-DIAGNOSTIC.md)은 새로운 TEST 결과가 아니다.

학습 난도는 원래 TRAIN7540장에만 v6 CPU 예측을 수행해 만들었다. VAL·TEST·새 미국100장은 가중치 계산에 사용하지 않았다. 게시자 GT 대비 FP/FN으로 가중치를1~3에 제한했으며 독립 재검수에서 모든7540개의 membership·TP/FP/FN·계산식을 확인했다. [난도 결과와 제약](reports/POTHOLE-TRAIN-HARDNESS-V8.md)을 참고한다. 이 가중치는 라벨의 물리적 진실성이나 사람의 개인정보 검수를 새로 보증하지 않는다.

매 epoch 복원 추출4096장은 RDD양성1024·RDD음성1024·MWPD양성2048로 정확히 나눈다. 각 집단 내부에서만 난도 가중치를 적용하고 별도의 고정 RNG로 섞는다. batch16에서 마지막 표본을 버리지 않는다. 적격 TRAIN7540장을 매 epoch 전부 사용했다는 뜻은 아니다.

입력·후처리·confidence0.1은 그대로다. BN running statistics는 고정하고 SGD/Nesterov·학습률0.0001·1epoch warmup/cosine·EMA를 쓴다. fullframe640·HSV·반전으로 학습하며 mosaic/mixup은 사용하지 않는다. epoch3부터 [공식 YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)의 native L1 localization loss를 더한다. 최대18epoch·최소8epoch·patience6이며 초기 v6 checkpoint도 선택 후보로 보존한다. 두 VAL 도메인 AP50 평균을 최대화하되 기존 RDD VAL 하락이3%p를 넘는 checkpoint는 선택하지 않는다.

실제 두 TRAIN 도메인의 최대 GT 사례를 포함한 CPU batch에서 GT 보존·L1 loss0.8942·유한 backward와 정확한 표집 quota를 확인했다. 이 계약은 optimizer0·GPU초기화0·TEST0이며 학습 정확도 결과가 아니다. [계약](reports/POTHOLE-LOCALIZATION-V8-CONTRACT.json), [사전 계획](reports/POTHOLE-LOCALIZATION-V8-PLAN.json)이 실행 기준이다.

포트홀 후보 가족의 최종 선택은 모든 후보의 개발 VAL 결과로만 결정한 뒤 새 미국100장과 과거 회귀 집합을 평가한다. 새 평가를 보고 후보·임계값을 다시 고르지 않는다. 가중치와 원자료는 로컬에만 두고 원래2026-11-10의 더 이른 자료 기한을 그대로 상속한다. 이 계획 자체가 AP50 80%나10FPS를 달성했다는 주장은 아니다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_localization.py prepare
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_localization.py preflight
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_localization.py train
```

진행·완료된 출력은 덮어쓰지 않는다. 소스·자료·난도 파일·초기 checkpoint·계획 SHA가 달라지면 학습을 거부한다. 실패와 초기 모델 유지도 결과로 기록한다.
