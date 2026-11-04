# 포트홀 Tiny 추가 개선 실험

기존 Nano의 RDD/MWPD VAL 평균 AP50과 작은 모델의 표현력 한계를 개선하기 위해 공식 YOLOX-Tiny를 별도 후보로 학습한다. [공식 YOLOX 저장소](https://github.com/Megvii-BaseDetection/YOLOX)의 COCO 초기 가중치와 고정한 로컬 SHA를 사용하며 COCO 모델을 포트홀 검출기로 대신하지 않는다. upstream 코드 라이선스는 Apache 2.0이고 실제 학습 데이터의 CC BY 4.0·CC BY-NC 3.0 및 연구 전용 제한은 별도로 상속한다.

기존 데이터의 TRAIN 7,540장·VAL 652장만 새 개발 release로 만들었다. 과거 TEST는 넣지 않았다. 새 외부 RDD2022 평가 자료는 별도 사전 프로토콜로 준비하며 학습에 뒤늦게 합류하지 않는다. 기존 MWPD 97장과 일반 노면 48장은 회귀용으로만 사용한다. 이미 알려진 결과에 대한 모델 선택을 새 TEST 성능으로 표현하지 않는다.

후보는 640 입력, COCO backbone/neck/head 공통 가중치 456개 텐서를 이식하고 80클래스 classifier 6개만 다시 초기화한다. batch16, trainable BN, SGD/Nesterov, 1epoch warmup과 cosine 학습률, native EMA, mosaic·HSV·반전을 사용한다. mosaic 바깥 표본은 RDD/MWPD 기대 비율을 같게 복원 추출하고 내부 donor는 적격 TRAIN에서 균등하게 뽑는다. 원본 GT 30개 이하·4장 mosaic 상한120·padding120을 확인한다. native 증강의 clipping/subpixel 필터링은 적용하지만 원본 라벨을 padding 한계 때문에 자르지 않는다.

최대36epoch·최소18epoch·patience10이며 이전 baseline도 epoch0 선택 후보로 유지한다. 두 도메인의 VAL AP50 평균이 높고 기존 RDD VAL 하락이 baseline 대비3%p 이내인 checkpoint를 선택한다. 임계값0.1과 AP confidence floor0.01은 고정한다. 학습·선택·source SHA를 사전 고정한 [계획](reports/POTHOLE-TINY-V7-PLAN.json)이 기준이다. 목표 달성은 실제 평가가 끝나야 판단할 수 있다.

실제 공개 TRAIN 두 도메인의 CPU 계약에서 모든 GT 보존, 유한 loss/backward와 decoded `(1,8400,6)` 출력을 확인했다. optimizer update0·TEST0·GPU 초기화0이며 이는 실제 GPU 학습이나 최종 정확도 증거가 아니다. [계약 결과](reports/POTHOLE-TINY-V7-CONTRACT.json)를 참고한다.

```powershell
.\.venv\Scripts\python.exe scripts/acquire_yolox_tiny.py
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_tiny.py prepare
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_tiny.py preflight
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_pothole_tiny.py train
```

완료·진행 중 실험을 덮어쓰지 않는다. 새 실험에는 별도 폴더와 새 사전 계획을 사용한다. 원자료 기한은 그대로 상속하고 사용자 주행 영상은 읽지 않는다. 원자료·초기/학습 가중치는 로컬에만 보관한다.
