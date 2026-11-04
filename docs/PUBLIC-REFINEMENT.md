# 공개 자료 후속 미세조정

사용자 녹화 영상과 그 프레임은 이 실험의 학습·검증·시험에 사용하지 않는다.
모든 원본·가중치·개별 예측은 프로젝트 내부에서만 처리하고 Git/외부 Release로
배포하지 않는다. 기존 모델·시험 기록·보관 기한은 보존한다.

## 데이터와 사전 고정

`public_refinement_data.py`는 A2D2 기존 train799/VAL100과 WIDER FACE/CCPD 기존
train4993/VAL508을 재사용한다. 이 VAL은 개발 자료이며 독립 성능 증거가 아니다.
과거 모든 TEST의 source ID/SHA가 fitting 자료에 들어오면 거부한다.

학습 전에 A2D2의 미사용 recording2개에서 정렬된 frame index를 균등 선택한
새80장, WIDER의 기존 heldout event partition에서 새100장과 CCPD의 미사용
heldout 번호판 identity100장을 별도로 동결했다. exact SHA 및 연결된 유사 영상
군(A2D2 DCT pHash≤6, 개인정보 original-source dHash≤3)을 모든 과거 split과
대조했다. 새 자료의 기존 자료 연결 중복은 없었다. WIDER의 잘못된 게시자 얼굴
GT가 있는 원본은 전체 제외했고 모든 유효 얼굴 bbox를 유지한다. 일부만
주석된 알려진 클래스에 한해 평가하며 미주석 타 클래스를 음성으로 세지 않는다.

새 A2D2 recording은 과거 자료와 같은 날짜이고 WIDER event partition도 과거
holdout과 공유한다. 따라서 새로 보지 않은 이미지/recording 평가이며 날짜·
실제 도로·인물의 독립성을 증명하는 시험이 아니다. old TEST는 회귀로만 보고한다.
도막은 Solid/Dashed semantic pixel proxy이며 마모율 O/G/V의 독립 GT가 아니다.

A2D2 CC BY-ND4.0, WIDER CC BY-NC-ND4.0, CCPD MIT의 기존 공식 근거와
비공개 연구 범위를 유지한다. [WIDER 공식 사용조건](https://shuoyang1213.me/WIDERFACE/index.html),
[A2D2 공식 공개 mirror 안내](https://registry.opendata.aws/aev-a2d2/).
기존 날짜를 새 다운로드 날짜로 갱신하지 않고 새 시험 자료에도 기존 최소
만료를 보수적으로 상속했다. 도막은2026-11-10 17:07:26UTC, 개인정보는 같은 날
16:55:19UTC까지이며 이후 사용·재검증 정책이 적용된다. ND/NC 연구 결과를
B2G 상업 운영이나 파생 가중치 배포 승인으로 해석하지 않는다.

## 도막 후보

단일 후보는 기존 DeepLabV3-MobileNetV3의 coarse 결과에 stride4 detail residual을
더한다. 기존 backbone의24채널 low feature와256채널 ASPP context를16/64채널로
투영하고 depthwise+pointwise decoder로 보정한다. 추가 파라미터는20,114개다.
마지막 residual conv를0으로 초기화해 시작 출력은 기존 checkpoint와 정확히 같다.
공식 DeepLabv3+ 가중치가 아니라, 경계 복원에 low-level feature를 활용하는
[원 논문](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html)
아이디어를 참고한 별도 연구 구조다.

계획은 Git `d83e020`에서 학습 전에 고정했다. 최대20epoch/min6/patience6,
VAL IoU 개선.001 기준, zero-step baseline도 checkpoint 후보에 포함한다.
patch640·추론1024·BN 통계 고정, AdamW backbone1e-5/coarse1e-4/detail1e-3,
EMA.99와 .45 weightedCE+.45 Dice+.1 boundaryDice를 사용한다. ignore255와
경계의 유효하지 않은 이웃은 손실에서 제외한다. checkpoint와 threshold는
기존 VAL100의 원본 좌표 IoU로만 선택한다. source/recipe SHA는 계획과
training-config에 기록하고 실행 중 소스를 바꾸지 않는다.

새 ONNX도 Windows AMD64 ORT1.31.1 CPU의 기본 rtol.001/atol.0001 raw 비교와
정확한 mask 일치를 별도로 검사한다. 고정 양성 VAL3 및 전체 VAL100에서 앱,
격리 ORT1.31.1, 원 FP32 Torch checkpoint를 비교한 뒤 fresh80을 평가한다.
기존 구ORT1.20 실패를 소급 통과로 바꾸거나 새 모델의 GPU 지원을 이전 모델
160프레임 증거로 대신하지 않는다. 원 모델·허용오차·threshold를 바꿔 실패를
숨기지 않는다.

## 개인정보 후보

한 후보는 기존 두 Nano specialist 중 얼굴만8epoch 미세조정하고 번호판
파라미터와 BN buffer를 byte digest로 동결한다. plate는 이미 기존 연구 시험의
Recall95% 목표를 넘었고, 얼굴은 기존 새 시험 Recall86.49%로 미달이었다.
입력1920의 global960 및4개960 tile/plate640 추론 구조는 유지한다.

학습 view는 실제 global branch와 동일한1920→960 bilinear50%, 실제960
quadrant50%다. tile은80% GT가 있는 quadrant/20% 균등 선택하며 모든 intersecting
GT를 clip해 보존한다. 512개를 넘으면 거부하고 조용히 자르지 않는다.
SGD1e-4·warmup/cosine·EMA.99, HSV/반전을 사용한다. 기존 faceVAL400의 실제
global+4tile AP50로 epoch0 포함 checkpoint를 선택한 후 전체 VAL508에서 미리
고정한 confidence grid를 평가한다. 최소 클래스 Recall95%가 안 되면 그 실패를
그대로 기록한다. 새200장과 과거400장은 선택 후 각각 fresh/regression으로만
실행한다.

원 개인정보 계획은 Git `d8905cf`와 원 SHA로 보존한다. 학습 시작 전
`test_evaluated:false`의 지속 상태 오해를 없애기 위해
`exported_before_final_test:true`로 바꾸는 metadata-only revision을 따로 고정했다.
부모 계획 SHA와 변경 이유를 새 계획에 남겼고 학습/선택 수학·epoch·데이터는
바꾸지 않았다.

## 재현과 결과

```powershell
.\.venv\Scripts\python.exe scripts/public_refinement_data.py lane
.\.venv\Scripts\python.exe scripts/public_refinement_data.py privacy
.\.venv\Scripts\python.exe scripts/refine_lane_train.py freeze-plan
.\.venv\Scripts\python.exe scripts/refine_privacy_train.py freeze-plan
.\.venv\Scripts\python.exe scripts/prepare_refinement_evaluation.py
# GPU slot 승인 후에만 하나씩 실행한다.
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/refine_lane_train.py train
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/refine_privacy_train.py train
```

새 출력 폴더가 필요하고 기존 학습 결과를 덮어쓰지 않는다. 공개 사전 계획은
`LANE/PRIVACY-REFINEMENT-PLAN.json`과각 FRESH/REGRESSION 계획, numeric gate 계획이다.
개별 양성 사례는 비공개 metadata SHA로 고정한다. 실제 학습·수치 검증·최종
비교가 끝난 뒤 `LANE/PRIVACY-REFINEMENT-RESULTS.json`에 집계를 기록한다.
이 문서의 계획과 합성 계약 통과 자체는 목표 정확도나 속도 달성 증거가 아니다.
