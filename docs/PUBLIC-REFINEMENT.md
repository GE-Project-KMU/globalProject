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

### 도막 완료 결과

사전 patience 조건으로12epoch에서 종료했고 실제 optimizer update는2,388회다.
선택은epoch6/threshold.2이며 기존 VAL IoU73.096%는 개발 수치다. 고정 양성
VAL3와 전체 VAL100의 앱/격리 ORT1.31.1/원 FP32 Torch raw 기본 허용오차 및
최종 마스크 정확 일치 검증을 통과한 뒤 최종 비교를 한 번씩 수행했다.

|자료|기존 IoU|후보 IoU|차이|후보 precision / recall|
|---|---:|---:|---:|---:|
|새 recording80장|49.28%|56.98%|+7.70pp|72.55% / 72.65%|
|이전에 본 TEST100장 회귀|48.75%|56.38%|+7.63pp|71.01% / 73.24%|

두 집합 모두 목표75%에 미달한다. 새 평가를 본 뒤 추가 checkpoint/threshold
선택이나 학습은 하지 않았다. 원본 출처 날짜·물리적 도로 독립성과 마모율 MAE는
검증하지 않았다. 개선은 구조·손실·미세조정·VAL threshold를 함께 바꾼 후보의
효과이며 개별 원인에 대한 ablation이 아니다.

새80장의 실제 CPU predict 중앙값은416.6→423.3ms, P95는463.3→444.2ms였다.
old TEST 회귀의 중앙값은453.3→466.2ms다. 입력 decode·전체 파이프라인을
제외한 공유 호스트의 관측 비용이므로 CPU 속도 개선이나 실차 FPS 합격을
주장하지 않는다. 새 artifact의 GPU 정합성은 이 CPU 증거로 대신하지 않는다.
학습·수치·최종 metric의 연결은 `LANE-REFINEMENT-COMPLETION.json`에 기록한다.

### 개인정보 평가 실행 병렬화

개인정보 모델의 최종 예측이0건인 시점에 별도
`PRIVACY-REFINEMENT-PARALLEL-PROTOCOL.json`을 고정했다. 이미 완료한 도막 결과와
실행 소스는 바꾸지 않는다. `compare_privacy_parallel.py`는 동일한 fresh200장과
회귀400장을 사용하되 baseline/candidate를 각각 CPU2thread 프로세스에서 실행한다.
모델별 sample_id 정렬과 각 이미지의 예측 순서를 보존하고 기존 `OnnxModel.predict`,
confidence floor.01/NMS, 기존 `score` 함수를 그대로 쓴다. 모형·threshold·GT·
선택 조건을 바꾸는 실험이 아니다. 실행 계획·모델·benchmark 해시를 전후 검사하고
실패/미완료 산출물을 자동 덮어쓰지 않는다.

독립 합성 계약8개에서 기존 직렬 scoring과2개 subprocess JSON 왕복의
AP/TP/FP/FN·동점 순서가 정확히 같았다. IoU.5 경계 float 왕복과3개 해상도의
resize thread1/2 픽셀 일치도 확인했다. 이는 실제 모델 정확도나 속도 개선 증거가
아니다. 실제 모델의 수치 검증은 별도로 먼저 통과해야 한다.

병렬 실행의 각 모델 median/P95에는 다른 CPU 프로세스와 공유 호스트 부하가
포함된다. 직렬 모델의 독립 속도와 직접 비교해 개선률을 주장하지 않는다.
`parallel_execution_wall_seconds`는 worker 시작~추론/worker JSON 완료 구간이다.
후속 공통 score/merge·최종 보고서 저장은 제외하므로 전체 CLI 시간은 외부
`Measure-Command`로 별도 기록한다. 픽셀 파일을 만들거나 GPU를 추가 사용하지 않는다.
