# 공개 자료 추가 개선 cycle2

이 cycle은 개발용 VAL 오류를 분석한 뒤 차선1개·비식별화1개 후보를 고정한다.
사용자가 제공한 주행 녹화는 전혀 쓰지 않는다. 이전 TEST는 모두 회귀용이며,
새 TEST의 점수로 학습·checkpoint·threshold를 재선택하지 않는다.

## 개발 오류와 출처

기존 차선 VAL100장은 모두 한 recording이고 유효 픽셀의 paint는1.582%다.
기존 v4 threshold.2에서 FP571,024/FN569,759였다. 얼굴 baseline v4의 VAL400에서
누락209개 중163개(78.0%)가 원본 sqrt(box area)32픽셀 미만이었다. 16픽셀 미만
재현율87.40%,64픽셀 이상96.86%로 작은 얼굴이 주요 병목이었다.
`NEXTCYCLE-VAL-ERROR-AUDIT.json`은 이 개발 진단이며 TEST 예측을 읽지 않았다.
원본 픽셀 기준 크기 구간은 WIDER 공식 difficulty 구분과 다르다.

원 출처는 [Audi A2D2](https://a2d2-dataset.github.io/),
[공식 AWS mirror](https://registry.opendata.aws/aev-a2d2/)와 기존 공식 WIDER/CCPD
archive다. A2D2 CC-BY-ND4.0, WIDER CC-BY-NC-ND4.0, CCPD MIT 조건을 그대로
기록한다. 파생 자료·가중치는 비공개 연구 범위로 보관하며 배포 권한을 새로
주장하지 않는다. publisher annotation은 machine QC이며 사람의 개인정보 검수나
촬영 동의로 바꾸어 적지 않는다.

## 차선 후보와 고정 자료

기존 v4 stride4 residual decoder checkpoint에서 시작한다. 추가 용량 대신
fullframe1024와 context1024 crop을50:50으로 학습해 기존 crop640의 문맥 간극을
줄이고, 희소 paint에는 상위25% 유효 weighted CE+.45Dice+.1boundaryDice를 쓴다
(CE계수.45). ignore255는 손실과 경계의 유효하지 않은 이웃에서 제외한다.

미사용 공식 recording9개를 픽셀/결과 확인 전에 seed20261106으로6TRAIN/1VAL/
2TEST로 나눴다. 균등 선택800쌍에서 모든 과거 split과 연결된 SHA/pHash<=6
중복4장을 제외했고 대체하지 않았다. 최종 TRAIN1,397/VAL179(기존100+새79),
새 TEST119장이다. 새 recording 사이의 cross-split 근접 연결은0개였다.
기존 TEST280개는 누출 blacklist에 포함하며 fitting에 사용하지 않는다.
새 자료도 날짜·물리적 도로 독립성은 검증하지 않았다.

max20/min6/patience6/delta.001, batch2, AdamW/EMA.99를 고정한다. 두 개발
recording의 IoU에 같은 가중치를 주고14개 고정 threshold grid에서 선택한다.
최초 warmstart epoch0도 후보에 포함한다. 계획 문구의 nominal 새VAL80/TEST120은
중복 QC 이전 목표이며 실제 고정 수는179/119다. 독립 최종 목표 IoU75% 달성은
학습 계획 자체로 주장하지 않는다. 이전100+직전80장은 별도 회귀180장으로 쓴다.

## 얼굴 P2 후보와 학습 전 V2

기존 v6은 새 평가에서 퇴보했으므로 warmstart하지 않는다. v4 얼굴의 active
backbone과 stride8/16 대응 head를 명시적으로 이식하고, 새 stride4 P2 stem/neck을
초기화한다. 396개 tensor를 byte검증해 이식하고186개를 새로 초기화한다.
전체 상태가 기존과 같다는 주장은 하지 않는다. 번호판 parameter/BN은 동결한다.

얼굴 stride4/8/16, shared1920에서 global640+네 crop960→640, 번호판640을 쓰는
새 모델 계약이다. 출력은1×176400×7 float32(4,939,200bytes)다. tile좌표는1.5배,
global/plate는3배로 원 shared 좌표에 돌린다. 전처리 precision 최적화로 기존 모델과
동등하다고 주장하지 않는다. 공식 YOLOX 모듈을 조합한 연구 구조이며 공식 제공
P2 checkpoint인 것처럼 표기하지 않는다.

기존 TRAIN3,000얼굴/1,993번호판 및 VAL400얼굴/108번호판을 그대로 유지한다.
새 blind200은 얼굴100장 GT950개와 번호판100장이다. 모든 과거4개 release의
sourceID/SHA/연결 dHash<=3/번호판 identity를 제외했다. WIDER event partition은
과거와 같으므로 새 인물·도로에 대한 독립 성능이라고 부르지 않는다. 기존
TEST400+직전200장은 별도 회귀600장으로만 평가한다.

학습 전 V1의 source2335f562…/plan c53c5556…과 CPU 증거는 별도 snapshot으로
보존했다. 공유 GPU 자원을 위한 V2는 batch8→4, body/head LR각 절반(5e-5/5e-4),
EMA sqrt(.99)로 바꾼다. 이 변경을 학습 동등성으로 표현하지 않는다.12epoch,
자료·아키텍처·VAL 선택·TEST는 그대로다. 작은 얼굴 포함률에 비례한 image
sampling과 실제 global/tile view를 쓴다. 모든 유효 GT를 보존하며512개 초과는
조용히 자르지 않고 거부한다. 마지막3epoch는 native localization L1을 추가한다.

VAL400 AP50로 선택하되 원본64픽셀 이상 얼굴 recall이 v4보다1pp 넘게 낮아지는
checkpoint는 부적격이다. 모두 부적격이면 연구 진단용 best AP로 남기고 후보
거부를 명시한다. confidence는 최종 VAL508의 사전 grid로만 선택한다.

## 검증과 자원 절차

CPU 합성11계약에서 실제 P2 loss/backward, 단일 bilinear pixel/좌표,512GT 거부,
번호판 byte동결, ONNX 변환과 기본 raw rtol.001/atol.0001, 차선 ignore gradient를
검사했다. 실제 개발 TRAIN 최대386GT를 포함한 batch4와 차선1024 loss/backward도
유한했다. 관련 metadata/누출/numeric gate/메모리 정책 pytest45개가 통과했다.
합성 계약과 loss smoke는 정확도 또는 속도 증거가 아니다.

root가 학습 전 코드를 commit한 후에만 다른 작업과 동시에 최대3batch GPU
메모리 probe를 한다. optimizer update0, TRAIN AMP 역전파2회+개발 VAL FP32 1회,
임시 BN 변화는 폐기하고 가중치는 저장하지 않는다. 전역9GiB/새 OOM 발생을
거부하며 Tiny 프로세스는 변경하지 않는다.20ms는 요청 sleep이며 실제 샘플 시각
간격을 별도 집계한다. in-flight CUDA 연산을 즉시 취소하는 보장이나 미래 모든
학습 조합의 최대 메모리 검증은 아니다. probe 후에도 학습은 root의 별도 slot
승인까지 시작하지 않는다. 최종 속도/지속 처리 검사는 학습 후 독점 자원에서 한다.

선택된 새 가중치는 고정 양성 VAL lane3/privacy2에서 앱/격리 ORT1.31.1/원 FP32
Torch 기본 raw와 최종 후처리를 먼저 검증한다. 차선은 모든 VAL179도 검사한다.
이전 GPU/수치 증거로 새 모델을 통과시키거나 허용오차를 완화하지 않는다.
통과 뒤 새 TEST와 회귀는 모델별 각1회만 실행한다. 목표 실패와 실제 비용을
그대로 공개하며10FPS 달성은 실제 별도 측정 전 주장하지 않는다.

새 scope의 원 기한은 차선2026-11-10T17:07:26.069653Z,
얼굴/번호판2026-11-10T16:55:19.285523Z를 상속한다. 이전17scope inventory는
바꾸지 않고 새 파생/개별 예측만 별도 inventory로 관리한다. 기한 연장·원본 삭제·
자동 등록은 하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/cycle2_public_data.py lane
.\.venv\Scripts\python.exe scripts/cycle2_public_data.py privacy
.\.venv\Scripts\python.exe scripts/train_lane_cycle2.py freeze-plan
.\.venv\Scripts\python.exe scripts/train_privacy_p2_cycle2.py freeze-plan
.\.venv\Scripts\python.exe scripts/evaluate_cycle2.py prepare lane
.\.venv\Scripts\python.exe scripts/evaluate_cycle2.py prepare privacy
.\.venv\Scripts\python.exe -m pytest tests/test_cycle2_data_audit.py tests/test_cycle2_numeric.py -q
# root commit 및 probe slot 승인 후에만
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/probe_cycle2_gpu_memory.py
# 별도 학습 slot 승인 후에만 하나씩 또는 승인된 동시 방식으로
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_privacy_p2_cycle2.py train
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_lane_cycle2.py train
```
