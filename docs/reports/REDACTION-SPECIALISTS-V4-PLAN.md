# 개인정보 검출 연구 후보 v4 사전 계획

기존 v3는 원본 영상의 얼굴 재현율이 낮아 운영 승인 기준을 충족하지 못했다. 이번 후보도 운영에 자동 적용하지 않으며, 이전 모델과 실패 보고서를 보존한다.

## 검증 자료로 확인한 원인

기존 검증 자료에서 입력640 기준 얼굴 최소 변 길이가 8픽셀 미만이면 재현율29.3%, 8~16픽셀이면68.5%, 16~32픽셀이면80.7%, 32픽셀 이상이면90.8%였다. 이는 검증 자료의 진단이며 현장 성능 추정값이 아니다. 기존 원본 얼굴 학습 자료는783장/GT5105개였다. 또한 얼굴 자료에는 번호판, 번호판 자료에는 얼굴이 완전하게 주석되어 있지 않아 공동 objectness 학습에 미주석 객체를 배경으로 취급하는 문제가 있었다.

## 고정한 후보 구조와 학습

이 후보는 **YOLOX-Nano 전문 검출기 두 개**이다. 단일 Nano라고 부르지 않는다. 공유 입력은 BGR float32[0,255]의960×960 상단·좌측 letterbox이며 얼굴 전문 검출기는960으로 실행한다. 번호판 전문 검출기는 이 공유 입력을 내부 bilinear Resize(align_corners=False)로640에 축소하고 decoded xywh를1.5배하여 공유 좌표로 반환한다. 학습·검증·ONNX 내보내기·실제 앱에서 동일한 연산 순서를 사용한다. 번호판의 이전640 성능이 그대로 보존된다고 가정하지 않는다.

결과는 기존 계약인 `[1,27300,7]` decoded xywh/objectness/face/license_plate로 합친다. `agent/redaction_network.py`가 이 구조의 공유 factory이다. 기존 앱의 후처리나 임계값을 변경하지 않는다. 두 검출기의 입력 픽셀 작업량은 단일640 Nano의 약3.25배이며, 이는 실측 처리량이나 지연 시간을 뜻하지 않는다.

기존 v3의 backbone 및 해당 클래스 출력 채널로 실제 미세조정한다. 얼굴60epoch, 번호판20epoch, batch4, 최대GT512개, SGD(.001, momentum.9, Nesterov, weight_decay.0005), 3epoch warmup/cosine, EMA 검증 AP50 checkpoint를 사용한다. 50% GT중심 확대와 HSV/좌우반전을 적용하며 확대에 교차하는 **모든** 알려진 클래스 GT를 clip하여 유지한다. 마지막5epoch는 증강하지 않는다. 512개 초과 또는 유효 면적을 갖는 publisher-invalid 얼굴 주석이 있는 영상은 통째로 제외한다. 라벨을 잘라 버려 완전 주석이라고 표시하지 않는다.

검증 전용 confidence 후보는 이전과 동일한 `.01,.03,.05,.1,.2,.35,.5`, NMS IoU는.5, 최소 클래스 재현율 목표는95%이다. 목표 미달 시 기준을 낮춰 성공이라고 표시하지 않는다. checkpoint와 confidence를 고정한 후 새 시험 자료를 평가한다.

## 자료와 시험 분리

`data/training/research-redaction-v4/release.json`의 SHA256은 `4eeae0afdf67fcaf727e3e8b3a9b35486ef6f1126ae486da64637a7aec5bbd16`이다. 얼굴 TRAIN3000/VAL400/새TEST200장, 번호판 TRAIN1993/VAL108/새TEST200장이다. 번호판2000장 계획은 기존 identity 분리·중복 차단을 유지한 실제 자료 부족으로 학습 전에1993장으로 고정했다. 예측이나 시험 성능에 따른 자료 변경은 없었다.

기존 WIDER 이벤트 분할과 CCPD 번호판 identity 그룹을 그대로 유지한다. 기존 모든 TRAIN/VAL/TEST 원본 UUID·SHA256·dHash Hamming≤3 이웃과 겹치는 새 시험 원본을 제외했고, 번호판은 이전 모든 자료의 identity도 제외했다. 새 자료도 다른 분할과 같은 기준으로 검사했다. WIDER 이벤트가 다르다는 사실은 사람 identity 독립을 증명하지 않는다. 새 시험 자료의 **원본은 미사용**이지만, 이벤트 분할 체계 자체는 이전 실험에서 사용한 체계이다. 독립 현장·노선·사람 분할로 과장하지 않는다.

기존 시험400장은 이미 확인한 회귀 벤치마크로 보존하며 TRAIN/VAL로 옮기지 않는다. 새 시험400장은 학습 전에 고정했다. 후보와 기존 baseline을 같은 새 시험 자료에서 각각 한 번 비교하여 분포가 다른 지표의 단순 전후 비교를 피한다. 시험 자료는 confidence·학습 epoch·checkpoint 선택에 사용하지 않는다.

## 권리와 적용 한계

WIDER 공식 CC-BY-NC-ND-4.0 및 CCPD 저자 MIT 증거를 그대로 보존한다. 모든 원본·학습 산출물은 로컬 비상업 연구에 한정하며, 파생 영상이나 학습 가중치의 외부 배포 권리가 정리되었다고 가정하지 않는다. 최초 취득한7일 보관 기한을 연장하지 않는다. 실제 배포나 B2G 상업 이용에 대한 개인정보 검토·권리 검토·기관 승인을 대신하지 않는다.

각 영상의 알려진 클래스 주석만 검증한다. 얼굴 원본에서는 얼굴, 번호판 원본에서는 번호판을 평가하고 미주석 다른 클래스를 FP나 정답 부재의 증거로 세지 않는다. 두 전문 검출기의 결합이 전체 개인정보 재현율·익명성·현장 승인 또는 미탐 부재를 보장하지 않는다.

자료 machine QC, 실제 메모리 smoke, 사전 recipe는 각각 `REDACTION-SPECIALISTS-V4-DATA.json`, `REDACTION-V4-SMOKE-B4.json`, `REDACTION-SPECIALISTS-V4-PLAN.json`에 기록했다. 밀집GT[386,380,246,200]의 실제960 batch4 학습 단계는 peak allocated1595MiB를 사용했다. 이 값은 다른 모델과 공유 중인 GPU에서 얻은 실측이며 고립된 추론 benchmark가 아니다.

초기 실행 `data/training/redaction-specialists-v4`는 첫 검증·시험 전에 중단했다. EMA에서 파라미터마다 source state dictionary를 재구성하는 Python O(N²) 비용이 확인되었기 때문이다. 같은 EMA mul/add와 integer copy를 수행한642개 tensor는 source dictionary를 step마다 한 번 만드는 수정 전후에 bit-identical이었다. CPU 갱신 시간은1.279초에서0.00937초였다. `scripts/improve_redaction_ema_check.py`와 `REDACTION-V4-EMA-OVERHEAD.json`에 재현 절차와 결과를 보존했다. 이는 모델 정확도나 GPU 추론 속도 비교가 아니다.

수정 후 실제 학습 출력은 `data/training/redaction-specialists-v4-efficient`이며, 동일 자료·seed·optimizer·증강·epoch·검증정책으로 처음부터 시작한다. 새 trainer SHA256은 `43ae1e92d49d51f2e52c3e22939fa3ccba8f18831557ea3e0832e3eb91c1dc26`이다. 구조 factory와 자료 manifest는 바뀌지 않았다. 중단한 실행의 configuration과 사유는 원래 폴더에 남겼으며, 이 구현 교정에 시험 자료는 사용하지 않았다.
