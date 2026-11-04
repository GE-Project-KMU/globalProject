# 기존 모델을 동결한 타일 P2 보조 헤드 학습 준비

이 후보는 v4 얼굴·번호판 모델의 모든 파라미터와 BN을 고정하고 보조 투영층·stride4 얼굴 헤드만 학습한다. 기존 global960·4tile960·plate640의 102,900개 출력을 앞에 유지하고, 보조 후보 230,400개를 뒤에 붙인다. 원 모델·진행 중이던 P2 소스·공통 검증기는 변경하지 않았다. 불완전하게 종료된 P2 실험은 진단 자료로 보존하며 최종 모델 가족의 승자 후보에서 제외한다.

학습은 고정한 얼굴 TRAIN 3,000장만 사용한다. 각 원본을 실제 앱과 같은 nominal scale, round resize, 좌상단 배치·114 padding으로 1920 정사각형에 넣은 뒤 네 개 960 타일을 만든다. 원본 단계에서만 HSV와 좌우 반전을 적용하며, 같은 원본의 네 타일은 epoch 안에서 같은 변환을 공유한다. 타일과 양의 면적으로 겹치는 모든 얼굴 GT를 clip해 유지하며 작은 조각을 음성 배경으로 버리지 않는다. 원본 GT가 512개를 넘으면 전체 입력을 거부하고 라벨을 자르지 않는다.

각 epoch는 12,000개 타일을 정확히 한 번씩 사용한다. 무작위로 짝지은 두 원본의 타일을 교차 배치해 batch2는 서로 다른 원본을 포함하고, 네 타일을 위해 같은 원본을 반복 디코딩하는 비용을 줄인다. 네 개 canvas를 넘게 캐시하지 않는다. 총 12epoch, 새 파라미터만 SGD/Nesterov(lr0.005, momentum0.9, BN·bias 제외 decay0.0005), 1epoch warmup·cosine, AMP, 새 상태만 EMA0.99를 적용한다. 기존 모델은 EMA로도 바꾸지 않는다.

CPU 계약 검사에서는 TRAIN 3,000장에 있는 GT 21,583개의 원본·반전 좌표를 모두 검사해 네 타일의 GT 포함과 잘린 면적 합 보존을 확인했다. 실제 최대 밀도 TRAIN 사진 한 장의 원본 GT386개 중 선택 타일의 GT186개를 `[1,512,5]` native 라벨로 전달했다. CPU2·OpenCV1에서 실제 역전파는 약 0.584초였으며 loss·gradient가 유한하고 새 파라미터 24개에 gradient가 생겼다. 기존 gradient는 0개, 기존 상태와 역전파 전후 raw prefix는 바이트 단위로 동일했다. 이 검사는 optimizer step을 실행하거나 새 가중치를 저장하지 않았고 VAL·TEST·사용자 영상을 읽지 않았다. 원본 사진과 GT 좌표도 새 파일로 저장하지 않았다.

checkpoint는 얼굴 VAL400의 AP50(confidence floor0.01)을 기준으로 선택하되, 원본 픽셀 박스 면적 제곱근 64 이상 얼굴의 **고정 confidence0.1 Recall이 epoch0 기존 모델보다 1%p 넘게 낮아지면 제외**한다. 초기 epoch0을 보존하고 AP가 같으면 앞선 epoch를 유지한다. 최종 운영 threshold는 얼굴400·번호판108의 완전한 VAL508과 동일 grid(0.01/0.03/0.05/0.1/0.2/0.35/0.5)에서만 선택한다. 완전한 가족 후보끼리 같은 큰 얼굴 gate를 적용하고, 두 클래스 Recall95%가 가능한 군에서는 macro Precision을 우선하며, 불가능하면 최소 클래스 Recall·macro Precision 순서로 선택한다. 동률이면 기존 baseline을 유지한다. 새 TEST는 선택·임계값 조정에 사용하지 않는다.

초기 얼굴 VAL 결과는 `best-validation-face.json`으로 저장하며 개선된 checkpoint가 나오면 바뀐다. 이를 나중에 초기 baseline 원본으로 간주하면 안 된다. 공정한 가족 비교에는 별도로 출처가 고정된 초기 얼굴400 결과와 번호판108을 결합하거나, 독립 평가 도구에서 기존 모델의 VAL508을 명시적으로 다시 확보해야 한다. 최종 선택·strict 얼굴/번호판 수치 검증·승자와 threshold 동결 후에만 root가 별도 승인한 TEST 비교를 진행한다.

실행은 프로젝트 루트에서 한다. CPU 계약과 계획은 이미 고정되어 있으므로 그대로 재사용하며, 재작성하면 계획의 해시 검증에 실패한다.

```powershell
# 이미 완료된 CPU 계약과 사전 계획을 확인하는 테스트
.\.venv\Scripts\python.exe -m pytest tests/test_redaction_rescue_training.py tests/test_redaction_rescue.py -q

# 다음 두 GPU 명령은 root의 단일 작업 슬롯 승인 후에만 실행한다.
# 현재는 GPU preflight와 학습 모두 실행하지 않았다.
$env:CUDA_VISIBLE_DEVICES = '0'
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_redaction_rescue_v8.py gpu-preflight --gpu-authorized
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/train_redaction_rescue_v8.py train --gpu-authorized
```

GPU preflight는 실제 학습과 별개로 최대 GT TRAIN 원본 등을 사용한 8개 batch의 디코딩·view·전송, forward/loss, backward/step과 VAL 타이밍을 나눠 기록한다. 원본3,000장·타일12,000개·얼굴VAL400·전체12epoch를 구분해 시간을 추정하며 짧은 자원 검사에서 정확도나 현장 FPS를 주장하지 않는다. `cudnn.benchmark=False`, TF32 끔, 전체 GPU 사용량 9GiB guard와 외부 root 감독을 적용한다. 다른 GPU 학습과 동시에 시작하지 않는다.

epoch0과 각 epoch 경계에는 모델·EMA·optimizer·scaler·Python/NumPy/Torch/CUDA RNG·sampler·선택 이력을 별도 `epoch-recovery.pt`로 원자적으로 보존한다. 모델 export용 `best.pt`와 구분하며 자동 재개하지 않는다. 중단 시 상태를 보존하고 새 retry 계획·root 승인이 필요하다. 완료하지 못한 실험은 가족 선택에서 제외한다. 초기 checkpoint가 최종 선택되면 기존 모델 fallback임을 별도 필드로 표시한다.

실험 경로는 `data/training/redaction-rescue-v8`이며 현재 CPU 계약·비공개 최소 사례 메타데이터·사전 계획만 있다. 전체 학습·검증 원자료의 최소 기한은 **2026-11-10 16:55:19.285523 UTC**를 상속한다. WIDER의 CC BY-NC-ND 4.0과 CCPD의 MIT 조건, 알려진 클래스에 한정된 라벨 범위, 연구 전용·개인정보 미승인 상태도 유지한다. 원자료 이동·삭제·기한 연장·기존 보관 목록 변경은 수행하지 않았다. 이번 준비를 실제 학습 완료나 정확도 향상으로 표시하지 않는다.
