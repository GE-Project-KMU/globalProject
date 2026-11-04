# 개인정보 모델 가족의 VAL 선택과 최종 평가 절차

독립 도구 `scripts/evaluate_privacy_family.py`를 준비했다. 현재는 코드·합성 계약 검사·사전 프로토콜 작성만 수행했다. 이 작업에서 새 사진 추론·GPU 실행·TEST 추론은 모두 0회다. 기존 Rescue trainer/data/factory와 불완전 P2 실험은 변경하지 않는다.

비교 대상은 완료된 v4 legacy와 완료된 Rescue v8이다. 중단된 P2는 진단 자료로 남기고 승자 후보에서 제외하며, P2의 이전 계획 해시도 새 프로토콜에 보존한다. 검증 범위는 동일한 얼굴400·번호판108의 전체 VAL508이다. 원 TEST 파일은 계획 단계에서 전체 파일 해시만 확인하며 GT나 사진을 디코딩하지 않는다.

공정한 baseline을 위해 다음 경로를 구분한다.

- 기본 native GPU 방식은 Rescue 학습 완료 후 root의 단일 GPU 슬롯에서 baseline VAL508을 평가하고, 완료된 고정 trainer의 selected VAL508 결과를 바인딩해 사용한다.
- CPU 방식은 두 모델 모두 같은 ORT1.31.1·CPU2·OpenCV1 경로로 VAL508을 재생한다. native 후보 결과와 CPU baseline을 섞어 선택하지 않는다. CPU 방식을 쓰려면 VAL 추론 전에 별도 사전 프로토콜을 고정해야 한다.
- 얼굴400과 번호판108을 합칠 때는 각각 원 baseline manifest·checkpoint·DEV·runtime profile·전체 case 멤버십이 확인된 불변 receipt가 있어야 한다. 현재 trainer의 `best-validation-face.json`은 개선 시 바뀌므로 초기 baseline snapshot으로 임의 재사용할 수 없다. 확실한 초기400 receipt가 없으면 전체508을 확보한다.

승자는 고정 confidence0.1에서 원본 픽셀 박스 면적 제곱근64 이상 얼굴의 Recall이 baseline보다 1%p 넘게 떨어지지 않는 완료 후보 중에서 선택한다. 같은 confidence grid(0.01/0.03/0.05/0.1/0.2/0.35/0.5)를 적용해 두 클래스 Recall95%가 가능한 군에서는 macro Precision·최소 클래스 Recall 순서로, 불가능하면 최소 클래스 Recall·macro Precision 순서로 선택한다. 정확한 동률에서는 기존 baseline을 유지한다. 모델 가족과 threshold 선택에 TEST를 사용하지 않는다.

선택 후 원 모델·가중치·기존 검증 파일은 보존하고 별도 child manifest/provenance를 만든다. 가중치 바이트를 바꾸지 않으며 새 학습이나 운영 승인을 기록하지 않는다. 복제 provenance는 원 소유 파일의 삭제 참조를 제거하고 출처 identity·기한·라이선스와 부모 해시를 유지한다. 승자·threshold·model/checkpoint·전체 VAL 예측 파일의 SHA를 `winner-freeze.json`에 한 번 고정하며 이후 재선택하지 않는다. 공개 freeze에는 집계와 전체 증거 파일의 해시만 기록한다.

canonical parity 사례는 클래스별로 baseline과 승자가 모두 실제 운영 threshold에서 IoU0.5로 맞힌 큰 GT가 있는 VAL 사진 중 source ID 사전순 첫 장으로 정한다. 이 규칙은 수치 분기의 양성 범위를 확보하기 위한 것이며 정확도를 대표하는 표본 선정이 아니다. 개별 경로·SHA·클래스는 비공개 `positive-VAL-cases.json`에만 저장한다.

root가 공통 verifier의 새 architecture 지원을 완료한 뒤 baseline child와 winner child 각각에 얼굴·번호판의 실제 canonical 검증을 수행한다. 기본 raw `rtol0.001/atol0.0001`, 원래 bbox·confidence 후처리 기준, 실제 같은 클래스 양성, app CPU ORT1.31.1을 모두 요구한다. 실패 시 허용 오차를 바꾸거나 다른 사례로 자동 교체하지 않는다. 이 gate가 실패하면 TEST 계획이나 픽셀을 읽기 전에 중단한다.

기본 native 방식의 후속 명령은 Rescue 완료와 root의 슬롯 승인 후 실행한다. 현재 실행하지 않았다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/evaluate_privacy_family.py evaluate --role baseline --authorized
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/evaluate_privacy_family.py evaluate --role rescue_v8 --authorized
.\.venv\Scripts\python.exe scripts/evaluate_privacy_family.py freeze
```

두 번째 명령의 native 후보 경로는 완료된 trainer의 전체 VAL508 파일을 가져와 해시를 고정하며 새 사진 추론은 하지 않는다. 구성 요소 방식을 선택한다면 `evaluate --role baseline --component face`와 `--component license_plate`로 동일한 receipt를 확보하고 `compose --face-component ... --plate-component ...`로 합친다. 이미 전체 baseline508을 확보한 후 중복 생성하지 않는다.

numeric 사례와 선택된 checkpoint 경로는 비공개 freeze에서 읽어 `scripts/verify_trained_models.py`에 전달한다. 모델별 결과는 각각의 child 폴더에 `verification-final.json`(얼굴), `verification-plate.json`(번호판)으로 저장한다. 기존 모델 폴더의 검증 파일을 덮어쓰지 않는다.

양 모델의 두 canonical gate가 통과하고 root가 별도로 승인한 뒤에만 최종 비교를 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_privacy_family.py run --role fresh --authorized
.\.venv\Scripts\python.exe scripts/evaluate_privacy_family.py run --role regression --authorized
```

fresh는 기존 cycle2의 새 blind200, regression은 기존 `cycle2-evaluation/privacy-regression-manifest.json`의 과거 평가1400을 참조한다. benchmark·승자·계획이 달라지면 거부하며 원자적인 일회 시도 receipt가 생긴 뒤에는 실패해도 자동 재시도하지 않는다. 최종 비교 결과를 보고 가족·threshold를 바꾸지 않는다. 과거 TEST는 회귀로만 취급하며 물리적 인물·차량·경로 독립성이나 완전한 개인정보 보호를 주장하지 않는다.

독립 자료 경로는 `data/training/privacy-family-v8-evaluation`이다. 새 계획·예측·child의 source 기한은 원 VAL의 최소 기한인 **2026-11-10 16:55:19.285523 UTC**를 상속한다. 연구 전용·개인정보 미승인·재배포 제한을 유지하고 원본 사진 복사·업로드·기간 연장·기존 보관 목록 등록이나 삭제를 수행하지 않는다.
