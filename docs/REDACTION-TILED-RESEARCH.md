# 선택 연구 후보: 얼굴 전역·타일 추론

2026-11-04, 새 가중치와 기존 학습 recipe를 바꾸지 않는 조건부 후보를 추가한다. 기본 모델로 자동 지정하지 않는다. `agent/redaction_tiled_network.py`만 새 factory이며 기존 `agent/redaction_network.py`의 SHA256은 `9d26b063b22b54b1606cb33be1291b6e71f57945849aab21119764e1c847ca5d`로 유지한다.

## 고정 구조

`build_redaction_tiled_network(yolox_root=None, shared_input_size=1920, face_input_size=960, plate_input_size=640)`는 기존과 같은 `.face`·`.plate` 속성 및 state_dict key/shape를 사용한다. 두 genuine Nano 가중치 한 세트이며 얼굴 가중치를 다섯 번 공유 실행한다. 새 타일 학습이나 가중치 복제 학습은 없다.

입력은 앱에서 만든 좌상단 letterbox1920 BGR float32 텐서다. 전역 얼굴은 bilinear1920→960(`align_corners=False`) 후 decoded xywh를2배, 4개960 quadrant의 얼굴은 좌표를 각 좌상단으로 offset, 번호판은 bilinear1920→640 후 xywh를3배 한다. 출력 순서는 전역 얼굴·좌상·우상·좌하·우하 얼굴·번호판이며 `[1,102900,7]` decoded xywh/objectness/face/plate다. 타 클래스 확률 열은0으로 채운다.

타일 안에서 박스를 미리 clipping하지 않는다. 전역 좌표의 전체 후보에 기존 class별 NMS를 적용하고 마지막에 원영상 경계 clipping과 퇴화 박스 거부를 수행한다. 입력에 이미 있는114 letterbox padding은 추가 이동·재패딩하지 않는다. 전역 얼굴 분기는 타일 경계에 걸친 얼굴을 보조하지만 경계 손실이 없어졌다고 보증하지 않는다.

기존 original→960/640 경로와 original→1920→960/640 경로의 리샘플링은 다르다. 같은 가중치여도 실제 validation에서 정확도를 다시 비교해야 한다. 얼굴5회·번호판1회의 CPU 비용을 별도로 보고하며 처리량 향상을 주장하지 않는다.

## 시험 전 선택 절차

기존 학습이 끝나면 가중치 SHA를 고정하고 표준960과 이 후보만 같은 validation에서 비교한다. 기존 confidence grid `[0.01,0.03,0.05,0.1,0.2,0.35,0.5]`, NMS IoU0.5, 클래스별 Recall95% 목표를 유지한다. 각 후보의 confidence는 기존 규칙대로 양 클래스 목표 달성 시 macro precision 우선, 미달 시 최소 클래스 recall 우선으로 정한다. 동점이면 추가 비용이 없는 표준 경로를 우선한다. 이 선택 규칙은 fresh TEST 점수를 읽기 전에 고정한다. validation상 이득이 없으면 타일 후보를 fresh TEST에서 평가하지 않을 수 있다.

얼굴·번호판 알려진 클래스 조건부 평가와 실제 개인정보 검수는 구분한다. 목표, 원본 TTL, 개인정보 처리와 운영 승인 조건은 완화하지 않는다.

표준 학습의 자동 평가 산출물은 선택 완료 전 봉인한다. heldout/baseline/regression 예측·summary·최종 점수 로그뿐 아니라 `heldout_map50`를 포함할 수 있는 원래 `model.json`도 읽지 않는다. validation 전용 구성은 source 계약과 `training-config.json`, `candidate-freeze.json`, validation calibration 및 checkpoint만으로 만든다. frozen 학습 소스나 자동 평가를 중간에 수정하지 않는다. 자동 생성된 시험 결과를 이후 공개할 때는 생성 시점과 봉인 절차도 명시한다.

## 현재 검증 범위

CPU 합성 좌표·클래스 열·padding·경계를 넘는 박스 보존·기존 strict checkpoint 구조 호환 및 random 초기화 forward의 finite 출력/shape만 확인한다. 실제 학습 자료·TEST·학습 가중치에 접근하지 않았으며 정확도 비교는 아직 하지 않았다. `reports/REDACTION-TILED-CPU-CONTRACT.json`은 이 범위의 구조 검증이다.

verifier·packager는 명시적 `architecture="two YOLOX-Nano specialists global plus four tiles"`, `input_size=[1920,1920]`, `shared_input_size=1920`, `face_input_size=960`, `plate_input_size=640`, `face_tile_grid=2`를 지원한다. 크기와 grid는 정수여야 하고 누락·다른 geometry·표준 architecture와 타일 설정 혼합은 거부한다. 새 모델 계약을 묶거나 설치할 때 이 필드를 보존한다. 실제 가중치 수치 비교·validation 정확도 평가는 별도 단계다.
