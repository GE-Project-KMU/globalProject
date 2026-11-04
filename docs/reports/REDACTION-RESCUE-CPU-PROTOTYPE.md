# 기존 모델 동결 + 타일 P2 보조 헤드 CPU 검토

기존 v4 얼굴·번호판 specialist를 모두 동결하고, 네 개 960 타일에 stride4 보조 얼굴 헤드를 붙이는 별도 연구 구조를 구현했다. 실제 학습 실행·GPU 사용·데이터 사진이나 TEST 사진 읽기는 모두 없었다. 기존 P2 학습 실험과 공통 추론·검증·학습 소스도 변경하지 않았다.

기존 얼굴 backbone이 이미 계산하는 dark2(32채널)와 기존 FPN의 stride8 출력(64채널)을 활용한다. stride8 맥락을 bilinear로 확대하고 dark2와 합친 96채널을 32채널로 투영한 뒤, 공식 YOLOX의 depthwise head(width0.125, stride4)를 호출한다. 네 타일이 새 투영층과 헤드를 공유한다. 새 파라미터는 **10,182개**, 기존 동결 파라미터는 1,793,508개다. 얼굴 backbone 호출은 기존처럼 5회, 번호판은 1회다.

FPN 연산 순서는 [공식 YOLOX YOLOPAFPN](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/models/yolo_pafpn.py)을 따르며 기존 모듈과 가중치를 그대로 재사용한다. 추가된 것은 dark2 참조 반환이다. 해당 upstream 구현은 “Copyright (c) Megvii Inc. All rights reserved.”, [Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/LICENSE)이며 데이터와 가중치의 연구용·재배포 조건은 별도로 유지한다.

출력은 기존 global·4개 tile·번호판의 **102,900개 proposal을 그대로 앞에 두고**, 보조 헤드의 230,400개를 뒤에 이어 총 333,300개로 구성한다. TopK는 사용하지 않는다. 새 objectness·class 최종 가중치는 0, bias는 sigmoid 확률 0.0001에 해당하도록 초기화하며, 추론 때 보조 objectness에 고정 0.5를 곱한다. 초기 최대 score는 약 **5×10⁻⁹**여서 현재 confidence0.01보다 낮다.

CPU 합성 입력과 원래 v4 checkpoint로 다음을 확인했다.

- 기존 102,900개 raw prefix가 바이트 단위로 동일하다.
- 합성 loss의 역전파에서 기존 파라미터 gradient는 0개이며, 기존 파라미터·BN buffer·eval 모드는 유지된다.
- 별도 합성 optimizer 계약 검사에서 새 파라미터 tensor 24개가 실제로 바뀌었다. 변경 후에도 기존 raw prefix는 바이트 단위로 동일하다.
- native loss에 512개 GT를 그대로 전달했고 라벨이 바뀌거나 잘리지 않았다. 이는 합성 계약 검사이며 실제 자료로 학습한 결과가 아니다.
- Torch와 애플리케이션 ONNX Runtime1.31.1의 raw 비교는 기본 `rtol=0.001`, `atol=0.0001`을 유지해 통과했다. 최대 절대 오차는 0.002319이며 상대 허용 오차를 포함한 원래 기준으로 판정했다.
- 실제 합성 패턴에서 초기 후처리는 동일했으며 검출은 0개였다. 별도 통제 raw fixture로 얼굴·번호판 양성 3개, 겹침 NMS, 경계 clip 분기도 회귀 검증했다. 그 fixture를 학습 모델의 실제 양성 결과로 표시하지 않는다.
- 관련 앱 테스트 102건이 통과했다.

Torch CPU 2스레드·워밍 후 순서를 교차한 각 3회 측정의 중앙값은 기존 1.179초, 보조 헤드 1.430초로 약 **21.3% 비용 증가**였다. 동시 GPU 학습과 CPU를 공유한 호스트의 단일 합성 입력 측정으로, 카메라·ONNX 파이프라인·GPU FPS를 뜻하지 않는다. 파라미터 수 증가가 작아도 고해상도 보조 feature와 proposal 수 증가에는 연산·메모리 비용이 있다.

현재 증거는 구조와 초기 수치·동결 계약에 한정된다. 학습한 보조 후보는 기존 낮은 score의 proposal과 NMS에서 경쟁할 수 있으므로, 학습 후 precision·최종 검출 결과가 동등하다고 주장하지 않는다. 실제 학습은 별도 승인과 사전 계획이 필요하며 고정 VAL에서 큰 얼굴 회귀·추가 FP·Recall·비용을 함께 확인해야 한다. 새 TEST는 그 선택 과정에 사용하지 않는다. 모델 등록이나 운영 활성화는 수행하지 않았다.

원 checkpoint와 기존 소스는 그대로 보존했고 새 ONNX·합성 tensor는 프로젝트 내부의 비공개 prototype 폴더에만 저장했다. 가중치·raw tensor는 Git과 공개 보고서에 포함하지 않았다. 보관 주석은 실행 후 별도로 추가했으며, 이번 원 실행에 없던 보관 가드를 소급해 검증했다고 주장하지 않는다. 기존 보관 목록 등록·변경·삭제도 수행하지 않았다.
