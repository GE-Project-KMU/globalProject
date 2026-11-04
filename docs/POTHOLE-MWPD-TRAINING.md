# 공개 데이터 포트홀 추가 학습과 평가

기존 YOLOX-Nano에서 시작해 MWPD v2의 새 도메인을 추가 학습했다. 원본·라벨·가중치는 프로젝트 폴더에만 보관하고, 재현 코드와 식별자 없는 집계만 Git으로 관리한다. 사용자 주행 영상은 사용하지 않았다.

## 출처와 분할

- [MWPD v2](https://data.mendeley.com/datasets/s5hx9n2jc3/2): Shahnaj Parvin·Foysal Munsy·Kamruddin Nur, 2025-01-15, DOI 10.17632/s5hx9n2jc3.2, CC BY 4.0. 게시자가 제공한 클래스 정의의 0번 Potholes를 사용했다.
- [RDD2020](https://data.mendeley.com/datasets/5ty2wb6gvg/1): 기존 학습 자료, CC BY-NC 3.0. 혼합 가중치는 연구 용도로 제한하며 상용 배포 승인을 뜻하지 않는다.

게시자 파일 SHA, 원본 이름의 증강 계열, 정확한 이미지 해시와 연결된 유사 이미지 군을 검사했다. 기존 VAL·TEST와 연결된 새 학습 이미지를 제외하고 새 VAL·TEST에서는 기존에 사용한 모든 자료와 연결된 이미지를 제외했다. 이 절차는 실제 촬영 도로·주행·날짜의 독립성을 보장하지 않는다.

MWPD의 고정 분할은 TRAIN 2,188장·VAL 250장·TEST 97장이며 모두 포트홀 양성이다. 별도로 이전에 사용하지 않은 RDD 체코 일반 노면 48장을 고정했다. 게시자 라벨에 D40이 없는 자료이며 다른 균열이 있을 수 있고, 사람이 모든 포트홀 부재를 다시 검수한 자료는 아니다. 이번 최종 평가는 양성 97장과 일반 노면 48장, 총 145장을 사용했다.

## 학습과 사전 고정

기존 TRAIN 5,352장과 새 TRAIN을 합친 적격 풀은 7,540장이다. 고정 12 epoch 동안 두 도메인의 기대 표본 수를 균등하게 한 복원 추출을 사용했으므로 매 epoch 모든 이미지를 봤다는 뜻은 아니다. SGD, 학습률 0.0001, Nesterov, 고정 BN 통계, HSV·수평 반전, 성공한 optimizer step의 EMA를 사용했다. Mosaic·L1은 사용하지 않았다.

기존·새 VAL AP50 평균으로 checkpoint를 고르고, 기존 VAL 하락이 기준 대비 3%p 이내여야 한다는 조건을 미리 정했다. epoch 0도 후보에 포함했다. 실제 optimizer update는 3,261회이고 선택된 checkpoint는 epoch 7이다. 기존 VAL AP50은 68.39%에서 65.68%로 2.72%p 하락했고, 새 VAL은 17.85%에서 59.15%로 상승했다. 개발 단계의 허용 조건 통과이며 기존 도메인의 새 독립 TEST 통과를 의미하지 않는다.

선택한 가중치와 confidence 0.1을 고정한 뒤 최종 TEST를 한 번 비교했다. TEST 결과로 모델·임계값·평가 표본을 바꾸지 않았다. 앱 ONNX Runtime CPU와 원 학습 프레임워크·참조 런타임의 양성 VAL 3건에서 원시 출력 허용 오차와 후처리 일치를 먼저 확인했다.

## 고정 TEST 결과

| 지표 | 기존 | 추가 학습 |
|---|---:|---:|
| 양성+일반 노면 AP50 | 18.04% | 60.03% |
| 양성 97장만의 AP50 | 18.47% | 60.25% |
| confidence 0.1 정밀도 | 31.11% | 46.30% |
| confidence 0.1 재현율 | 23.97% | 68.49% |
| 일반 노면 48장의 오탐 박스 | 12 | 8 |
| 일반 노면의 오탐/이미지 | 0.250 | 0.167 |

AP는 IoU 0.5·101점 보간, 낮은 confidence 0.01에서 계산하고 운영 지표는 고정 0.1에서 별도로 계산한다. 목표 AP50 80%에는 미달한다. 이 작은 해외 공개 자료의 결과를 국내 ZED 주행 성능이나 오탐/km로 해석하지 않는다. 과거 다른 평가 세트의 수치와 직접 비교하지 않는다. 상세 집계는 [최종 결과](reports/POTHOLE-MWPD-RESULTS.json), [수치 검증](reports/POTHOLE-MWPD-CANONICAL.json)에 보존한다.

## 재현 도구

획득·분할은 `scripts/prepare_mwpd.py`, 일반 노면 고정은 `scripts/prepare_fresh_pothole_negatives.py`, 학습은 `scripts/train_pothole_mwpd.py`, canonical 검증은 `scripts/verify_pothole_mwpd.py`, 최종 평가는 `scripts/evaluate_pothole_mwpd.py`를 사용한다. 각 도구의 `--help`를 확인하고 별도 출력 폴더에서 실행한다. 기존 최종 TEST를 개발용 재선택에 사용하지 않는다. 원 계획·출처 만료 시각·SHA는 로컬 증거에 유지하며 재학습이나 복사로 원자료 보관 기한을 연장하지 않는다.
