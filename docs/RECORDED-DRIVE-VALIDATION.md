# 제공 주행 영상의 로컬 실행 검증

사용자는 이번 검증에 한해 **2026년 11월 5일 23:59(KST)**까지 처리·보관을
허용했다. 촬영일은 사용자 추정 10월 28일이며 독립적으로 확인된 UTC가 아니다.
원래 촬영일·기본 7일 정책은 변경하지 않는다. 자동 삭제·보관 등록, 외부 업로드,
학습 승인을 이 예외에서 도출하지 않는다. 개별 승인 receipt와 파일별 결과는
Git에서 제외한 `data/recorded-drive-validation/`에만 둔다.

## 검증 방법

전 구간 재생 검사와 모델 실행 검사를 분리한다. 전 구간 검사는 좌우 메모리
디코딩, 원본 메시지와 프레임 PTS 대응, 정상 EOF, 제공된 CRC, 원본 SHA256
전후 일치를 확인한다. 밝기·표준편차·에지·이전 축소 영상과의 동일 여부는
진단 지표이며 도로 장면·장착 상태·유효 학습 데이터의 판정을 대신하지 않는다.

모델 검사는 결과를 보기 전에 파일별 영상 메시지 수 N을 기준으로
`floor(k*(N-1)/4), k=0..4`를 고정했다. 중복 인덱스는 제거한다. 제공한 16개
파일에서는 총 80개 시점이며 좌측 영상만 사용한다. 원본 픽셀·예측 마스크·주행
영상은 파일로 내보내지 않는다. 개인식별 가능성이 있는 후보 좌표·기록시각·개별
원본 해시는 비공개 결과에만 남기며 API 관측이나 공유 승인 자료를 생성하지 않는다.

## 실제 결과

[전 구간 집계](reports/RECORDED-MCAP-FULL-DECODE.json): 16개 파일의
14,753프레임을 끝까지 디코딩했다. 프레임 시각 불일치 0, 원본 전후·인계 SHA
16개 일치, 제공된 nonzero Chunk CRC 2,531개가 통과했다. DataEnd CRC는
제공되지 않아 검증했다고 주장하지 않는다. 좌측 기준 1920×1080 2,901프레임,
1280×720 11,852프레임이며 우측 해상도도 각각 일치했다.

[80프레임 모델 집계](reports/RECORDED-DRIVE-MODEL-PROBE.json): 세 모델이
모두 정상 실행됐다. 포트홀 후보는 8개 프레임에서 8개, 얼굴 후보 403개,
번호판 후보 14개였다. 프레임 사이 중복과 오탐을 제거한 실제 객체 수가 아니다.
기본 영상 품질 검사는 73개 통과, 7개 underexposed였다. 전 구간 검사와 샘플
검사의 밝기 휴리스틱은 해상도·조건이 다르며 동일한 품질 판정 척도가 아니다.

CPU 추론 중앙값은 포트홀 28.94ms, 차선 480.85ms, 비식별화 400.20ms였다.
합산만으로도 현재 CPU 직렬 모델 경로는 10FPS 목표를 충족했다고 볼 수 없다.
추적·마스킹·저장까지 포함한 연속 처리 시험이나 GPU 성능 결과는 아니다.

실제 모델 검사는 소스 `e1951fd`의 사전 고정 계획으로 실행했다. 종료 후
계획 SHA 및 소스 SHA가 실행 전 공개 기록과 그대로 일치함을 확인했다.
이후 코드 검토에 따라 비공개 계획/결과의 저장 경로를
`data/recorded-drive-validation/`으로 제한하고, 시작 시 읽은 계획의 SHA를
결과에 고정하며 종료 시 변경을 거부하는 검사를 추가했다. 이 후속 보강은 합성
회귀로 검증했고 실제 80프레임 추론을 반복한 결과로 제시하지 않는다. 기존 실행
계획을 새 코드로 덮어쓰거나 결과를 새 실행의 증거로 바꾸지 않는다.

최종 코드 전체 Python 회귀 검사는 **534개 및 하위 검사 189개 통과**,
실제 symlink 생성 1개는 Windows 계정 권한으로 건너뛰었다. 비공개 경로 이탈,
기한 도달, 실행 중 계획 변경과 원본/모델 변경 거부를 합성 데이터로 검증했다.
의존성 검사도 통과했다. 프론트엔드 코드는 변경하지 않았다.

사용 모델은 기존에 선택한 포트홀 RDD2020 v4, 지원 ORT1.31.1 차선 모델,
개인정보 비식별화 tiled 모델이다. 모델/manifest/검증 소스 SHA와 샘플 계획을
[실행 전 계획](reports/RECORDED-DRIVE-MODEL-PROBE-PLAN.json)에 고정했다.
이 영상의 결과로 가중치·임계값·모델 선택을 바꾸지 않는다. 새 학습도 수행하지 않는다.

`scripts/probe_recorded_models.py`는 모델별 CPU 추론 지연과 후보 수, 기본 영상
품질, 차선 도색으로 예측된 픽셀 비율을 기록한다. 도색 픽셀 비율은 원근·배경을
포함한 영상 면적 대비 값으로, 물리적 마모율이나 도색 잔존율이 아니다.
이번 실행은 별도 전 구간 디코딩 검사와 병렬로 진행하므로 지연 수치에는 다른
검사와의 자원 경쟁과 첫 추론 초기화가 포함된다. 독립 성능 벤치마크가 아니다.

## 재현

의존성은 `requirements.txt`와 `requirements-vision.txt`로 설치한다. 계획 생성과
실행은 분리되며 기존 계획/결과 덮어쓰기를 거부한다. 개인 원본을 읽는 명령은
승인된 처리 기간 안에서만 실행한다. `--permission`은 이 대화의 명시 승인에
대응하는 비공개 receipt이며 현재 도구는 해당 승인보다 늦은 기한을 거부한다.
새 코드로 재현할 때는 새로운 계획을 생성해야 한다. 계획에는 모델 가중치와
manifest뿐 아니라 모델 런타임 소스 SHA도 함께 기록한다.

```powershell
.\.venv\Scripts\python.exe scripts/probe_recorded_models.py plan --inventory data/PRIVATE-INVENTORY.json --audit data/PRIVATE-AUDIT.json --permission data/recorded-drive-validation/permission.json --pothole data/training/pothole-nano-rdd2020-v4/model.json --lane data/training/lane-a2d2-supported-ort131-v3/lane.json --redaction data/training/redaction-inference-options-v4/option-tiled/model.json --output data/recorded-drive-validation/NEW-RUN/plan.json
.\.venv\Scripts\python.exe scripts/probe_recorded_models.py run data/recorded-drive-validation/NEW-RUN/plan.json
.\.venv\Scripts\python.exe -m pytest tests/test_recorded_models_probe.py tests/test_recorded_mcap_full.py -q
```

기한은 원본 열기·해시 블록·디코딩 레코드/프레임·추론 전후에 검사한다. 진행 중인
네이티브 함수는 반환 후 다시 확인하고 다음 처리와 결과 저장을 차단한다.
만료는 자동 삭제를 실행하는 조건이 아니다.

## 판정할 수 없는 요구사항

파일별 내부 기록시각 span 합은 약 12분 10초다. 파일 사이 중복·연속성이나
실제 취득 시 기대 프레임 수를 검증한 값이 아니므로 60분 실차 취득·누락률
합격 근거로 쓰지 않는다. 전체 디코딩 속도와 80개 샘플 추론은 10분 연속으로
두 인식·추적·마스킹·저장까지 수행하는 10FPS 시스템 시험을 대체하지 않는다.

정답 라벨이 없어 포트홀 AP/정밀도/재현율, 개인정보 종류별 누락률, 차선 IoU와
마모 오차를 산출할 수 없다. 검출 후보는 확인된 포트홀이나 고유 객체 수가 아니다.
별도 GNSS·보정 파일과 검증된 장착 정보가 없어 지도 위치·거리별 오탐·10m 마모
등급·물리 깊이를 생성하지 않는다. SDK 재생과 rectification도 미검증이다.
이 단계는 실제 영상에 대한 실행 경로의 검증이며 현장 성능 합격 판정이 아니다.
