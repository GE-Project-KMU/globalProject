# 장비 연결 및 실행 인계

구현된 CLI는 프로젝트 폴더 안의 경로만 사용한다. 아래는 실물 장비를 연결한 뒤 수행할 순서다. **장비 연결만으로 승인 모델·보정·기관 수집 근거가 자동으로 준비되는 것은 아니다.** 예시의 UUID/포트/경로는 실제 설정으로 바꾼다. 개발 서버의 합성 자료와 실제 모델 결과는 별도로 유지한다.

1. 운영 가이드대로 `.venv`·런타임·영상 의존성을 설치하고 서비스를 시작한다. Windows와 Ubuntu의 가상환경은 각각 새로 생성한다.
2. 웹 관리자 메뉴에서 기기, 실제 카메라 보정 프로파일, 잠금한 모델 번들을 등록한다. 기기 토큰은 발급 시 한 번만 표시하며 별도 관리한다.
3. VISION.md 규격의 로컬 profile 파일을 준비한다. 서버의 calibration과 실제 검수자, BEV 크기·미터/픽셀·검증된 차선 외곽·색상/종류·점선 위상을 연결한다. `validate-profile`과 `calibrate` 도구를 사용한다.
4. 승인된 데이터로 `agent.train_detector`/`agent.train_lane`을 실행하고 ONNX·manifest를 준비한다. 모델별 task/classes/hash/입력 규약과 번들 weights_hashes가 같아야 한다. `model-status`와 `agent.evaluate`로 준비 상태와 독립 평가 결과를 구분해 확인한다.
5. ZED 2i UVC 좌우 결합 영상과 NMEA 포트를 확인한다. SDK·깊이는 선택 사항이며 기본 RGB 수집은 SDK 없이 동작한다. 하드웨어 드라이버·CUDA의 전역 설치는 이 작업에서 수행하지 않았다.
6. 구조화 수집계획과 정차 점검을 준비하고 `collect`를 실행한다. 기관·연락처·담당·근거·목적·범위·고지·보관기간이 필요하다. 고정·안전줄·렌즈·케이블·조도·건조 노면은 사람이 확인한다. 운전 중 조작하지 않는다.
7. 정차 후 종료·postcheck를 수행한다. 장치 단절과 Ctrl+C에서도 이미 수집한 암호화 조각과 주행 정보를 보존한다. 문제 구간은 판정 불가로 남긴다.
8. postcheck가 끝난 manifest를 `python -m agent register-capture data/agent/runs/RUN_UUID/manifest.json --gateway https://서버주소 --ca-bundle data/tls/localhost.pem`으로 먼저 등록한다(`ROAD_DEVICE_TOKEN` 필요). 이 요청은 주행 메타데이터/해시만 POST하고 원본을 보내거나 주행을 완료하지 않는다. 등록 후에는 manifest를 바꾸지 않는다. 준비 설정 파일로 `workflow`를 실행하거나, 같은 기관의 주행 ID·번들 ID와 준비 설정 UUID를 웹 분석 작업 화면에 입력한다. 원본 영상을 공개 API에 업로드하는 흐름은 제공하지 않는다.
9. 생성된 비식별 사진을 사람이 전량 확인하고 누락 가림을 보완한다. 최종 이미지 해시와 관측 ID를 decisions 파일에 기록하고 `approve-review`를 실행한다.
10. `sync-run`으로 주행·경로/완료를 동기화한 뒤 `upload`로 승인→이미지→관측을 전송한다. ACK 전에는 완료로 표시하지 않는다. 웹에서 사진·위치 품질·수치·검토·보고서를 확인한다.

정확한 인수와 파일 스키마는 다음 명령과 VISION.md를 참고한다.

```powershell
.\.venv\Scripts\python.exe -m agent --help
.\.venv\Scripts\python.exe -m agent collect --help
.\.venv\Scripts\python.exe -m agent workflow --help
.\.venv\Scripts\python.exe -m agent model-status data\models\pothole\model.json
```

기본 서버 주소는 로컬 개발용이다. 기관 배포에서는 발급된 HTTPS 인증서를 사용한다. 프로젝트 내부 CA를 명시적으로 신뢰할 때는 `--ca-bundle data/tls/localhost.pem`을 사용하고 인증서 검증을 끄지 않는다. 토큰은 `ROAD_DEVICE_TOKEN`·`ROAD_REVIEWER_TOKEN`으로 전달하며 원본·사진·토큰·키를 Git에 넣지 않는다.

`purge-local --apply`와 철회 도구를 기관 운영 일정에 등록한다. 원본 7일, 비식별 상세 90일, 최소 감사 180일이라는 프로젝트 기본 정책을 유지하고 실제 기관 방침이 더 짧으면 먼저 적용한다. 데이터 철회·만료로 영향받은 모델은 재검증 전 사용할 수 없게 표시한다. 권한·촬영 근거·보정·모델·장비 준비가 충족되지 않으면 실행을 차단하는 것이 정상 동작이다.
