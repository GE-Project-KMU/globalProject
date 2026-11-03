# 개발 MVP 검증 기록

2026-11-03, Windows에서 프로젝트 내부 가상환경·캐시·합성 자료로 검증했다. 실물 주행 정확도 또는 운영 적합성 인증 결과가 아니다.

| 검증 | 결과 | 범위 |
| --- | --- | --- |
| 실제 Gateway HTTP | 34/34 통과 | 인증·권한·기관 격리, 기기, 승인 이미지, 관측·차선 경계값, 검토, 중복, 보고서, 삭제 |
| 로컬 영상 도구 | 34/34 통과 | 암호화 촬영/재생, GNSS, 차선 면적·판정 불가, 분할 누출 검출, 모델 검증, outbox |
| Chromium Playwright | 8/8 통과 | UI 독립 시험 7건, 실제 Gateway 연동 1건, 데스크톱·모바일 |
| TypeScript / Vite | 통과 | 타입 검사 및 production build |
| 최종 HTTP 실행 중 서버 로그 | 예외 0건 | 해당 실행에서 추가된 로그의 Traceback/ERROR/unhandled/HTTP 500 검사 |

HTTP 시험은 실제 TCP 요청과 상태 코드·응답 본문을 검사했다. 4개 동시 동일 관측의 멱등 수신, 동시 검토의 200/409 충돌, If-Match 428/409/200, PDF 한글 본문 추출, 공유 이미지 삭제의 관련 관측 전파, 삭제 후 410과 재수신 차단을 포함한다.

브라우저에서는 로그인·보호 이미지·상세 보기·검토 PATCH·서비스 상태·CSV 생성/다운로드와 모바일 배치를 확인했다. 정상 시나리오의 console.error/pageerror, 예상하지 않은 요청 실패와 API 4xx/5xx는 없었다. 개발 중 발견한 날짜 범위, 잘못된 JSON/숫자 처리, Leaflet 화면 해제 예외, 모바일 가로 넘침을 수정하고 재검증했다. 이전 오류 로그는 지우지 않고 최종 실행의 추가 구간을 별도로 검사했다.

로컬 증거는 Git 제외 경로 `data/http-test-results.xml`, `data/http-log-check.json`, `frontend/test-results/`에 있다. 실제 연동 화면은 `frontend/test-results/live-live-gateway-login-in-2c3f8-ealth-and-responsive-layout/`의 desktop/mobile/service-status PNG에 보관한다. 자격 증명·이미지·원본 로그는 CI artifact로 업로드하지 않는다.

## 재실행

프로젝트 루트에서 다음을 실행한다. Windows 명령이며 Ubuntu에서는 `.venv/bin/python`을 사용한다. Playwright 브라우저와 임시 경로는 운영 가이드에 따라 프로젝트 내부로 지정한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-vision.txt
.\.venv\Scripts\python.exe scripts/manage.py start --frontend
.\.venv\Scripts\python.exe scripts/seed_demo.py
.\.venv\Scripts\python.exe -m pytest tests/test_http.py tests/test_vision.py -q
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run test:e2e
```

GitHub Actions에는 Ubuntu Python 3.14 HTTP/영상 도구와 Node 24 build/실제 연동 UI job을 추가했다. 로컬 통과 결과와 원격 CI 결과는 구분한다.

## 남은 검증과 구현 제한

- ZED 이름의 연결 장치를 현재 호스트에서 탐지하지 못했다. 실제 ZED 2i SDK 캡처, 장착·진동·조도·주행 속도별 검증, GPU 추론은 수행하지 않았다.
- 포트홀 Nano 및 차선 paint·얼굴/번호판 비식별화의 승인된 학습 가중치와 독립 현장 GT가 없다. 학습 진입점과 ONNX 어댑터를 구현했으나 학습·정확도 합격은 주장하지 않는다.
- 고정 revision·SHA256의 외부 YOLOX-S 참고 모델을 내려받아 합성 영상 한 장에서 CPU 실행을 확인했다. 이는 목표 모델과 다르며 게시자 mAP50 72.1%도 목표 80%에 미달한다. 실제 도메인 성능 증거가 아니다. 출처와 사용법은 VISION.md를 참고한다.
- Docker 실행 환경이 없어 Compose/PostgreSQL의 실제 실행은 미검증이다. 현재 SQLite 실제 통합 결과이며, 저장 계층은 서비스별 JSON aggregate/낙관적 version을 사용한다. 정규화된 PostGIS 설계 전체를 구현한 것은 아니다.
- PDF는 Windows 이식성을 위해 ReportLab을 사용한다. 설계서의 WeasyPrint 선택과 다르며 한글 추출은 검증했다.
- VWorld 키가 없으면 지도는 배경지도 없음 안내와 좌표 격자를 표시한다. 개인정보 열람·정정은 관리자 수동 검토 상태로 접수되며 자동 완료하지 않는다.
- 외부 기관 운영 전 보관 정책 전체의 물리 삭제·백업 삭제, TLS·비밀키 운영, 대규모 처리량과 실제 카메라·모델 수용 시험이 필요하다.
