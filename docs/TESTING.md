# 개발 MVP 검증 기록

## 2026-11-04 학습·추론 변경 후 재검증

Windows 전체 Python/실제 HTTP/영상 도구 시험 **180/180 통과**(43.79초, 실패·skip0), TypeScript/Vite build 및 Chromium **20/20 통과**(실제 Gateway2건 포함). 이번 HTTP 실행의 GET130/POST446/PATCH29/PUT2/DELETE1 요청에서 예상 상태 코드와 응답을 검사했다. 인증·입력 오류 시험의 4xx는 예상 결과이며 500은0건이다. 이번 실행 이후 증가한 서버 로그의 ERROR/Traceback/unhandled는0건, 실제 연동 브라우저의 console.error/pageerror/예상하지 않은 요청 실패도0건이다. 데스크톱·모바일 화면도 확인했다.

소스 변경 도중 첫 실행에서 이전 models import와 새 pipeline이 섞여4건 실패했다. 파일 변경을 완료한 뒤 전체155건을 통과했고, 검출 후처리 비교와 경로 검증을 추가한157건, 이어 경계 NMS6건과 보관 도구17건을 포함한 최종180건을 통과했다. 첫 실패 원인은 이 기록에 남기며 로컬 최종 로그는 마지막 실행 결과다. 새 합성 모델/평가 수식 검사는 수치·좌표·보관 계약을 검증하며 도로 정확도 근거가 아니다.

실제 GPU 학습·공개 자료 평가 결과는 [모델 학습 기록](MODEL-TRAINING.md)과 `reports/MODEL-TRAINING-SUMMARY.json`, 검출기 수식 검사9건은 `reports/DETECTION-TRAINING-CONTRACT.json`에 구분한다. [Ubuntu CI 41684055596](https://github.com/GE-Project-KMU/globalProject/actions/runs/41684055596)도 Python/HTTP/영상·PostGIS/HTTP·build/Chromium 세 job을 통과했다. GPU 학습은 Windows에서 수행했으며 Ubuntu GPU 학습 실행 결과는 없다.

로컬 증거: `data/training-backend-junit.xml`, `data/training-backend-test.log`, `data/training-http-log-check.json`, `frontend/test-results/revalidation-summary.json`. 정제된 합계는 `reports/TRAINING-REGRESSION.json`에 보관한다. 아래11월3일 기록은 변경 전의 검증 이력이다.

2026-11-03, Windows에서 프로젝트 내부 가상환경·캐시·합성 자료로 검증했다. 실물 주행 정확도 또는 운영 적합성 인증 결과가 아니다.

| 검증 | 결과 | 범위 |
| --- | --- | --- |
| 실제 Gateway HTTP | 53/53 통과 | 인증·권한·기관 격리, 관리·설정·분석 작업·엄격한 UUID/JSON/true·요청 ID·오류 envelope |
| 개인정보·전송 HTTP | 15/15 통과 | 열람 승인/다운로드·불변 정정·거절·삭제·보고서·100건 3회 전송·4중 이미지 업로드·다른 기관 파일 덮어쓰기 차단 |
| 운영·보관·HTTPS | 6/6 통과 | 실제 TLS/인증서·로그인·백업 안전성·물리 TTL·과거 검토 사유 기한 |
| 스키마·레거시 | 3/3 통과 | 멱등 마이그레이션·기존 자료 보존·잘못된 설정 격리·응답 검증 로그 입력값 차단 |
| 로컬 영상 도구 | 54/54 통과 | 메모리 MP4/AES-GCM·GNSS·보정·고유 구간·검수/경로/전송·모델·학습 자료 TTL/철회 |
| Python 통합 합계 | **131/131 통과** | 41.00초, 실패/오류/skip 0 |
| Chromium Playwright | **20/20 통과** | UI 독립 시험 18건, 실제 Gateway 연동 2건, 데스크톱·모바일 |
| TypeScript / Vite | 통과 | 타입 검사 및 production build |
| 최종 HTTP 실행 중 서버 로그 | 예외 0건 | 해당 실행에서 추가된 로그의 Traceback/ERROR/unhandled/HTTP 500 검사 |

HTTP 시험은 실제 TCP 요청과 상태 코드·응답 본문을 검사했다. 4개 동시 동일 관측의 멱등 수신, 동시 검토의 200/409 충돌, If-Match 428/409/200, PDF 한글 본문 추출, 공유 이미지 삭제의 관련 관측 전파, 삭제 후 410과 재수신 차단을 포함한다.

100개 관측을 실제 HTTP로 3회 전송해 ID/점검 건 보존과 유실·추가 생성 0건을 확인했다. 동시에 같은 사진을 4번 올리면 201 한 건·200 세 건이며 승인된 실제 바이트가 유지된다. 다른 기관이 같은 파일 UUID와 별도 유효 승인을 사용해도 기존 사진을 덮어쓸 수 없음을 검사했다. 열람은 관리자 승인 후 제한 다운로드, 정정은 원본 보존+부가 기록, 삭제는 주행 경로·연결 이미지·파생 보고서 정리 후 검증까지 시험했다.

브라우저에서는 로그인·보호 이미지·상세 보기·검토 PATCH·서비스 상태·CSV 생성/다운로드, 계정/기기·설정·주행·감사·분석 작업·개인정보 승인과 데스크톱/모바일 배치를 확인했다. 포트홀 박스·도막 RLE/외곽, 날짜/bbox 필터·보고서 조건·종류별 차트도 검사했다. 정상 시나리오의 console.error/pageerror, 예상하지 않은 요청 실패와 API 4xx/5xx는 없었다. 최종 점검에서 발견한 이전 버전의 잘못된 프로파일 목록 500을 실제 자료 격리로 해결하고 재검증했다. 이전 오류 로그는 지우지 않고 최종 실행의 추가 구간을 별도로 검사했다.

로컬 증거는 Git 제외 경로 `data/backend-junit.xml`, `data/final-backend-summary.json`, `data/http-log-check.json`, `frontend/test-results/`에 있다. 실제 연동 화면은 `frontend/test-results/live-live-gateway-login-in-2c3f8-ealth-and-responsive-layout/`의 desktop/mobile/service-status PNG 및 `live-live-management-opera-9fbbe-cy-access-use-real-services/`의 관리 화면 PNG에 보관한다. 자격 증명·이미지·원본 로그는 CI artifact로 업로드하지 않는다.

## PostgreSQL/PostGIS 별도 실제 실행

공식 Windows PostgreSQL 16.15·PostGIS 3.6.2 바이너리를 `.cache/postgres-probe`에서 비관리자 계정으로 실행했다. 5개 독립 DB JSONB·마이그레이션1/2, inspection 마이그레이션3/4·공간 점·거리·GiST를 확인했다. 실제 inspection HTTP 18103에서 bbox 포함/제외·위도 경계/전 세계 범위·기관 분리404·관측 삭제 시 공간 점 제거를 통과했다. 기존 SQLite 서비스를 바꾸지 않았고 별도 클러스터는 시험 후 종료했다. 증거는 `data/postgres-probe-results.json`, 재현은 `scripts/postgres_probe.py`와 ADR-002를 참고한다.

## 재실행

프로젝트 루트에서 다음을 실행한다. Windows 명령이며 Ubuntu에서는 `.venv/bin/python`을 사용한다. Playwright 브라우저와 임시 경로는 운영 가이드에 따라 프로젝트 내부로 지정한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-vision.txt
.\.venv\Scripts\python.exe scripts/manage.py start --frontend
.\.venv\Scripts\python.exe scripts/seed_demo.py
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=data/backend-junit.xml
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run test:e2e
```

GitHub Actions는 Ubuntu Python 3.14 통합 시험, 실제 PostGIS 공간 조회, Node 24 build/실제 연동 UI의 세 job으로 구성한다. 외부 ONNX 참고 모델은 고정 revision/SHA256으로만 내려받아 실행 시험하며 정확도 합격으로 해석하지 않는다. 원격 CI 실행 결과는 아래 확인 기록에 추가한다.

[Ubuntu 검증 실행 41625044328](https://github.com/GE-Project-KMU/globalProject/actions/runs/41625044328)은 커밋 `bf4026b`의 세 job을 모두 통과했다. 통합 Python 시험과 TypeScript/build/실제 연동 브라우저, PostgreSQL/PostGIS 공간 조회를 각각 별도 job에서 실행했다. 실물 카메라·GPU·독립 현장 정확도 시험과는 구분한다.

## 남은 검증과 구현 제한

- ZED 이름의 연결 장치를 현재 호스트에서 탐지하지 못했다. 실제 ZED 2i SDK 캡처, 장착·진동·조도·주행 속도별 검증, GPU 추론은 수행하지 않았다.
- 포트홀 Nano 및 차선 paint·얼굴/번호판 비식별화의 승인된 학습 가중치와 독립 현장 GT가 없다. 학습 진입점과 ONNX 어댑터를 구현했으나 학습·정확도 합격은 주장하지 않는다.
- 고정 revision·SHA256의 외부 YOLOX-S 참고 모델을 내려받아 합성 영상 한 장에서 CPU 실행을 확인했다. 이는 목표 모델과 다르며 게시자 mAP50 72.1%도 목표 80%에 미달한다. 실제 도메인 성능 증거가 아니다. 출처와 사용법은 VISION.md를 참고한다.
- Docker 실행 환경이 없어 Compose 자체 실행은 미검증이다. SQLite 전체 통합과 네이티브 PostgreSQL/PostGIS의 별도 실제 검증을 구분한다. 저장 계층은 서비스별 JSON aggregate/낙관적 version과 생성 공간 열을 사용하며 원래 단일 DB 정규 스키마를 그대로 구현한 것은 아니다.
- PDF는 Windows 이식성을 위해 ReportLab을 사용한다. 설계서의 WeasyPrint 선택과 다르며 한글 추출은 검증했다.
- VWorld 키가 없으면 지도는 배경지도 없음 안내와 좌표 격자를 표시한다. 기관의 실제 지도 키와 모델/보정·수집 근거 준비가 필요하다.
- 개인정보·TTL·설정 전용 복구와 개발 HTTPS는 실제 시험했다. 외부 기관 운영 전 기관 TLS·키 운영, 별도로 도입하는 상세 백업의 삭제 전파, 대규모 처리량·5명 사용자 과제·실제 카메라/모델 수용시험이 필요하다. 현재 도구는 상세자료 백업을 만들지 않는다.
