# RoadWatch MSA 실행·검증 운영 가이드

이 저장소의 파일·캐시·가상환경·로그·데이터는 프로젝트 디렉터리 안에 둔다. 아래 명령은 `C:\Users\user\proj\globalProject` 또는 Ubuntu에서 복사한 동일 프로젝트 루트에서 실행한다. 원본 영상·암호·실제 개인정보를 Git에 커밋하지 않는다.

## 서비스 경계

| 서비스 | 로컬 포트 | 책임 | 소유 저장소 |
| --- | --- | --- | --- |
| identity | 8101 | 기관·사용자·세션·기기 | identity DB |
| media | 8102 | 사진 공유 승인·무결성·비공개 파일 | media DB와 파일 볼륨 |
| inspection | 8103 | 주행·관측·점검 건·검토·중복 | inspection DB |
| analysis | 8104 | 프로파일·모델 번들·분석 가능 상태 | analysis DB |
| reporting | 8105 | 보고서 작업·개인정보 처리 조정 | reporting DB |
| gateway | 8000 | Bearer 검증·내부 서비스 라우팅 | 도메인 DB 없음 |
| frontend | 5173 / Compose 8080 | 웹 UI | 브라우저 메모리 세션 |

도메인 서비스는 서로의 DB 파일/테이블을 열지 않는다. API Gateway만 브라우저에 공개하고, 서비스 간 API 호출에는 만료·대상 서비스가 있는 서명된 내부 principal을 전달한다. 외부에서 보낸 내부 principal 헤더를 신뢰하면 안 된다.

개발 기본 저장소는 서비스별 SQLite `data/<service>/service.db`다. `IDENTITY_DATABASE_URL`, `MEDIA_DATABASE_URL`, `INSPECTION_DATABASE_URL`, `ANALYSIS_DATABASE_URL`, `REPORTING_DATABASE_URL`을 각각 지정하면 독립 PostgreSQL 저장소를 사용한다. 이는 같은 DB를 나눠 읽는 구조가 아니다.

## Windows 로컬 실행

```powershell
Set-Location C:\Users\user\proj\globalProject
$env:TEMP = "$PWD\.cache\tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
$env:npm_config_cache = "$PWD\.cache\npm"
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\.cache\playwright"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
.\.venv\Scripts\python.exe scripts\manage.py start --frontend
```

실행 중인 서비스가 있으면 새로 생성하지 않는다. `status`로 확인하고 프로젝트 매니저가 시작한 프로세스만 `stop`한다. PID와 생성 시각을 비교하므로 이미 종료된 PID가 다른 프로그램에 재할당되어도 종료하지 않는다. Windows 자식 프로세스는 숨겨진 창으로 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\manage.py status
.\.venv\Scripts\python.exe scripts\manage.py stop
```

게이트웨이: `http://127.0.0.1:8000`, 개발 UI: `http://127.0.0.1:5173`. 최초 기관 코드는 `demo`, 로그인은 `admin@road.local`이다. 암호는 프로젝트 `.env`의 `ROAD_BOOTSTRAP_PASSWORD`에 자동 생성한다. 매니저는 암호·토큰을 콘솔에 출력하지 않는다. `.env`는 외부 공유하지 않는다.

실행 로그는 `data/logs/<service>.log`, 소유 프로세스 상태는 `data/processes.json`이다. 로그인 토큰·암호·전체 이미지/원본 데이터는 로그에 넣지 않는다.

## Ubuntu 로컬 실행

지원되는 Python·Node·시스템 라이브러리가 설치된 Ubuntu에서 프로젝트를 복사하고 프로젝트 내부 가상환경을 새로 생성한다. Windows `.venv`와 `node_modules`를 그대로 복사하지 않는다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend ci
.venv/bin/python scripts/manage.py start --frontend
.venv/bin/python scripts/manage.py status
.venv/bin/python scripts/manage.py stop
```

실제 ZED SDK/CUDA 설치는 장치 드라이버와 시스템 영역을 변경할 수 있다. 현재 프로젝트 디렉터리만 허용된 작업에서는 그런 전역 설치를 수행하지 않는다. 카메라·GPU·학습 가중치 준비 상태는 소프트웨어 HTTP 시험 결과와 별도로 확인한다.

## Docker Compose 참조 배포

현재 개발 호스트에 Docker 실행 파일이 없으면 이 구성의 실제 빌드·실행은 검증하지 못한 것이다. YAML 구문 검토와 Docker 런타임 검증을 구분한다. Docker Desktop/Engine 설치나 전역 저장소 변경도 현재 프로젝트 내부 파일 작업과 별도다.

```sh
python scripts/manage.py env
docker compose config
docker compose up --build -d
docker compose ps
docker compose logs --tail=100
docker compose down
```

Compose는 서비스별 PostgreSQL 16 컨테이너와 별도 데이터 경로를 사용한다. DB 포트와 도메인 API 포트는 호스트에 공개하지 않고, 게이트웨이 8000/UI 8080을 localhost에만 바인딩한다. 서비스 파일·DB 데이터는 프로젝트 `data/containers`와 `data/postgres` 아래에 둔다. `down -v`는 필요 없이 실행하지 않는다.

Linux bind mount에서는 `ROAD_CONTAINER_UID`, `ROAD_CONTAINER_GID`를 실제 프로젝트 소유자 UID/GID에 맞춘다. 기본값은 1000이다. 컨테이너는 비root 애플리케이션 사용자로 실행한다. `.env`에 서비스별 DB 암호가 필요하며 `manage.py env`가 새 프로젝트 설정을 만들 때 생성한다.

이 Compose 파일을 검증 없이 인터넷 운영 환경에 노출하지 않는다. 실제 기관 운영에는 TLS 종료·신뢰 프록시 설정·서비스별 비밀키/회전·권한별 DB 계정·백업/복원/삭제 전파·작업자 운영과 모니터링을 확인한다. 서명된 내부 요청은 암호화 전송이나 네트워크 격리를 대체하지 않는다.

## HTTP·오류 테스트

먼저 로컬 MSA 서비스를 실행한다. 테스트는 실제 TCP/HTTP로 게이트웨이에 연결하고, 테스트용 프로파일·기기·합성 JPEG·관측을 생성한다. 실물 카메라를 썼다거나 포트홀 모델 정확도를 측정했다는 뜻이 아니다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_http.py -v
```

주요 시험: 인증/권한·기관 분리·이미지 해시/승인·포트홀/차선 스키마·null 판정 불가·경계값·멱등 재수신/본문 충돌·상태 version 충돌·계정/토큰 회수·보고서 작업/다운로드·크기 제한이다. 테스트용 `isolation` 기관은 identity가 제공하는 관리 bootstrap 명령으로만 생성한다.

```powershell
Get-ChildItem data\logs\*.log | Select-String -Pattern 'Traceback|ERROR|unhandled|INTERNAL_ERROR'
```

예상한 4xx 검증 실패와 서버의 처리되지 않은 예외(5xx/Traceback)를 구분한다. 테스트 결과와 로그 검사 결과를 함께 기록한다. UI 테스트는 `frontend`의 Playwright 설정을 사용하며 브라우저 파일도 `.cache/playwright`에 저장한다. 실제 실행 결과는 프로젝트 테스트 결과 보고서에서 확인한다.

## 보관·삭제 주의점

자체 원본/식별 가능 중간 파일은 촬영 후 7일, 마스킹 점검 상세는 촬영 후 90일, 최소 감사로그는 행위 후 180일이라는 프로젝트 기본 정책을 적용한다. 재시도·병합으로 촬영 자료 기한을 연장하지 않는다. 파일 삭제·DB 삭제·파생 보고서·복구 사본까지 검증되기 전 개인정보 삭제 요청을 완료로 처리하지 않는다.

실제 기관의 승인·데이터 사용조건·배포 보안 점검·실물 장착/주행 시험은 이 소프트웨어 실행 가이드만으로 완료되지 않는다.
