# 격리 운영 진단

2026-11-04 Windows native 실행에서 프로젝트 내부 PostgreSQL16.15/PostGIS3.6에 서비스별 새 DB·비관리자 DB 계정5개, 별도 포트의 MSA6개를 생성했다. 기존 서비스 PID·DB·원자료는 변경하지 않았다. [실행 집계](reports/OPERATIONAL-PROBE-RESULTS.json)는 합성 자료만 포함한다.

4개 동시 요청으로 관측 목록·통계·주행·감사 조회160건이 모두200이었다. 이번 실행 median163.4ms, p95 189.2ms, 최대215.7ms다. 각 요청은 소량의 합성 관측1개만 있는 격리 DB를 사용했다. 이는 수용량·장시간 안정성·실제 현장 성능·5명 사용자 사용성 시험을 입증하지 않는다.

실제 HTTP에서 익명401, 기기 토큰의 관리자 기능403, 다른 기관 자료404, 잘못된 UUID422, 잘못된 Method405 및 오류 envelope/request ID를 확인했다. 서비스별 migration5개와 다른 서비스 계정의 테이블 읽기 권한 거부도 실제 PostgreSQL로 확인했다. localhost의 개발용 trust 인증 클러스터를 사용했으므로 배포 DB 비밀번호/TLS/네트워크 보안 검증을 대신하지 않는다.

보고서 작업을 생성·poll·CSV 다운로드한 뒤, 해당 진단의 reporting→media 삭제 요청에만503을1회 주입했다. 삭제 작업이 실패해도 조회 차단은 유지되었고, 의존성을 복구한 뒤 관리자의 명시적 retry로 완료되었다. 관측·이미지는410, 무효화된 보고서는409로 다운로드를 거부했다. 자동 재시도 성공이나 모든 실패 유형을 시험했다는 뜻이 아니다. 격리 서비스 로그의 처리되지 않은 오류/Traceback은0이었다.

설정 백업11개를 해시 검증하고 빈 임시 프로젝트에 실제 적용 복원하여 원본 바이트와 일치를 확인했다. 실제 작업 트리 설정은 교체하지 않았다. `.env`·키·상세 DB·사진·토큰은 백업하지 않았고 상세 관측 복원은0이다. 현재 상세 데이터 백업·복원은 지원하지 않으므로 삭제 전파를 포함한 상세 DB 복구 성공을 주장하지 않는다.

## 재현

프로젝트 내부 portable PostgreSQL 클러스터 또는 CI의 격리 PostGIS 서비스를 localhost:55432에서 실행한 후 아래 명령을 사용한다. Windows에서는 DB PID 실행파일과 실제 data_directory가 `.cache/postgres-probe` 안인지 검사한다. PostgreSQL 시작·중단은 해당 운영자가 자신이 시작한 클러스터에 대해서만 수행한다. 도구는 기존 포트가 사용 중이면 거부하며 서비스별 새 DB/계정만 생성한다.

```powershell
.\.venv\Scripts\python.exe scripts/operational_probe.py
.\.venv\Scripts\python.exe -m pytest tests/test_operational_probe.py -q
```

도구는 자신이 실행한 서비스6개만 종료한다. 진단 DB/계정과 로컬 로그는 점검용으로 보존하며 자동 삭제하지 않는다. 출력은 집계만 포함하고 기관·사용자 ID, 로그인 비밀, 원본 응답을 공개하지 않는다. 반복 실행은 별도 DB를 추가하므로 운영자가 로컬 보존량을 관리해야 한다.

Docker 실행파일이 현재 Windows 환경에 없어 Docker/Compose build·실행은 수행하지 않았다. Ubuntu CI PostgreSQL/PostGIS job에 같은 격리 운영 진단 단계를 추가했다. CI 설정 추가와 실제 CI 성공은 다르며 해당 revision의 GitHub Actions 결과를 따로 확인한다. 전역 Docker·DB 서비스·드라이버 설치는 수행하지 않았다.
