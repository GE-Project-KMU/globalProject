# ADR-002: 서비스별 스키마 버전과 공간 조회

날짜: 2026-11-03

기존 JSON 집합체를 유지하면서 서비스마다 독립적인 `schema_migrations`를 적용한다. 버전 1은 기존 집합체 테이블을 인수하고, 버전 2는 기관·종류·생성/만료 시각 복합 인덱스를 추가한다. 시작 시와 `python -m common.migrations inspection` 같은 운영 명령에서 같은 멱등 마이그레이션을 실행한다. PostgreSQL은 서비스별 advisory lock으로 동시 적용을 직렬화한다. 기존 행을 삭제하거나 다른 서비스 DB를 조회하지 않는다.

inspection PostgreSQL에서 `ROAD_POSTGIS_ENABLED=1`이면 버전 3이 PostGIS 확장, WGS84 차량 관측 위치의 generated geography 열과 GiST 인덱스를 만든다. 이 공간 열은 원본 관측 JSONB에서 계산되므로 독립 복제본 동기화가 필요 없고 관측 삭제와 함께 제거된다. bbox 조회는 공간 인덱스를 사용하고 정확한 경계 조건은 기존 검증도 유지한다. 위치가 없는 관측에는 공간 점을 만들지 않는다. 이는 손상물 정밀 좌표가 아니라 차량 관측 좌표다.

버전 4는 경도·위도 사각형 bbox의 geometry 캐스트에 별도 GiST 인덱스를 추가한다. bbox는 위도·경도 경계를 포함하는 평면 사각형 의미로 조회한다. geography 다각형의 대권 경계는 같은 위도의 중간 지점이나 전 세계 bbox를 누락할 수 있으므로 bbox 필터에 사용하지 않는다. 실제 PostgreSQL HTTP 시험에 위도 경계와 전 세계 bbox를 포함한다.

참조 Compose는 inspection에 `postgis/postgis:16-3.6`, 다른 서비스에 PostgreSQL 16을 사용한다. [공식 이미지 지원 버전](https://github.com/postgis/docker-postgis)에서 PostgreSQL 16 / PostGIS 3.6 조합을 확인했다. PostGIS 확장 설치 권한이 없는 운영 계정은 관리자가 해당 DB에 확장을 사전 설치해야 한다. 기존 PostgreSQL 데이터 디렉터리의 이미지 태그를 바꾸기 전 백업/복구 절차를 검증한다.

로컬 SQLite는 동일 공개 API와 서비스별 저장 경계를 제공하며 버전 1·2를 적용한다. SQLite 연결마다 `secure_delete=ON`을 설정한다. PostgreSQL 물리 저장소/백업에서의 완전 삭제는 운영 보관·백업 만료 정책과 별도로 검증해야 한다.

이 결정은 SDD의 단일 DB 정규 스키마를 그대로 구현했다는 뜻이 아니다. 각 도메인의 집합체와 낙관적 version을 정본으로 삼고, 실제 운영 부하에 따라 서비스 내부 정규 테이블로 이행할 수 있다.

현재 호스트에는 Docker가 없어 Compose 실행은 검증하지 못했다. 대신 [PostgreSQL 공식 Windows 다운로드가 연결하는 EDB 바이너리](https://www.enterprisedb.com/download-postgresql-binaries) PostgreSQL 16.15(332,441,502bytes)와 [PostGIS 공식 Windows 바이너리](https://postgis.net/documentation/getting_started/install_windows/released_versions/) PostGIS 3.6.2(123,957,610bytes)를 프로젝트 `.cache/postgres-probe` 안에 내려받아 비관리자 계정으로 실행했다. 전역 설치·Windows 서비스 등록 없이 localhost:55432의 별도 클러스터를 사용하고 검증 후 종료했다.

실제 검증: 5개 독립 PostgreSQL DB의 JSONB와 버전1·2 이전, inspection 버전3·4·PostGIS 점 생성·거리 계산·GiST 인덱스, 별도 inspection HTTP 서버(18103)의 bbox 포함200/제외빈목록·위도 경계/전 세계 포함·기관 분리404·관측 삭제 후 공간 점 제거. 기존 SQLite 서비스는 변경하지 않았다. 재현 코드는 `scripts/postgres_probe.py`, 상세 실행 증거는 무시되는 `data/postgres-probe-results.json`이다. 이는 컨테이너 네트워킹·볼륨·전체 PostgreSQL 운영 부하를 검증한 결과와 구분한다.
