# 구현 API 추가 계약

2026-11-03. 원본 API 명세 스냅샷의 32개 작업에 아래 운영·웹 조회 계약을 추가한다. 인증·기관 경계·오류 envelope는 원본 계약과 같다. `/openapi.json`은 관리자 세션으로 조회하며 서비스별 요청·응답 모델과 바이너리 다운로드 형식을 합친다.

| 경로 | 권한 | 응답/조건 |
|---|---|---|
| GET `/api/v1/users` | admin | data.items: 공개 사용자 id/login/role/active |
| GET `/api/v1/devices` | admin/reviewer | data.items; credential_id·원문 토큰 제외 |
| GET `/api/v1/profiles` | 같은 기관 사용자/기기 | data.items; 등록 후 불변 보정 manifest |
| GET `/api/v1/model-bundles` | 같은 기관 사용자/기기 | data.items; manifest 등록은 실제 가중치 검증과 구분 |
| GET `/api/v1/runs` | admin/reviewer | data.items, next_cursor:null; status 선택 필터, limit 1~1000; 차단·만료 자료 제외 |
| GET `/api/v1/audit-logs` | admin | data.items, next_cursor:null; limit 1~1000; 서비스별 내부 서명 HTTP로 집계 |
| GET `/api/v1/summary` | admin/reviewer/viewer | 전체 집계; from 포함/to 미포함 선택 기간; UTC 일자별 trend와 종류별 trend_by_type |
| GET `/api/v1/system/status` | 사용자 | data.services의 name/status; 도메인 헬스 확인 |
| GET `/api/v1/analysis/capabilities` | 같은 기관 사용자 | 로컬 원본 처리·작업 API·모델별 검증 방식을 안내 |
| POST `/api/v1/analysis/jobs` | admin/reviewer | {run_id,model_bundle_id,prepared_config_id};202 queued |
| GET `/api/v1/analysis/jobs` | admin/reviewer | data.items; 같은 기관 작업만 |
| GET `/api/v1/analysis/jobs/{id}` | admin/reviewer | 같은 기관 작업 상태·시각·관측 수·안전한 오류 코드 |

감사 항목은 id,action,actor_hash,target_hash,at,service만 반환한다. 자유 서술 사유·로그인명·파일 위치·토큰은 반환하지 않는다. 한 서비스가 응답하지 않으면 전체 감사 목록을 성공한 완전한 목록으로 표시하지 않고 의존성 오류를 반환한다.

trend/trend_by_type 항목은 `{date,candidates,confirmed,completed}`다. 사례별 대표 관측 일자에 현재 상태를 묶은 집계이며 상태 변경 발생 일별 통계가 아니다. confirmed는 confirmed/action_planned/completed를 포함한다. 종류별 배열은 pothole과 lane_segment를 구분하고 서로 다른 계수 단위를 하나의 정밀도 지표로 해석하지 않는다.

차선은 선택 `paint_mask_rle:{size:[height,width],counts:[...]}`와 `outline_polygon_px:[[x,y],...]`를 제공할 수 있다. RLE은 최대512×512, counts 최대8192, 비음수 정수, 0부터 시작하는 이진 교대 run, C 행 우선 순서이며 합계가 픽셀 수와 일치해야 한다. outline은 3~64점이며 원본 이미지 안쪽이어야 한다. 이 필드를 생략하거나 null로 보내도 기존 관측의 멱등 해시가 변하지 않는다.

분석 작업은 운영자가 프로젝트 내부에 준비한 설정 ID만 받는다. 누락404, 기관/주행/번들/보정 불일치422, 동일 주행 진행 중 작업409, 권한403이다. 실제 agent workflow를 실행하고 실제 결과만 completed로 기록한다. 가중치 준비 부족은 blocked/MODEL_NOT_READY이며 오류는 failed/WORKFLOW_FAILED 또는 WORKFLOW_TIMEOUT이다. 재시작 중단은 failed/SERVICE_RESTARTED다. 원본 영상과 검수 전 결과를 공개 API로 업로드하거나 자동 공유 승인하지 않는다.

설치 보정 생성 시 calibrated_by는 인증 사용자 ID로 서버가 기록한다. 보정·번들 생성/목록/상세는 created_at UTC 시각과 공개 ID를 반환한다. 새 번들은 code_commit과 environment_hash를 반드시 포함한다. config_hash는 승인된 고정 manifest의 선언 해시이며 실제 바이너리 검증과 구분한다. 운영자는 정규화한 설정·코드·환경·가중치 해시를 함께 잠그고 변경 시 새 버전을 등록한다.

초기 검증 버그로 잘못 저장된 레거시 설정은 typed 공개 응답을 검사하여 격리한다. 문자열이 아닌 name을 임의 문자열로 바꾸거나 원본 값을 덮어쓰지 않는다. 목록은 정상 items와 quarantined_count를 반환하며 잘못된 설정 하나로 전체 목록을500 처리하지 않는다. 격리 ID 상세는410 INVALID_LEGACY_PROFILE 또는 INVALID_LEGACY_BUNDLE이다. 관리자가 새 유효 버전을 등록해야 하며 격리된 설정의 원문은 생성 후90일까지, 최소 감사·ID tombstone은 격리 후180일까지 보관한다. 운영 분석은 격리 설정을 선택할 수 없다.
