# 구현 API 추가 계약

2026-11-03. 원본 API 명세 스냅샷의 32개 작업에 아래 운영·웹 조회 계약을 추가한다. 인증·기관 경계·오류 envelope는 원본 계약과 같다. `/openapi.json`은 관리자 세션으로 조회하며 서비스별 요청·응답 모델과 바이너리 다운로드 형식을 합친다.

2026-11-04 추가: GET `/api/v1/observations`, `/api/v1/summary`, `/api/v1/reports`는 선택 query `data_scope=all|presentation|test|synthetic|field`를 지원한다. API 기본은 기존과 같은 `all`이고 UI 기본 `presentation`은 명백한 HTTP 시험만 제외한다. 잘못된 값은422이며 권한·기관 경계는 그대로다. 목록 cursor는 범위와 출처 정책 snapshot hash에 묶고 보고서 캐시도 범위를 구분한다. 정책 변경이나 다른 범위의 cursor를 재사용할 수 없다.

관측 응답의 추가 `data_origin`은 `http_test`(교차 확인된 이전 HTTP 시험 또는 명시적 합성 HTTP fixture), `synthetic_demo`(선언된 합성 시연), `field`(그 외 출처 미확인 일반 관측)이다. `field`를 실제 카메라 촬영 보증으로 해석하지 않는다. 원 관측의 payload·demo·리뷰를 분류 목적으로 바꾸지 않으며 CSV/PDF에도 같은 출처를 표시한다. [시연 자료 분리](DEMO-DATA-ISOLATION.md)에 정책 생성과 화면 사용법을 기록했다.

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

## 개인정보 처리 완료와 보고서

| 경로 | 권한 | 계약 |
|---|---|---|
| GET `/api/v1/privacy-requests` | admin | data.items: request_id/job_id/version/request_type/status/created_at/resolved_at/result_code/target_refs/reason/decision_reason/actions/download_path |
| POST `/api/v1/privacy-requests` | admin | 기존 PrivacyCreate;202 received. 삭제는 조회/재전송 차단 후 정리 작업 예약 |
| GET `/api/v1/privacy-requests/{id}` | admin | 같은 기관 요청의 현재 version/처리 이력. 실제 resolved_at을 고정 보존 |
| PATCH `/api/v1/privacy-requests/{id}` | admin | expected_version≥1,action,reason(1~500),corrections 선택. 오래된 version409 |
| GET `/api/v1/privacy-requests/{id}/download` | admin | 열람 승인된 요청만 인증 JSON; no-store. 최대24시간 및 원자료 만료/삭제/정정에 따른 무효화 |

action은 `approve_access`, `apply_correction`, `reject`, `retry`다. 접근·정정은 작업이 in_progress/관리자 검토 대기 상태일 때 결정하며 완료/거절 요청을 다시 처리하지 않는다. 삭제 실패는 차단을 유지한 채 retry하고 reject로 차단을 풀지 않는다. reason은 내부 티켓 참조와 최소 근거만 포함하며 실제 요청자의 민감정보를 입력하지 않는다.

corrections는 최대100개의 `{observation_id,field,value}`다. field=`location`이면 value는 쌍으로 유효한 lat/lon(null 가능)과 quality(valid/degraded/unavailable), 선택 HDOP/age/accuracy를 가진다. field=`annotation`이면 1~500자의 부가 설명이다. 요청 대상 밖 관측의 정정은422다. 원본 관측을 교체하지 않고 검수자·시각·사유와 별도 정정 기록을 보존한다. 정정/삭제로 영향받은 기존 보고서·제한 열람 사본은 무효화한다.

GET `/reports`는 from/to/type/review_status와 bbox/quality/grade/run_id를 지원한다. `snapshot_at`은 timezone 포함 시각이며 새로운 예약 구분자다. `public_demo=true`일 때만 위치3자리·날짜·단축 해시 ID로 내보내고 검토 자유 서술·정정 상세를 제외한다. CSV는 이를 public_demo 열로, PDF는 안내 문장으로 표시한다. 상세 이동 경로는 두 형식 모두 내보내지 않는다. 상태 조회와 다운로드는 기존 owner/admin·기관 경계를 적용한다.

## 설정 격리와 오류 응답

프로파일/모델 목록의 `quarantined_count`는 현재 계약으로 검증할 수 없어 격리한 이전 설정의 수다. 잘못된 원문을 정상 값으로 바꾸지 않고 원문을 별도 종류로 격리하고 최소 감사/tombstone을 남긴다. 해당 ID를 사용하면410이며 정상 설정을 새 버전으로 등록해야 한다. 정상 목록은 계속200으로 제공한다. 프로파일은 인증된 검수자의 calibrated_by, 두 설정 모두 created_at과 id/유형별 ID를 반환한다. 새 모델 번들의 code_commit/environment_hash는 필수다.

게이트웨이 자체 인증 실패·라우팅 실패·405·요청 검증 오류도 `{error,meta}` envelope를 사용한다. 성공과 실패의 meta.request_id는 X-Request-ID 헤더와 일치한다. 응답 검증 실패는 안전한500을 반환하고 서버 로그에는 loc/type만 남기며 실제 입력/토큰/응답 원문은 남기지 않는다.

설치 보정 생성 시 calibrated_by는 인증 사용자 ID로 서버가 기록한다. 보정·번들 생성/목록/상세는 created_at UTC 시각과 공개 ID를 반환한다. 새 번들은 code_commit과 environment_hash를 반드시 포함한다. config_hash는 승인된 고정 manifest의 선언 해시이며 실제 바이너리 검증과 구분한다. 운영자는 정규화한 설정·코드·환경·가중치 해시를 함께 잠그고 변경 시 새 버전을 등록한다.

초기 검증 버그로 잘못 저장된 레거시 설정은 typed 공개 응답을 검사하여 격리한다. 문자열이 아닌 name을 임의 문자열로 바꾸거나 원본 값을 덮어쓰지 않는다. 목록은 정상 items와 quarantined_count를 반환하며 잘못된 설정 하나로 전체 목록을500 처리하지 않는다. 격리 ID 상세는410 INVALID_LEGACY_PROFILE 또는 INVALID_LEGACY_BUNDLE이다. 관리자가 새 유효 버전을 등록해야 하며 격리된 설정의 원문은 생성 후90일까지, 최소 감사·ID tombstone은 격리 후180일까지 보관한다. 운영 분석은 격리 설정을 선택할 수 없다.
