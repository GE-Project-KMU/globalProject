Here is the result of "fetch" for the Page with URL https://app.notion.com/p/40d8ac7c2dde812ab381cd1b2d45bea2 as of 2026-11-03T09:38:24.031Z:
<page url="https://app.notion.com/p/40d8ac7c2dde812ab381cd1b2d45bea2" icon="🔌">
<ancestor-path>
<parent-page url="https://app.notion.com/p/40d8ac7c2dde801095cdca9e832444f8" title="글로벌공학설계프로젝트"/>
</ancestor-path>
<properties>
{"title":"API 명세서"}
</properties>
<iconMetadata>{"type":"emoji","emoji":"🔌"}</iconMetadata>
<content>
<callout icon="🔌" color="blue_bg">
API 명세서 v1.0 · 작성일: <date start="2026-11-03" /><br />Base path /api/v1 · 관측 payload schema\_version=0.4 · 기획서/SRS v0.4·SDD v1.0 기준.<br />서버·로컬 업로더·웹 구현자가 공유하는 독립 계약이다. 아래 URL·UUID·토큰·해시는 설명용 예시이며 실제 서버 배포나 호출 시험 완료를 뜻하지 않는다.
</callout>

<table-of-contents />

기준 문서: <mention url="https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf" />, <mention url="https://app.notion.com/p/40d8ac7c2dde81f3b7a4f3ae95edb804" />. 1\~2장은 시스템 설계서의 API 기본 계약을 독립 문서로 정리한 것이고 3장 이후는 필드·검증·호출 순서·시험 기준을 구체화한다. 실제 개발용 OpenAPI는 본 계약에 맞춰 구현에서 생성·검증한다.

# 1. API 공통 규약

## 1.1 기본 규약·인증

Base path는 /api/v1, JSON은 UTF-8, 시각은 offset 포함 RFC3339, UUID는 문자열이다. 응답 시간은 UTC Z로 내보내며 UI에서 Asia/Seoul로 변환한다. 조회 기간은 from 포함·to 미포함이다. 필드 이름·enum·요청/응답 스키마는 Pydantic으로 정의하고 OpenAPI 문서를 생성한다.

사용자 인증은 HTTPS의 POST /auth/login으로 발급한 고엔트로피 불투명 Bearer 세션 토큰을 사용한다. 비밀번호는 Argon2id, 세션/기기 토큰은 서버에 해시만 보관한다. 사용자 세션은 30분 만료이며 만료 시 재로그인한다. 웹은 토큰을 메모리에만 보관하고 로그아웃 시 서버에서 폐기한다. 토큰을 URL·일반 로그·CSV·브라우저 영구 저장소에 넣지 않는다. [FastAPI 인증·비밀번호 해시 참고](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)

- admin: 기관 계정/장치/프로파일 관리, reviewer 기능, 개인정보 요청 처리.
- reviewer: 조회·사진 공유 승인·점검 상태/우선순위 변경·중복 승인·보고서.
- viewer: 허용된 기관 결과·비식별 사진·보고서 조회. 원본 접근 권한 없음.
- device: 자기 기기의 run·승인된 이미지·관측 전송, 자기 프로파일/번들 조회만 가능.

org\_id·reviewer\_id는 서버 인증 문맥에서 결정한다. 요청에서 다른 기관을 지정하면 거부한다. 다른 기관의 단일 자원 ID는 404, 같은 기관에서 역할 부족은 403으로 처리한다. /docs·/openapi.json도 시연 외부 공개 시 관리자 인증으로 제한한다.

## 1.2 공통 응답·오류

```json
{
  "data": {
    "observation_id": "10000000-0000-4000-8000-000000000001",
    "case_id": "60000000-0000-4000-8000-000000000001",
    "deduplicated": false
  },
  "meta": {
    "request_id": "req-example",
    "schema_version": "0.4"
  }
}
```

```json
{
  "error": {
    "code": "VERSION_CONFLICT",
    "message": "다른 사용자가 점검 상태를 변경했습니다.",
    "details": {
      "current_version": 4
    }
  },
  "meta": {
    "request_id": "req-example"
  }
}
```

<table header-row="true">
<colgroup>
<col width="160" />
<col width="530" />
<col width="500" />
</colgroup>
<tr>
<td>
HTTP
</td>
<td>
코드 예
</td>
<td>
호출자 동작
</td>
</tr>
<tr>
<td>
200 / 201
</td>
<td>
정상 조회·같은 요청 재수신 / 신규 생성
</td>
<td>
200 deduplicated=true와 최초 ID로 같은 결과 확인
</td>
</tr>
<tr>
<td>
202
</td>
<td>
REPORT\_PENDING / DELETE\_QUEUED
</td>
<td>
job\_id로 진행 조회, 아직 완료로 표시하지 않음
</td>
</tr>
<tr>
<td>
401 / 403 / 404
</td>
<td>
UNAUTHENTICATED / FORBIDDEN / NOT\_FOUND
</td>
<td>
재로그인 또는 권한 확인, 원본/다른 기관 자원 추측 금지
</td>
</tr>
<tr>
<td>
409
</td>
<td>
IDEMPOTENCY\_CONFLICT / VERSION\_CONFLICT / STATE\_CONFLICT
</td>
<td>
동일 ID 변경·동시 수정·상태 불일치, 사용자 확인 전 재시도 중지
</td>
</tr>
<tr>
<td>
410
</td>
<td>
DATA\_EXPIRED
</td>
<td>
보관기한 지난 데이터·삭제 대상 재생성 금지
</td>
</tr>
<tr>
<td>
413 / 415 / 422
</td>
<td>
PAYLOAD\_TOO\_LARGE / UNSUPPORTED\_MEDIA / VALIDATION\_ERROR
</td>
<td>
영구 오류 표시 후 수정. 원래 ID의 payload를 바꿔 재전송하지 않음
</td>
</tr>
<tr>
<td>
428 / 429 / 503
</td>
<td>
VERSION\_REQUIRED / RATE\_LIMITED / SERVICE\_UNAVAILABLE
</td>
<td>
version 누락 보완 / Retry-After / 일시 장애 재시도
</td>
</tr>
</table>

관측 본문은 64,000 bytes, 이미지 파일은 5,000,000 bytes 상한이다. multipart 요청 전체 상한은 메타데이터 오버헤드를 감안해 별도로 설정한다. 이미지 형식은 JPEG/PNG만 허용하고 MIME 선언·magic byte·실제 디코딩·최대 4096×4096 픽셀을 함께 검증한다. 사용자 파일명으로 저장 경로를 만들지 않는다.

목록은 limit 기본 50·최대 200, cursor 방식이다. cursor는 정렬 키·필터 해시·기관·기준 시각을 서버 서명으로 묶는다. 필터가 바뀌면 새 조회로 시작한다. 기본 정렬은 긴급→미확인 포트홀→마모 후보→관찰→최초 관측 시각→ID이며 위치 누락·판정 불가는 별도 품질 필터로 항상 접근 가능하게 한다. 각 요청의 항목·카운트는 같은 읽기 트랜잭션에서 산출한다. cursor의 기준 시각은 새 관측 유입을 제한하며, 페이지 사이의 검토 상태 변경까지 장기 DB 스냅샷으로 보장하지는 않는다. 변경이 감지되면 목록 새로고침을 안내한다.

# 2. API 명세

## 2.1 인증·장치·설정 API

<table header-row="true">
<colgroup>
<col width="280" />
<col width="110" />
<col width="510" />
<col width="560" />
</colgroup>
<tr>
<td>
메서드·경로
</td>
<td>
권한
</td>
<td>
요청
</td>
<td>
정상 응답·검증
</td>
</tr>
<tr>
<td>
POST /auth/login
</td>
<td>
익명
</td>
<td>
\{org\_code,login,password\}
</td>
<td>
200 \{access\_token,token\_type:Bearer,expires\_in:1800,user:\{id,role\}\}. 실패 동일 메시지·호출 제한
</td>
</tr>
<tr>
<td>
POST /auth/logout
</td>
<td>
사용자
</td>
<td>
본문 없음
</td>
<td>
204, 현재 세션 즉시 폐기
</td>
</tr>
<tr>
<td>
GET /me
</td>
<td>
사용자
</td>
<td>
없음
</td>
<td>
200 \{id,org\_id,role,expires\_at\}
</td>
</tr>
<tr>
<td>
POST /devices
</td>
<td>
admin
</td>
<td>
\{name\}
</td>
<td>
201 \{device\_id,device\_token\}. 토큰 최초 1회만 표시
</td>
</tr>
<tr>
<td>
PATCH /devices/\{id\}
</td>
<td>
admin
</td>
<td>
\{active,name?\}
</td>
<td>
200, 비활성 기기 요청 즉시 차단·감사
</td>
</tr>
<tr>
<td>
POST /profiles
</td>
<td>
admin
</td>
<td>
\{name,version,camera\_serial,lens\_label,calibration,config\_hash\}
</td>
<td>
201 \{profile\_id\}; 같은 버전 수정 불가·중복 409
</td>
</tr>
<tr>
<td>
POST /model-bundles
</td>
<td>
admin
</td>
<td>
\{version,weights\_hashes,preprocess,thresholds,config\_hash\}
</td>
<td>
201 \{model\_bundle\_id\}; 실제 가중치 바이너리를 공개 API에 업로드하지 않음
</td>
</tr>
<tr>
<td>
GET /profiles/\{id\}; GET /model-bundles/\{id\}
</td>
<td>
같은 기관 사용자/device
</td>
<td>
경로 UUID
</td>
<td>
200 등록 manifest·해시. device에는 운용에 필요한 필드만 제공
</td>
</tr>
</table>

프로파일의 calibration에는 image\_size, height\_m, pitch\_deg, camera\_matrix, distortion, homography, roi\_polygon\_px, visible\_distance\_m, calibrated\_at·검수자 정보가 필요하다. 행렬 크기·유한성·역변환 가능·ROI 영상 경계를 검증한다. 모델 번들에는 3개 모델 해시와 코드/환경 해시, resize·정규화·thresholds를 포함한다.

## 2.2 주행·이미지 API

<table header-row="true">
<colgroup>
<col width="280" />
<col width="140" />
<col width="500" />
<col width="600" />
</colgroup>
<tr>
<td>
메서드·경로
</td>
<td>
권한
</td>
<td>
요청
</td>
<td>
정상 응답·검증
</td>
</tr>
<tr>
<td>
POST /runs
</td>
<td>
device
</td>
<td>
\{run\_id,device\_id,profile\_id,started\_at,manifest\_hash,precheck\}
</td>
<td>
201 \{run\_id,status:recording\}. 등록 프로파일·device·기관 일치. 동일 ID/동일 고정 시작 정보는 200
</td>
</tr>
<tr>
<td>
PATCH /runs/\{id\}
</td>
<td>
해당 device
</td>
<td>
\{ended\_at,status,manifest\_hash,summary\}
</td>
<td>
200. status=completed/interrupted, 종료≥시작, 관측 시각 포함 확인
</td>
</tr>
<tr>
<td>
PUT /runs/\{id\}/path
</td>
<td>
해당 device
</td>
<td>
\{segments:\[\{coordinates,from,to,quality,visible\_scope\}\],manifest\_hash\}
</td>
<td>
200. 상세 NMEA/원본 영상 미수신. 끊어진 fix는 선분 단위로 저장하며 경로 대체는 manifest 충돌 검사
</td>
</tr>
<tr>
<td>
GET /runs/\{id\}
</td>
<td>
사용자·해당 device
</td>
<td>
UUID
</td>
<td>
200 주행·프로파일·유효 범위·누락/오염·종류별 결과 수
</td>
</tr>
<tr>
<td>
POST /image-approvals
</td>
<td>
reviewer/admin
</td>
<td>
\{file\_id,device\_id,sha256,size\_bytes,mime,captured\_at,reviewed\_at,redaction\_model\_version\}
</td>
<td>
201 \{approval\_id,expires\_at\}. 로그인한 검수자 기록, 최초 전송 유효기간 15분
</td>
</tr>
<tr>
<td>
POST /images
</td>
<td>
device
</td>
<td>
multipart: file + metadata JSON 문자열 \{file\_id,approval\_id,sha256,captured\_at\}
</td>
<td>
201 \{image\_id,state:ready,sha256,expires\_at\}; 같은 ID·해시 ready면 200. 승인·해시·크기·기관·기기 일치
</td>
</tr>
<tr>
<td>
GET /images/\{id\}/content
</td>
<td>
사용자
</td>
<td>
UUID
</td>
<td>
200 비식별 파일 스트림. 매번 권한·기한 검사, Cache-Control:private,no-store
</td>
</tr>
</table>

multipart에서는 file과 함께 metadata라는 폼 필드에 JSON 문자열을 보낸다. 같은 요청에 application/json 본문과 파일이 동시에 온다고 정의하지 않는다. [FastAPI 파일·폼 공식 문서](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/)

사진 검수는 오프라인에서도 가능하며 검수 이력·최종 해시를 로컬에 보존한다. 온라인 전환 후 검수자가 로그인해 해당 파일의 공유 승인을 등록해야 전송한다. 승인 대기/만료는 로컬 blocked 사유로 표시하며 파일이 바뀌지 않은 경우에도 검수자 확인을 거쳐 승인 갱신한다. 이미 ready로 저장된 같은 파일의 멱등 재수신은 원래 승인 만료 뒤에도 허용하되, 촬영 후 90일·삭제 여부를 우선 검사한다.

경로 공백은 UI에서 이어 그리지 않는다. runs.path\_geom은 geometry(MultiLineString,4326)으로 저장하고 path\_segments에 정상 구간별 시각·품질·가시 범위를 연결한다. PUT /runs/\{id\}/path의 최종 본문은 \{segments:\[\{coordinates,from,to,quality,visible\_scope\}\],manifest\_hash\}이며 각 정상 구간을 별도 선으로 그린다. 무효 GNSS 구간을 두 점으로 가로질러 연결하지 않는다.

## 2.3 관측 등록 API

<b>POST /observations · device 권한 · application/json.</b> SRS v0.4 스키마를 따르며 원본 경로·외부 image\_url·임의 기관 ID는 받지 않는다. 아래는 포트홀 예시다.

```json
{
  "schema_version": "0.4",
  "observation_id": "10000000-0000-4000-8000-000000000001",
  "event_id": "20000000-0000-4000-8000-000000000001",
  "run_id": "30000000-0000-4000-8000-000000000001",
  "device_id": "40000000-0000-4000-8000-000000000001",
  "type": "pothole",
  "captured_at": "2026-11-03T01:15:30Z",
  "frame_id": 1234,
  "location": {
    "lat": 37,
    "lon": 127,
    "source": "vehicle_observation",
    "quality": "valid",
    "hdop": 1.2,
    "age_ms": 100,
    "accuracy_m": null
  },
  "profile_id": "50000000-0000-4000-8000-000000000001",
  "model_version": "bundle-0.4.1",
  "config_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "pothole": {
    "bbox_xywh_px": [
      100,
      200,
      160,
      80
    ],
    "confidence": 0.91,
    "track_observations": 5,
    "candidate_state": "multi_frame",
    "occluded": false
  },
  "lane": null,
  "quality": {
    "usable": true,
    "reasons": []
  },
  "image": {
    "file_id": "70000000-0000-4000-8000-000000000001",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "redacted": true,
    "reviewed_for_sharing": true
  },
  "review": {
    "status": "unreviewed",
    "priority": "normal"
  }
}
```

<table header-row="true">
<colgroup>
<col width="260" />
<col width="520" />
<col width="650" />
</colgroup>
<tr>
<td>
필드
</td>
<td>
필수·제약
</td>
<td>
서버 처리
</td>
</tr>
<tr>
<td>
schema\_version / IDs
</td>
<td>
0.4 / UUID, frame\_id≥0
</td>
<td>
같은 org/device/run의 고정 event에 연결. JSON canonicalization 후 SHA-256 request\_hash
</td>
</tr>
<tr>
<td>
profile\_id / model\_version / config\_hash
</td>
<td>
등록 프로파일 UUID·번들 버전·64자리 hex
</td>
<td>
run.profile 및 등록 번들 manifest와 일치. model\_version을 model\_bundle\_id FK로 해석
</td>
</tr>
<tr>
<td>
captured\_at
</td>
<td>
run 구간 내 유효 시각, 초기 시계 허용 오차 ±2초
</td>
<td>
기본 expires\_at=captured\_at+90일. 종료 메타데이터 도착 전에는 주행 시작·서버 시각으로 임시 검증
</td>
</tr>
<tr>
<td>
location
</td>
<td>
lat/lon 모두 값 또는 모두 null, 범위·finite 검사
</td>
<td>
valid/degraded/unavailable, source=vehicle\_observation. 실제 객체 좌표로 표시하지 않음
</td>
</tr>
<tr>
<td>
pothole / lane
</td>
<td>
type에 해당하는 쪽만 object, 다른 쪽 null
</td>
<td>
종류별 Pydantic 모델·금지 필드 검사. bbox가 실제 원영상 밖이면 422
</td>
</tr>
<tr>
<td>
image
</td>
<td>
ready file\_id·SHA-256 필수
</td>
<td>
본문의 redacted/reviewed\_for\_sharing는 호환 필드. 승인 기록·파일 해시·기기·촬영 시각을 별도로 확인
</td>
</tr>
<tr>
<td>
review
</td>
<td>
초기 unreviewed/normal만 허용
</td>
<td>
device가 confirmed·completed·urgent로 생성 불가. 이후 권한 있는 별도 API로만 변경
</td>
</tr>
</table>

응답 201/200 data에는 observation\_id,event\_id,case\_id,case\_version,review,expires\_at,deduplicated를 포함한다. 최초 대표 관측을 만들고 같은 event의 후속 관측은 같은 case에 연결한다. type·run·기기·request\_hash가 충돌하면 409이며 자동 덮어쓰지 않는다.

차선 등록은 type=lane\_segment, pothole=null, lane에 SRS의 segment\_id·estimated\_length\_m·retention\_pct·grade·color·line\_type·valid\_area\_ratio·outline\_supported·patch\_ids·unassessable\_reason을 채운다. 구간 근거 비율 segment\_support\_ratio(산출 가능 시 0\~1, 분모 미정의 시 null)를 lane에 추가해 저장하며 0.8 미만 또는 미정의이면 grade=unassessable·R=null로 강제 검증한다. 이 값은 SRS의 구간 근거 80%를 API에 구체화한 필드다. 인식 종류가 불분명한 경우 unknown과 사유를 사용하고 법정 등급을 전송하지 않는다. valid\_area\_ratio도 외곽 분모 미정의 시 null을 허용하고, 유효 판정에는 두 비율의 산출 근거·임계값 충족이 필요하다.

## 2.4 조회·상태 변경·중복 API

<table header-row="true">
<colgroup>
<col width="320" />
<col width="130" />
<col width="540" />
<col width="600" />
</colgroup>
<tr>
<td>
메서드·경로
</td>
<td>
권한
</td>
<td>
요청·필터
</td>
<td>
정상 응답·검증
</td>
</tr>
<tr>
<td>
GET /observations
</td>
<td>
사용자
</td>
<td>
bbox=west,south,east,north; from,to; type; review\_status; quality; run\_id; grade; limit; cursor
</td>
<td>
200 \{items,next\_cursor,counts\_by\_type\}. bbox는 좌표 있는 것만, quality=unavailable·bbox 동시는 422. 정밀 위치 없는 항목은 별도 조회
</td>
</tr>
<tr>
<td>
GET /observations/\{id\}
</td>
<td>
사용자
</td>
<td>
UUID
</td>
<td>
200 원관측·case 현재 검토 상태·version·비식별 이미지 ID·이력. 만료/삭제는 410
</td>
</tr>
<tr>
<td>
PATCH /observations/\{id\}/review
</td>
<td>
reviewer/admin
</td>
<td>
\{expected\_case\_id,expected\_version,status?,priority?,reason,override\_transition?\}
</td>
<td>
200 \{case\_id,version,review\}. 같은 event·대표 case에 속한 관측들의 표시 상태도 일치. case 이동/버전 충돌 409
</td>
</tr>
<tr>
<td>
GET /cases/\{id\}/history
</td>
<td>
사용자
</td>
<td>
cursor·limit
</td>
<td>
200 검토·우선순위·병합 이력, 원본/토큰/삭제된 사진 미포함
</td>
</tr>
<tr>
<td>
GET /duplicate-links
</td>
<td>
reviewer/admin
</td>
<td>
status;source\_case\_id;limit;cursor
</td>
<td>
200 근접 후보와 거리·사진·진행 방향 근거, 최종 동일성 확률로 표시하지 않음
</td>
</tr>
<tr>
<td>
POST /duplicate-links
</td>
<td>
reviewer/admin
</td>
<td>
\{source\_observation\_id,target\_observation\_id,decision,source\_version,target\_version,reason\}
</td>
<td>
201/200 suggested/approved/rejected. 같은 type·기관·정상 case·버전 검증. 자동 후보는 worker가 suggested만 생성
</td>
</tr>
<tr>
<td>
PATCH /duplicate-links/\{id\}
</td>
<td>
reviewer/admin
</td>
<td>
\{decision:revoked,expected\_version,reason\}
</td>
<td>
200 수동 연결 취소, 원관측·과거 상태 이력 보존. 합병/분리 기록 추가
</td>
</tr>
</table>

bbox는 WGS84 \[경도 최소,위도 최소,경도 최대,위도 최대\]이고 west≤east·south≤north다. 최초 시제품은 날짜 변경선을 가로지르는 bbox를 지원하지 않는다. 전체 카운트와 항목 목록은 같은 기관·필터·스냅샷 기준이며 포트홀 건수와 차선 구간 수를 합산하지 않는다.

상태 API의 status enum은 unreviewed/confirmed/false\_positive/recapture/action\_planned/completed, priority는 urgent/normal/low다. reason은 1\~500자이며 개인식별정보 입력을 피하도록 안내한다. 금지 상태 전이 409, version 누락 428, 다른 기관 ID 404다.

중복 승인은 두 case를 UUID 순으로 잠근 뒤 source의 merged\_into를 대표 target으로 연결한다. 관측/event 원본은 이동·삭제하지 않는다. 순환·자기참조·다른 유형 합치기를 금지하고 이미 별칭이면 대표 case로 정규화한다. 별칭의 과거 상태는 보존하며 이후 화면은 대표 case의 현재 상태를 표시한다. 취소는 특정 source 연결을 복구하고 이력을 추가한다. 완료 후 새 시각의 관측은 새 재발 case로 만들며 단순 근접으로 과거 완료 건에 자동 연결하지 않는다.

## 2.5 보고서·삭제 API

<table header-row="true">
<colgroup>
<col width="280" />
<col width="160" />
<col width="440" />
<col width="640" />
</colgroup>
<tr>
<td>
메서드·경로
</td>
<td>
권한
</td>
<td>
요청
</td>
<td>
정상 응답·검증
</td>
</tr>
<tr>
<td>
GET /reports
</td>
<td>
사용자
</td>
<td>
format=csv/pdf + 관측 조회와 동일 필터, snapshot\_at 선택
</td>
<td>
완성된 같은 요청의 유효 결과가 있으면 200. 없으면 202 \{job\_id,status\_url\}, DB 작업 큐 등록
</td>
</tr>
<tr>
<td>
GET /jobs/\{id\}
</td>
<td>
원 요청자/admin
</td>
<td>
UUID
</td>
<td>
200 \{status,progress,error\_code?,download\_path?,expires\_at?\}. 다른 사용자 작업은 404
</td>
</tr>
<tr>
<td>
GET /reports/\{id\}/download
</td>
<td>
원 요청자/admin
</td>
<td>
job UUID
</td>
<td>
200 파일 스트림, 현재 권한·원자료 TTL 재확인, 캐시 금지
</td>
</tr>
<tr>
<td>
POST /privacy-requests
</td>
<td>
admin
</td>
<td>
\{request\_type,target\_refs,reason\}
</td>
<td>
202 \{request\_id,job\_id\}; 대상 최소 연결정보, 열람/정정/삭제 처리
</td>
</tr>
<tr>
<td>
GET /privacy-requests/\{id\}
</td>
<td>
admin
</td>
<td>
UUID
</td>
<td>
200 처리 상태·조치 기록, 삭제 완료는 파일·DB·파생·백업 정책 결과 확인 뒤 기록
</td>
</tr>
</table>

GET /reports는 보고서 산출 작업만 예약하며 도메인 점검 상태는 변경하지 않는다. 같은 사용자·정규화 필터·형식·템플릿 버전의 진행 중 작업/유효 결과는 같은 작업 키를 재사용해 반복 생성하지 않는다. 최초 예약 트랜잭션에서 대상 ID·검토 version·출력 필드를 최대 10,000건으로 잠근 스냅샷을 저장하고, worker는 이를 출력한다. 새로운 상태로 다시 생성하려면 새 snapshot\_at을 명시한다. 원자료 삭제·만료는 스냅샷에도 즉시 전파한다. 보고서는 최대 10,000개 대표 점검 건, 기간 최대 90일로 제한한다. 초과 시 422로 조건 축소를 안내한다. 작업 결과는 최대 24시간이며 포함 원자료 중 가장 이른 만료시각을 넘기지 않는다.

보고서에는 종류별 수, 관측 범위·위치 품질·판정 불가율·검토/조치 상태·비식별 사진·모델/프로파일 버전을 담는다. CSV의 텍스트 셀은 수식 시작 문자를 무해화하고 숫자 좌표는 숫자 타입으로 출력한다. PDF는 정해진 템플릿과 승인된 내부 파일만 읽으며 사용자 HTML·외부 URL·원본 파일 경로를 받지 않는다.

보고서의 공개 시연 모드는 상세 이동경로를 제외하고 필요한 관측 위치만 적절히 축약한다. 직접 파일 URL을 공개하지 않으며 다운로드할 때도 기관·권한·삭제 상태를 확인한다.

# 3. 필드·관리 API 상세

## 3.1 공통 타입·처리 순서

M=필수, O=선택, N=null 허용이다. M/N 필드는 키를 포함하고 값만 null로 보낼 수 있다. 요청 스키마에 없는 키·중복 JSON 키·NaN/Infinity·문자열로 위장한 boolean/숫자는 422로 거부한다. UUID는 유효한 UUID 문자열, sha256은 소문자 16진수 64자, 시각은 offset/Z 포함 date-time이다. 정수는 0 이상·JSON 안전 정수 범위로 제한한다.

요청 처리 순서는 전송 크기/MIME→인증/기기 활성→스키마→기관/참조·기한→멱등성/동시성→도메인 저장→COMMIT→응답이다. 존재 여부를 이용해 다른 기관 자원 정보를 노출하지 않는다. HTTP 204와 파일 스트림에는 JSON data/meta 래퍼가 없다. validation의 details에는 field\_path·안전한 code만 넣고 비밀번호/토큰·전체 본문을 되돌려주지 않는다.

응답의 request\_id는 서버가 발급하고 X-Request-ID로도 반환한다. Bearer 토큰 종류를 서버에서 구분해 사용자 권한을 device 토큰으로 우회할 수 없도록 한다. 사용자 변경/비활성·로그아웃은 기존 세션에도 적용한다.

## 3.2 계정 관리 구체화

SRS의 사용자 권한 관리를 위한 관리자 계약을 추가한다. 초기 기관과 최초 admin은 운영자 부트스트랩으로 등록하고 공개 회원가입 API는 제공하지 않는다.

| 메서드·경로 | 권한 | 요청 | 응답·제약 |
| --- | --- | --- | --- |
| POST /users | admin | login:string(1\~254), password:string(12\~128), role:admin/reviewer/viewer | 201 data:\{id,login,role,active:true\}. 기관 내 login 중복409, 비밀번호는 응답/로그 미포함 |
| PATCH /users/\{id\} | admin | role?,active?,password? 중 하나 이상 | 200 동일 사용자 공개 정보. 비밀번호/권한/비활성 변경 시 해당 사용자 세션 폐기. 마지막 활성 admin 제거/강등409 |
| POST /devices/\{id\}/credentials | admin | reason:string(1\~500) | 200 data:\{device\_id,device\_token\}. 새 토큰 1회 표시, 이전 토큰 즉시 무효·감사 기록 |

비밀번호의 임의 공백 제거·잘라내기를 하지 않는다. 로그인 실패는 계정 존재 여부를 구분하지 않는 동일 응답으로 처리한다. 계정/토큰 생성·교체 응답이 유실되면 관리자 확인 후 재발급하며 오래된 토큰을 평문으로 조회하는 API를 만들지 않는다.

## 3.3 프로파일·모델 번들

요약 API 표의 필드는 핵심 목록이며 실제 필수 타입은 다음 계약을 적용한다.

| 객체 | 필수 필드·타입 | 검증 |
| --- | --- | --- |
| ProfileCreate | name:string(1\~100),version:int≥1,camera\_serial:string,lens\_label:string,calibration:object,config\_hash:sha256 | name/version 기관 내 유일, 실제 장착 프로파일. 사용 후 불변 |
| calibration | image\_size:\[width,height\],height\_m:number,pitch\_deg:number,camera\_matrix:3×3,distortion:number\[\],homography:3×3,roi\_polygon\_px:좌표쌍\[\],visible\_distance\_m:\[near,far\],calibrated\_at:date-time | 양의 크기·높이·near\<far, finite·유효 행렬·ROI. 검수자 ID는 서버 인증 사용자로 기록 |
| ModelBundleCreate | version:string,weights\_hashes:\{pothole,lane,redaction\}:sha256,preprocess:object,thresholds:object,config\_hash:sha256,code\_commit:string,environment\_hash:sha256 | 동일 버전 수정409. 바이너리/원본·인증 URL 제외 |
| preprocess | pothole:\{input\_size:\[416,416\],resize:letterbox\},lane:\{long\_edge:512\},redaction:\{input\_size:\[416,416\]\},각 normalization·역변환 규약 | 정규화·보간 방식 등 실제 모델 입력 규약을 잠근 manifest에 명시 |
| thresholds | pothole\_confidence:0.35,pothole\_nms\_iou:0.5,single\_frame\_confidence:0.8,lane\_monitor:60,lane\_good:80,min\_valid\_ratio:0.7,min\_segment\_support\_ratio:0.8 | 초기값은 검증셋에서 동결 전 변경 가능. 각 값의 범위·서열 검사 |
| DeviceCreate/Patch | name:string(1\~100) / name?,active?:boolean | org\_id·credential\_hash는 요청 불가. inactive 기기 전송403 |

GET 응답은 ProfileCreate/ModelBundleCreate 공개 manifest에 id,created\_at를 추가한다. 비밀키·해시 전 원문 토큰·서버 파일 경로는 반환하지 않는다. Profile config\_hash는 해당 프로파일 정규 manifest의 해시, 관측 config\_hash는 <b>모델 번들</b>의 설정 해시다. 서로 바꿔 쓰지 않는다.

## 3.4 주행·경로

| 객체 | 필드·타입 | 규칙 |
| --- | --- | --- |
| RunCreate | run\_id,device\_id,profile\_id:UUID;started\_at:date-time;manifest\_hash:sha256;precheck:object | device\_id=인증 기기, profile 같은 기관. 동일 ID의 불변 시작 정보 충돌409 |
| precheck | camera\_preview\_ok,mount\_secure,lens\_clear,profile\_valid,gnss\_device\_connected,storage\_ge\_10gb,notice\_present:boolean | 필수 모두 true. GNSS 장치 연결과 실제 fix 품질은 구분 |
| RunComplete | ended\_at:date-time,status:completed/interrupted,manifest\_hash:최종sha256,summary:object | 영상 관측 범위 포함, 종료≥시작. summary는 captured\_frames,dropped\_frames,quality\_failures,gnss\_gap\_seconds |
| RunPath | segments:\[\{coordinates:\[\[lon,lat\],...\],from,to,quality,visible\_scope\}\],manifest\_hash | 정상 연속 구간별 ≥2점, 경도±180·위도±90·finite, from\<to. 한 요청 총≤1,000좌표·64,000bytes |
| visible\_scope | profile\_id:UUID,roi\_id:string,assessment\_state:assessed/unassessable,reason:string/null | 촬영 경로와 실제 판정 범위를 구분. 지도 궤적을 전 차로 점검 범위로 표시하지 않음 |

POST /runs 응답은 run\_id,status,manifest\_hash와 ETag를 포함한다. PATCH 종료의 If-Match 헤더는 이전 manifest\_hash를 인용부호로 감싼 값이다. 이전 해시가 다르면409, 누락428이며 새 manifest\_hash로 갱신한다. 같은 최종 본문이 이미 적용된 재수신은200으로 처리한다.

경로 PUT의 manifest\_hash는 연결할 주행 manifest 버전이다. 새 경로는 종료 확정 전 전송하고, 같은 경로·같은 버전 재수신은200이다. 이미 완료된 주행의 다른 경로를 자동 덮어쓰지 않는다. 1,000점을 넘으면 서버에 보낼 요약 경로를 생성하되 단절 구간·유효 범위를 보존하고 원시 NMEA를 대신 보내지 않는다. 1장 본문 크기 상한은 이 JSON에도 적용한다.

# 4. 관측·이미지 스키마 상세

## 4.1 공통 관측

| 필드 | 타입·필수 | 제약 |
| --- | --- | --- |
| schema\_version | string M | 0.4 |
| observation\_id,event\_id,run\_id,device\_id,profile\_id | UUID M | run/device/profile 참조 일치. 같은 event는 같은 run/type |
| type | enum M | pothole/lane\_segment |
| captured\_at,frame\_id | date-time M,int M | run 시간 범위·원영상 frame. 종료 확정 전에는 시작/현재 시각으로 임시 검증 |
| model\_version,config\_hash | string M,sha256 M | 등록 번들 version·그 설정 해시와 일치 |
| location | Location M | 아래 규칙 |
| pothole,lane | Pothole N / Lane N | 종류에 해당하는 한쪽만 object, 반대쪽 반드시 null |
| quality | \{usable:boolean,reasons:string\[\]\} M | reasons≤20개·각1\~100자, 불가인 경우 이유 필수 |
| image | \{file\_id:UUID,sha256:sha256,redacted:true,reviewed\_for\_sharing:true\} M | 호환 플래그가 true여도 서버 승인·ready·해시 검증 별도 |
| review | \{status:unreviewed,priority:normal\} M | 기기가 긴급/확인/완료 상태를 생성할 수 없음 |

Location은 lat:number/N,lon:number/N,source:vehicle\_observation,quality:valid/degraded/unavailable,hdop:number/N,age\_ms:int/N,accuracy\_m:number/N을 필수 키로 가진다. lat/lon은 함께 null 또는 유효 좌표다. unavailable은 좌표 null이며 관측을 버리지 않는다. degraded는 좌표가 있을 수 있으나 지도/보고서에 품질 저하를 표시한다. HDOP·age·accuracy는 제공되지 않으면 null이고 음수·비유한 값은422다. HDOP를 미터 정확도로 바꾸지 않는다.

## 4.2 포트홀 객체

bbox\_xywh\_px:\[x,y,w,h\] M(원영상 좌표·w/h>0), confidence:number M(0\~1), track\_observations:int M(≥1), candidate\_state:multi\_frame/single\_frame\_review/held M, occluded:boolean M이다.

multi\_frame은 1초 내 추적 관측≥3, single\_frame\_review는 관측1\~2·confidence≥0.8 조건을 만족해야 한다. 그 밖의 불명확 후보는 held와 quality 사유로 처리한다. 서버는 요약 필드의 일관성을 확인하며 프레임 집계 근거는 로컬 event manifest에서 추적 가능하게 한다. 물리적 깊이·부피·자동 위험도 필드는 받지 않는다.

## 4.3 차선 객체·판정 불가

| 필드 | 타입·필수 | 규칙 |
| --- | --- | --- |
| segment\_id | UUID M | 로컬 고유 구간 |
| estimated\_length\_m | number M/N | 양수, 거리 근거 부족은null |
| color,line\_type | enum M | white/yellow/unknown;solid/dashed/unknown |
| patch\_ids | UUID\[\] M | 중복 없는 고유 패치, 최대200개 |
| outline\_supported | boolean M | 외곽 근거 여부 |
| valid\_area\_ratio | number M/N | 계산 가능하면0\~1. 외곽 분모 자체가 없으면null |
| segment\_support\_ratio | number M/N | 계산 가능하면0\~1. 도색 예정 길이 분모 미정의이면null |
| retention\_pct | number M/N | 산출 가능하면0\~100, 판정 불가면null |
| grade | enum M | visible\_good/monitor/worn\_candidate/unassessable |
| unassessable\_reason | string M/N | unassessable이면1\~200자, 유효 판정이면null |

유효 grade에는 outline\_supported=true, q≥0.70, segment\_support\_ratio≥0.80, 구간 거리/분모 근거, 알려진 색/종류가 필요하다. R≥80=visible\_good, 60≤R\<80=monitor, R\<60=worn\_candidate이며 반올림 전에 판정한다. 필요한 근거가 없으면 unassessable·R=null로 보낸다. 비율이 정의되지 않았다는 뜻의 null을 실제 측정값0으로 대신하지 않는다. 이는 SRS의 “판정 불가”를 필드로 구체화한 허용 규칙이다.

R은 같은 좌표계의 예상 외곽에서 정상 점선 공백·무효 영역을 제외한 도막 면적비다. q는 유효 면적/도색 예정 면적, 구간 근거 비율은 중복 제거한 판정 가능한 도색 예정 길이/전체 도색 예정 길이다. 정의·마스크·분할 근거는 라벨링 가이드를 따른다.

## 4.4 차선 등록 예시

아래는 형식 예시이며 사진 ID·해시는 승인된 실제 파일로 바꿔 사용해야 한다.

```json
{
  "schema_version": "0.4",
  "observation_id": "10000000-0000-4000-8000-000000000002",
  "event_id": "20000000-0000-4000-8000-000000000002",
  "run_id": "30000000-0000-4000-8000-000000000001",
  "device_id": "40000000-0000-4000-8000-000000000001",
  "type": "lane_segment",
  "captured_at": "2026-11-03T01:15:30Z",
  "frame_id": 1234,
  "location": {
    "lat": 37,
    "lon": 127,
    "source": "vehicle_observation",
    "quality": "valid",
    "hdop": 1.2,
    "age_ms": 100,
    "accuracy_m": null
  },
  "profile_id": "50000000-0000-4000-8000-000000000001",
  "model_version": "bundle-0.4.1",
  "config_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "pothole": null,
  "lane": {
    "segment_id": "80000000-0000-4000-8000-000000000001",
    "estimated_length_m": 10,
    "retention_pct": 55,
    "grade": "worn_candidate",
    "color": "white",
    "line_type": "dashed",
    "valid_area_ratio": 0.9,
    "segment_support_ratio": 0.85,
    "outline_supported": true,
    "patch_ids": [
      "90000000-0000-4000-8000-000000000001"
    ],
    "unassessable_reason": null
  },
  "quality": {
    "usable": true,
    "reasons": []
  },
  "image": {
    "file_id": "70000000-0000-4000-8000-000000000001",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "redacted": true,
    "reviewed_for_sharing": true
  },
  "review": {
    "status": "unreviewed",
    "priority": "normal"
  }
}
```

판정 불가일 때도 같은 외부 스키마를 사용하고 lane만 아래처럼 바꾼다. 외곽·길이 분모가 모두 없는 예시다.

```json
{
  "segment_id": "80000000-0000-4000-8000-000000000002",
  "estimated_length_m": null,
  "retention_pct": null,
  "grade": "unassessable",
  "color": "unknown",
  "line_type": "unknown",
  "valid_area_ratio": null,
  "segment_support_ratio": null,
  "outline_supported": false,
  "patch_ids": [],
  "unassessable_reason": "missing_outline_and_distance_basis"
}
```

## 4.5 이미지 승인·멀티파트

ApprovalCreate 필수 필드는 file\_id/device\_id:UUID,sha256:64hex,size\_bytes:int(1\~5,000,000),mime:image/jpeg 또는 image/png,captured\_at/reviewed\_at:date-time,redaction\_model\_version:string이다. reviewed\_at이 촬영 전이거나 현재 시각보다 허용 오차 이상 미래이면422다. 승인자·기관·approved\_at은 서버가 정한다. 최초 전송 유효기간15분이며 파일의 촬영 후90일 만료가 우선한다.

파일은 기기와 승인된 file\_id·해시·크기·MIME·촬영 시각에 바인딩한다. 같은 파일을 여러 관측의 근거로 참조할 수 있으나 다른 촬영 시각/기관/기기 자료를 같은 해시라는 이유만으로 합치지 않는다. 파일을 재압축/가림/오버레이로 바꾸면 새 해시로 재승인한다.

multipart의 metadata는 JSON 문자열이고 file은 바이너리다. 클라이언트가 boundary를 포함한 Content-Type을 직접 잘못 고정하지 않도록 업로드 라이브러리가 생성하게 한다. 전송된 바이트 해시를 계산하고 실제 디코딩·크기·4096×4096 상한·승인 상태를 검증한 뒤 ready로 저장한다. raw 영상·NMEA·외부URL·사용자 파일 경로는 업로드 대상이 아니다.

# 5. 조회·응답 객체·필터

| 질의 | 타입·기본값 | 규칙 |
| --- | --- | --- |
| bbox | 쉼표 구분4수치, O | west,south,east,north. finite·범위·순서 검사, 날짜변경선 횡단 미지원 |
| from,to | date-time O | from포함/to미포함, from\<to. 보고서 최대90일 |
| type | pothole/lane\_segment O | 두 단위 합산 금지 |
| review\_status | 상태enum O | 대표 case의 현재 상태 |
| quality | valid/degraded/unavailable O | 위치 품질 필터. 영상 usable은 별도 응답 quality 객체 |
| grade | 차선grade O | type=pothole와 조합하면422 |
| run\_id | UUID O | 같은 기관 주행 |
| limit,cursor | int기본50·최대200 / 서명문자열 O | 필터 변경 시 cursor재사용422 |
| snapshot\_at | date-time O, 보고서 전용 | 요청 중/유효 결과 재사용 구분값. 과거 DB 상태로 시간여행하는 기능 아님 |

bbox와 unavailable 조합은422다. 위치 없는 후보를 찾으려면 bbox 없이 quality=unavailable을 사용한다. list의 data=\{items:\[ObservationView\],next\_cursor:null/string,counts\_by\_type:\{pothole,lane\_segment\}\}다. counts\_by\_type의 각 값은 해당 필터에 맞는 관측 수이며 counts\_unit=observation을 반환한다. 보고서 집계는 대표 점검 건(case) 기준으로 counts\_unit=case를 별도 표시하고 두 수를 혼동하지 않는다.

ObservationView는 불변 관측 스키마에 case\_id,case\_version,review:\{status,priority,reason?,updated\_at\},expires\_at,image\_content\_path를 추가한다. image\_content\_path는 /api/v1/images/\{id\}/content 같은 보호된 상대 경로이며 공개 주소가 아니다. 같은 case에 속한 모든 관측의 review는 대표 case의 현재 상태를 읽는다.

이력 응답은 entries:\[\{at,actor\_id,from\_status,to\_status,from\_priority,to\_priority,reason,case\_version,action\}\],next\_cursor다. 삭제된 사진·원본·토큰·다른 기관 정보를 포함하지 않는다. 개인정보 관련 상세는 일반 이력과 분리한다.

# 6. 검토·중복·보고서 계약

## 6.1 검토 상태 변경

ReviewPatch는 expected\_case\_id:UUID M,expected\_version:int≥1 M,reason:string(1\~500) M,status/priority 중 하나 이상 O,override\_transition:boolean O 기본false다. 원관측에 연결된 대표 case가 달라졌거나 version이 다르면409다. version키 누락은 사용자 오류 처리에서428로 매핑한다.

표준 전이는 unreviewed→confirmed/false\_positive/recapture,recapture→confirmed/false\_positive,confirmed→action\_planned,action\_planned→completed다. 기각 취소(false\_positive→unreviewed)·완료 취소(completed→confirmed)와 그 밖의 허용 예외에는 override\_transition=true·검토 권한·명시적 사유가 필요하다. 변경이력·version증가·감사는 한 트랜잭션이다. 단계 건너뛰기로 개인 식별 원본/다른 기관 접근을 승인할 수는 없다.

```json
{
  "expected_case_id": "60000000-0000-4000-8000-000000000001",
  "expected_version": 3,
  "status": "action_planned",
  "priority": "urgent",
  "reason": "담당자 현장 확인 후 조치 일정 등록",
  "override_transition": false
}
```

```json
{
  "data": {
    "case_id": "60000000-0000-4000-8000-000000000001",
    "version": 4,
    "review": {
      "status": "action_planned",
      "priority": "urgent",
      "updated_at": "2026-11-03T02:00:00Z"
    }
  },
  "meta": {
    "request_id": "req-review-example",
    "schema_version": "0.4"
  }
}
```

## 6.2 중복 링크

DuplicateCreate는 source\_observation\_id/target\_observation\_id:UUID,decision:suggested/approved/rejected,source\_version/target\_version:int≥1,reason:string(1\~500)를 필수로 한다. 버전은 연결된 대표 case의 버전이다. 다른 type/기관·자기참조·순환·유효하지 않은 source/target은409/404다. 기기 토큰은 호출403이며 자동 worker는 suggested만 생성한다.

응답은 \{id,source\_case\_id,target\_case\_id,status,version,distance\_m,evidence,decided\_at,decided\_by\}다. PATCH 취소는 decision=revoked,expected\_version,reason을 요구한다. 승인/취소는 원관측 삭제 없이 별칭 연결·버전·이력을 원자적으로 변경한다. 완료된 과거 건 근처의 새 관측은 재발 후보로 만들고 근접만으로 자동 합치지 않는다.

## 6.3 보고서·작업·개인정보 요청

보고서 GET은 도메인 검토 상태를 수정하지 않는 산출 작업 예약/조회다. 같은 사용자·필터·형식·템플릿의 진행 중 작업/유효 결과는 재사용한다. 자동 브라우저 prefetch를 적용하지 않고 요청/기관별 작업 수를 제한한다. 최대10,000 대표 점검 건·90일 기간, 동시 worker 생성1개를 초기 운영 상한으로 둔다.

202 응답 data=\{job\_id,status:pending,status\_url,expires\_at:null\}이며 완료 후 GET /jobs/\{id\}는 \{job\_id,status:succeeded,progress:100,download\_path,expires\_at\}를 반환한다. pending/running/succeeded/failed만 사용하고 실패는 안전한 error\_code를 포함한다. 사용자 입력 snapshot\_at은 새 예약 구분자이며 최초 예약 시 실제 대상/출력값 스냅샷을 저장한다. 임의 과거시점 재구성 약속은 하지 않는다.

PrivacyCreate는 request\_type:access/correction/deletion,target\_refs:\[\{kind:observation/image/run/case,id:UUID\}\],reason:string(1\~500)를 요구한다. 최대100참조, 같은 기관/권한 검사, 접근·정정·삭제 작업을 구분한다. 정정 요청은 추가 근거 수집·관리자 결정으로 처리하고 불변 관측을 몰래 덮어쓰지 않는다. 민감한 요청자 정보는 이 본문/일반로그에 넣지 않고 별도 제한 채널의 내부 티켓 참조만 남긴다.

PrivacyView=\{request\_id,request\_type,status,job\_id,created\_at,resolved\_at,result\_code\}다. status는 received/in\_progress/completed/rejected/failed다. 삭제는 조회/전송 차단→원자료·파생·관련 보고서/사본 정리→검증 후 completed다. 행을 큐에 넣은 것만으로 완료라고 응답하지 않는다.

# 7. 멱등성·동시성·전송 순서

## 7.1 정상 전송

사용자/기기 등록→프로파일/모델 번들 등록→run 등록→로컬 분석·마스킹·사람 확인→image-approvals(검토자)→images(device)→observations(device)→path 요약→run 종료 확정 순서다. 파일 ready 전에는 관측을 저장하지 않는다. 이미지가 준비되지 않은 요청은409 IMAGE\_NOT\_READY이며 큐는 의존 이미지부터 완료한다.

관측 응답은 COMMIT 후201, 같은 요청 재수신은200/deduplicated=true다. 정상 예:

```json
{
  "data": {
    "observation_id": "10000000-0000-4000-8000-000000000002",
    "event_id": "20000000-0000-4000-8000-000000000002",
    "case_id": "60000000-0000-4000-8000-000000000002",
    "case_version": 1,
    "review": {
      "status": "unreviewed",
      "priority": "normal"
    },
    "expires_at": "2027-02-01T01:15:30Z",
    "deduplicated": false
  },
  "meta": {
    "request_id": "req-ingest-example",
    "schema_version": "0.4"
  }
}
```

## 7.2 관측 요청 해시

관측 ID는 재전송해도 바꾸지 않는다. SHA-256 request\_hash는 스키마 검증·명시적 기본값 적용 후 정규화된 관측 JSON을 JCS(RFC8785) 규칙으로 직렬화해 계산한다. 중복 키·비유한 수치·허용 범위 밖 정수를 먼저 거부한다. 단순 문자열 키 정렬만으로 모든 수치/문자열 정규화가 끝난다고 가정하지 않는다. [RFC8785](https://www.rfc-editor.org/rfc/rfc8785)

동일 기관·기기·observation\_id·동일 request\_hash면 최초 관측 ID와 현재 연결 case 정보를 반환한다. 키 순서·JSON 공백 차이는 같은 요청으로 본다. 다른 값·다른 모델/파일/시각은409이며 원래 관측을 덮어쓰지 않는다. 정정/재분석 자료는 새 ID와 별도 lineage로 관리한다. API는 기존 ID 내용 교체용 PUT을 제공하지 않는다.

이미지 멱등 키는 기관·기기·file\_id이고 해시/승인 관계도 비교한다. 같은 해시라도 다른file\_id·시각·기관을 자동으로 같은 파일에 합치지 않는다. 원자료 기한과 삭제 상태는 멱등 재수신보다 먼저 검사해 만료 자료 부활을 막는다.

## 7.3 실패별 큐 처리

네트워크 오류·408·5xx는5초→15초→60초→이후5분 간격으로 같은 본문을 재시도한다. 429는 Retry-After 우선이다. 401/403,승인 대기/만료,409충돌,422는blocked로 남기고 재인증/재승인/사용자 확인 전 반복하지 않는다. 413/415는 파일 오류로 표시한다. 410은expired로 종결한다.

같은 API키의 ACK 유실과 도메인 version충돌은 다르다. 관측의 중복 재수신은200이어야 하지만 stale case변경을200으로 숨기면 안 된다. 동시 상태 변경은 UPDATE version비교+history+audit을 한 트랜잭션에서 수행하고 패자가409를 받는다.

# 8. 보안·개인정보·보관

원본·SVO·NMEA·식별 가능 중간 파일은 암호화 로컬7일 이내다. 자체 마스킹 사진/관측/상세 경로는 촬영 후90일, 최소 감사로그는 행위 후180일, 보고서/파생 스냅샷은최대24시간·원자료기한우선이다. 법정 일률 기한이 아닌 프로젝트 설계값이다. 세부 공개 자료/학습 파생본은 데이터 관리 계획서 기준을 따른다.

관측은 expires\_at을 클라이언트가 임의 지정하지 못한다. 삭제 요청·목적 종료·기관 정책이 더 이르면 더 이른 시점을 적용한다. 재전송·병합·서버수신·모델재분석이 기존 사진 기한을 갱신하지 않는다. 기한이 지난 자료는 worker 삭제 전이라도 즉시 조회/다운로드를 차단한다.

로그에 Authorization·password·device\_token·원본·상세 좌표·전체 JSON을 넣지 않는다. 사진 승인 플래그를 신뢰하는 대신 승인자·해시·기관·기기·파일 준비 상태를 확인한다. 파일은비공개 경로에 저장하고 임의 외부URL 다운로드·사용자HTML/PDF 템플릿을 실행하지 않는다.

CSV 수식 주입·PDF 외부리소스 요청·파일 경로 순회·다른기관ID열람·MIME위장·대용량 디코딩을 입력 검증 대상으로 한다. 브라우저/중간 캐시의 민감 파일 저장을 제한한다. 개인정보 삭제는 DB·이미지·썸네일·보고서·백업 전파 검증 후 완료한다.

# 9. 계약 시험·검수

다음은 구현 후 실행할 시험 목록이며 문서 작성만으로 통과했다고 표시하지 않는다.

| 시험 | 입력·상황 | 예상 결과 |
| --- | --- | --- |
| AT-01 인증·기관 | 익명/만료/다른기관UUID/device로검토변경 | 401/404/403. 파일·존재정보·토큰 누출 없음 |
| AT-02 스키마 | 잘못된UUID·64hex·NaN·좌표한쪽null·반대type object | 422, 안전한 field\_path·code |
| AT-03 승인 | 기기단독승인·승인만료·해시변경·다른기기·원본형식 | 403/409/415/422, ready/관측 생성 없음 |
| AT-04 정상수신 | ready파일+유효 run/모델·포트홀/차선 | 201, 참조/기한·case 생성·JSON래퍼 일치 |
| AT-05 멱등 | 같은 관측100건을3회전송·키순서변경·ACK유실 | 관측100건 유지, 재수신200·같은ID. 본문변경409 |
| AT-06 차선경계 | R=59.999/60/79.999/80,q=.6999/.70,support=.7999/.80 | 등급·판정불가가라벨가이드와일치. 근거없음null허용 |
| AT-07 검토충돌 | 동일 case/version으로2명동시변경·version누락 | 한쪽200·한쪽409, 누락428, history 1회 |
| AT-08 중복 | 가까운별개물체·다른type·순환·수동승인/취소 | 자동삭제없음, 위반409, 원관측/과거상태보존 |
| AT-09 조회 | bbox순서오류·unavailable+bbox·필터변경cursor | 422. 위치누락은독립목록에남음 |
| AT-10 파일·보고서 | 5,000,001byte·잘못된MIME·최대행수초과·원자료삭제 | 413/415/422·보고서기한/삭제전파, 무단파일열람없음 |
| AT-11 복구 | 이미지저장후DB실패·10분단절·worker재시작 | 고아파일정리·동일ID재시도·유실/이중생성0 |
| AT-12 계정·회수 | 마지막admin비활성·기기토큰교체·사용자비활성 | 409·이전토큰차단·기존세션회수 |

필수 API 시험은 SRS TC-07/08/09, NFR-14/23/25와 연결한다. 실제 성능 지표(FPS·mAP·IoU)는 이 API 계약 시험과 별도다. 구현 OpenAPI에는 모든 경로·security·요청/응답·오류·nullability·multipart를 반영하고 예제 JSON도 스키마 검증한다. 문서버전v1.0, 경로/v1, payload0.4, 데이터라벨1.0을 혼용하지 않는다.

# 10. 근거·변경 이력

- <mention url="https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf" />, <mention url="https://app.notion.com/p/40d8ac7c2dde81f3b7a4f3ae95edb804" />
- [FastAPI 폼·파일 요청](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/), [Pydantic 타입·검증](https://pydantic.dev/docs/validation/latest/api/pydantic/types/)
- [JSON Canonicalization Scheme, RFC8785](https://www.rfc-editor.org/rfc/rfc8785)
- [OpenAPI 3.1.1 형식 참고](https://spec.openapis.org/oas/v3.1.1.html) — 생성 문서 구현 기준. 실제 생성/배포된 스키마 파일은 구현 단계에서 검증한다.

| 버전 | 날짜 | 변경 |
| --- | --- | --- |
| v1.0 | <date start="2026-11-03" /> | 기존 API 계약 독립 문서화, 관리 API·필드/null·승인·재전송/버전·상태·보고서·시험 상세 작성 |

연계 문서: <mention url="https://app.notion.com/p/40d8ac7c2dde818aa89ed3a2bc9b24a7" />, <mention url="https://app.notion.com/p/40d8ac7c2dde81b5af5cc462aa535923" />.
</content>
</page>
