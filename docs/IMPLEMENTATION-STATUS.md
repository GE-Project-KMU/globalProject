# 기능 구현 대응표

2026-11-03 · 기준: SRS v0.4, API v1.0 및 API-IMPLEMENTATION.md. 이 표의 **구현**은 실행 코드와 오류 처리 경로를 작성했다는 뜻이다. 학습 완료·목표 정확도 달성·실물 수용시험 합격과 구분한다. 실제 검증 결과는 TESTING.md에 기록한다.

## 데이터 취득

| 요구사항 | 구현 경로 | 검증 범위 |
|---|---|---|
| FR-ACQ-01 | agent/capture.py: 좌우 메모리 MP4 인코딩, AES-GCM, 실제 시각/프레임 로그 | 합성 SBS 영상 코덱·암복호화; 실물 FPS 별도 |
| FR-ACQ-02 | capture.py, workflow.run_samples: 주행 ID·PTS·암호화 조각 재생 | 메모리 디코딩·무결성; 장시간 실물 운용 별도 |
| FR-ACQ-03 | gnss.py, workflow.py: NMEA·UTC·fix·HDOP·수신 시각 정렬 | 유효/무효/시간 공백 주입 |
| FR-ACQ-04 | equipment.preflight, collect/postcheck CLI: 정차·고지·저장·장치·사람 확인 | 장치 오류·중단 계약 시험; 실제 장착 별도 |
| FR-ACQ-05 | models.frame_quality, workflow.py: 흐림·노출·가림·정렬/장착 이상 사유 | 불량 입력·판정 불가 계약 |
| FR-ACQ-06 | profile.py, workflow.py: 프로파일/번들 연결·10 FPS 샘플링 | 서로 다른 설정·해시/해상도 불일치 차단 |

## 포트홀·차선·위치

| 요구사항 | 구현 경로 | 검증 범위 |
|---|---|---|
| FR-PH-01 | models.OnnxModel: 포트홀 박스·신뢰도 | 실제 외부 ONNX 실행과 별도의 테스트용 모델 계약 |
| FR-PH-10 | models.py, analysis: 잠금된 confidence/NMS 설정 | 범위·모델 명세·해시 검사 |
| FR-PH-11 | pipeline.Tracker, workflow.py: 1초/3회·고신뢰 단발·보류 이벤트 | 추적 수·시간 경계 |
| FR-PH-12 | models.py, workflow.py: 품질/확정 불가 보류 | 보류 사유 처리; 어려운 음성 구별 성능은 실제 학습·평가 필요 |
| FR-PH-13 (C) | equipment.stereo_depth/road_relative_step: 선택적 SGBM·주변 평면 잔차 | 합성 시차 실제 SGBM; 실물 깊이 정확도 미검증 |
| FR-LN-01 | profile.py: 독립 보정 외곽·가시 도막 정렬 대조 | 근거 없음/완전 소실 시 판정 불가 |
| FR-LN-02 | models.py, workflow.py, ProtectedImage: 도막 마스크·행 우선 RLE·오버레이 | 마스크 크기/합계/좌표 및 UI 토글 |
| FR-LN-03 | profile.geometry: 왜곡 보정·ROI·homography·BEV | 설정 누락/해상도·정렬 불일치 차단 |
| FR-LN-04 | lane.py, segments.py: 면적 잔존율·10m 구간 | 독립 패치 가중 집계·거리 근거 부족 null |
| FR-LN-05 | lane.py, inspection/schemas.py | 60/80% 등급 경계 |
| FR-LN-06 | profile.py, segments.py: 백/황 대조·검증된 실선/점선·의도된 공백 제외 | 색상/방향/점선 위상 불일치 차단; 현장 인식률 별도 |
| FR-LN-09 | lane.py, workflow.py, schema/UI | 유효 면적 70%·지원율 80%, 판정 불가 null/사유 |
| FR-LN-10 | segments.py: 거리/epoch/차선별 고유 패치 | 반복 프레임과 중복 패치 검사 |
| FR-LOC-01 | gnss.py, workflow.py: vehicle_observation·null 위치 | 오래된 fix·무효 fix·시간 공백 |
| FR-LOC-02 (C) | equipment.object_coordinate | 좌표·heading·장착 변환이 있을 때만 선택 추정; 실측 오차 별도 |
| FR-LOC-03 | pipeline.Tracker | IoU·시간·진행과 관측 수; 현장 ID 정확도 별도 |
| FR-LOC-04 | inspection/app.py: 거리·가능한 사진 dHash·경로 방향 후보 | 수동 승인·취소·순환/타입/버전 충돌; 자동 합병 없음 |
| FR-LOC-05 (S) | 점검 상세·Duplicates·case history | 대표 사진·원관측 보존·병합/취소 이력 |
| FR-LOC-06 | inspection.rank/review, 대시보드 | 종류/상태/오래된 순서·수동 우선순위 이력 |

## 개인정보·서버·웹

| 요구사항 | 구현 경로 | 검증 범위 |
|---|---|---|
| FR-PRV-01 | models.redact, MediaReview, workflow.approve_review, media | 자동 마스킹 어댑터·수동 가림·검수자/파일 해시 승인; 실제 검출 Recall 별도 |
| FR-PRV-02 | capture.py, outbox.py, media 승인 API | 평문 원본 파일 작성 차단·메모리 복호화·미검수 업로드 거부 |
| FR-PRV-03 | agent.retention/governance, 서비스 worker, config_backup | 원본/상세/감사 TTL·철회·파생/보고서/모델 차단·상세자료 백업 금지 |
| FR-PRV-04 | equipment.collection_plan, collect CLI | 기관·문의처·담당·근거·목적·범위·보관·고지·정차 확인 |
| FR-PRV-05 | reporting privacy API, inspection/privacy.py, PrivacyRequests | 열람 승인·제한 다운로드·불변 원관측에 정정 부가·삭제 차단/검증·공개 보고서 축약 |
| FR-SRV-01 | media/inspection, gateway | SHA/JCS 멱등·파일 승인·관측 스키마·100건 3회 재전송 |
| FR-SRV-02 | outbox.py | 암호화 큐·단계별 ACK·재시도·lease·영구 오류·만료 |
| FR-SRV-03 | inspection.listing, Inspections | 기간·영역·종류·검토·품질·위치 없는 독립 목록 |
| FR-SRV-04 | identity/gateway, Accounts/Devices/Registry | 기관/역할·세션 회수·토큰 교체·비공개 파일·승인 |
| FR-SRV-05 | reporting/app.py, Reports | UTF-8 BOM CSV·JSON 계약·실제 한글 PDF |
| FR-WEB-01 | ObservationMap, Operations.RunMap | 종류별 점·품질·가시 범위·경로 공백; VWorld 키 없으면 격자 안내 |
| FR-WEB-02 | Inspections, App | 필터·기본 정렬·우선순위·조건 전달 |
| FR-WEB-03 | Detail, ProtectedImage, history API | 사진·박스/마스크/외곽·수치/품질·모델·원관측/대표 case/정정·이력 |
| FR-WEB-04 | ReviewPatch·상태 전이·Detail | 확인→조치 예정→완료·사유·작성자/시각·되돌림 이력 |
| FR-WEB-05 | Reports, reporting | 현재 조건과 출력 일치·종류/단위·위치 품질/신뢰도/보류/모델·공개 축약 |
| FR-WEB-06 (C) | summary.trend_by_type, TrendChart | 대표 case의 관측 일자·현재 상태, 종류별 별도 차트 |

## 추가 운영 기능

계정·기기·프로파일·모델의 관리 화면, 주행·감사 조회, 실제 분석 작업 큐/상태 UI와 로컬 작업 실행을 구현했다. 공개 API 요청/응답과 nullable 필드를 명시하고 UUID·엄격한 true·동시 수정·요청 ID를 검사한다. 서비스별 독립 스키마 마이그레이션, PostGIS generated geography/GiST, 프로젝트 내부 인증서를 이용하는 HTTPS, 설정만 백업하는 복구 도구를 제공한다.

## 장비 연결 전후에 필요한 실제 준비

- ZED 2i·GNSS 장치/포트, 차종별 실제 보정·가시 ROI·거리 스케일·점선 위상, 기관 수집계획·고지를 설정한다. 임의 예제 보정값으로 실측 완료를 대신하지 않는다.
- 목표 포트홀 Nano·차선 paint·얼굴/번호판 모델의 승인 데이터와 가중치가 필요하다. 학습/export·독립 평가 코드는 제공하지만 가중치나 학습 실적을 만들어 넣지 않는다. 이 조건이 없으면 분석은 명시적으로 blocked다.
- 실제 SDK/GPU 추론·목표 정확도/처리량·60분 주행·진동·조도·장착 안전·2개 차종 검증은 미수행이다. NFR-01/04/05/06/07/09/10/11/12/13/17~22의 현장 수치는 합격으로 표시하지 않는다.
- NFR-15의 5명 사용자 시험 역시 브라우저 자동 시험으로 대체하지 않는다. Docker/컨테이너 자체 실행은 Docker가 없어 미검증이며, Windows 네이티브 PostgreSQL/PostGIS와 Ubuntu CI 결과를 별도로 기록한다.

설치·장비 실행: VISION.md 및 DEVICE-INTEGRATION.md. 실제 시험 근거: TESTING.md. 원본 Notion 문서 스냅샷은 보존하고, MSA 저장 구조 변경은 ADR-001/002에 기록한다.
