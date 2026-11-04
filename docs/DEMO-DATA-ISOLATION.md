# 시연 출처 분리와 실노면 로컬 참고사진

검증: `pytest tests/test_http.py tests/test_presentation.py`는 전용 새 기관으로 HTTP·분류·출력 범위를 검사한다. 프론트 `npm run test:e2e`는 사진 로딩·출처 배지·로컬 검수 확인·악성 파일과 만료 차단·지연 응답 경합을 검사한다. 실제 RTK 사진 미리보기 테스트는 로컬 pack이 있을 때만 수행하며, 원자료가 없는 CI에서 실제 사진을 내려받거나 사진 artifact를 공개하지 않는다. 파일 형식·보안 검증은 별도의 **명백한 합성 validator fixture**로도 실행한다. 합성 fixture를 실노면 사진의 증거로 사용하지 않는다. 브라우저 fixture Python 경로는 Windows/Unix 프로젝트 venv를 구분한다.

HTTP 통합 시험과 쓰기를 수행하는 브라우저 시험은 실행마다 새로운 UUID 기관을 만든다. 지원되는 identity bootstrap CLI를 사용하며 기존 demo 기관이나 사용자 기록을 삭제·수정하지 않는다. 가상 사진/관측은 `demo=true`이고 사진 자체에도 합성 시험임을 표시한다. 브라우저의 기존 demo 기관 점검은 사진 조회·보고서 생성만 수행하며 관측 검토 상태를 바꾸지 않는다.

## 기존 HTTP 시험 기록

`python scripts/configure_presentation.py`는 기본 dry-run이다. `--apply`는 지원 API로 기기 이름, 프로파일 이름/카메라 serial, 모델 code_commit을 교차 확인한 정확한 기관·run·device·profile·model 식별자만 `data/presentation/policy.json`에 기록한다. 원 관측 payload, `demo` 값, 리뷰, 이미지, 케이스 버전은 변경하지 않는다. 이름 접두사 하나만으로 일반 기록을 숨기지 않는다. 설정 파일은 프로젝트 밖 경로를 허용하지 않는다.

API `GET /observations`, `GET /summary`, 보고서 `GET /reports`의 `data_scope`는 `all`(기존 기본값), `presentation`(HTTP 시험 제외), `test`(HTTP 시험만), `synthetic`(합성 시연만), `field`(분류되지 않은 일반 관측) 중 하나다. 권한/기관 격리는 동일하다. 목록은 정책 snapshot 한 번으로 분류하며 cursor는 범위와 정책 SHA에 연결한다. 보고서 캐시는 범위를 구분하고 CSV/PDF에도 자료 출처를 표시한다.

화면 기본값은 `presentation`이다. 필요하면 전체 또는 HTTP 시험만 조회할 수 있다. `data_origin=http_test`는 명백한 합성 HTTP 시험, `synthetic_demo`는 선언된 합성 시연이다. `field`는 실제 촬영의 증명이 아니므로 화면에 **일반 관측 · 출처 미확인**으로 표시한다. 원본 `demo=false`인 이전 시험도 읽기 전용 출처 배지를 통해 구분한다.

## 실제 도로 참고사진의 별도 로컬 미리보기

`python scripts/prepare_reference_preview.py`는 이미 내려받은 RTK의 TRAIN 원본 중 publisher semantic road class1..12에 해당하는 영역만 보수적으로 잘라 로컬 묶음을 만든다. 모델 실행, 새 검출 bbox 생성, 사람에 의한 개인정보 검수 또는 공유 승인으로 표시하지 않는다. 원본·기존 split·가중치·기존 보관 인벤토리는 바꾸지 않는다. JPEG crop과 pack의 기한은 원본 acquisition의 고정 기한을 그대로 상속하며 자동 연장하지 않는다. pack이 이미 있으면 덮어쓰지 않는다.

사용자는 **로컬 참고사진** 화면에서 프로젝트 폴더 → `data` → `demo-reference` → `road-reference-pack.json`을 직접 선택하고, 출처·기한을 확인했다는 로컬 열람 확인 후 사진을 연다. 이미지는 브라우저 안의 내장 JPEG이며 업로드·관측·지도·보고서·기관 이미지 승인을 생성하지 않는다. 화면은 **실노면 참고사진 / 개인정보 검수 미완료 / AI 분석 미실행 / 공유 승인 없음**을 고정 표시한다. semantic 도로 영역이라는 기준은 개인정보 검수를 대체하지 않는다.

실사진이 포함된 로컬 브라우저 검증 스크린샷은 `data/demo-reference/browser-evidence`의 실행별 새 폴더에만 저장한다. 해당 사진과 `private-evidence.json`도 원본 기한을 상속하는 비공개 보관 대상이다. 일반 프론트 테스트 artifact에 원자료 사진을 넣지 않으며, 기존 인벤토리는 수정하지 않는다. 운영자는 최종 통합 테스트 이후 새 파일을 별도 보관 인벤토리에 등록한다.

브라우저의 현재 시각으로 만료를 검사하고 한국 표준시(UTC+09:00)로 기한을 표시한다. 열린 상태에서 만료되면 미리보기를 닫는다. schema/키, 원본 기한 상한, 5MB 묶음 한도, 1~6장 제한, 내장 JPEG·실제 JPEG frame 크기 확인을 적용하며 외부 이미지 URL·경로·잘못된 승인 주장·만료 및 기한 연장을 거부한다. 이는 로컬 참고 편의 기능이며 서버 공유 승인 권한을 부여하지 않는다.

RTK 출처: Thiago Rateke, Aldo von Wangenheim, Rafael Toledo, *Road Traversing Knowledge*, UFSC / Mendeley Data, DOI10.17632/hssswvmjwf.1, [CC BY4.0](https://creativecommons.org/licenses/by/4.0/), [발행 원본](https://data.mendeley.com/datasets/hssswvmjwf/1). 수정 사항은 도로 영역 crop 및 JPEG 변환이다. 개인정보·초상권 등 제3자 권리는 라이선스로 대체되지 않는다. 원본/파생/label SHA와 crop 좌표·보관 기한은 로컬 `data/demo-reference/preview-items-receipt.json`에 남기고 사진 자체는 Git이나 공개 보고서에 포함하지 않는다.
