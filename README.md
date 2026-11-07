# Road Insight

ZED 2i 기반 포트홀·마모차선 인식과 도로점검 업무 지원 프로젝트.

## 현재 상태

MSA 기반 개발 MVP입니다. 인증·기기·설정 관리, 촬영/분석/검수/전송,
점검·중복·이력·통계·보고서, 개인정보 열람·정정·삭제와 보관 작업을 연결했습니다.
관리자 화면과 실제 분석 작업 운영 UI, HTTPS·스키마 마이그레이션·설정 백업도 제공합니다.
요구사항별 코드 경로는 [기능 대응표](docs/IMPLEMENTATION-STATUS.md),
실제 검증 결과와 제한은 [테스트 결과](docs/TESTING.md)에 기록했습니다.

공개 자료로 포트홀 YOLOX-Nano·도막 DeepLabV3-MobileNetV3·얼굴/번호판 Nano를 실제 GPU 학습하고 연구용 ONNX 가중치를 준비했습니다.
재현·시험 수치·기준 미달과 사용조건은 [초기 모델 학습 기록](docs/MODEL-TRAINING.md)에 있습니다.
후속 [모델 개선 통합 기록](docs/MODEL-IMPROVEMENT.md)은 포트홀·도막·얼굴·번호판의 같은 시험군 비교, 회귀, 재현 절차를 담습니다.
이번 공개 데이터 추가 학습·평가의 결과는 [포트홀](docs/POTHOLE-CYCLE2-RESULTS.md), [차선](docs/LANE-CYCLE2-RESULTS.md), [비식별화](docs/reports/privacy-family-v10-count-retry1-results.md)에 기록했습니다. 새 시험의 포트홀 AP50은42.25%, 차선 도막 IoU는73.07%, 비식별화 얼굴/번호판 Recall은92.42%/96.00%입니다. 각 목표80%·75%·양 클래스95%는 충족하지 못했습니다. 차선은 개선됐지만 포트홀 새 시험 점수는 하락했고, 얼굴은 Recall 소폭 상승과 함께 정밀도가 하락했습니다.
최종 세 모델의 [CPU 성능 검증](docs/FINAL-CPU-PIPELINE-BENCHMARK.md)에서는 기본1.05→병렬2.03FPS로 측정됐습니다. 이는 합성1280×720 입력의 짧은 처리량이며10FPS·실차 성능을 뜻하지 않습니다. 새 GPU 모델 구성은 strict 수치 검증을 통과하지 못해 등록·승격하지 않았습니다. 기존 모델의 과거 GPU 속도를 새 모델의 성능으로 사용하지 않습니다.
가중치는 프로젝트 내부에 보관하며 Git에는 코드·출처·정제된 결과를 올립니다.
제공 주행 영상에서는 [대표80장 AI 시각 라벨 초안과 로컬 검수 화면](docs/RECORDED-DRIVE-AI-LABELING.md)을 작성했습니다. 사진·세부 라벨은 로컬에만 보관하며, 전체 영상 라벨 완료나 사람 검수 정답·정확도 상승으로 표시하지 않습니다.
운영 승인·독립 ZED 실차 정확도·전체 처리량 검증은 남아 있습니다. 모델 미준비·만료·철회 상태는 명시적으로 차단합니다.

## 구성

- `services/identity`: 사용자·기관·기기·인증 세션
- `services/media`: 비식별 이미지 공유 승인·파일 접근·보관
- `services/inspection`: 관측·점검 건·검토·중복 후보
- `services/analysis`: 설치 프로파일·모델 manifest·분석 작업
- `services/reporting`: 보고서·비동기 작업
- `services/gateway`: 외부 API 진입점과 내부 인증 전달
- `frontend`: React/TypeScript 점검 워크스페이스
- `agent`: 로컬 ZED/UVC 촬영·분석·검수 업로드 도구
- `common`: 업무 도메인을 포함하지 않는 공통 기술 기반
- `docs`: 기준 문서와 운영 기록

서비스별 저장소와 실행 프로세스를 분리합니다. 다른 서비스의 DB를 직접 읽지 않습니다.
로컬 개발 저장소는 SQLite, 참조 배포 구성은 서비스별 PostgreSQL/PostGIS입니다.
Windows의 별도 PostgreSQL/PostGIS에서 스키마·공간 조회·기관 격리·삭제를 실제 확인했습니다.
Docker 컨테이너 자체 실행과 전체 PostgreSQL 운영 부하는 별도 검증 대상입니다.

## 개발·커밋 규칙

`main`은 기준 브랜치, 개발은 `feat/msa-mvp` 및 후속 기능 브랜치에서 진행합니다.
API 또는 컴포넌트와 해당 테스트를 한 기능 단위로 커밋합니다.
타입·범위는 영어, 설명은 한국어로 작성합니다. 예: `feat(identity): 로그인 구현`,
`fix(media): 이미지 검증 보완`, `chore(deploy): 실행 환경 구성`, `test(api): 오류 응답 검증`.
통과하지 않은 테스트·실물 미검증 항목은 완료로 기록하지 않습니다.

## 실행

실행·중지·테스트 절차는 [운영 가이드](docs/OPERATIONS.md)를 참고하세요.
실제 장비 연결 순서는 [장비 연동 인계](docs/DEVICE-INTEGRATION.md)와 [영상 도구 가이드](docs/VISION.md)에 있습니다.
개발 UI는 `http://127.0.0.1:5173`, Gateway는 `http://127.0.0.1:8000`입니다.
합성 예제 생성은 `scripts/seed_demo.py`로 수행하며 화면에 합성 데이터임을 표시합니다.
생성된 `.env`는 로컬에만 보관하며 원본 촬영 자료·모델 가중치·캐시·로그는 Git에 넣지 않습니다.

## 작업 범위

현재 로컬 파일 작업과 다운로드 저장 위치는 이 프로젝트 폴더 안으로 제한합니다.
운영체제 전역 드라이버·CUDA·SDK 설치는 이 저장소 생성 작업에 포함하지 않습니다.

## 연구 원본·개인별 파생물 정리

취득 근거의 고정 SHA·기한으로 명시적으로 등록한 파일은 `scripts/research_retention.py purge`로 dry-run 점검합니다.
실제 만료 파일 정리는 학습·평가 종료 확인 후 `purge --apply`를 명시해야 하며 자동 예약 작업은 아닙니다.
원본 archive/test/crop·개인별 preview/예측·파일명을 담은 release JSON과 보존할 가중치/최소 provenance/공개 집계를 구분합니다.
등록·누락 점검·수동 purge 절차는 [연구 자료 보관 정리](docs/RESEARCH-RETENTION.md)를 참고하세요.
