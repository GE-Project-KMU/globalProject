# 장비 연결 전 후속 구현·검증

2026-11-04. 시연 자료 혼입, 실제 사진 미리보기, 모델 수치·효율, 추가 학습, 운영·컨테이너 검증을 수행했다. 초기 실험 및 [이전 모델 개선 기록](MODEL-IMPROVEMENT.md)은 당시 결과로 보존한다. 정확도 인수나 실차 검증을 완료했다는 의미는 아니다.

| 작업 | 완료한 내용 | 현재 제한 |
| --- | --- | --- |
| 회색 HTTP 시험 사진 혼입 | 새 시험은 UUID 전용 기관, 기존 명백한 시험은 원자료 변경 없이 출처 정책으로 분리. 화면 기본은 시험 제외, 전체/시험 조회 유지 | 일반 관측이라는 분류가 실제 카메라 촬영의 증명은 아님 |
| 사진 화면 | 실제 RTK 노면 crop3장의 로컬 참고사진, 출처 배지, 만료·파일 검증, 데스크톱/모바일 확인 | AI 미실행·개인정보 검수 및 공유 승인 없음. 관측 사진으로 허위 교체하지 않음 |
| 차선 수치 정합성 | 같은 checkpoint·threshold, FP64 내부 평균 누산 그래프와 정확한 지원 런타임 강제. 고정 양성 VAL3 및 전체 VAL100 raw/마스크 통과 | Windows AMD64·ORT1.31.1 CPU만 검증. 구ORT1.20 실패 보존. 새 도막/마모 정확도 향상을 주장하지 않음 |
| 개인정보 추론 효율 | 사전 고정3후보를 동일 VAL508에서 표준·전체타일과 실제 비교 | 선택타일은 비용↓/오탐↓/Recall↓. 어느 후보도95% 목표 미달, 자동 적용 없음 |
| 포트홀 추가 학습 | 고정10epoch/3,334 optimizer updates, 선택에 epoch0 포함, tensor/ONNX 동일성 및 고정 VAL canonical 확인 | VAL 개선 없어 기존 v4 유지. 새 공식 자료 HTTP403, 독립 holdout 미확보 |
| 운영 | 새 PG/PostGIS DB5개·서비스6개에서 권한·160조회·작업/삭제 실패 후 재시도·설정11개 복원 확인 | 작은 합성 데이터/설정 복원 범위. 상세 DB 백업·복원/운영 부하 인수 아님 |
| Docker | 격리 Ubuntu CI에서 실제 backend/frontend build, Compose 기동, Nginx HTTP8항목 통과 | Windows에 Docker를 설치한 것은 아님. 운영 보안·처리용량 인수 아님 |

## 실행 중인 화면

[개발 UI](http://127.0.0.1:5173)에서 점검 목록의 자료 출처 필터로 시연/시험을 구분한다. API 기본 `all`은 기존 호환성을 유지하고 UI는 `presentation`으로 시작한다. 관측 원본·리뷰·이미지는 분류를 위해 덮어쓰거나 삭제하지 않았다. scope와 정책 hash는 cursor에 묶고 보고서에도 같은 범위를 전달한다.

실제 노면 참고사진은 **로컬 참고사진** 메뉴에서 프로젝트의 `data/demo-reference/road-reference-pack.json`을 선택해 연다. 브라우저 안에서만 열리며 서버 업로드·관측·지도·AI 검출 결과·공유 승인을 생성하지 않는다. [시연/사진 사용법](DEMO-DATA-ISOLATION.md)에 정확한 범위와 출처를 기록했다. 테스트의 실제 사진 캡처는 Git 제외 `data/demo-reference/browser-evidence`에만 남긴다.

## 모델 결과의 해석

포트홀 추가 학습의 초기/선택 VAL AP50은68.3948%, 마지막 epoch10은58.0196%였다. 기존 가중치를 유지해 선택된 ONNX 바이트도 원v4와 같다. 새 정확도가 개선됐다고 하지 않는다. [추가 학습·동일성 기록](POTHOLE-REFINEMENT-V5.md)과 [추가 자료 취득 상태](reports/RDD2022-REVISION-ACQUISITION.json)를 참고한다. 과거 Japan86/Czech231을 새 시험으로 재사용하거나 재추론하지 않았다.

개인정보 선택 타일 후보는 같은 실행의 전체 타일 대비 median439.0→257.3ms, 얼굴 Precision22.415→25.654%, Recall92.498→91.816%였다. 성능/비용 교환관계이며 이전 실행349ms와 직접 비교하지 않는다. [효율 연구](PRIVACY-EFFICIENCY.md)에3후보 모두와95% 목표 미달을 기록했다.

차선의 새 지원 프로파일은 실제 app ORT1.31.1/격리 reference ORT1.31.1/원PyTorch FP32를 비교한 결과다. 세 raw 비교쌍과 최종 마스크가 전체 VAL100에서 통과했다. 기본 rtol0.001/atol0.0001·paint threshold는 유지했다. 원ORT1.20.1의4픽셀 불일치는 해결한 것으로 바꾸지 않았다. 다른 OS·아키텍처·버전·provider를 거부하며, 기존 manifest에는 새 제약을 소급 적용하지 않는다. [수치 재현·지원 범위](LANE-NUMERIC-REPRODUCTION.md)를 함께 읽는다.

새 독립 도로/사람 시험과 국내 ZED2i 정확도, 기대 윤곽·진짜 마모율 GT는 여전히 필요하다. 이미 본 VAL의 개발 결과를 독립 최종 정확도나 운영 승인으로 바꾸지 않는다. 가중치·원본·개별 예측·사진 hash는 로컬에만 보관한다.

## 최종 소프트웨어·보관 검증

Windows에서 **440tests와189subtests**, 실패/skip0,47.88초였다. GET148/POST448/PATCH29/PUT2/DELETE1 요청과 예상 오류 응답을 검사했고 새 HTTP5xx 및 서버 오류 로그는0이었다. 첫 통합 실행에서 새 기관 fixture를 다른 테스트 모듈에 연결하지 않아15개 setup 오류가 있었으나, 수정 후 개인정보15개 및 전체440개를 통과했다. 이 실패를 모델·서비스 장애로 기록하지 않는다.

프론트는 TypeScript/build와24E2E를 통과했다. 이후 실제 사진 테스트의 메뉴 전환/스크롤 안정화 검증도 통과했고 두 viewport를 직접 확인했다. 원자료가 없는 CI에서는 실제 연구 사진 테스트1건을 skip하며 합성 validator로 악성 파일·만료 차단을 검사한다. 기존 잘린 mobile 캡처는 닫힘 transition 중이었으며, desktop fixed sidebar도 스크롤된 fullpage 캡처 문제였다. DOM 위치·메뉴 닫힘·본문 실제 클릭을 확인해 제품 코드 변경 없이 캡처를 안정화했다.

[운영 검증](OPERATIONAL-VERIFICATION.md), [실제 Docker CI 증거](reports/COMPOSE-CI-VERIFICATION.json), [후속 소프트웨어 집계](reports/FOLLOWUP-SOFTWARE-REGRESSION.json)를 구분한다. Docker 최초 통과 소스는f2ac8e4이며 뒤에 작성한 모델·시연 변경을 그 실행에 소급 포함하지 않는다. 이후 통합 소스의 CI는 해당 commit의 GitHub Actions를 기준으로 확인한다.

새9개 명시 경로의 보관 후보57개(약72.1MB)를 별도 followup inventory에 등록했다. 기존 inventory2개/39,053개 등록 행의 snapshot은 불변이고 새 경로는 이전 scope와 겹치지 않는다. 이번 작업은 과거 원파일 전체를 다시 hash한 검사라고 주장하지 않는다. 새 파일의 hash와 고정 기한을 검사했고 미등록 새 후보·수정 거부·범위 거부는0, 실제 삭제도0이었다. 노면 참고사진/pack/스크린샷은 RTK 원90일 기한, 모델 진단은 원연구 기한을 상속한다. [보관 집계](reports/FOLLOWUP-RETENTION-VERIFICATION.json)에 구분했다. 자동 삭제 예약은 설치하지 않았다.

## 아직 필요한 입력·검증

- 녹화 주행 원본과 촬영 시각·SDK 버전·카메라 보정·GNSS: [자료 인계](RECORDED-DRIVE-INTAKE.md).
- SVO/SVO2 실제 파일 입력 연결 및 SDK 실행. 현재 live-camera 어댑터와 녹화 파일 검증은 구분한다.
- 승인된 국내 도로 라벨, 새로운 독립 정확도 시험, 실제 마모율과 개인정보 전체 GT.
- 새 차량 수집60분·진동·조도·장착·2개 차종 및 담당자5명 사용자 시험.
- 운영 DB 인증/TLS·최소 권한·규모 부하·삭제 전파를 포함한 상세 복구 정책. 현재 Compose smoke와 설정 복원을 운영 승인으로 대신하지 않는다.
