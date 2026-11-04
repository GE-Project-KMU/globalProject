# 장비 없는 주행 분석·업로드 통합 검증

장비 없이 다음 실제 소프트웨어 경로를 검증했다. 학습 모델의 정확도 검증과 구분되는 **합성 계약 검사**이며, 사용자 주행 기록·공개 원본 사진·학습된 가중치는 입력하거나 서버로 전송하지 않았다.

1. 인물·문자 번호판이 없는 기하학적 합성 좌우 영상 6프레임을 생성한다.
2. 실제 ONNX Runtime CPU에서 고정 Constant/Conv 그래프 3개를 실행한다. 기존 `OnnxModel.predict` 전처리·후처리, 추적, 마스킹을 사용한다.
3. `analyze_video` 진단 경로와 암호화 취득 저장→복호화 재생→`analyze_samples` 검토 패키지 경로를 모두 실행한다.
4. 동일 후보의 6프레임 추적, 합성 체크무늬 영역의 실제 마스킹 픽셀 변경, 공유 미승인 초기 상태와 참조 사진 정리를 확인한다.
5. 새 테스트 기관과 합성 장비·프로파일·번들을 생성하고 실제 Gateway의 주행 POST·경로 PUT·상태 PATCH를 호출한다. 위치/거리 근거가 없으므로 위치 미상과 차선 평가 불가 상태를 유지한다.
6. 미검토 관측 전송이 거부되고 검토자 로그인 없는 큐가 차단되는지 확인한다. 합성 픽셀에 한해 검토 계약을 모의한 뒤 로컬 전송 계층에 503을 한 번 주입한다. 암호화 SQLite 큐를 다시 열고 백오프 후 사진·관측 재전송이 ACK되는지 확인한다.
7. 실제 관측 상세·사진 조회, 검토 버전 충돌, 해당 주행·합성 출처로 제한한 CSV 보고서를 확인한다.
8. 브라우저에서 합성 배지·`NOT-TRAINED` 모델 표기·64×48 사진 로딩·보고서 다운로드·390px 화면을 검증한다. 콘솔·페이지·요청 실패와 예기치 않은 API 오류도 수집한다.
9. 해당 테스트 기관에서 이번에 생성한 관측만 개인정보 삭제 API로 처리한다. 완료된 작업·상세 410·사진 접근 불가·이전 보고서 무효·동일 승인 payload 재전송 410을 확인한다. 전용 로컬 철회 레지스트리로 검토 사진 및 큐도 제거하고 재큐잉을 거부하는지 확인한다. 실제 학습 레지스트리는 스캔하거나 수정하지 않는다.

## 재현

프로젝트 루트에서 기존 서비스 상태를 확인한다. 중단돼 있다면 프로젝트 관리 도구로 시작한다.

```powershell
.\.venv\Scripts\python.exe scripts/manage.py status
.\.venv\Scripts\python.exe scripts/manage.py start --frontend
```

프로브 자체는 새 패키지를 설치하지 않는다. 고정된 작은 ONNX 바이트가 소스에 포함되며, 운영 모델의 준비 조건을 통과했다고 위장하지 않는다. `target_trained=false`, `production_approved=false`를 유지한다. 명시적 테스트 adapter 주입점에서만 합성 그래프를 허용하며 운영 `readiness`가 이를 거부하는 테스트도 별도로 수행한다. 모델 번들의 해시는 합성 그래프 해시이며 학습 모델의 해시가 아니다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_workflow_probe.py tests/test_vision.py -q --basetemp .cache/pytest-synthetic-workflow-final
.\.venv\Scripts\python.exe -m pytest tests/test_http.py tests/test_synthetic_workflow_probe.py -q --basetemp .cache/pytest-workflow-http-final
cd frontend
npm run build
npx playwright test
```

브라우저 검사 1건만 재현할 때는 `npx playwright test e2e/workflow.spec.ts`를 사용한다. 이 검사가 준비와 삭제를 함께 수행한다. Windows의 프로젝트 브라우저 캐시는 다음과 같이 설정할 수 있다.

```powershell
$env:TEMP=(Resolve-Path '..\.cache\tmp').Path
$env:TMP=$env:TEMP
$env:PLAYWRIGHT_BROWSERS_PATH=(Resolve-Path '..\.cache\playwright').Path
npx playwright test e2e/workflow.spec.ts
```

브라우저 없이 분리 실행하려면 루트에서 `scripts/probe_synthetic_workflow.py prepare`를 실행한다. 반환된 `private_receipt`를 사용해 `scripts/probe_synthetic_workflow.py cleanup --receipt <private_receipt>`를 반드시 이어서 실행한다. 인계 파일에는 해당 테스트 기관의 일회성 생성 계정 및 장비 토큰이 있으므로 외부 공유하거나 커밋하지 않는다. 검토 모의는 `AUTOMATED-SYNTHETIC-CONTRACT-NOT-HUMAN`으로 명시한다. 실제 사람의 개인정보 검수나 사용자 사진의 공유 승인을 의미하지 않는다.

## 결과와 경계

기계 판독 집계는 [HARDWARE-FREE-WORKFLOW-VERIFICATION.json](reports/HARDWARE-FREE-WORKFLOW-VERIFICATION.json)에 있다. 신규 오프라인 7건과 기존 영상 검사 61건을 합한 68건, 기존 HTTP 54건과 신규 7건을 합한 61건, 전체 브라우저 25건이 통과했다. TypeScript 및 Vite 빌드도 통과했다. 새 브라우저 통합 시나리오의 콘솔 오류·페이지 오류·요청 실패·예기치 않은 API 오류는 각각 0건이었다.

검증용 503은 로컬 테스트 전송 계층에서 주입한 값이다. 실제 서버의 5xx와 혼동하지 않는다. 고정 그래프가 반환한 후보를 실제 AI 검출·개인정보 검출 정확도·실제 결함 판정으로 해석하지 않는다. 합성 교정 프로파일·생성 시각·시험 precheck 또한 현장 설치 검증을 대체하지 않는다. ZED SDK·카메라·GNSS·깊이·물리 잔존율·10분 지속 취득·운영자 승인·현장 성능은 이 검사로 입증하지 않았다.

제품 기능과 공유 runtime 코드는 변경하지 않았다. 기존 단위 검사에서 분리돼 있던 ONNX 실행·로컬 큐·실제 HTTP·브라우저·삭제 경로를 하나의 재현 시나리오로 연결한 검증 도구와 회귀 검사만 추가했다. 모든 사진은 이 시나리오가 생성한 비인물 합성 자료다.
