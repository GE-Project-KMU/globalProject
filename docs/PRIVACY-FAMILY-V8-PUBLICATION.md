# 개인정보 평가 결과의 공개 집계 준비

`scripts/finalize_privacy_family_results.py`는 평가 실행기가 아니다. 현재 실제 개인정보 TEST는 아직 실행되지 않았으며, 이 도구 준비에서는 합성 JSON·GT·예측 fixture만 사용했다. 실제 성능 JSON/Markdown은 생성하지 않았다. 원 family·parallel·trainer·runtime·모델·자료·운영 큐는 수정하지 않는다.

실제 사용은 root가 독립 운영 큐 완료를 확인하고 별도로 허가한 뒤다. 직접 Python3.11.9, NumPy1.26.4, OpenCV4.10.0, ORT1.31.1 환경만 허용한다. 기존 canonical verifier의 CVthreads2와 parallel worker의 CVthreads1은 그대로 구분해 기록한다.

```powershell
# 현재 실행하지 않는다. 완료 및 공개 집계 허가 후에만 사용한다.
$env:CUDA_VISIBLE_DEVICES = ''
.\.cache\lane-eval-311-ort1311\python.exe scripts/finalize_privacy_family_results.py --authorized
```

입력은 비공개 control JSON과 가중치·소스의 파일 해시다. 사진·픽셀·NPY는 reader 자체가 거부하며 모델/session을 만들지 않는다. 최대512MB의 control JSON을 읽되 중복 key·비정상 수치를 거부하고, 읽은 파일은 전체 검사가 끝날 때 SHA를 다시 확인한다. 원 자료 사진을 다시 디코딩하거나 모델 예측을 추가하지 않는다.

경로는 먼저 문자열 연산만으로 프로젝트 안인지 확인한다. 외부 absolute 경로, `..` 이탈, 프로젝트 이름과 접두사만 같은 형제 디렉터리, 다른 drive·UNC·drive-relative 경로는 `Path.resolve()`·`stat()`·공통 resolver를 호출하기 전에 거부한다. 이 lexical 검사를 통과한 뒤에만 기존 workspace resolver와 실제 파일·해시 검사를 수행한다. 활성 `agent/paths.py`는 수정하지 않는다.

공개 전 다음 조건이 모두 필요하다.

- 동결 family5777/소스95b5, parallel3171/프로토콜fbbc와 원 계획이 그대로여야 한다.
- active retry1 운영 큐와 모든 phase가 completed/exit0/자기 자식 정리 완료여야 한다. 원래 보존된 preflight 실패는 이전 이력으로 유지하며 active 실패와 혼동하지 않는다.
- 완료된 모델·가중치·checkpoint·원 및 파생 provenance·source release·VAL 예측·canonical 양성 사례의 SHA 연결이 유지되어야 한다.
- 전체 VAL508에 대해 원 순서·동일 GT의 가족 선택, confidence grid, 큰 얼굴 gate를 다시 확인한다. 이 검사는 TEST로 모델이나 임계값을 재선정하지 않는다.
- baseline/winner 각각 얼굴·번호판의 기본 strict raw·후처리 gate가 모두 통과하고 실제 같은 클래스 양성이어야 한다.
- fresh200와 과거 regression1400의 원 permanent attempt, 실행 계획, operator PID, 원 exclusive-start snapshot, 두 CPU worker 결과, per-image JSON, 최종 비교 수치가 일치해야 한다.

점수 일치 검사는 원 canonical AP101의 GT 목록 순서와 전역 GT index를 보존하면서 sample index로 lookup만 줄인다. 예측 confidence 정렬은 stable이고 동률 순서가 유지된다. 최대 IoU의 동률은 원 `(IoU, 전역 index)` max 규칙을 그대로 쓰며 한 GT를 한 번만 사용한다. 알려진 클래스만 평가한 입력을 그대로 사용한다. 합성 oracle에서 원 `score()`와 dict/수치 단위 완전 일치를 확인한다. 재계산 수치는 일치 검사에만 사용하고, 공개 AP50·TP/FP/FN·Precision/Recall·지연은 **완료된 원 comparison의 값**을 복사한다.

출력은 `docs/reports/PRIVACY-FAMILY-V8-RESULTS.json`과 같은 이름의 `.md`다. 명시적 whitelist만 허용한다. cohort 수, 클래스별 GT support/AP50/operating counts/Precision/Recall, 95% 최소 Recall 목표 충족 여부, 전체 모델·weights·control의 SHA, 실제 runtime 및 해석 한계가 포함된다. 개별 image SHA·UUID·경로·bbox·원 예측/score 배열은 공개하지 않는다.

동일 입력과 같은 helper 소스로 다시 호출하면 JSON/Markdown bytes와 SHA가 동일하며 기존 결과를 다시 쓰지 않는다. 하나만 존재하는 부분 공개나 다른 bytes의 기존 결과는 덮어쓰지 않고 거부한다. 입력 실패·활성 phase 실패·미완료·해시 불일치가 있으면 성공 보고서를 만들지 않는다.

fresh200는 한 번만 비교하는 새 benchmark, 과거1400은 회귀 전용이다. publisher가 아는 클래스의 조건부 GT이며 다른 클래스의 완전한 주석이나 전체 개인정보 보호를 입증하지 않는다. 물리적 인물·차량·경로 독립성은 미확인이다. 공유 CPU 호스트의 predict/worker wall 지연은 단독 모델 속도·차량 전체 pipeline10FPS·카메라 처리·현장 인수의 근거가 아니다. 원 만료일과 연구·비상업·재배포 제한을 그대로 유지한다.
