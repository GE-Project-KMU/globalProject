# 공개 연구 개선 주기 보관 목록 준비

**준비 단계이며 새 보관 목록은 등록하지 않았다.** 현재 학습·평가 writer가 종료되기 전에는 이번 주기의 원본·파생 파일을 스캔하거나 최종 해시를 확정하지 않는다. 기존 보관 inventory 3개와 대응 계획 3개의 고정 SHA는 별도로 감사한다. 기존 목록·계획은 수정하지 않는다.

`scripts/freeze_public_cycle_retention.py`는 기본 실행 시 이 과거 제어 파일만 읽는 `audit` 동작을 수행한다. 신규 파일 등록은 명시적 writer 종료 인계서와 그 SHA가 있을 때에만 별도 `freeze` 동작으로 허용한다. 인계서는 운영자의 실제 확인 기록이며 도구가 모든 학습 프로세스의 종료를 스스로 증명한다는 뜻은 아니다. 자동 스케줄 설정이나 원본·가중치 삭제 동작은 없다.

## 대상과 기한

이번에 추가한 MWPD 원자료·포트홀 학습 자료·모델, A2D2 추가 자료·차선 학습/기준 모델, 개인정보 모델 개선 자료·모델, 해당 평가 산출물 및 **지정한 신규 RTK 브라우저 증거 폴더 하나**만 포함한다. 학습·선택에 쓰지 않는 별도 RDD2020 일반 노면 평가 코호트도 추가 대상이며, 이 폴더는 준비가 성립하지 않아 존재하지 않으면 건너뛴다. 폴더가 존재하면 원래 RDD2020 receipt와 해당 release 기한을 검증해야 한다. 임의 경로·태그·기존 RTK 폴더 전체 확장은 허용하지 않는다. 사용자 촬영 기록과 그 검증 폴더는 대상이 아니다.

기한은 각 대상의 원래 source receipt, release 행과 provenance에 기록된 기한의 최솟값으로 상속한다. 수정 시각이나 오늘 날짜로 기한을 만들지 않는다. MWPD 원자료는 원기한인 11월 11일, 기존 A2D2·WIDER·CCPD 및 혼합 모델·평가 결과는 더 이른 원기한인 11월 10일, RTK 증거는 원기한인 2027년 2월 1일을 넘길 수 없다. 이 날짜 상한은 실제 원기한을 대체하지 않는다. 같은 날짜 안에서도 증거의 정확한 UTC 시각을 상속한다. 더 이른 원기한이 발견되면 그 기한을 사용한다. 근거에 기한이 없으면 거부한다.

기존 보관 정책의 가중치·최소 provenance·법적 근거 보존 규칙을 그대로 사용한다. 그 밖의 원본 이미지·주석·파생 이미지·개별 예측·식별 가능한 manifest는 파일 종류를 검증한 후 새 목록에만 등록한다. 분류할 수 없는 파일은 보수적으로 유지하고 집계로 알린다. 최소 증거를 보존한다는 정책이 개인정보 포함 파일의 무기한 공개·배포 승인을 의미하지는 않는다.

## 수동 절차

현재 실행 가능한 읽기 전용 감사:

```powershell
.\.venv\Scripts\python.exe scripts/freeze_public_cycle_retention.py audit
```

모든 해당 학습·평가·보고서·숫자 검증 writer가 종료된 뒤 담당자가 `data/research-retention/` 아래의 비공개 JSON 인계서를 준비한다. 필드는 `cycle="public-refinement-v6"`, 실제 확인한 `writers_stopped=true`, 시간대를 포함한 `confirmed_at`, 도구의 정확한 대상 11개(선언한 선택 폴더 포함)를 정렬한 `roots`다. 확인하지 않고 true를 입력하지 않는다. 파일의 SHA를 고정한 후 아래 동작을 수동으로 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts/freeze_public_cycle_retention.py freeze --writers-stopped-receipt <비공개 인계서 경로> --receipt-sha256 <확인한 SHA256>
```

새 `public-cycle-frozen-plan.json`, `public-cycle-inventory.json`, `public-cycle-summary.json`은 기존 제어 파일과 분리하여 같은 비공개 보관 관리 폴더에 생성한다. 기존 출력이 있으면 덮어쓰지 않고 거부한다. 준비 중 파일·근거·파일 목록 변경, 링크/junction, 과거 등록 파일 중복, 기존 제어 파일 변조를 거부한다. 경로·개별 파일 해시·표본 식별자는 비공개 계획과 목록에만 남긴다. stdout 집계에는 개수·바이트·종류·기한별 개수·안전 검사 결과만 포함한다.

`freeze`는 새 목록 등록 후 삭제 **dry-run** 집계만 수행한다. 실제 삭제의 별도 승인·정책 판단을 대신하지 않는다. 진행 중 오류가 나면 증거를 보존하고 원인을 확인한다. 이미 생성된 계획이나 목록을 덮어써 재시도하지 않는다.

## 준비 도구 검증

합성 작업 공간만 사용해 기한 최솟값 상속, 날짜 상한 초과, 기한 누락, writer 인계 누락·변조·범위 변경, 과거 제어 파일 변조, 알 수 없는 경로·태그, 가중치 보존, 원본 변경 감지, junction 차단, 재실행 덮어쓰기 거부와 비공개 경로가 없는 집계를 검사한다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_public_cycle_retention.py tests/test_research_retention.py -q --basetemp .cache/pytest-public-cycle-retention
```

실제 자산 등록·기한별 파일 수·최종 해시 검증은 writer 종료 후 수행할 별도 작업이며, 이 문서는 그 완료를 주장하지 않는다.
