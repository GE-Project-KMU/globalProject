# 미검토·승인 사본의 기한 상속

미검토 자동 마스킹 결과는 누락된 얼굴·번호판이 있을 수 있으므로 검수된 비식별 사진으로 취급하지 않는다. 로컬 `analysis.json`·`review.json`과 참조 PNG의 기본 기한은 7일이며, 명시된 원자료 계약 기한이 더 이르면 그 기한을 적용한다. 촬영 시각을 아는 workflow에서는 촬영+7일을 사용한다. 진단 `analyze`는 촬영 시각을 파일 이름·수정 시각·분석 시작일로 추정하지 않는다.

`analyze`의 Python API와 CLI는 `source_expires_at`/`--source-expires-at`에 원래의 시간대 있는 기한을 받거나, 실제 확인한 `source_captured_at`/`--source-captured-at`으로 촬영+7일을 계산한다. 둘 다 없거나 만료·naive 시각이면 모델 로드와 픽셀 읽기 전에 거부한다. 명시된 기한만 제공하는 경우에도 새 미검토 파생 사본 자체의 7일 상한을 함께 적용하며 이를 원본 촬영일로 표시하지 않는다. workflow는 aware·단조 증가 시각과 현재+2초를 넘지 않는 촬영 시각을 확인한다. 선택적인 `run.source_expires_at`은 촬영+7일보다 긴 기한을 만들 수 없다.

## 승인과 전송

`approve-review`는 아직 만료되지 않은 원본 검토 패키지와 해당 사진 SHA, 명시적인 승인 결정을 모두 확인한다. 미검토 원본 파일의 수명을 늘리지 않는다. 승인한 사진은 작업 폴더의 `approved/<관측 식별자>/` 아래 별도 바이트 동일 사본과 최소 `review.json`으로 저장한다. 이 승인 사본과 암호화 큐의 기한은 촬영+90일과 원자료 계약 기한 중 이른 때다. 큐 재시도는 이미 승인된 사본을 전송하며 미검토 기한이 지난 파일에 새 승인으로 수명을 주지 않는다.

패키지·주행·개별 항목의 source cap은 모두 최솟값으로 상속한다. 항목의 cap 누락이나 기한 메타데이터 수정으로 원래 기한을 늦출 수 없다. `enqueue --confirm-human-review`도 새 검토 시점에 촬영+7일 이내 및 원래 기한을 확인하고, 원본/미검토 플래그를 가진 이미지는 거부한다. 계약 cap이 있다면 `--source-expires-at`으로 전달한다.

큐는 cap을 암호화 payload에 보존하고 실제 `image-approvals` 요청의 optional `source_expires_at`으로 전달한다. 서버는 승인 ledger에서 이 기한을 사진과 관측에 상속하므로 업로드로 다시 90일이 생기지 않는다. upload multipart metadata의 추가 필드는 필요 없다. cap이 없는 기존 승인 요청은 촬영+90일을 유지한다. 서버가 공유 검수 승인을 부여하는 절차와 로컬 모의 검사는 별개다.

## 정리 범위와 실패 시 동작

`purge-local`은 기본 dry-run이다. 기본 생성 결과 루트는 `data/agent-jobs/results`이며, 다른 진단 루트는 `--results`로 명시한다. 프로젝트 루트 전체, symlink/junction, 패키지 폴더 밖의 이미지, 수정된 SHA, 정책보다 긴 기한을 거부한다. 알려진 `analysis.json`·`review.json`과 직접 참조하는 생성 `*.redacted.png`만 대상으로 하며 관련 없는 파일을 지우지 않는다. 원래 source 영상·사진은 이 파생 정리 목록에 넣지 않는다.

```powershell
.\.venv\Scripts\python.exe -m agent purge-local
.\.venv\Scripts\python.exe -m agent purge-local --results data/agent/analysis
```

실제 정리는 운영자가 결과를 검토한 뒤 `--apply`로 실행한다. 기한·SHA 근거 없는 과거 결과와 중단 중 만들어져 묶음에 등록되지 못한 파일은 추정 삭제하지 않고 별도 확인한다. 생성·정리 도중 오류는 성공으로 바꾸지 않으며, 삭제되지 않은 파일의 보관 의무가 사라진다는 뜻은 아니다. 도구는 자동 스케줄을 만들지 않는다.

## 검증과 이전 실행의 구분

새 회귀 검사는 누락·naive·만료 source 문맥의 픽셀 읽기 전 거부, 미래 촬영, 7일 미검토/90일 별도 승인 사본, 짧은 계약 상속, 뒤늦은 승인·수정 기한 우회 거부, 만료 큐 전송 차단, analysis 정리, SHA 변경·폴더 이탈·junction·legacy 거부와 관계없는 파일 보존을 합성 자료로 확인한다. 실제 Gateway 브라우저 시나리오에는 1시간 기한의 생성 자료를 사용해 큐→승인 ledger→관측의 기한 상속을 확인하고 자체 테스트 자료만 삭제한다.

10분 연속 추론 결과와 과거 CPU/CUDA 처리량은 이 보완 이전의 실행 증거다. 원래 결과·수치·실행 source SHA를 그대로 보존하며 새 정책이 그 실행에 포함됐다고 소급하지 않는다. 합성 벤치 helper의 생성 시각·1시간 계약 문맥 연결은 새로운 실행을 위한 변경이다. 이전 실행을 변경하거나 다시 수행했다는 뜻이 아니다. [10분 소크](INFERENCE-SOAK.md)와 [처리량 범위](INFERENCE-PERFORMANCE.md)를 함께 확인한다.

보완 후 전체 비HTTP 회귀 599건·하위 검사 189건과 전체 브라우저 25건이 통과했고, 이어 추가 CLI 경계 검사를 포함한 정책 검사 18건도 통과했다. 비HTTP 실행의 기존 skip 2건은 승인된 단독 GPU 슬롯과 Windows symlink 생성 권한이 없어 수행하지 않은 검사다. 별도의 서버 HTTP 검사는 60건을 통과했으며, 실제 15초 기한 경과 후 사진·관측·캐시 보고서의 접근 차단도 확인했다. 공개 집계는 [LOCAL-MATERIAL-RETENTION-VERIFICATION.json](reports/LOCAL-MATERIAL-RETENTION-VERIFICATION.json)이다. 원래 10분 실행은 commit `39e1106`에 별도로 보존돼 있다.
