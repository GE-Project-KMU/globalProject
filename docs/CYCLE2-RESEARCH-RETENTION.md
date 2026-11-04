# 새 cycle 보관 준비

이 문서는 **실제 asset scan·freeze·등록·이동·삭제를 아직 실행하지 않은 준비 상태**를 기록한다. 모든 ML·평가·성능·soak writer가 종료된 뒤 root 담당자가 확인하고 수동 실행한다. 기존17개 범위의 inventory/plan과 과거6개 control pin은 수정하지 않는다. 새 helper는 과거6개와 직전17범위 inventory/plan2개, 총8개 control hash를 별도로 확인한다.

`scripts/freeze_cycle2_retention.py`는 새 `cycle2-frozen-plan.json`, `cycle2-inventory.json`, `cycle2-summary.json`만 생성한다. 같은 이름이 있으면 거부한다. 기본 `audit`는 control JSON만 읽고 자료를 탐색하지 않는다. source 원기한을 mtime·현재 날짜·재다운로드로 연장하지 않는다. 원증거 hash가 바뀌거나 기록 중인 파일이 바뀌면 거부한다.

## 명시 범위

| 범위 | 근거와 기준 |
| --- | --- |
| RDD2022 blind source / 별도 release | 고정 source.json, 2026-11-10 17:07:24.286159 UTC. 원2개국200 release는 전송 실패로 미생성·선택 범위 |
| research-pothole-rdd2022-us100-v7 | 별도 US100(50+50) 계획, 원source.json·완료 후 split-evidence.json. 원11/10 17:07:24.286159 UTC 유지 |
| pothole-validation-diagnostic-v7 | VAL-only GT/예측·개별 그림, 원v6 release/provenance 최소11/10 17:14:46.862846 UTC. aggregate 보존 |
| pothole-tiny-v7 | 고정 release와 완료 후 provenance, 최소 원행기한·최대11/10 17:14:46.862846 UTC |
| pothole-train-hardness-v8 / pothole-localization-v8 / pothole-small-v9 | 원v6 release/provenance·각 실험 plan 최소값, 최대11/10 17:14:46.862846 UTC. Small은 계획 단계에서 미생성이면 optional |
| redaction-rescue-cpu-prototype-v8 | 원root 바로 아래 지정 NPY3개만11/04 17:37:03.942455 UTC cap. 이미 만료됐으며 추가 재사용 금지. ONNX·context·집계·최소 provenance 보존 |
| redaction-rescue-v8 | 계획 root(optional), 원 cycle2 privacy release 최소11/10 16:55:19.285523 UTC. 만료된 prototype NPY를 학습/추론에 재사용하지 않음 |
| a2d2-cycle2-v5 / lane-cycle2-v5 | 고정 source.json·완료 후 provenance, 11/10 17:07:26.069653 UTC |
| research-redaction-cycle2-v7 / redaction-p2-cycle2-v7 | 고정 release 행별 expires_at·완료 후 provenance, 최소11/10 16:55:19.285523 UTC |
| public-nextcycle / cycle2-evaluation / cycle2-logs | task별 근거의 최소값, 보수적으로 privacy 최소기한 상속 |
| gpu-model-sets/io-overlap-v1 | 원 FP32 case plan, 로컬 analysis가 더 짧으면 그 cap 상속 |
| gpu-model-sets/cycle2-final-v7 및 하위 soak | 새 모델·case plan·원증거 최소값, 로컬 analysis cap도 상속 |

원경로는 코드의 정확한 allowlist로 고정한다. `.cache`나 `data/research` 전체를 부모 범위로 허용하지 않는다. 개인정보 `source-qc.json`의 `expiry`는 기존 retention key가 아니므로 파일을 수정하거나 범용 key를 늘리지 않고, 고정 `release.json`의 행별 `expires_at`을 주근거로 사용한다.

prototype의 정확한 `redaction-rescue-synthetic-artifact-context-v1` format에만 두 비표준 날짜 key의 adapter를 둔다. `synthetic_tensor_temporary_expiry`와 `legacy_source_earliest_expiry`의 최소를 읽으며, 일반 JSON의 같은 key나 mtime/현재시각으로 새 기한을 만들지 않는다. source context의 기존 SHA를 고정한다. 대상은 `synthetic-input.npy`, `baseline-torch.npy`, `candidate-torch.npy`의 정확한 세 파일뿐이다. 중첩 폴더·다른 NPY/NPZ는 임의 등록하지 않는다. 관리용 hash 확인과 달리, 만료 파일을 `np.load`하여 재추론/학습하는 경로는 없다. 이 만료 cap은 새 trainer에 사용한 적 없는 NPY를 새로 유효하게 만드는 근거가 아니다.

## ZIP range-cache 전용 종류

새 `source_range`는 정확히 `data/research/rdd2022-blind-v7/range-cache/<start>-<end>.bin`만 허용한다. 각 범위는4MiB 이하, 길이는 `end-start+1`과 같아야 하며 archive13,264,172,619바이트를 넘을 수 없다. link/junction·다른 디렉터리·모델 `.bin`·다른 이름은 거부한다. 옆의 동일 stem `.sha256`가 정확한64자 SHA이며 실제 바이트와 일치해야 한다. source JSON의 고정 원기한 evidence도 필수다.

일반 `source_archive`에는 `.bin`을 추가하지 않았다. 가중치·최소 provenance·source/hash 근거·비식별 aggregate는 보존한다. 범위를 알 수 없는 `.partial`이나 binary 등은 임의 삭제 대상으로 삼지 않고 미분류 건수와 검토 필요를 보고한다. 미분류 자료를 완전 관리했다고 표시하지 않는다.

## 종료 인계와 수동 freeze

인계 JSON은 ignored `data/research-retention`에 두고 `cycle=public-cycle2-v7`, `writers_stopped=true`, timezone-aware `confirmed_at`, 코드의 **전체 정확 root 목록**을 담아 hash로 고정한다. 이는 외부 writer를 자동 탐지하거나 종료하는 기능이 아니라 운영자의 확인 기록이다. 인계가 틀리면 source proof·자료 목록을 읽기 전에 중단한다.

```powershell
# control만 확인; asset scan/등록/삭제 없음
.\.venv\Scripts\python.exe scripts/freeze_cycle2_retention.py audit
# 모든 writer 종료와 root 확인 뒤에만 실행. 등록 후 dry-run만, 삭제 없음.
.\.venv\Scripts\python.exe scripts/freeze_cycle2_retention.py freeze --writers-stopped-receipt LOCAL_HANDOFF_JSON --receipt-sha256 PINNED_HANDOFF_SHA
```

summary에는 건수·기한 집계·변경 거부·삭제0·미분류 현황만 넣는다. 개별 원파일명·이미지 ID·경로·예측 좌표는 ignored plan/inventory에만 둔다. 자동 예약이나 purge/apply 기능은 이 도구에 없다.

## cache 로그 이관 준비

모든 writer가 끝나면 담당자가 정확한 `.cache/cycle2*.log` 파일명과 SHA 목록을 제공한다. `prepare_log_migration()`은 pin된 목록·global 종료 인계를 확인하고 원/대상 resolved path가 프로젝트 안인지 검사한다. 대상은 `data/training/cycle2-logs/<동일이름>`이며 이미 있으면 거부한다. 개별 자료를 포함하지 않는 숫자 로그는 최소 provenance로 남긴다.

현재 구현은 원/대상과 SHA를 담은 **가역 이동 계획만** 만들며 실제 파일 이동은 하지 않는다. root가 계획을 검토한 뒤 정확한 파일만 수동 이동하고 결과 mapping receipt/SHA를 보존해야 한다. parent directory를 재귀 이동하거나 전체 `.cache`를 허용하지 않는다. 현재 실제 이동0이다.

합성 회귀는 정확 범위·sidecar·archive 경계·link 거부·dry-run 불변·모델 binary 보존·종료 전 scan 거부·과거 control 불변·짧은 cap 상속·로그 계획의 이동0을 검사한다. 실제 자료는 모든 writer 종료 후 별도로 확인한다.
