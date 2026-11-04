# 새 cycle 보관 준비

이 문서는 **실제 asset scan·freeze·등록·이동·삭제를 아직 실행하지 않은 준비 상태**를 기록한다. 모든 ML·평가·성능·soak writer가 종료된 뒤 root 담당자가 확인하고 수동 실행한다. 기존17개 범위의 inventory/plan과 과거6개 control pin은 수정하지 않는다. 새 helper는 과거6개와 직전17범위 inventory/plan2개, 총8개 control hash를 별도로 확인한다.

`scripts/freeze_cycle2_retention.py`는 새 `cycle2-frozen-plan.json`, `cycle2-inventory.json`, `cycle2-summary.json`만 생성한다. 같은 이름이 있으면 거부한다. 기본 `audit`는 control JSON만 읽고 자료를 탐색하지 않는다. source 원기한을 mtime·현재 날짜·재다운로드로 연장하지 않는다. 원증거 hash가 바뀌거나 기록 중인 파일이 바뀌면 거부한다.

## 명시 범위

| 범위 | 근거와 기준 |
| --- | --- |
| RDD2022 blind source / 별도 release | 고정 source.json, 2026-11-10 17:07:24.286159 UTC. 원2개국200 release는 전송 실패로 미생성·선택 범위 |
| rdd2022-domain-v11 | 별도 full-transfer phase. 원blind source.json과 새 transfer-plan.json의 최소값, 최대11/10 17:07:24.286159 UTC. 정확한 `RDD2022_released_through_CRDDC2022.zip` 및 같은 basename의 `.zip.download`만 archive 분류. 검증 실패/부분파일도 동일기한 관리하며 실제 등록은 전송·CRC writer 종료 후 |
| research-pothole-rdd2022-us100-v7 | 별도 US100(50+50) 계획, 원source.json·완료 후 split-evidence.json. 원11/10 17:07:24.286159 UTC 유지 |
| pothole-validation-diagnostic-v7 | VAL-only GT/예측·개별 그림, 원v6 release/provenance 최소11/10 17:14:46.862846 UTC. aggregate 보존 |
| pothole-tiny-v7 | 고정 release와 완료 후 provenance, 최소 원행기한·최대11/10 17:14:46.862846 UTC |
| pothole-train-hardness-v8 / pothole-localization-v8 / pothole-small-v9 | 원v6 release/provenance·각 실험 plan 최소값, 최대11/10 17:14:46.862846 UTC. Small은 계획 단계에서 미생성이면 optional |
| pothole-small-v9-retry1 | 실제 순차 재시작 root. 원v6 release/provenance·복사 plan 최소값, 최대11/10 17:14:46.862846 UTC. 실패 원실험·receipt를 덮어쓰지 않음 |
| lane-cycle2-v5-retry1 | 별도 순차 retry root(optional). 원 lane source 최소11/10 17:07:26.069653 UTC |
| lane-cycle2-retry1-evaluation | 별도 CPU-gated artifact receipt·119 fresh/180 회귀 개별 수치 출력 root(optional). 원 lane source 최소11/10 17:07:26.069653 UTC. 개별 sample 정보가 없는 exact freeze/comparison control만 최소 provenance 보존 |
| pothole-family-v9 | family/control·US100/회귀의 개별 GT/예측 root(optional). 원blind source와 v6 최소값, 최대11/10 17:07:24.286159 UTC |
| pothole-resolution-v10 | 동일 CPU VAL652의 해상도 옵션·개별 예측. 원v6 release/provenance 최소값, 최대11/10 17:14:46.862846 UTC. 모델·비식별 집계 보존 |
| pothole-family-v10 | 별도 family/control·US100/회귀 root(optional). 원blind source와 v6 최소값, 최대11/10 17:07:24.286159 UTC |
| redaction-rescue-cpu-prototype-v8 | 원root 바로 아래 지정 NPY3개만11/04 17:37:03.942455 UTC cap. 만료 여부는 실제 UTC clock과 비교하며 이번 cycle의 추가 재사용0. ONNX·context·집계·최소 provenance 보존 |
| redaction-rescue-v8 | 계획 root(optional), 원 cycle2 privacy release 최소11/10 16:55:19.285523 UTC. prototype NPY 추가 재사용이나 기한 연장 없음 |
| privacy-family-v8-evaluation | 별도 family VAL·개별 metadata root(optional). 원 privacy release와 protocol.json의 최소값, 최대11/10 16:55:19.285523 UTC |
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

원실험/재시작 감독 receipt·소스·숫자 로그·GPU device memory/관측 timestamp는 개별 이미지 정보가 없으면 최소 provenance로 보존한다. 개별 이미지 ID·SHA·경로·좌표를 포함한 로그/예측은 원TTL 대상이다. family/control의 정확한 V9/V10 freeze·comparison format은 모델·코드·JSON proof pointer만 있고 개별 이미지 필드가 없을 때만 보존한다. 같은 format이라도 image ID/좌표가 섞이면 보호하지 않는다.

## 종료 인계와 수동 freeze

인계 JSON은 ignored `data/research-retention`에 두고 `cycle=public-cycle2-v7`, `writers_stopped=true`, timezone-aware `confirmed_at`, 코드의 **전체 정확 root 목록**을 담아 hash로 고정한다. 이는 운영자의 수동 확인 기록이며 계속 필수다. 추가 읽기 전용 guard는 이 프로젝트의 알려진 연구 script·대기 queue·감독/자손·메모리 observer·GPU worker·평가/성능 helper를 시작 전과 publication 전후에 검사하여 발견 시 차단한다. 자체 PID와 일반 MSA resident는 제외한다. 조회할 수 없는 프로세스나 알려지지 않은 writer까지 없다고 증명하지 않으며 프로세스를 종료하지 않는다. 인계가 틀리면 source proof·자료 목록을 읽기 전에 중단한다.

```powershell
# control만 확인; asset scan/등록/삭제 없음
.\.venv\Scripts\python.exe scripts/freeze_cycle2_retention.py audit
# 모든 writer 종료와 root 확인 뒤에만 실행. 등록 후 dry-run만, 삭제 없음.
.\.venv\Scripts\python.exe scripts/freeze_cycle2_retention.py freeze --writers-stopped-receipt LOCAL_HANDOFF_JSON --receipt-sha256 PINNED_HANDOFF_SHA
```

summary에는 건수·기한 집계·변경 거부·삭제0·미분류 현황만 넣는다. 개별 원파일명·이미지 ID·경로·예측 좌표는 ignored plan/inventory에만 둔다. 자동 예약이나 purge/apply 기능은 이 도구에 없다.

## cache 로그 이관 준비

모든 writer가 끝나면 담당자가 정확한 `.cache/cycle2*.log` 파일명과 SHA 목록을 제공한다. `prepare_log_migration()`은 pin된 목록·수동 종료 인계·알려진 writer guard를 확인하고 원/대상 resolved path가 프로젝트 안인지 검사한다. 대상은 `data/training/cycle2-logs/<동일이름>`이며 이미 있으면 거부한다. 개별 자료를 포함하지 않는 숫자 로그는 최소 provenance로 남긴다.

현재 구현은 원/대상과 SHA를 담은 **가역 이동 계획만** 만들며 실제 파일 이동은 하지 않는다. root가 계획을 검토한 뒤 정확한 파일만 수동 이동하고 결과 mapping receipt/SHA를 보존해야 한다. parent directory를 재귀 이동하거나 전체 `.cache`를 허용하지 않는다. 현재 실제 이동0이다.

합성 회귀는 정확 범위·sidecar·archive 경계·link 거부·dry-run 불변·모델 binary 보존·종료 전 scan 거부·과거 control 불변·짧은 cap 상속·로그 계획의 이동0을 검사한다. 실제 자료는 모든 writer 종료 후 별도로 확인한다.


## retry1 동결 입력과 정확 operator 로그

새 감독 계획에 고정된 원 preflight 로그 두 개와 `redaction-rescue-v8-env-repair`의 정확한 7개 control은 원경로·고정 SHA로 검사한다. 변경·누락이면 준비를 거부한다. 두 preflight 로그는 비개인 warning/error_type 및 빈 observer이며 설치 로그 두 개도 원경로 의존성이다. 이 예외는 임의 로그나 같은 namespace의 새 파일을 비개인 자료로 인정하지 않는다. 코드·원 queue/stop/spawn/import-control JSON은 이동 입력으로 허용하지 않는다.

`.cache/privacy-family-v8-operator[-retry1]/`에서는 코드에 명시한 phase 로그만 정확 파일명+SHA manifest로 받는다. strict verifier stdout의 개별 이미지 SHA/좌표는 원 PRIVACY_CAP 이하로 관리한다. private 로그의 이동 목적지는 `cycle2-logs/operator-original--<이름>` 또는 `operator-retry1--<이름>`으로 분리한다. 비개인 aggregate 검사에 통과한 observer 로그는 보존한다. 실제 이동은 계속 수동이다. 전/후 guard는 TOCTOU를 완전히 없애는 락이 아니므로 writer를 새로 시작하지 않는 운영자 통제도 필요하다.

### 미승인 synthetic MP4 처리 제안

soak의 `generated-nonpersonal.mp4`는 일반 영상 suffix 허용에 추가하지 않았다. 후속 승인 시 정확 basename·생성 설정·파일 SHA·순수 합성 증거에 결합하고 생성 후 2시간과 원 source cap의 최소로 제한하는 방안을 제안한다. 현재는 미분류 검토 대상이며 이 문서만으로 등록·삭제하거나 2시간 cap을 기존 자료에 소급하지 않는다.
