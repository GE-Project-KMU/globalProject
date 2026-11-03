# 연구 자료의 실제 보관 정리

`scripts/research_retention.py`는 공개 연구자료 원본·취득 조각·개인별 파생물의 **수동 개별 파일 정리 도구**다. 자동 예약·백그라운드 삭제 작업을 설치하지 않는다. 기본은 dry-run이며 `purge --apply`에서만 만료된 등록 파일을 실제 unlink한다. 현재 학습 중인 자료는 등록 목록을 먼저 검토하고 학습·평가·이미지 출력 완료 후 최종 인벤토리를 고정한다.

기존 `agent.governance`의 학습 출처 만료·철회 차단을 대신하지 않는다. 두 도구는 역할이 다르다. 이 도구는 원본 archive/test 이미지/취득 부분 파일/개인별 결과의 물리 정리를 담당하고, 모델 가중치와 최소 provenance는 보존한다. 원본이 만료됐다고 모델을 운영 승인 상태로 바꾸지 않는다.

## 고정 기한과 범위

- 원래 acquisition/source/release/provenance JSON의 SHA와 expires_at을 고정한다. 현재 시각·파일 mtime·새로 기록한 다운로드 시각으로 기한을 늘리지 않는다.
- WIDER/CCPD 등 얼굴·번호판 원본과 RDD 취득 기록에는 기존7일 기한이 있다. A2D2도 기존 source/release 기한을 따른다. RTK는 `data/public/rtk/acquisition.json`의 원래90일 공개 연구자료 기한을 유지하며 새 취득시각이나 임의7일 소급값을 만들지 않는다.
- 여러 고정 근거가 있으면 가장 이른 expiry를 사용한다. expiry가 없는 근거에는 현재 날짜나 mtime을 대신 넣지 않고 등록을 거부한다.
- 실제 경로·SHA·기한을 한 번 등록한 뒤 같은 경로의 다른 내용이나 더 늦은 기한을 덮어쓰지 않는다. 새 영수증을 만들어도 기존 기한은 연장되지 않는다.
- `data/research/redaction`, `data/research/rdd2020`, `data/research/rdd2022`, `data/public/rtk`, `data/public/a2d2`와 명시한 `data/training/<experiment>`만 대상이다. `data`, `data/training`, 프로젝트 전체를 대상으로 지정할 수 없다.

학습 원본/test/crop/fullframe, source_member 등 개별 파일명이 담긴 release JSON, 개인별 예측 이미지/JSON, `*.preview.jpg`, `heldout-predictions`/`selected-validation-predictions`의 개별 파일을 명시적으로 등록한다. archive의 `.download`/`.part`/`.assembled` 취득 조각도 고정 SHA와 원본 source 기한으로 등록할 수 있다. 확장자 중 `.pt/.pth/.onnx` 등이 포함되면 부분 다운로드라도 모델 가중치로 보호한다.

모델 manifest, `provenance.json`, 취득 근거, 라이선스 근거와 개별 식별자 없는 집계 보고서는 보존한다. JSON/JSONL/CSV에 개별 이미지·source_member·sample/source ID·bbox 등 연결 정보가 있으면 개인별 자료로 분류하고, precision/recall/count만 있는 집계 자료와 구분한다. 알 수 없는 파일은 자동으로 삭제하지 않는다.

## 등록

private 경로를 담은 계획 파일과 인벤토리는 Git에서 무시되는 `data/research-retention` 등에 둔다. 실제 자료를 읽어 확인한 SHA로 아래 예시의 자리표시자를 교체한다. 예시는 실제 개인별 파일명이 아니다.

```json
{
  "schema_version": 1,
  "roots": ["data/public/rtk"],
  "entries": [{
    "path": "data/public/rtk/EXAMPLE_IMAGE.jpg",
    "kind": "source_image",
    "sha256": "PINNED_FILE_SHA256",
    "evidence": [{
      "path": "data/public/rtk/acquisition.json",
      "sha256": "PINNED_ACQUISITION_JSON_SHA256"
    }]
  }]
}
```

kind는 source_archive/source_image/source_annotation/derived_image/per_image_prediction/identifiable_manifest 중 지정한다. `expires_at`을 계획에 함께 넣으면 고정 근거의 가장 이른 기한과 정확히 같아야 한다. 원본 annotation의 TXT/XML/MAT도 명시적으로 등록할 수 있다. 모델·최소 provenance/공개 aggregate를 개인별 자료로 잘못 지정하면 거부한다.

```powershell
.\.venv\Scripts\python.exe scripts\research_retention.py register data/research-retention/plan.json
.\.venv\Scripts\python.exe scripts\research_retention.py coverage data/research-retention/plan.json --details data/research-retention/coverage-hashes.json
```

coverage는 명시한 roots만 읽기 전용으로 살펴본다. 누락 후보 개수와 종류, 보존/미분류 파일 개수를 점검하며 삭제 대상을 자동 등록하지 않는다. 등록/계획 여부 점검과 콘텐츠 해시 재검증은 구분한다. 출력·details에는 개인별 파일명 대신 개수/경로 hash를 남긴다. 미분류/누락 후보는 원래 취득 근거를 확인해 계획을 보완한 뒤 최종 등록한다. 동시 register/apply는 lock으로 거부하므로 한 작업씩 실행한다.

## dry-run과 명시적 실제 정리

```powershell
# 기본: 삭제하지 않고 만료 여부·고정 해시·범위를 확인한다.
.\.venv\Scripts\python.exe scripts\research_retention.py purge

# 학습·평가 종료를 확인한 운영자가 명시적으로 실제 정리한다.
.\.venv\Scripts\python.exe scripts\research_retention.py purge --apply
```

원본/파생 파일의 재귀 디렉터리 삭제는 수행하지 않는다. 만료 시점에도 고정 SHA와 내용이 같고 링크가 아니며 등록 scope 안에 있는 정확한 개별 파일만 삭제한다. 바뀐 파일, 새 미등록 파일, 알 수 없는 형식, symlink/junction과 가중치는 삭제하지 않는다. modified_refused/scope_refused가 있으면 남은 자료를 별도로 검토하며 강제 삭제로 우회하지 않는다.

계획 파일 자체도 private 파일명을 담으므로 등록 시 그 파일의 SHA와 가장 이른 기한을 함께 고정한다. `--apply`는 만료된 계획도 개별 삭제하고, 삭제/누락 처리한 인벤토리 행에서 원문 경로를 제거한다. 최소 path/file/evidence hash·기한·상태·감사만 보존한다. 감사180일 기록은 수동 apply 때 함께 정리한다. 인벤토리를 다시 쓰는 행위로 원본 자료 기한은 바뀌지 않는다.

실제 원본이 아직 만료되지 않았다면 `--apply`를 실행해도 정리 대상이 아니다. 이 문서 작성 과정의 회귀 테스트는 `.cache` 아래 별도 가짜 프로젝트의 합성 파일만 지웠으며 실제 연구 원본·학습 파일·모델은 삭제하거나 수정하지 않았다.
