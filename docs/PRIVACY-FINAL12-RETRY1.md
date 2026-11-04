# fixed12 CPU export retry1

초기 exporter의 provenance 검사에 오류가 있었습니다. 원 학습은 `load_development()`가 반환한 얼굴 TRAIN3,000장과 전체 VAL508장, 총3,508개 항목을 사용했습니다. exporter가 development 전체5,501개를 요구하여 사용하지 않은 번호판 TRAIN1,993개까지 provenance에 있어야 한다고 잘못 판정했습니다. 초기 시도는 이 검사에서 실패했고 새 checkpoint·ONNX·manifest를 쓰지 않았습니다.

retry1은 이 소프트웨어 검사와 출력 namespace만 분리해 수정합니다. 원 학습 소스7bc, 학습 계획6bb, 사전에 고정한 최종12epoch EMA, 실제71,969회 업데이트, recovery SHA `0d72cb7f8655cef70c2f99f3fab45e884479eab945aada56454bd51bb9697e98`는 유지합니다. 새 학습·데이터·epoch 선택·threshold 선택은 없습니다. 최종12epoch 선택은 원 [사전 선언](reports/PRIVACY-FAMILY-V9-PREDECLARATION.json)과 [초기 export 계획](reports/PRIVACY-RESCUE-V8-FINAL12-EXPORT-PLAN.json)의 이력을 그대로 참조합니다. 학습 종료 뒤 생성한 retry 계획은 선택을 다시 선언하는 문서가 아니라 검사 오류를 고친 후속 실행 계획입니다.

원 exporter e8de, 계획 bca, CPU 감독3b449, 실패 receipt·log·attempt는 수정하지 않았습니다. 원 실패 receipt의 `parent_artifacts_preserved_verified:false`도 그대로 남깁니다. root의 별도 보존 감사7a4a는 원9개 입력 SHA가 모두 유지됐음을 독립적으로 확인했으며 retry 계획이 이 감사와 중지 증거를 연결합니다. 실패를 성공으로 바꿔 기록하지 않습니다.

수정한 검사는 단순히 항목 수를 줄이는 방식이 아닙니다. 원 trainer의 실제 predicate인 `(split=train and evaluation_class=face) or split=validation`를 적용한 항목과 부모 provenance를 정확히 대조합니다. sample/source/observation identity, source 경로·SHA, 원 만료일과 identity hash가 모두 같아야 합니다. 같은3,508개 수량이라도 미사용 번호판 TRAIN을 대신 넣거나 source identity·TTL을 바꾸면 거부합니다. 실제 JSON metadata 감사는 [공개 집계](reports/PRIVACY-FINAL12-RETRY1-MEMBERSHIP.json)에 기록했고, 사진·recovery·모델을 읽지 않았습니다. 합성5,501행 검사도 원 `load_development`의 AST predicate와 trainer의 `trainrows + valrows` 호출을 직접 대조합니다.

새 [export 계획](reports/PRIVACY-RESCUE-V8-FINAL12-RETRY1-EXPORT-PLAN.json)과 `scripts/export_redaction_final12_retry1.py`를 고정했습니다. 실제 CPU 실행은 root의 검토·승인 후에만 다음 감독 도구로 수행합니다.

```powershell
.venv/Scripts/python.exe scripts/supervise_final12_cpu_export_retry1.py --cpu-export-approved
```

새 출력은 `data/training/redaction-rescue-v8-final12-retry1/`, log는 `.cache/cycle2-final12-cpu-export-retry1.log`입니다. `root-export-execution.json`과 기존과 같은 export receipt·candidate freeze 형식을 사용하며, 원 실패 execution SHA와 원 export 계획 SHA를 추가로 연결합니다. `original_trainer_subset`은 얼굴 TRAIN3,000/얼굴 VAL400/번호판 VAL108/합계3,508로 명시합니다.

완료epoch12·정확한 자식 종료·원 source/plan·TTL·동일 recovery와 전체 EMA state 검증은 유지합니다. legacy parameter·BN bytes와 합성 CPU raw prefix102,900개 일치 검사를 그대로 수행하며 출력은333,300×7입니다. 새 provenance는 실제 사용한3,508개 identity와 원 TTL을 상속하고 부모 삭제 참조를 비우며 새 모델만 참조합니다. confidence.01은 가족 VAL 평가 전 임시 floor입니다. 엄격한 CPU 수치 검증과 V10의 별도 VAL·blind 평가가 필요하고 운영 승격은 없습니다.

이 준비 과정에서 실제 재실행·모델·recovery·사진 읽기·추론·GPU 실행은 하지 않았습니다. 원 보관기한과 모든 실패 자료를 유지합니다. retry1에도 자동 재시도나 기존 namespace 덮어쓰기는 허용하지 않습니다.
