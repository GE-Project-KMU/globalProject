# 비식별화 추론 옵션의 시험 봉인·validation 선택

`scripts/select_redaction_inference.py`는 CPU export와 앱 CPU validation을 분리한다. 실제 실행은 가중치 학습 완료를 확인한 후 수행한다. 도구 작성 과정에서는 합성 파일과 가짜 모델로만 검사했으며 실제 가중치·TEST·최종 모델 manifest를 읽지 않았다.

## CPU export

새 출력 폴더만 허용한다. 프로젝트 안의 전용 학습 Python을 사용하고 CUDA를 숨긴다.

```powershell
.\.venv\Scripts\python.exe scripts/setup_training.py run scripts/select_redaction_inference.py export --training-dir data/training/redaction-specialists-v4-efficient --release data/training/research-redaction-v4/release.json --validation-manifest data/training/research-redaction-v4/validation-only.json --validation-sha256 754958da4b13d1a80abcd40a64467342eb229f7df1b5e6e0e74132dc2963f7f8 --output data/training/redaction-inference-options-v4
```

전체 release는 내용을 JSON으로 해석하지 않고 원래 고정 SHA `4eeae0afdf67fcaf727e3e8b3a9b35486ef6f1126ae486da64637a7aec5bbd16`를 검사한다. validation-only508행(face400/plate108)과 출처 증거·기한을 따로 확인한다. 학습 폴더의 허용 입력은 training-config/candidate-freeze/confidence-calibration/provenance JSON, best.pt, model.onnx다. 원래 model.json·summary·로그·heldout/regression/baseline 시험 결과는 읽지 않는다.

candidate-freeze의 checkpoint·ONNX·calibration 해시와 학습·factory 해시를 검증한다. 표준 옵션은 원래 ONNX를 참조하고, 타일 옵션은 같은 checkpoint를 CPU에 strict load하여 새 ONNX로 내보낸다. 새 gradient 갱신은 없다. 원래 새 학습 provenance에 두 옵션의 manifest 경로를 연결하되 assets·기한·기존 데이터는 변경하지 않는다. 기존 baseline provenance로 바꾸지 않는다.

`option-standard/model.json`, `option-tiled/model.json`은 먼저 confidence0.01의 validation 전용 상태로 작성한다. `export-receipt.json`의 SHA는 명령 출력에서 기록한다. 변경·기존 출력·동시 실행을 감지하면 덮어쓰지 않는다. 중간 실패 폴더를 재사용하지 않는다.

## 앱 CPU validation과 최종 선택 동결

```powershell
.\.venv\Scripts\python.exe scripts/select_redaction_inference.py validate --options-root data/training/redaction-inference-options-v4 --export-receipt-sha256 EXPORT_RECEIPT_SHA256 --public-report docs/reports/REDACTION-INFERENCE-OPTION-SELECTION.json
```

validation-only의 고정 이미지 바이트 SHA를 확인하여 decode하고 실제 앱 ONNX Runtime CPU 경로로 예측한다. 같은7개 confidence grid와 NMS IoU0.5를 사용한다. 각 원본에서 알려진 클래스만 채점하며 다른 클래스의 미주석 검출을 FP로 세지 않는다. 이는 완전한 개인정보 검출 보증이 아니다. 같은 실물 GT와 예측은 IoU0.5에서 일대일 대응한다.

각 옵션의 confidence를 validation에서 고정한 뒤, 최소 클래스 Recall95%를 달성한 옵션이 있으면 그중 macro precision→최소 recall 순으로 선택한다. 둘 다 미달하면 최소 recall→macro precision 순이다. 모든 지표가 같으면 표준을 선택한다. 목표를 낮추거나 시험 결과를 보고 선택하지 않는다.

선택 완료 시 새 두 manifest의 confidence를 선택 값으로 갱신한다. 준비 단계 manifest SHA와 최종 manifest SHA는 다른 단계의 증거이며 둘 다 최종 영수증에 남는다. `option-selection-freeze.json`에 선택·두 threshold·ONNX/모델/validation 해시를 고정하며 재실행·변조·lock 변경은 거부한다. 이후 검증에는 이 최종 manifest SHA를 사용한다. 선택 전 export 영수증으로 수정된 최종 manifest를 재실행하려 하면 거부되는 것이 정상이다.

개별 bbox·GT·sample ID는 무시되는 로컬 `validation-details.json`에만 보관하며 원래 기한을 상속한다. 공개 aggregate에는 클래스별 점수·선택·해시·CPU predict latency만 기록한다. latency는 공유 호스트에서 전처리·추론·후처리를 포함한 predict 시간이며 decode·전체 시스템 처리량·실차 FPS 인수 결과와 다르다. source TTL·철회·배포 권한·production_approved=false는 유지한다. 최종 시험 봉인은 이 선택 동결 전까지 유지한다.
