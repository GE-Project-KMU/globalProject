# 학습 모델 검증·이식

모델 학습 완료와 프로젝트 인수 통과를 구분한다. 이 도구는 연구용 가중치의 출처·입출력·수치 일치와 안전한 파일 전달을 검증한다. 공개 데이터 성능이나 합성 입력의 출력 일치를 ZED 2i 실차 성능으로 표시하지 않는다.

## 가중치를 읽기 전

```powershell
.\.venv\Scripts\python.exe scripts\verify_trained_models.py data\training\lane-rtk-v1\lane.json --metadata-only
```

완료 manifest가 존재할 때만 실행한다. metadata-only는 ONNX와 checkpoint를 열지 않는다. research_only=true, production_approved=false, project_acceptance_eligible=false, source_licenses, train/validation/test 비어 있지 않은 분할을 확인한다. 원래 release 파일의 SHA256을 계산해 manifest의 dataset_release 및 provenance의 release_sha256과 대조하고, provenance에 기록된 학습 분할·만료·철회 상태를 검사한다. provenance에 test 자료가 포함되면 거부한다. 이 설명은 provenance 파일 자체의 별도 고정 SHA256 검증을 뜻하지 않는다. 이 단계 결과의 weight_hash_verified는 false다.

## 실제 런타임 비교

학습 및 validation threshold 고정이 끝난 뒤 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\verify_trained_models.py data\training\lane-rtk-v1\lane.json --checkpoint data\training\lane-rtk-v1\lane_best.pt --output data\training\lane-rtk-v1\verification.json
```

앱 Python3.14/ONNX Runtime CPU adapter와 별도 Python3.11/ONNX Runtime, checkpoint를 지정하면 PyTorch CPU 결과를 비교한다. worker는 CUDA_VISIBLE_DEVICES를 비워 GPU 학습과 경합하지 않는다. 모델 SHA 확인, ONNX checker, 외부 tensor 파일 차단, 단일 입출력과 finite 값, lane N×2×H×W 또는 YOLOX decoded N×anchors×(5+classes), 학습 전처리 일치를 확인한다. 기본 허용 오차는 rtol0.001/atol0.0001이며 최대·평균·p99 오차, RMSE, cosine과 상대 L2를 남긴다. 실제 마스크 일치율·foreground 교집합/합집합도 기록하고 양성 예측이 없으면 IoU는 null이다. 실패를 숨기기 위해 임의로 허용 오차를 늘리지 않는다.

이 프로젝트의 실제 RTK 모델에서 OpenCV5 기본 새 DNN engine이 일부 입력에 약1e23 크기의 잘못된 logit과 빈 마스크를 내는 문제를 재현했다. 같은 ONNX/checkpoint·전처리 입력을 받은 ONNX Runtime/PyTorch는 서로 같은 양성 마스크를 냈다. [OpenCV 공식 migration 안내](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)의 classic engine을 강제한 진단에서는 이 폭주가 사라졌다. 앱의 기본 추론은 ONNX Runtime1.31.1 CPU로 고정하고 OpenCV는 영상 처리에 사용한다. `OnnxModel(path,backend="opencv-classic")`는 명시적 진단 경로이며 자동 fallback하지 않는다. 과거 engine 오류가 포함된 평가를 모델의 최종 성능으로 인용하지 않는다. 변경 전 진단 보고서는 그대로 보존한다.

기본 입력은 합성 난수 이미지이며 정확도 평가가 아니다. `--image`에 프로젝트 안의 실제 validation 이미지를 지정할 수 있다. 임시 입력/출력 배열은 검사 종료 시 제거하고 보고서에 원본 픽셀·이미지 파일명·bbox 목록을 넣지 않는다. 별도 독립 holdout 평가는 평가 도구로 수행하며 모델·threshold 선택에 test를 사용하지 않는다. paint_threshold와 validation-only threshold_calibration은 이식 과정에서도 보존한다.

별도 학습 인터프리터 기본값은 Windows의 `.venv-training/python.exe`, Linux의 `.venv-training/bin/python`이며 `--training-python`으로 프로젝트 내부의 다른 경로를 지정할 수 있다. 검출기는 ORT/PyTorch raw tensor를 동일한 실제 앱 후처리에 넣고 class별 NMS 뒤 개수·클래스·최대 박스 오차·최대 신뢰도 오차·매칭 IoU도 보고한다. 후처리 일치가 raw allclose 실패를 덮어쓰지 않는다. 양쪽 예측이 비면 no_positive_predictions를 명시하며 양성 사례의 근거로 사용하지 않는다. 현재 후처리 비교 허용 오차는 박스0.05pixel/신뢰도0.0001이다.

## 연구용 모델 묶음

원시 출력이 일치해도 lane adapter 또는 비교한 엔진의 최종 마스크가 다르면 전체 artifact 검증은 실패한다. 패키징도 전체 artifact 판정과 lane 후처리 판정을 모두 요구한다. 검사 보고서의 실제 입력 `image_sha256`는 로컬 원본 기한을 적용하는 영상별 진단 자료이며 공개 package 요약에는 복사하지 않는다. 체크포인트·ONNX 해시는 모델 출처 증거로 구분한다. 도막의 [진단용 정확도 평가](LANE-PORTABILITY-DIAGNOSIS.md)는 이 엄격한 패키징 통과를 대신하지 않는다.

`artifact_redistribution_allowed:true`가 출처 사용조건 검토를 통해 명시된 모델만 build가 가능하다. 미지정 또는 false이면 로컬 묶음 생성 단계에서 차단한다. WIDER 등 제한이 확인된 자료의 파생 가중치를 허용 플래그 변경으로 우회하지 않는다. 현재 외부 업로드·Release 게시를 수행하는 명령은 제공하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts\model_artifacts.py build data\training\lane-rtk-v1\lane.json data\model-artifacts\lane-rtk-v1.zip --verification data\training\lane-rtk-v1\verification.json
```

전체 수치 검증을 통과하고 원래 manifest/가중치 SHA가 같은 verification 보고서를 요구한다. 묶음에는 ONNX와 모델 계약, 정제된 provenance, release 분할/라이선스 요약, 검증 요약, 파일별 SHA 목록만 포함한다. sample/source/observation 원문 ID, 이미지 경로·전체 파일명, 이미지/영상, 키·토큰·`.env`는 포함하지 않는다. 학습 출처의 identity hash/source hash, 만료·철회 표시와 train/validation 구분은 보존한다. original release/manifest SHA를 보존하면서 기존의 무시된 data/datasets 경로 없이도 검증할 수 있게 한다.

## 깨끗한 checkout에 설치

```powershell
.\.venv\Scripts\python.exe scripts\model_artifacts.py install data\model-artifacts\lane-rtk-v1.zip --sha256 PUBLISHER_CONFIRMED_SHA256 --id lane-rtk-v1
```

고정 SHA256을 필수로 받으며 `data/models/research/<id>`에 새 설치만 허용한다. 기존 모델을 덮어쓰지 않는다. ZIP의 파일 목록·각 SHA·원본 manifest SHA·경로·중복 항목·심볼릭 링크·압축 해제 크기를 확인한다. 가중치/provenance/release summary 참조를 실제 프로젝트 상대 경로로 바꾸고 설치 영수증에 원래와 설치된 manifest SHA를 함께 기록한다. 입력 provenance를 검증한 뒤에만 설치 디렉터리를 게시한다.

설치 후 기존 앱 환경에서 `python -m agent model-status data/models/research/lane-rtk-v1/model.json`과 위 verifier를 다시 실행한다. 자료 만료·철회로 needs_revalidation이 되면 이식된 모델에도 상태를 전파한다. 새 연구용 설치가 자동으로 운영 승인이나 독립 주행 검증 완료를 뜻하지 않는다.

향후 게시자가 허용한 공개 파일을 내려받으려면 저장소에서 검토된 catalog를 사용한다. catalog 형식은 `{schema_version:1,artifacts:[{id,url,sha256,research_only:true,artifact_redistribution_allowed:true}]}`이다. URL은 인증정보나 query가 없는 HTTPS 공개 주소여야 하며 고정 SHA와 파일 목록을 확인한 후에만 다운로드 결과를 확정한다.

```powershell
.\.venv\Scripts\python.exe scripts\model_artifacts.py download docs\approved-model-catalog.json --id lane-rtk-v1 --output data\model-artifacts\lane-rtk-v1.zip
```

예시의 catalog와 publisher SHA는 실제 게시/검토 전에는 존재하지 않는다. 이 문서는 외부 모델 게시 완료를 의미하지 않는다.
