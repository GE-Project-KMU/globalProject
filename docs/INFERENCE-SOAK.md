# 장비 없는 10분 연속 CPU 추론 검사

`scripts/probe_inference_soak.py`는 기존 검증된 연구 모델 3개를 실제 ONNX Runtime CPU로 실행하는 소프트웨어 검사다. 새 학습 모델이나 현장 정확도 검증을 대신하지 않는다. 입력은 프로그램이 만든 64×48 비인물 기하학 패턴이며 사용자 촬영 자료나 공개 원본 사진을 열지 않는다. 외부 업로드와 운영 관측 생성은 없다.

실제 `pipeline.analyze_video`의 준비 조건·전처리·추론·추적·마스킹·PNG 및 JSON 저장을 사용한다. 프로브 한 프로세스에서만 `replay`를 명시적 합성 프레임 생성기로 대체한다. 실제 MP4 디코딩·카메라 프레임 취득의 증거는 아니다. 생성기의 PTS는 `frame_index/10`인 가상 시각이며 실제 초당 처리량이나 카메라 FPS가 아니다.

실행은 old 포트홀 모델, 기존 수치 검증된 차선 모델, 기존 선택된 개인정보 타일 모델을 사용한다. 원래 manifest·가중치의 SHA를 전후 대조한다. 운영 준비 검사를 우회하거나 `production_approved`를 변경하지 않는다. 기본 CPU 프로파일의 모델별 ORT intra-op 2개·inter-op 1개 스레드와 CPUExecutionProvider만 허용하며, OpenCV 스레드는 2개다. GPU worker·학습 작업·기존 서버에는 접근하거나 재시작하지 않는다.

## 시간과 메모리

세 모델 로드 후 합성 생성기가 시작될 때 단조 시계로 600초 기한을 설정한다. 기한에 도달하면 다음 프레임을 시작하지 않는다. 이미 시작한 한 프레임의 CPU 추론과 최종 JSON 저장은 기한 뒤에 끝날 수 있으므로 실제 완료 시간도 별도로 기록한다. 출력은 새 `data/training/hardware-free-soak-v6/` 한 곳으로 제한하며 기존 출력은 덮어쓰지 않는다. 전체 산출물 예산은 250MiB다.

메모리는 Windows에서 ctypes의 현재 프로세스 Working Set으로 측정한다. 다른 OS의 `resource` 대체 값은 현재 RSS가 아닌 프로세스의 누적 최고 메모리라고 명시한다. 첫 30초의 워밍업 표본을 제외하고 최솟값·최댓값·중앙값·첫값 대비 마지막값과 표본 시간을 집계한다. 이 값에는 프로브 자체의 저장·결과 목록 관리 비용도 포함되며 메모리 누수가 전혀 없다는 증명이 아니다.

실행 후 모든 프레임의 PNG 수와 저장된 SHA를 대조하고 모델 객체의 close 호출·회수, 새 Python 스레드 잔류 없음, 이전 manifest·가중치 불변을 확인한다. 도구는 subprocess를 만들지 않으며 정상 종료 시 해당 프로브 프로세스도 끝난다. 소프트웨어 처리 FPS는 동시 학습 부하 아래의 합성 검사 값이고 독립된 장비 성능 벤치마크가 아니다.

## 실행

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:TEMP="$PWD\.cache\tmp"
$env:TMP=$env:TEMP
$env:PYTHONPYCACHEPREFIX="$PWD\.cache\pycache"
.\.venv\Scripts\python.exe scripts/probe_inference_soak.py
```

진행 집계는 비공개 출력 폴더의 `progress.json`, 완료 결과는 `summary.json`, 메모리 표본은 `memory-samples.json`에 저장한다. 개별 생성 사진·bbox·추적 식별자가 있는 분석 JSON은 로컬에만 둔다. 결과의 기한은 기존 모델 provenance의 원래 기한 중 최솟값으로 강화하며 취득일이나 보관 기한을 새로 추정하지 않는다. 오류 시 `failure.json`을 남기고 성공으로 표시하지 않는다. 원본 삭제와 자동 보관 스케줄은 없다.

이 검사는 60분 카메라 연속 취득, 10FPS 목표, GNSS·거리·마모 측정, 실도로 인식 정확도, 실제 개인정보 검수·공유 승인을 주장하지 않는다. 실제 완료 수치와 합격 여부는 실행이 끝난 뒤 별도 집계 보고서에서 확인해야 한다.

## 완료 결과

수정 전 소크 실행은 600.78초에 정상 종료했고 633프레임 및 PNG 633개를 기록했다. 오류는 0건이며 모델 전후 SHA가 같고 모델 객체·프로브 Python 스레드는 정리됐다. 30초 워밍업 제외 603개 메모리 표본의 첫값 대비 마지막값 증가량은 73,728바이트였으며 Working Set은 약856MiB였다. 소프트웨어 합성 처리량은 약1.054FPS다. 실제 카메라 처리 성능으로 인용하지 않는다.

개별 생성 사진 및 분석 JSON의 기한은 기존 모델 출처의 최솟값인 2026-11-10 16:55:19.285523 UTC로 강화했다. 원래 실행 스크립트는 비공개 실행 폴더에 별도 보존했다. 집계는 [HARDWARE-FREE-INFERENCE-SOAK.json](reports/HARDWARE-FREE-INFERENCE-SOAK.json)에 있다. 이 실행은 이후 발견한 제품의 미검토 이미지 보관 정책 수정이 포함된 실행이 아니며, 후속 수정을 소급 입증하지 않는다.
