# 도막 정확도 개선 실험

2026-11-04. 기존 모델·시험·보관기한을 유지하고 새 후보 하나를 학습한다. 개발 자료799장 train/100장 validation만 학습 프로세스가 읽는다. 전체999장 release는 SHA 및 provenance 식별에만 사용하며, 새 기록의 TEST100장은 최종 후보·threshold 고정 후 평가한다. 이전 TEST100장은 이미 확인한 회귀 benchmark로 구분한다. 촬영 날짜·recording은 분리됐지만 실제 물리 도로 독립성은 확인하지 못했다.

원인 검토에서 RGB 정규화·마스크 클래스 대응의 명백한 오류는 발견하지 못했다. 원본 해상도의 GT를 nearest로 축소/복원하는 개발 자료 진단은512에서 IoU65.96%,1024에서77.27%였다. 이는 모델 성능이나 달성 가능한 수학적 상한이 아니라 라벨 aliasing 진단이다. 시험 자료는 이 진단에 쓰지 않았다.

후보는 다음 조합을 미리 고정했다. 개별 변경의 인과 효과를 분리한 ablation 결과로 주장하지 않는다.

- DeepLabV3-MobileNetV3-Large pretrained 초기화, 추론1024
- 원래 픽셀 두께를 더 유지하는640 학습 patch, scale·광도·반전 및 train-only 도막 중심75% sampling
- batch4, encoder LR1e-4/decoder1e-3 AdamW,3epoch warmup/cosine, EMA.99
- 매 train() 호출 후 BN.eval()로 running statistics 고정, affine gradient는 유지
- CE foreground weight4+Dice를 반씩 사용, ignore255 제외
- ASPP dilation6/12/18 연구 설정을 metadata로 명시; 기존12/24/36 모델은 변경하지 않음
- 최대60epoch, 최소15, validation IoU 유의 개선.001 없이12epoch면 종료
- checkpoint·threshold 평가 좌표를 앱과 같은 원본 해상도로 통일

[Torchvision0.22.1 소스](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/segmentation/deeplabv3.py)의 MobileNet backbone은 output stride16을 사용한다. [공식 저장소의 dilation 논의](https://github.com/pytorch/vision/issues/7956)를 참고해 작은 도막을 위한 대안을 실험한다. 이 설정 변경을 기존 학습 실패의 확정 원인으로 단정하지 않는다.

```powershell
.venv\Scripts\python.exe scripts/setup_training.py run scripts/improve_lane.py data/public/a2d2-improvement-v2/development-manifest.json data/public/a2d2-improvement-v2/manifest.json data/training/lane-a2d2-v2 --epochs 60 --min-epochs 15 --patience 12 --batch-size 4 --patch-size 640 --input-size 1024
```

실행 전 `pretraining-config.json`에 source/factory/development/full SHA·seed·recipe를 기록한다. 소스와 입력을 학습 중에 바꾸지 않는다. 기존 `begin_training`의 만료·철회·출력 폴더 재사용 거부를 유지하고 원본을 불필요하게 복사하지 않는다.

완료 후 validation-only 앱 calibration, 실제 양성 validation 표본의 Torch/ORT/앱 수치·마스크 비교를 먼저 수행한다. 새100장에서는 이전 baseline과 후보를 같은 자료로 비교한다. fresh/regression 평가 계획의 JSON은 새 시험 추론 전에 고정했다. 공개 보고서에는 집계만, 개별 결과와 모델 가중치는 프로젝트 내부에 보관한다. 도막 원본 annotation은 마모율 O/G/V가 아니므로 R MAE를 산출하지 않는다.
