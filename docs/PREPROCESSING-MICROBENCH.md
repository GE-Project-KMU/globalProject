# 합성 입력 전처리 복사 비용 비교

공개·사용자 사진, 모델 manifest 및 가중치를 읽지 않고 uint8 합성 패턴으로 비교했다. 실제 network forward와 GPU 작업은 0회다. 크기 변경·색 변환·정규화 값·연산 순서·임계값을 유지하면서 불필요한 float32 배열 복사를 줄이는 후보만 조사했다.

검출 입력은 transpose 후 `astype(float32)`와 `setInput`의 contiguous 사본을 만들던 경로를 직접 `ascontiguousarray(..., dtype=float32)`로 바꾼다. 차선은 원래의 `/255`, `-mean`, `/std` 순서를 그대로 유지해 새로 할당한 float32 RGB 배열에서 처리하고 마지막 NCHW 변환을 contiguous로 만든다. 원 입력 프레임은 바꾸지 않는다. 역수를 곱하거나 연산을 합치지 않는다.

모든 uint8 값 0~255, 범위 전체 패턴·체커보드·난수·검정·흰색, 64×48·65×49·720p·1080p를 검사했다. 앱 환경 NumPy2.5.4/OpenCV5.0과 별도 NumPy1.26.4/OpenCV4.10 환경에서 각각 828건이 **비트 단위로 일치**했다. 이는 각 환경 안에서의 전후 비교이며 OpenCV 버전 간 결과 동일성을 새로 주장하지 않는다. 적용 후 실제 `OnnxModel.predict`도 `setInput`에서 중단하여 추론 없이 동일 828건을 확인했다. 출력은 float32 C-contiguous이고 후속 `ascontiguousarray`가 같은 객체를 반환하며 원 입력도 불변이었다.

## CPU 마이크로벤치 결과

각 입력 크기의 고정 패턴에 대해 후보/기존을 각각 3회 워밍업하고 ABBA 순서 8블록, 각 16표본을 측정했다. 표는 네 가지 원본 크기에 걸친 중앙값의 범위다. 변경하지 않은 `setInput`의 finite 검사와 contiguous 마샬링까지 포함한다.

| 입력 텐서 | 기존 중앙값 | 후보 중앙값 | 절약 시간 |
|---|---:|---:|---:|
| 포트홀 640² | 4.00~5.58ms | 2.30~3.01ms | 1.57~2.58ms |
| 개인정보 1920² | 41.12~41.70ms | 22.26~23.56ms | 17.77~19.25ms |
| 차선 1024² | 33.16~34.74ms | 21.17~22.41ms | 12.00~12.66ms |

각 종류의 32개 짝지은 블록 모두에서 후보가 빨랐다. 동시 학습의 CPU·메모리 대역폭 경합은 완전히 제거하지 못했으므로 독립된 장비 벤치마크가 아니다. 세 종류의 절약 시간을 더해 전체 처리량 증가로 환산하지 않는다. 병렬 실행·GPU 전송·추론·후처리·PNG/JSON 저장은 이 마이크로벤치로 측정하지 않았다.

측정 후 승인된 두 전처리 위치에 복사 변경을 적용했다. 후보 검사 15건 및 관련 프레임워크 검사 147건이 통과했고, 적용 경로를 추가 검사하는 3개 테스트도 통과했다. 승인된 단독 GPU 슬롯이 필요한 기존 검사는 건너뛰었고 학습된 외부 baseline 추론 검사는 이번 조사에서 명시적으로 제외했다. GPU 지원이나 새로운 학습 모델 검증의 대체 증거로 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_preprocessing_microbench.py -q --basetemp .cache/pytest-preprocessing-applied
.\.venv\Scripts\python.exe scripts/benchmark_preprocessing.py --output docs/reports/NEW-PREPROCESSING-MICROBENCH.json
```

원 측정 집계는 [PREPROCESSING-MICROBENCH.json](reports/PREPROCESSING-MICROBENCH.json)에 보존한다. 측정 당시 runtime SHA와 적용 후 SHA를 구분한다. 과거 `afb734a`의 약6FPS 및 CPU 기준 처리량은 변경 전의 실행 증거이며 이번 복사 최적화가 포함된 수치로 소급하지 않는다. 새로운 전체 처리량·학습 모델 수치 일치는 별도 고정 실행으로 확인해야 한다.
