# 개인정보 가족 최종 CPU 비교의 두 프로세스 실행

`scripts/compare_privacy_family_parallel.py`는 선택된 두 모델을 각각 하나의 CPU 프로세스에서 실행한다. 원 `evaluate_privacy_family.py`, 가족 프로토콜, 학습기, 공통 비교·추론·점수 함수는 수정하지 않았다. 이번 준비에서는 합성 GT·예측의 계약 검사와 실행 프로토콜만 작성했으며 실제 모델·평가 사진·GPU 추론은 실행하지 않았다.

각 프로세스는 동일한 전체 cohort를 `sample_id` 순서로 처리한다. 한 모델의 사례를 여러 작업으로 나누지 않고 이미지 안의 예측 순서도 바꾸지 않는다. 클래스가 알려진 범위의 GT·예측 필터, floor0.01, 기존 NMS·전처리·가중치·정밀도·운영 threshold를 유지한다. 결과 점수는 원 공통 `score()`와 기존 조립 함수를 그대로 호출한다. 각 worker는 app ORT1.31.1, CPU intra2/inter1, OpenCV1, CPUExecutionProvider만 허용한다.

실행 전 원 가족 프로토콜·동결 승자·원 `comparison_plan()` 재구성의 동일성·양 모델 양 클래스의 `numeric_gates()`를 검사한다. 이후 원 serial 실행과 같은 `{role}-attempt.json`을 exclusive 생성하고 flush·fsync한다. 따라서 serial과 parallel이 같은 cohort를 중복 실행할 수 없다. gate 실패 때는 TEST 라벨·픽셀을 읽거나 시도 파일을 만들지 않는다.

worker는 실제 실행 중인 부모의 직접 자식이어야 하고, 같은 `started` 시도 파일의 SHA·원 계획·실행 프로토콜에 묶여야 한다. 부모·계획·수치 gate를 재검사하며 오래된 worker 직접 재실행도 거부한다. 모델·가중치·출처 release·provenance·양성 proof·benchmark·source 해시를 시작과 종료에 확인한다. 두 worker 중 하나가 실패하면 이 실행기가 생성한 두 프로세스만 정리한다. 실패 시도와 부분 산출물을 보존하며 자동 재시도하지 않는다.

결과는 원 `fresh-comparison.json`, `regression-comparison.json`과 동일한 필드 및 모델별 AP·운영 점수·고정0.35·지연 구조를 사용한다. 새 스케줄 설명은 기존 `limits`에 추가하고 worker 구간 walltime은 별도 `execution-receipt.json`에 기록한다. 이 시간은 **worker 시작부터 종료까지**이며 입장 검사·파일 해시·최종 점수 조립·직렬화는 제외한다. 모델별 지연에는 두 프로세스의 CPU 경합이 포함되므로 단독 모델 속도 향상·10FPS·현장 처리 성능으로 해석하지 않는다.

합성 검사에서는 실제 두 자식 프로세스로 예측을 전달한 뒤 canonical serial 점수 조립과 dict 단위 완전 일치를 확인했다. 같은 confidence에서 FP가 TP보다 먼저 나오는 순서를 유지해 AP가 2/3인 경우, 모델별 서로 다른 운영 threshold, 고정0.35, median·p95도 확인했다. 공유 시도의 양방향 중복 차단·fsync, 불완전 worker·행 순서·runtime 차이 거부, 부모/worker/model/source/benchmark 변경 거부, 실패한 자식의 동료 정리와 실패 시도 보존을 검사했다. 이 합성 결과를 실제 학습 모델의 정확도나 병렬 추론 실측으로 표시하지 않는다.

사전 프로토콜은 이미 작성했으며 실제 실행은 root가 승자·threshold·canonical proof와 CPU 자원 사용을 확인한 후 별도로 승인한다.

```powershell
# 실제 실행 허가 전에는 아래 run 명령을 실행하지 않는다.
$env:CUDA_VISIBLE_DEVICES = ''
.\.venv\Scripts\python.exe scripts/compare_privacy_family_parallel.py run --role fresh --authorized
.\.venv\Scripts\python.exe scripts/compare_privacy_family_parallel.py run --role regression --authorized
```

원 fresh200·과거 regression1400 benchmark와 가족 동결은 그대로 참조한다. 모델당 worker 산출물·실행 snapshot·로그는 `data/training/privacy-family-v8-evaluation/parallel-{role}-v8`에만 저장한다. 원본 사진을 복사하지 않고 source의 최소 보관 기한을 상속한다. 실제 비교 전까지 원 시도·비교 파일을 만들지 않았으며 기존 모델·승인 상태·보관 목록·DPM8 관련 핀도 변경하지 않았다.
