# FP32 추론과 PNG·품질검사 overlap

`--io-profile overlap`은 같은 FP32 추론을 유지하면서 PNG 저장·SHA 계산 및 품질검사를 추론과 겹치는 선택 기능이다. 기본 `default`는 직렬 처리다. 가중치·정규화·threshold·NMS·마스킹·PNG 압축은 바꾸지 않고 프레임을 생략하지 않는다. [공개 집계](reports/PIPELINE-IO-OVERLAP-RESULTS.json)에 실행 source·profile·원수치증거·사전 plan·결과 SHA와 단계별 시간을 기록했다.

## 구현과 안전 조건

한 worker에서 최대 PNG snapshot1개와 품질검사1개만 대기/실행한다. 쓰기가 밀리면 다음 snapshot 제출을 기다리므로 저장 queue가 무제한 커지지 않는다. 품질 결과는 관측에 쓰기 전에 기다리며 다음 프레임으로 넘어가기 전에 해당 프레임 처리를 끝낸다. PNG는 snapshot을 사용하여 재사용된 frame buffer가 저장 결과를 바꾸지 못하게 한다.

overlap은 새 출력 경로만 허용한다. 임시 PNG를 동일 encoder로 저장하고 SHA·원기한을 확인한 뒤 최종 경로로 옮긴다. 모든 쓰기·품질 작업을 flush한 후에만 analysis/review JSON을 저장하거나 성공을 반환한다. 원소스 기한·미검토7일 cap·승인 사본 기준은 연장하지 않는다. 예외가 나면 작업을 기다리고, 해당 시도가 새로 만든 바이트가 변하지 않은 PNG만 SHA 확인 후 정리한다. 다른 파일이나 변경된 파일을 임의 삭제하지 않는다.

측정 후 파일 게시를 같은 parent의 원자적 hard link로 보강했다. 목적지가 동시에 생성되면 실패하며 POSIX `rename`처럼 다른 파일을 덮어쓰지 않는다. 정리 guard 실패는 원래 write/processing 예외의 note로 남겨 primary failure를 보존한다. 이는 filesystem 동작 변경이며 metadata-only 수정이 아니다. 이전 실행 writer bytes/SHA는 별도 보존했고 공개 집계에 변경 전후 SHA가 있다. 아래11.44FPS는 **보강 이전 소스**의 결과다. 현재 소스에 소급하지 않으며 최종 새 모델의 GPU 성능·soak에서 다시 확인해야 한다. 보강 후 CPU102개 회귀는 통과했고 추가 GPU 실행은0이다.

```powershell
.\.venv\Scripts\python.exe -m agent workflow LOCAL_PREPARED_CONFIG --cpu-profile fast --io-profile overlap
.\.venv\Scripts\python.exe -m agent analyze LOCAL_SOURCE --output NEW_OUTPUT --pothole-manifest LOCAL_POTHOLE --lane-manifest LOCAL_LANE --redaction-manifest LOCAL_PRIVACY --inference-backend cuda --gpu-profile LOCAL_REGISTERED_PROFILE --cpu-profile fast --io-profile overlap --source-expires-at ORIGINAL_AWARE_UTC_DEADLINE
```

prepared config에는 `"io_profile":"overlap"`을 추가할 수 있다. 모델의 CPU/GPU profile guard는 그대로 적용된다. IO 기능이 다른 가중치의 GPU 승인이나 새 precision 증거를 부여하지 않는다.

## 실제 GPU ABBA

고정된 v6 연구 후보 세 모델과 기존 FP32 native160 proof를 사용했다. exact inference source6개·부모 manifest·weights·case·proof를 실제 admission guard로 재확인했고 우회하지 않았다. 추론 수학·backend 소스는 변경하지 않았으며 새 pipeline 출력은 별도로 검사했다. 이 후보 조합은 운영 모델 승격이 아니다.

GPU를 독점한 시간대에 모델별5회 워밍 후 비개인 합성1280×720 프레임16개씩 직렬/overlap/overlap/직렬을 한 번 실행했다.

| 구간 | warmed 처리율 |
| --- | ---: |
| 직렬 A / A | 7.48998 / 7.42976FPS |
| overlap B / B | 11.53690 / 11.34273FPS |
| median | 직렬7.45987 / overlap11.43981FPS |

같은 실행의 median 비율은1.5335배다. 네 구간의 PNG 바이트·관측 순서·품질·추적·차선 결과가 모두 같았다. 약26ms의 PNG와 약10~11ms 품질 작업 자체를 생략한 것이 아니라 추론 중 실행했다. 단계별 작업 시간의 합은 overlap wall time과 같지 않다.

측정 당시 별도의2-thread Tiny CPU forward/backward preflight가 겹쳤을 가능성이 뒤늦게 확인됐다. 정확한 중첩은 독립 확정하지 않았으며 metadata 해시 I/O·다운로드 경합 가능성도 있다. 이 문맥을 결과에 보존하고 유리한 값으로 재실행하지 않았다. 따라서 완전히 격리된 host의 확정 처리율로 확대하지 않는다.

위 warmed 값에는 전처리·shared-memory 왕복·추론·후처리·품질·추적·마스킹·PNG/JSON 및 마지막 flush가 포함된다. 모델/process load·5회 워밍·teardown은 별도 측정했고 제외했다. capture·실제 video decode·GNSS·보정·API·10분 지속 처리는 제외했다. **이 짧은 처리 구간에서10FPS를 넘었지만 SRS의 지속/전체 장비·현장10FPS 인수는 아직 완료하지 않았다.** 최종 새 모델 조합의 수치·정확도·지속 처리율은 따로 확인해야 한다.

## 회귀 확인

실제 PNG byte/pixel 동일성, snapshot·backpressure, 품질 동시 실행, 오류 전파, 원기한 중간 만료, 변경된 파일 보존, JSON 전 flush 및 CLI를 검사했다. workflow는 CPU default/fast와 IO default/overlap 네 조합에서 capture→review→queue 계약을 통과했다.

프로젝트 TEMP/basetemp를 지정한 최종 확대 실행은161 passed +189 subtests, 실제 GPU opt-in1 skipped였다(18.36초). 앞선 잘못된 basetemp 실행에서 material-policy fixture18개가 workspace guard로 거부된 것은 성공에 합산하지 않았다. GPU 수치160 증거와 이번 실제 GPU ABBA는 CPU 합성 회귀와 구분한다.

```powershell
# 승인된 GPU slot과 같은 exact local model-set 증거가 필요하다.
.\.venv\Scripts\python.exe scripts/benchmark_verified_pipeline.py NEW_OUTPUT --gpu-profile data/gpu-runtime-profiles/final-v6.json --native-proof data/gpu-model-sets/final-v6/native/aggregate.json --model-set data/gpu-model-sets/final-v6/plan.json --comparison io
```

실행별 raw 사진·weights·개별 관측은 Git에 넣지 않는다. 이번 입력은 비개인 합성이며 사용자 영상·TEST를 읽지 않았다. [기존 GPU 연구 후보의 검증 범위](FINAL-MODEL-GPU-VALIDATION.md)와 구분하여 읽는다.
