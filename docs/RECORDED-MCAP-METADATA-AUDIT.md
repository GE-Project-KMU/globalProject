# 제공 녹화16개: MCAP metadata 감사

제공 파일16개를 인계 manifest의 개별 size/SHA와 감사 전후 대조해 모두 일치함을 확인했다. 총1,778,740,737bytes이며 내용이 서로 다른16개다. 원본 쓰기·이동·삭제는0, 이번 감사의 영상 pixel decode·ML추론·학습도0이다. 공개 결과는 [개별정보를 제외한 집계](reports/RECORDED-MCAP-METADATA-AUDIT.json)에 기록했다.

파일은 MCAP magic/header·summary·schema/channel·전체 message를 공식 Python reader로 읽었다. SVO JSON header에서 관측한 SDK 표기는5.5.0, header version은2.0.5다. 이는 파일 내부 표기이며 해당 SDK가 설치되었거나 실제 장치·보정·IMAGE timestamp가 검증되었다는 뜻이 아니다. 해상도·카메라 모델·장착 상태·encoded codec은 이름만으로 추정하지 않았다.

각 파일에는 SVO JSON header, custom encoded video, 센서 주제2개, footer 주제가 있었다. 별도로 GNSS/GPS/NMEA/NavSat을 명명한 topic은 발견하지 못했다. 센서 payload 값이나 임의 custom payload를 의미 해석하지 않았으므로 파일 내부 어디에도 GNSS가 없다고 단정하지 않는다. 카메라 SN을 포함한 원래 channel/schema 이름과 원본 경로·개별 SHA·정확한 기록시각·JSON header는 로컬 private 감사에만 두었다.

## 영상 message의 기록시각 범위

영상 topic message14,753개의 MCAP `log_time`을 사용했다. 파일별 최초~최후 span 합은730.207682초, 약12분10초다. 최소 파일 span0.803967초, 최대111.803366초이며 file-order에서 영상 log time 역전은 관측되지 않았다. 이 값은 각 파일 내부 기록시각 차이의 합이다. 파일 사이 중복 장면·연속 촬영 여부나 정확한 촬영 UTC를 입증하지 않는다. **60분 실주행 합격 근거가 아니며 반복 재생으로60분 주행을 만들 수 없다.**

MCAP의 log/publish time은 사용자가 정한 epoch를 바탕으로 기록될 수 있으므로 내부 숫자를 실제 UTC나 ZED SDK IMAGE timestamp로 자동 취급하지 않는다. [MCAP 공식 writer 설명](https://mcap.dev/docs/python/mcap-apidoc/mcap.writer)은 두 시각을 구분하며 epoch 선택도 명시한다. 사용자는 촬영일을10월28일로 추정했지만 이는 기기시계 관측과 별도 근거다. 외부 시계 동기화·확정 촬영일·원래 보관기한은 확인되지 않았다.

현재 정책 확인을 기다리며 이 metadata 감사에서 expiry를 새로 만들거나 mtime/인계일로 기한을 연장하지 않았다. 자동 보관/삭제 등록과 원본 삭제도 수행하지 않았다. 시각과 기한이 미확정인 원본을 확정적으로 만료됐거나 계속 사용할 수 있다고 판정하지 않는다. 이후 pixel decode·ML 사용 등은 사용자와 루트의 별도 정책 판단을 따른다.

## 무결성과 검사 범위

summary channel/message count를 전체 streaming 관측과 대조했다. `NonSeekingReader(validate_crcs=True)`로 사용 가능한 nonzero chunk/data CRC를 검사하도록 요청했다. CRC가 생략된 부분에 checksum이 존재한다고 주장하지 않는다. [공식 reader 설명](https://mcap.dev/docs/python/mcap-apidoc/mcap.reader)은 streaming·CRC·파일 순서 옵션을 설명한다.

합성 회귀4개는 잘린 파일 거부, 변조한 Chunk CRC 거부, 역전된 clock에서도 UTC/expiry를 발명하지 않는 처리, public 집계에서 SN·경로·개별 SHA·정확한 시각 제외를 확인했다. 이는 영상 내용의 정확도·비식별화·GNSS 품질이나 실물 장착 검증을 대신하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/audit_recorded_mcap.py LOCAL_PRIVATE_HANDOFF_MANIFEST NEW_PRIVATE_AUDIT_DIRECTORY
.\.venv\Scripts\python.exe -m pytest tests/test_recorded_mcap_audit.py -q
```

원본은 읽기만 하며 인계 manifest와 source hash가 바뀌면 중단한다. 감사 출력은 새 디렉터리에만 생성한다. 개별 camera SN·path·hash·timestamp가 있는 private 결과는 Git에 넣지 않는다. 공식 MCAP 구조를 읽은 것과 ZED SDK를 통한 실제 재생·정류·depth 검증은 구분한다.
