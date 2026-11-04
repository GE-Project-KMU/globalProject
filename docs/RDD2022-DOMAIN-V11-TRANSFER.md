# RDD2022 V11 별도 bulk 전송·무결성·XML 감사

앞선 3회 availability probe는 원 receipt와 함께 보존한다. 이후 승인된 별도 phase의 사전 transfer-plan SHA는 `495c1872562cffc9e33c2c6adcaa3953f990c094f2a5e44bfb124a10d7947514`이다. 원 archive는 공식 file38030910의 **13,264,172,619바이트**, publisher MD5 **b62bd51d2ffcfaa76c60f234f0cc2bb3**로 고정한다. source 원기한 **2026-11-10 17:07:24.286159 UTC**는 연장하지 않는다.

`scripts/acquire_rdd2022_domain_v11.py`는 새 root `data/research/rdd2022-domain-v11`에서만 동작한다. 최대2 transfer attempts, 전체 wall45분, inactivity read timeout60초, archive+5GiB free reserve를 고정한다. 새 공식 downloader nonce로 받은 signed redirect는 변경하지 않고 즉시 따라가며 signed query·HTTP error body를 출력하거나 receipt에 저장하지 않는다. signature lifetime10초는 요청 시작 가능 시간이며 stream 전체 성공 보장이 아니다.

임시 경로는 `RDD2022_released_through_CRDDC2022.zip.download`이다. 중단 부분은 보존한다. resume는 같은 공식 object의 정확한 offset·끝·전체 크기인206과 남은 Content-Length를 모두 확인한 경우만 append한다. Range를 무시한200 응답을 기존 부분 파일에 이어 붙이지 않는다. 첫 body ZIP magic·전체 byte 상한과 최종 전체 MD5/SHA256를 확인한다. publisher MD5 실패나403·예산 만료는 보존 후 중단하며 사용가능 source로 표시하지 않는다.

전체 공식 MD5가 일치한 뒤에만 같은 root의 `.zip`으로 atomic no-replace publish한다. 이어 ZIP outer/국가별 nested archive 전체의 CRC를 확인하고, entry 총200,000개·중첩 container를 포함한 uncompressed50GiB·개별 JPG32MiB/XML2MiB 상한과 ZipSlip·link·암호화 거부를 적용한다. 이미지 파일을 디코딩하거나 디스크로 추출하지 않는다. MD5 receipt의 `usable_source=false`는 CRC가 끝나기 전 상태이며 최종 `full-transfer-results.json`의 성공 상태와 전체 CRC까지 확인해야 한다.

## 이후 metadata 감사의 별도 gate

`scripts/audit_rdd2022_domain_v11.py`는 실제 전체 MD5+CRC 성공 receipt와 archive SHA를 재확인한 뒤에만 새 `annotation-audit-plan.json`을 작성한다. 이 gate 전에 기존 release metadata나 원 archive XML을 감사하지 않는다. 감사 대상 국가7개는 Japan, India, Czech, Norway, United_States, China_MotorBike, China_Drone이다.

XML2MiB 상한과 DOCTYPE/ENTITY 거부를 적용하고, 국가·publisher train/test별 이미지/annotation 매칭, literal D40 양성 이미지·박스, noD40 annotation 이미지, 좌표 범위·차원·XML 오류를 집계한다. **무주석 TEST 사진을 noD40 음성으로 간주하지 않는다.** 과거 고정 `history-exclusions.json`의 release 목록과 SHA를 확인하여 country/source_member 기반 TRAIN/VAL/TEST 구성과 고유 image SHA 개수만 집계한다. 물리 촬영 위치를 검증한 것으로 표현하지 않으며 알 수 없는 country는 unknown으로 남긴다.

US100과 모든 과거 TEST는 향후 TRAIN 금지 상태를 유지한다. 이 감사는 새 TRAIN membership·split이나 독립 holdout을 만들지 않는다. 모델·threshold·V10 winner 재선택, 모델 추론·새 학습·GPU 실행은 모두 범위 밖이다. 공개 집계에는 국가/GT 수·archive 무결성·원기한·시간만 기록하며 개별 이미지 ID·경로·좌표는 공개하지 않는다.

새26번째 보관 root도 준비 범위에 추가했다. 정확 ZIP/동일 `.zip.download`와 고정 transfer-plan sidecar만 source_archive로 분류한다. 다른 `.download`·모델 binary·중첩 이름을 넓게 허용하지 않는다. 무개인 acquisition control은 최소 provenance로 보존하며 실제 inventory 등록·dry-run은 모든 writer 종료 후 root가 별도로 수동 실행한다. 전송 중 scan·등록·이동·삭제는 하지 않는다.

실제 결과는 로컬 최종 receipt를 근거로 별도 공개 집계한다. 합성 계약은 resume200 거부·정확206·크기/MD5·CRC·ZipSlip·원기한 및 metadata gate를 검사하며, 실제 archive 취득 성공의 대체 증거가 아니다.
