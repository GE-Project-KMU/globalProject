# RDD2022 V11 별도 bulk 전송·무결성·XML 감사

앞선 3회 availability probe는 원 receipt와 함께 보존한다. 이후 승인된 별도 phase의 사전 transfer-plan SHA는 `495c1872562cffc9e33c2c6adcaa3953f990c094f2a5e44bfb124a10d7947514`이다. 원 archive는 공식 file38030910의 **13,264,172,619바이트**, publisher MD5 **b62bd51d2ffcfaa76c60f234f0cc2bb3**로 고정한다. source 원기한 **2026-11-10 17:07:24.286159 UTC**는 연장하지 않는다.

`scripts/acquire_rdd2022_domain_v11.py`는 새 root `data/research/rdd2022-domain-v11`에서만 동작한다. 최대2 transfer attempts, 전체 wall45분, inactivity read timeout60초, archive+5GiB free reserve를 고정한다. 새 공식 downloader nonce로 받은 signed redirect는 변경하지 않고 즉시 따라가며 signed query·HTTP error body를 출력하거나 receipt에 저장하지 않는다. signature lifetime10초는 요청 시작 가능 시간이며 stream 전체 성공 보장이 아니다.

임시 경로는 `RDD2022_released_through_CRDDC2022.zip.download`이다. 중단 부분은 보존한다. resume는 같은 공식 object의 정확한 offset·끝·전체 크기인206과 남은 Content-Length를 모두 확인한 경우만 append한다. Range를 무시한200 응답을 기존 부분 파일에 이어 붙이지 않는다. 첫 body ZIP magic·전체 byte 상한과 최종 전체 MD5/SHA256를 확인한다. publisher MD5 실패나403·예산 만료는 보존 후 중단하며 사용가능 source로 표시하지 않는다.

전체 공식 MD5가 일치한 뒤에만 같은 root의 `.zip`으로 atomic no-replace publish한다. 이어 ZIP outer/국가별 nested archive 전체의 CRC를 확인하고, entry 총200,000개·중첩 container를 포함한 uncompressed50GiB·개별 JPG32MiB/XML2MiB 상한과 ZipSlip·link·암호화 거부를 적용한다. 이미지 파일을 디코딩하거나 디스크로 추출하지 않는다. MD5 receipt의 `usable_source=false`는 CRC가 끝나기 전 상태이며 최종 `full-transfer-results.json`의 성공 상태와 전체 CRC까지 확인해야 한다.

## 이후 metadata 감사의 별도 gate

`scripts/audit_rdd2022_domain_v11.py`는 실제 전체 MD5+CRC 성공 receipt와 archive SHA를 재확인한 뒤에만 새 `annotation-audit-plan.json`을 작성한다. 이 gate 전에 기존 release metadata나 원 archive XML을 감사하지 않는다. 감사 대상은 Japan, India, Czech, Norway, United_States, China_MotorBike, China_Drone의7개 archive 그룹이다. 중국의 두 장비군을 분리한 것이며 공식 자료의 국가 수는6개국이다.

XML2MiB 상한과 DOCTYPE/ENTITY 거부를 적용하고, 국가·publisher train/test별 이미지/annotation 매칭, literal D40 양성 이미지·박스, noD40 annotation 이미지, 좌표 범위·차원·XML 오류를 집계한다. **무주석 TEST 사진을 noD40 음성으로 간주하지 않는다.** 과거 고정 `history-exclusions.json`의 release 목록과 SHA를 확인하여 country/source_member 기반 TRAIN/VAL/TEST 구성과 고유 image SHA 개수만 집계한다. 물리 촬영 위치를 검증한 것으로 표현하지 않으며 알 수 없는 country는 unknown으로 남긴다.

US100과 모든 과거 TEST는 향후 TRAIN 금지 상태를 유지한다. 이 감사는 새 TRAIN membership·split이나 독립 holdout을 만들지 않는다. 모델·threshold·V10 winner 재선택, 모델 추론·새 학습·GPU 실행은 모두 범위 밖이다. 공개 집계에는 국가/GT 수·archive 무결성·원기한·시간만 기록하며 개별 이미지 ID·경로·좌표는 공개하지 않는다.

새26번째 보관 root도 준비 범위에 추가했다. 정확 ZIP/동일 `.zip.download`와 고정 transfer-plan sidecar만 source_archive로 분류한다. 다른 `.download`·모델 binary·중첩 이름을 넓게 허용하지 않는다. 무개인 acquisition control은 최소 provenance로 보존하며 실제 inventory 등록·dry-run은 모든 writer 종료 후 root가 별도로 수동 실행한다. 전송 중 scan·등록·이동·삭제는 하지 않는다.

실제 결과는 로컬 최종 receipt를 근거로 별도 공개 집계한다. 합성 계약은 resume200 거부·정확206·크기/MD5·CRC·ZipSlip·원기한 및 metadata gate를 검사하며, 실제 archive 취득 성공의 대체 증거가 아니다.

## 실제 완료 결과

첫 transfer attempt에서 **13,264,172,619바이트**를 받았고 공식 MD5가 일치했다. 전체 SHA256은 `5d230a2941e8f2ac5fbca1a0faddee393314368dd17f9304effc11ec532845b6`이다. 85,852개 entry, 중첩 container를 포함한 uncompressed27,096,647,528바이트에 대해 경로·크기 제한과 전체 CRC를 통과했다. 전송+stream hash+publish+구조/CRC 총 wall은 **1,687.30초(약28분7초)**이며 network-only 시간으로 표시하지 않는다. 남은 partial은0바이트다.

이어 별도 audit plan SHA `790cdfeb6a23401adc5b5883067d933b8d820ffd20036785cfdcc2797be59173`를 고정하고 XML·고정 과거6release metadata를 감사했다. 실제 사진 decode·모델 추론·새 split·학습은 모두0이다. 집계는 [RDD2022-DOMAIN-V11-VERIFIED-AUDIT.json](reports/RDD2022-DOMAIN-V11-VERIFIED-AUDIT.json)에 있다.

| Archive 그룹 | 전체 JPG | Publisher TRAIN 이미지/XML 매칭 | D40 양성 이미지 | D40 box | noD40 주석 이미지 | 무주석 publisher TEST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| China_Drone | 2,401 | 2,401 | 64 | 86 | 2,337 | 0 |
| China_MotorBike | 2,477 | 1,977 | 164 | 235 | 1,813 | 500 |
| Czech | 3,538 | 2,829 | 154 | 197 | 2,675 | 709 |
| India | 9,665 | 7,706 | 1,530 | 3,187 | 6,176 | 1,959 |
| Japan | 13,133 | 10,506 | 1,390 | 2,243 | 9,116 | 2,627 |
| Norway | 10,201 | 8,161 | 256 | 461 | 7,905 | 2,040 |
| United_States | 6,005 | 4,805 | 116 | 135 | 4,689 | 1,200 |

전체 JPG47,420개와 XML38,385개를 확인했다. literal D40 양성은 총3,674장/6,544box다. XML/image 매칭과 D40 좌표 범위는 모두 통과했다. 일본의 **비D40 box1개**는 범위를 벗어났다. 주석이 없거나 D40가 기록되지 않았다는 사실은 사람의 검수로 포트홀 부재가 확인됐다는 뜻이 아니다.

과거 고정6release의19,069행은 country 필드가 모두 존재했고 **US의 supervised TRAIN·VAL 행은0**이었다. 과거 release 합집합의 TRAIN 고유 image SHA 수는 India4,149/Japan4,441/Bangladesh-source-unverified-location2,188이다. 별도 US100은 기존 TEST100(양성50/noD40 50)이며 계속 TRAIN 금지다. 국가 필드는 publisher/source metadata이며 물리 촬영 위치를 입증하지 않는다. 이 관측만으로 낮은 AP의 단일 원인을 미국 데이터 부재라고 확정할 수 없다.

과거 합집합에는 초기·중단·격리 실험도 포함한다. TRAIN↔TEST 고유 SHA 중복163, VAL↔TEST35, TRAIN↔VAL606은 이 전체 역사 기록의 교차 노출 집계다. 이를 현재 하나의 최종 release에서 발생한 누출이라고 표현하지 않는다. 모든 과거 TEST의 향후 TRAIN 금지는 그대로 유지한다.

**미국 추가 학습은 실행하지 않는다.** 새 원본 전체에서도 D40 양성이116장으로 추가 학습 기준 최소200장에 못 미친다. 기존 US100의 양성50장을 제외한 조건부 상한은66장이며 추가 중복·block·identity QC 전 숫자다. 이66장을 실제 eligible TRAIN release로 만든 적은 없다. 무주석 TEST1,200장을 음성 TRAIN이나 새 정확도 평가용으로 전환하지 않는다. V10 winner·threshold·기존 TEST 결과는 변경하지 않는다.

source 만료는 원11/10 17:07:24.286159 UTC이며 새 취득 시각으로 연장하지 않았다. 과거 control8개(4inventories/4plans)의 실제 SHA를 대조해 모두 불변임을 확인했고 asset scan·보관 등록·삭제는0이다. 모든 archive/XML 감사 writer는 종료됐다. 공개 집계에는 개별 이미지 ID·원경로·좌표·signed query를 포함하지 않는다.
