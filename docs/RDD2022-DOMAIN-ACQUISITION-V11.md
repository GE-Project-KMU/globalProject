# RDD2022 대상 도메인 추가 취득 가능성

이번 조사는 V10 TEST100을 확인한 뒤 기존 winner나 threshold를 재선택하는 작업이 아니다. 향후 독립 cycle의 공개 TRAIN 후보 취득 가능성만 확인했다. 과거 TEST 전체는 향후 TRAIN 영구 제외 대상으로 유지한다. 모델 추론·이미지 decode·학습·새 split 생성은 모두 0이다.

공식 [Figshare API](https://api.figshare.com/v2/articles/21431547)에서 article21431547/file38030910의 크기13,264,172,619바이트, MD5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`, CC BY4.0을 재확인했다. [저자 공식 repository](https://github.com/sekilab/RoadDamageDetector/blob/master/README.md)는 Figshare 원본과 미국·중국 등 국가별 S3 ZIP을 현재도 안내한다. 저자 안내상 원 TRAIN에는 이미지와 XML annotation이 있고 원 TEST에는 이미지가 제공된다. 따라서 archive 취득 자체가 새 독립 평가 split 확보나 완전한 포트홀 GT 확보를 뜻하지 않는다.

새 로컬 `data/research/rdd2022-domain-v11`에서 원기한 **2026-11-10 17:07:24.286159 UTC**를 상속하고 사전 acquisition plan을 고정했다. 기존 실패 파일·706MB cache·inventory는 수정하지 않았다. source 파일 전송 attempt를 합계3회, attempt timeout60초, 각 Range payload64바이트로 제한했다. 원 archive 크기와2GiB disk reserve도 상한 계약에 기록했다. signed redirect URL은 메모리에서만 그대로 사용하고 query/credential을 로그에 남기지 않았다.

관측 결과는 다음과 같다.

| 공식 경로 | 관측 | 의미 |
| --- | --- | --- |
| 새 nonce의 Figshare downloader | 302→206, `bytes 0-63/13264172619`, ZIP local-header prefix 확인 | 현재 fresh signed redirect에서 제한된 Range 접근 가능 |
| 저자 공식 United_States S3 | 403 | 해당 요청으로는 취득 불가 |
| 저자 공식 China_MotorBike S3 | 403 | 해당 요청으로는 취득 불가 |

Figshare 관측 서명 lifetime은10초, 응답 시점 서명 age는 약2.6초였다. 서명 URL 자체는 기록하지 않았다. 이 짧은 성공은 과거403 원인이 영구적으로 해결되었거나 장기 전송이 성공할 것이라는 증거가 아니다.

64바이트 Range 성공은 archive 무결성 증거가 아니다. 전체 MD5·ZIP CRC는 미검증이며 관측 multipart ETag도 공식 MD5의 대체로 사용하지 않았다. **검증된 archive0개, 새 사용가능 이미지0개**다. 3회 probe 한도를 소진해 중단했으며 자동 bulk transfer나 재시도는 하지 않는다. 추가 취득은 새 전송 phase의 예산·시간·체크섬 검증 절차를 따로 고정한 뒤 판단해야 한다. 원본을 실제로 받더라도 공식 전체 MD5를 통과하기 전에는 사용가능 source로 표시할 수 없다.

코드는 `scripts/probe_rdd2022_domain_v11.py`, 공개 집계는 [RDD2022-DOMAIN-V11-AVAILABILITY.json](reports/RDD2022-DOMAIN-V11-AVAILABILITY.json)이다. 합성 HTTP 계약15개는 bounded range·redirect host·크기·ZIP prefix·deadline·오류 body 비기록·200 전체응답 미수용을 확인했다. 이는 실제 전체 archive 취득 성공 시험이 아니다. GPU·기존 TEST 추론·원기한 연장·기존 파일 삭제는 수행하지 않았다.

위 수치는 3회 probe phase의 보존된 snapshot이다. 이후 별도로 승인된 bulk phase는 전체 MD5/CRC를 통과했으며, 새 split·학습 없이 XML/history metadata 감사를 마쳤다. 현재 결과와 미국 추가 학습을 실행하지 않은 판단은 [별도 전송·감사 문서](RDD2022-DOMAIN-V11-TRANSFER.md)에 기록했다. 원 probe receipt와 예산을 사후 수정하지 않았다.
