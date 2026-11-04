# 미국 포트홀 도메인 자료 준비 초안

**현재 계획은 표본 부족으로 차단되었다.** 완료된 공식 XML 통계에서 미국 publisher TRAIN의 D40 양성은116장이다. 기존 US100 평가의 양성50장을 제외하면 새 양성은 최대66장이고, 추가 과거 노출·근접중복·분할 경계 검수 전의 상한이다. 본 기준의 최소 양성200장(TRAIN100+VAL50+TEST50)을 충족할 수 없다. 기준은 완화하지 않으며 실제 사진 읽기·분할·stage·파일 생성·학습을 진행하지 않는다. 코드도 `execution_blocked:true` 정책을 확인하여 `--authorized`가 있더라도 자료 소비 전에 중단한다. 미래 실행은 별도 새 계획이 필요하다. 아래 실행 구조는 미실행 설계와 합성 계약 설명이다.

`scripts/prepare_pothole_us_domain_v11.py`는 아직 실행되지 않은 자료 준비 정책과 합성 계약만 구현한다. 현재 실제 아카이브 추출, 사진 디코딩, 새 분할, 모델 예측, 학습은 모두 0회다. 공식 파일 획득·XML 통계 감사는 별도 작업이며 그 성공이 새 자료 분할 허가를 대신하지 않는다. 실제 자료 읽기와 분할, 이후 파일 생성은 root의 별도 승인 단계다.

출처는 [저자 RDD2022 Figshare 자료](https://doi.org/10.6084/m9.figshare.21431547.v1)이며 데이터 라이선스는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)이다. 라이선스 근거와 저자 출처를 상속하지만 개인정보 검수·기관 승인·현장 배포 승인을 만들지 않는다. 이번 산출물은 비공개 연구 전용으로 제한한다. 원 자료의 더 이른 기한 **2026-11-10 17:07:24.286159 UTC**를 그대로 사용하며 재다운로드·새 분할 시점을 기준으로 연장하지 않는다.

실행 gate는 전체 공식 ZIP의 MD5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`, 전체 SHA와 바이트 수, 모든 멤버 CRC·경로·압축해제 크기 검증 완료 receipt, 획득 소스와 계획, 완료된 annotation audit의 소스·계획·원 archive 연결을 요구한다. MD5만 확인된 중간 receipt나 `.download` 파일은 허용하지 않는다. 실행 직전과 멤버십 확정 전에 원 파일·증거 SHA를 확인한다. 디코딩 전 JPEG/PNG 헤더의 픽셀 수와 파일 크기도 제한한다.

대상은 `United_States/train/images`의 JPG와 `annotations/xmls`의 같은 원 파일명 XML이 모두 있는 경우뿐이다. 원 publisher TEST의 무주석 사진을 음성 자료로 해석하지 않는다. D40만 포트홀 foreground로 사용한다. D00/D10/D20은 알려진 다른 도로 손상으로 제외하지만 그 박스도 모두 유효해야 한다. 알 수 없는 클래스, 비유한 좌표, 역전·0면적·영상 경계를 벗어난 박스는 해당 사진 전체를 제외하고 사유를 센다. 박스를 자르거나 일부 GT를 버리지 않는다.

현재 좌표 초안은 publisher 숫자를 연속 edge 좌표로 보존하여 `xmin, ymin, xmax-xmin, ymax-ymin`으로 쓴다. 근거 없이 1을 빼거나 좌표 원점을 추정하지 않는다. 이는 원 자료의 좌표 관례가 확인되기 전의 명시적 초안이다. XML 메타데이터 검토 후 다른 관례가 입증된다면 실제 실행 전에 별도 정책 revision이 필요하며, 실행 후 결과에 맞춰 바꾸지 않는다.

기존 고정 `history-exclusions.json`의 여섯 release는 TRAIN·VAL·TEST를 모두 제외한다. 기존 US100도 원 release SHA를 확인하여 전부 제외한다. 이 목록에는 과거 평가145가 포함된 원 자료가 유지되며 원 blacklist를 수정하지 않는다. 제외 기준은 원 이미지 SHA, 정규화한 publisher 파일명/member, 과거 미국 frame-block100, 확인 가능한 과거 미국 TEST 프레임의 ±20이다. 이어 과거 모든 분할의 실제 이미지로 pHash를 계산해 새 자료와 비교한다. 과거 TRAIN과 근접한 자료도 새 pool에서 빠진다.

64bit pHash는 회색조 32×32 INTER_AREA 축소 → DCT의 왼쪽 위 8×8 → DC를 제외한 중앙값 비교로 정의하며 DC bit는 0이다. 원본과 수평 반전을 모두 계산하고 Hamming 거리6 이하를 연결한다. 기존 dHash를 pHash로 이름만 바꾸어 사용하지 않는다. 새 sequence group은 국가와 `floor(원 숫자 프레임 ID/100)`이다. 같은 sequence, 같은 SHA, 근접 pHash를 DSU로 연결하며 과거 자료에 닿은 연결 성분은 전체 제외한다. 숫자 파일명은 물리적 주행·경로 독립성을 입증하지 않는다.

DSU 성분 식별자를 정렬하고 seed **20261111**로 섞은 뒤 성분 수의 floor70%/floor15%/나머지를 TRAIN/VAL/TEST에 배정한다. 분할이 다른 프레임 ID의 차이가20 이하이면 양쪽 사진을 모두 제거하며 재배정하지 않는다. 모든 클래스별 cap 선택은 `SHA256(sample_id)` 순서다.

- TRAIN: 양성100·음성100 이상, 각 최대1500, 선택 후 독립 DSU 성분5개 이상.
- VAL: 양성50·음성50 정확100, 선택 후 독립 DSU 성분5개 이상.
- TEST: 양성50·음성50 정확100, 선택 후 독립 DSU 성분5개 이상.

조건 미달은 실패다. 남은 표본 수·제외 이유를 비공개 실패 receipt에 기록하고 release를 만들지 않으며 조건 완화·재배정·자동 재시도하지 않는다. 모델 예측이나 성적은 선택에 전혀 사용하지 않는다.

실행 인터페이스는 세 단계다. `draft`는 코드와 기존 증거 파일의 SHA만 확인해 공개 초안을 쓴다. `stage --authorized`는 별도 허가 후에만 전체 gate를 통과해 사진/QC/분할을 수행한다. UUID·GT·원 멤버·CRC·이미지 SHA·원기한·그룹·분할을 포함한 비공개 `membership-plan.json`을 **사진 쓰기 전에** exclusive로 고정한다. `materialize --authorized --membership-sha256 <고정 SHA>`는 다시 원 gate와 멤버십 바인딩을 검사하고 `data/training/research-pothole-us-domain-v11/source/<UUID>.jpg`만 새로 쓴다.

실제 파일의 이미지·SHA·기하·pHash 검사와 `validate_research_release()`는 바이트 쓰기 후 실행한다. 파일이 아직 없는데 validator를 먼저 호출하지 않는다. 검증을 통과한 `release.json`, `development.json`, `fresh100.json`은 임시 bundle에 함께 쓴 뒤 `validated-release-bundle` 디렉터리를 한 번 rename하여 공개한다. 따라서 release 파일 위치는 다음과 같다.

```text
data/training/research-pothole-us-domain-v11/
  membership-plan.json
  source/<UUID>.jpg
  validated-release-bundle/
    release.json
    development.json
    fresh100.json
  materialization-completed.json
```

멤버십 schema는 `format`, `policy_sha256`, `source_sha256`, `bindings`, `source_expires_at`, `rows_sha256`, `rows`, `counts`, `removed`, `QC`를 포함한다. `rows`는 기존 연구 release 필드에 원 publisher member/XML/CRC/기하/모든 D40 GT/sequence 및 DSU group/분할/pHash 쌍을 보존한다. 성공 release에서 내부 `_` QC 필드만 제거하며 GT는 그대로 보존한다. 연구 scope는 `restricted_research`, `publisher_annotations_machine_checked`, `offline_research`, `human_privacy_reviewed:false`, `not_for_deployment:true`다.

실패 시 부분 파일과 실패 receipt는 원기한으로 남기며 성공 bundle을 만들지 않는다. 원 RAW, 과거 release, blacklist, 기존 보관 inventory·pins의 수정·삭제·재등록은 하지 않는다. 새로운 모델 학습은 이 도구의 범위에 포함되지 않는다.
