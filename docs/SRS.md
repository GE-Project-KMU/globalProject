Here is the result of "fetch" for the Page with URL https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf as of 2026-11-03T09:37:49.524Z:
<page url="https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf" icon="📋">
<ancestor-path>
<parent-page url="https://app.notion.com/p/40d8ac7c2dde801095cdca9e832444f8" title="글로벌공학설계프로젝트"/>
</ancestor-path>
<properties>
{"title":"요구사항 정의서 (SRS)"}
</properties>
<iconMetadata>{"type":"emoji","emoji":"📋"}</iconMetadata>
<content>
<callout icon="📝" color="gray_bg">
문서 버전: v0.4 · 개정일: <date start="2026-11-03" /> · 상태: 주제 변경 반영 설계 기준안<br /><b>대상:</b> ZED 2i 스테레오 카메라를 사용하는 차량 부착형 포트홀·마모차선 인식 시스템 / B2G(지자체·한국도로공사)<br /><b>우선순위:</b> M=8주 시제품 필수, S=권장·미달 시 핵심 합격을 대신하지 않음, C=후속 확장. 수치·예산·장착값·보관기간은 작성자 제안 기준이며 시험 실적·기관 합의가 아니다.<br />기존 치수·부피·ASTM 위험도 중심 요구사항은 폐기하고, 인식·근거 기록·담당자 검토를 검수 기준으로 삼는다.
</callout>

<table-of-contents />

# 1. 개요

## 1.1 목적

본 문서는 「ZED 2i 스테레오 카메라 기반 차량 부착형 포트홀·마모차선 인식 및 도로점검 지원 시스템」의 기능·데이터·인터페이스·검수 기준을 정한다. 기획서 v0.4와 함께 사용하며, 기획서는 목적·일정·비용을, 본 문서는 구현과 검수를 위한 동작 조건을 정의한다.

## 1.2 범위

<b>포함:</b> 좌우 RGB 취득과 GNSS 로그, 포트홀 박스 탐지, 차선 외곽·남은 도막 기반 시각적 잔존율, 품질·판정 불가 처리, 관측 위치·추적·중복 후보, 로컬 비식별화·공유 전 확인, 업로드 큐, 인증·권한, 지도/목록/검토·조치 이력, CSV/PDF.

<b>필수 운용 조건:</b> 주간·건조 노면·아스팔트 위 백색/황색 실선·점선, 차량 속도 20 km/h 이하, 주 차량 1대의 가시 노면. 추가 차종 1대는 장착·시야·저속 촬영으로 범용성을 검증한다.

<b>제외:</b> 차선 그리기·도색·자동 보수, 차량 제어, 포트홀 깊이·실제 면적·부피·자재량 산출과 치수 기반 위험도, 재귀반사도 합격 판정, 고속도로 본선 운용 검증, RTK, 전국 도로 전수 점검, 행정 시스템 실제 연동. 야간·우천·30 km/h·스테레오 보조 깊이·실시간 차량 추론은 S/C 평가로 분리한다.

## 1.3 용어

| 용어 | 정의 |
| --- | --- |
| 관측(observation) | 한 시점의 물체/차선 후보와 사진·위치·모델 결과. observation\_id로 전송 중복 방지 |
| 이벤트(event) | 같은 주행에서 같은 포트홀을 추적으로 묶은 후보. 별도 실제 객체와 혼동하지 않음 |
| 점검 구간 | 동일 차선에 대해 누적 주행거리 약 10 m로 묶은 관측 범위. 정밀 측량 길이가 아님 |
| 시각적 잔존율 R | 유효한 예상 도색 외곽 안의 남은 도막 면적 비율. 재귀반사도와 다른 지표 |
| 판정 불가 | 외곽 복원·영상 품질·범위가 기준을 충족하지 못해 잔존율/분류를 내지 않는 상태 |
| 관측 위치 | 촬영 시점 차량의 GNSS 위치. 손상 객체의 정밀 좌표가 아님 |
| BEV·프로파일 | 평면 보정 영상과 차종·카메라·높이·각도·보정값·ROI를 저장한 설정 |
| 검토 우선순위 | 사람이 확인할 순서. AI 신뢰도·마모 지표를 법정 위험등급으로 해석하지 않음 |

# 2. 이해관계자 및 사용 흐름

| 사용자 | 책임·권한 | 사용 흐름 |
| --- | --- | --- |
| 지자체 도로관리 담당자 | 관할 후보 조회·검토·조치 관리 | 목록/지도 → 사진 확인 → 오탐/확인/재촬영 → 조치 예정/완료 |
| 한국도로공사 유지관리 담당자 | 후속 실증의 요구·연계 조건 검토 | 저속 시제품 평가 후 실제 운용 속도·연계·보안 요건 별도 설정 |
| 운전자·수집 담당 | 정차 상태에서 시작/종료·장치 점검 | 고정·렌즈·저장 상태 확인 → 촬영 → 정차 후 종료/업로드 |
| 시스템 관리자 | 장치·계정·권한·삭제·버전 관리 | 프로파일 등록, 오류·접근로그 확인, 보관 정책 실행 |
| 조회 사용자·보수업체 역할 | 허가된 결과 조회, 담당자가 허용한 조치 기록 | 비식별 결과 열람. 원본 접근은 기본 차단 |

최종 보수·도색 여부와 긴급성은 담당자가 판단한다. 미검출·미촬영·판정 불가를 도로 정상이나 조치 완료로 자동 처리하지 않는다.

# 3. 운영 환경 및 설계 결정

<table header-row="true">
<colgroup>
<col width="160" />
<col width="480" />
<col width="550" />
</colgroup>
<tr>
<td>
구분
</td>
<td>
v0.4 기준
</td>
<td>
검증·제약
</td>
</tr>
<tr>
<td>
카메라
</td>
<td>
ZED 2i 유지, 좌우 RGB 동기 취득, 인식은 좌영상
</td>
<td>
렌즈·일련번호·취득 모드는 실제 장비 기준 기록. UVC 좌우 결합 분리 후 시간표 보존
</td>
</tr>
<tr>
<td>
영상·연산
</td>
<td>
눈당 1280 × 720, 30 FPS 녹화, 10 FPS 분석
</td>
<td>
포트홀 416 × 416, 차선 입력 긴 변 512 기준. 원영상 좌표로 결과 환산
</td>
</tr>
<tr>
<td>
PC·SDK
</td>
<td>
기존/대여 PC, 성능 기준 RTX 3060 12 GB·RAM 16 GB급
</td>
<td>
실제 장비 확보·성능은 W1 확인. 기본 RGB는 UVC·OpenCV, 선택 SVO/깊이는 ZED SDK·지원 NVIDIA 환경
</td>
</tr>
<tr>
<td>
장착
</td>
<td>
전방 범퍼 하단부, 초기 높이 0.6\~0.9 m·하향 20°, ±10° 조절
</td>
<td>
이중 고정·안전줄·케이블 완화. 구조 변경 없는 장착, 차종 간섭·시야 확인 후 촬영
</td>
</tr>
<tr>
<td>
가시 범위
</td>
<td>
초기 목표 노면 1.5\~6 m, 유효 종방향 길이 ≥ 3 m
</td>
<td>
차종별 실제 보정 ROI만 점검. 반대편 차선·가려진 영역을 검사했다고 기록하지 않음
</td>
</tr>
<tr>
<td>
GNSS
</td>
<td>
USB NMEA 수신기, 목표 5 Hz, 단독측위
</td>
<td>
fix·HDOP·수신 시각 기록. 2초 초과 공백/무효 fix/HDOP > 3은 품질 저하
</td>
</tr>
<tr>
<td>
스택
</td>
<td>
Python·OpenCV·PyTorch, YOLOX-Nano, DeepLabV3-MobileNetV3, IoU 추적
</td>
<td>
서버 FastAPI·PostgreSQL/PostGIS, React·Leaflet, WeasyPrint·CSV. 환경·모델 해시 W1/W6 잠금
</td>
</tr>
<tr>
<td>
운용
</td>
<td>
녹화 후 PC 배치 분석 필수, 차량 내 실시간 분석 권장
</td>
<td>
운행 후 Wi-Fi 업로드 가능. Jetson·RTK·LTE 신규 구매 필수 아님
</td>
</tr>
<tr>
<td>
지도
</td>
<td>
브이월드 우선, 서비스별 약관·키·출처 표시 조건 적용
</td>
<td>
키 없으면 배경지도 없이 좌표·목록·자체 경로 선. 무제한 이용·공공 납품 적합 보장 안 함
</td>
</tr>
</table>

카메라 본체 IP66·USB 3·롤링 셔터는 제조사 사양이다. 완성 장치 방수·고속 주행 내구성으로 확장해 주장하지 않는다. [Stereolabs 사양](https://support.stereolabs.com/articles/9573039616-where-can-i-find-a-comparison-table-for-the-zed-stereo-cameras), [UVC와 SDK 기능 구분](https://support.stereolabs.com/articles/7166665108-can-i-use-the-zed-without-cuda)

# 4. 기능 요구사항

## 4.1 데이터 취득 (ACQ)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-ACQ-01
</td>
<td>
ZED 2i 좌우 RGB를 저장하고 프레임 번호·취득 시각·실제 FPS를 기록한다. 깊이 동시 취득은 선택이다.
</td>
<td>
M
</td>
<td>
취득·재생 시 좌우·프레임 시간 대조
</td>
</tr>
<tr>
<td>
FR-ACQ-02
</td>
<td>
MP4와 프레임-시각 매핑 로그를 같은 주행 ID로 저장·재생한다. 선택 SVO에도 같은 주행 메타데이터를 유지한다.
</td>
<td>
M
</td>
<td>
10분 파일 재생·무결성 확인
</td>
</tr>
<tr>
<td>
FR-ACQ-03
</td>
<td>
GNSS 좌표·UTC·fix·HDOP·수신 시각을 저장하고 영상과 정렬한다. GNSS 없음/품질 저하를 0,0 좌표로 채우지 않는다.
</td>
<td>
M
</td>
<td>
수신 공백·무효 fix 주입
</td>
</tr>
<tr>
<td>
FR-ACQ-04
</td>
<td>
정차 상태에서 시작/종료하고 취득·저장·위치·오염 경고를 확인한다. 운전자 주행 중 조작을 요구하지 않는다.
</td>
<td>
M
</td>
<td>
기본 사용자 과제·오류 시나리오
</td>
</tr>
<tr>
<td>
FR-ACQ-05
</td>
<td>
흔들림·심한 흐림·과다 노출·가림을 품질 플래그와 사유로 남기고 해당 영역을 자동 정상으로 처리하지 않는다.
</td>
<td>
M
</td>
<td>
품질 불량 표본 대조
</td>
</tr>
<tr>
<td>
FR-ACQ-06
</td>
<td>
장착 프로파일과 모델 버전을 주행에 연결하고 10 FPS 분석 입력으로 배치 재생한다.
</td>
<td>
M
</td>
<td>
다른 차종 프로파일·재현 분석
</td>
</tr>
</table>

## 4.2 포트홀 인식 (PH)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-PH-01
</td>
<td>
포트홀 바운딩박스·클래스·confidence를 출력한다. 인스턴스 분할과 실제 치수는 필수가 아니다.
</td>
<td>
M
</td>
<td>
박스 mAP@0.5·이벤트 Recall
</td>
</tr>
<tr>
<td>
FR-PH-10
</td>
<td>
기본 confidence 0.35, NMS IoU 0.5로 시작하고 검증셋에서 정한 최종값을 W6 잠근다. 모델 confidence를 위험 확률로 표시하지 않는다.
</td>
<td>
M
</td>
<td>
설정·출력·화면 비교
</td>
</tr>
<tr>
<td>
FR-PH-11
</td>
<td>
1초 이내 3회 이상 추적 관측한 후보를 기본 이벤트로 저장한다. 1\~2회만 보인 confidence ≥ 0.8 후보는 단발 검토로 별도 저장한다.
</td>
<td>
M
</td>
<td>
관측 수 경계 1/2/3회 테스트
</td>
</tr>
<tr>
<td>
FR-PH-12
</td>
<td>
그림자·맨홀·패치·웅덩이와 구분하고, 물·가림·품질 부족으로 확정할 수 없는 후보는 보류 사유를 출력한다.
</td>
<td>
M
</td>
<td>
어려운 음성·보류 표본 검수
</td>
</tr>
<tr>
<td>
FR-PH-13
</td>
<td>
스테레오 깊이를 이용한 노면 대비 단차 보조 확인을 실험할 수 있다. 결측 시 기본 인식 동작을 막지 않는다.
</td>
<td>
C
</td>
<td>
선택 평가로 별도 보고
</td>
</tr>
</table>

이전 FR-PH-02\~05(평면·깊이·면적·부피), FR-PH-07(치수 등급), FR-PH-09(깊이 히트맵)은 v0.4 범위에서 제외한다. FR-PH-06 다중 프레임 측정은 FR-LOC-03 추적 기능으로 대체하고, FR-PH-08 측정 불가 기능은 FR-PH-12·FR-LN-09의 인식 보류/품질 처리로 대체한다. 삭제된 치수 요구사항을 새로운 필수 의미로 재사용하지 않는다.

## 4.3 마모차선 인식 (LN)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-LN-01
</td>
<td>
가시 차선 경계·방향으로 예상 도색 외곽을 복원하고, 외곽 추정 근거가 없는 곳은 판정 불가로 둔다.
</td>
<td>
M
</td>
<td>
합의 외곽 라벨·완전 소실 사례 비교
</td>
</tr>
<tr>
<td>
FR-LN-02
</td>
<td>
외곽 안의 남은 도막을 픽셀 마스크로 분할하고 원영상 오버레이를 제공한다.
</td>
<td>
M
</td>
<td>
도막 클래스 IoU
</td>
</tr>
<tr>
<td>
FR-LN-03
</td>
<td>
차종별 노면 평면 보정(BEV)을 적용한다. 보정 프로파일이 없거나 영상 정렬이 틀리면 잔존율 공개를 차단한다.
</td>
<td>
M
</td>
<td>
격자·차종 프로파일 검증
</td>
</tr>
<tr>
<td>
FR-LN-04
</td>
<td>
유효 외곽 면적 대비 남은 도막 면적으로 R(%)를 산출한다. 구간 기본 길이는 추정 주행거리 10 m이며 실제 분석 범위·불확실성을 남긴다.
</td>
<td>
M
</td>
<td>
수작업 마스크 대비 MAE
</td>
</tr>
<tr>
<td>
FR-LN-05
</td>
<td>
R ≥ 80 시각상 양호, 60 ≤ R \< 80 관찰, R \< 60 마모 후보를 출력한다. 법정 안전/재도색 판정으로 표시하지 않는다.
</td>
<td>
M
</td>
<td>
59.9/60/79.9/80 경계값
</td>
</tr>
<tr>
<td>
FR-LN-06
</td>
<td>
백색/황색 및 실선/점선을 구분하고 점선의 의도적 빈 구간을 분모에서 제외한다. 종류가 불명확하면 판정 불가로 둔다.
</td>
<td>
M
</td>
<td>
점선·소실·혼합 사례
</td>
</tr>
<tr>
<td>
FR-LN-09
</td>
<td>
외곽 근거 부족 또는 가림·오염·그림자 등으로 유효 면적 \< 70%이면 R과 등급을 null로 저장하고 사유를 표시한다.
</td>
<td>
M
</td>
<td>
69.9/70%·완전 가림·완전 소실
</td>
</tr>
<tr>
<td>
FR-LN-10
</td>
<td>
같은 노면이 겹쳐 찍힌 프레임은 같은 패치로 연결해 이중 면적 합산하지 않는다. 구간 대표값은 고유 유효 패치 면적 가중 평균으로 계산한다.
</td>
<td>
M
</td>
<td>
반복 프레임 삽입 전후 값 비교
</td>
</tr>
</table>

이전 FR-LN-07 대비 지표는 선택 연구로만 유지하고, FR-LN-08 횡단보도·정지선은 후속 범위로 둔다. 영상 잔존율 80%/60%는 내부 공학 임계값이며 법령·재귀반사 수치로 주장하지 않는다.

## 4.4 위치 및 이벤트 관리 (LOC)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-LOC-01
</td>
<td>
유효 GNSS를 영상 시각에 연결하고 좌표 출처를 vehicle\_observation으로 기록한다. 위치가 없으면 null과 사유를 저장한다.
</td>
<td>
M
</td>
<td>
시간 공백·좌표 없음 재생
</td>
</tr>
<tr>
<td>
FR-LOC-02
</td>
<td>
객체 좌표 보정은 선택 기능으로 분리한다. 적용 시 보정 방식·오차를 저장하고 관측 위치와 구별한다.
</td>
<td>
C
</td>
<td>
별도 기준점 시험
</td>
</tr>
<tr>
<td>
FR-LOC-03
</td>
<td>
같은 주행에서 IoU·시간·진행 방향으로 포트홀 추적 ID와 관측 수를 유지한다.
</td>
<td>
M
</td>
<td>
일대일 객체 ID 대조
</td>
</tr>
<tr>
<td>
FR-LOC-04
</td>
<td>
반복 주행의 10 m 이내·같은 종류·방향·사진 유사 후보를 중복 후보로 연결한다. 자동 합병/삭제하지 않고 담당자가 승인·분리한다.
</td>
<td>
M
</td>
<td>
가까운 별개 포트홀·반복 주행
</td>
</tr>
<tr>
<td>
FR-LOC-05
</td>
<td>
반복 주행의 사진·검토 이력을 조회하되 미검출만으로 보수 완료를 추론하지 않는다.
</td>
<td>
S
</td>
<td>
보수 전후·가림 시나리오
</td>
</tr>
<tr>
<td>
FR-LOC-06
</td>
<td>
기본 검토 순서를 미확인 포트홀→마모 후보→관찰, 같은 군은 오래된 순으로 정한다. 수동 긴급도 변경 사유·작성자·시각을 보존한다.
</td>
<td>
M
</td>
<td>
정렬·수정 이력 검사
</td>
</tr>
</table>

## 4.5 개인정보 보호 (PRV)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-PRV-01
</td>
<td>
별도 YOLOX-Nano 얼굴·번호판 2클래스 검출로 로컬 마스킹하고 검수 상태를 저장한다. 학습 자료의 이용조건을 확인하며, 검수 화면에서 누락 영역을 수동 가림 처리할 수 있어야 한다. 서버/외부 공유 이미지는 사람이 최종 확인한 파일만 허용한다.
</td>
<td>
M
</td>
<td>
마스킹 지표·미검수 업로드 차단
</td>
</tr>
<tr>
<td>
FR-PRV-02
</td>
<td>
비식별화 전 원본은 업로드 API에서 거부하고 암호화 로컬 저장소의 수집 담당자만 접근하게 한다.
</td>
<td>
M
</td>
<td>
원본·검수 우회 전송 시도
</td>
</tr>
<tr>
<td>
FR-PRV-03
</td>
<td>
원본 7일, 마스킹 사진·상세 주행경로 90일, 감사로그 180일을 기본 TTL로 적용한다. 매일·앱 시작 시 삭제하고 이미지·파생파일·경로·백업을 같은 정책으로 정리한다.
</td>
<td>
M
</td>
<td>
기한 경계·재시작·백업 정리 시험
</td>
</tr>
<tr>
<td>
FR-PRV-04
</td>
<td>
차량에 촬영 목적·운영 주체·문의처를 표시하고 수집 계획서에 근거·범위·보관기간을 기록한다.
</td>
<td>
M
</td>
<td>
운용 전 점검 기록
</td>
</tr>
<tr>
<td>
FR-PRV-05
</td>
<td>
열람·정정·삭제 요청을 관리자가 접수하고 대상 원본/관측을 찾을 수 있는 최소 식별정보와 처리 기록을 남긴다. 공개 시연 자료에는 상세 이동경로를 넣지 않는다.
</td>
<td>
M
</td>
<td>
모의 요청·시연 파일 검사
</td>
</tr>
</table>

보관기간은 <b>프로젝트 기본 설계값</b>이며 일률적인 법정 7/30/90일 규정이라는 뜻이 아니다. 표시 의무와 적법한 수집·이용 근거는 별개로 검토한다. [개인정보 보호법 제25조의2](https://www.law.go.kr/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1029334871), [개인정보위 안내서](https://pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS217&mCode=D010030000&nttId=10679)

## 4.6 서버 (SRV)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-SRV-01
</td>
<td>
관측 JSON·마스킹 이미지·주행 메타데이터를 저장하고 observation\_id·image 해시로 중복 수신을 방지한다.
</td>
<td>
M
</td>
<td>
같은 자료 3회 전송 시 관측 1건
</td>
</tr>
<tr>
<td>
FR-SRV-02
</td>
<td>
암호화 로컬 큐에 pending/uploading/acked를 기록한다. 타임아웃·재시작 후 재시도하고 서버 ACK 전 파일을 전송 완료로 지우지 않는다.
</td>
<td>
M
</td>
<td>
단절·충돌·재시작·재전송
</td>
</tr>
<tr>
<td>
FR-SRV-03
</td>
<td>
범위·종류·기간·검토 상태·위치 품질로 조회하고 위치 없는 후보도 별도 목록에 남긴다.
</td>
<td>
M
</td>
<td>
정상·null 위치·권한 필터
</td>
</tr>
<tr>
<td>
FR-SRV-04
</td>
<td>
관리/검토/조회 권한을 구분하고 계정·기관 범위를 서버에서 검사한다. 익명 업로드·이미지 직접 URL 접근을 차단한다.
</td>
<td>
M
</td>
<td>
권한별 허용/거부 시나리오
</td>
</tr>
<tr>
<td>
FR-SRV-05
</td>
<td>
검토된 결과를 UTF-8 CSV로 내보내고 스키마 버전을 포함한다. 행정 시스템 직접 연결과 GeoJSON은 후속 범위다.
</td>
<td>
M
</td>
<td>
인코딩·좌표·행수·권한 대조
</td>
</tr>
</table>

## 4.7 웹 대시보드 (WEB)

<table header-row="true">
<colgroup>
<col width="150" />
<col width="720" />
<col width="80" />
<col width="260" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
요구 동작
</td>
<td>
우선순위
</td>
<td>
검증
</td>
</tr>
<tr>
<td>
FR-WEB-01
</td>
<td>
지도에 포트홀 관측 점과 차선 관측 경로를 표시한다. 추정 구간·위치 품질·미점검 영역을 구분하고 지도 미제공 시 목록으로 동작한다.
</td>
<td>
M
</td>
<td>
범위·위치 없음·지도 키 없음
</td>
</tr>
<tr>
<td>
FR-WEB-02
</td>
<td>
기본 검토 순서와 기간·종류·구역·상태 필터, 수동 우선순위를 제공한다.
</td>
<td>
M
</td>
<td>
정렬·필터 과제
</td>
</tr>
<tr>
<td>
FR-WEB-03
</td>
<td>
대표 이미지·박스/마스크·R·판정 근거·품질·관측 수·프로파일/모델 버전·이력을 제공한다. 치수·부피·자동 자재량은 노출하지 않는다.
</td>
<td>
M
</td>
<td>
JSON·사진·화면 대조
</td>
</tr>
<tr>
<td>
FR-WEB-04
</td>
<td>
미확인→확인됨/오탐/재촬영→조치 예정→조치 완료 흐름을 제공한다. 완료는 담당자·시각·조치 사유가 필요하며, 기각/완료를 되돌리면 감사 이력을 남긴다.
</td>
<td>
M
</td>
<td>
상태 전이·필수값·권한 검증
</td>
</tr>
<tr>
<td>
FR-WEB-05
</td>
<td>
검토 결과·사진·관측 위치·품질·종류별 집계·범위·판정 불가율을 CSV/PDF로 출력한다. 카메라 점검의 한계를 함께 표시한다.
</td>
<td>
M
</td>
<td>
출력과 조회 조건·건수 대조
</td>
</tr>
<tr>
<td>
FR-WEB-06
</td>
<td>
기간별 후보/확인/조치 현황 차트는 선택 기능으로 제공한다. 포트홀 개수와 차선 구간 수를 섞지 않는다.
</td>
<td>
C
</td>
<td>
집계 단위·중복 후보 확인
</td>
</tr>
</table>

# 5. 비기능 요구사항

모든 성능 수치는 개발 합격 목표다. 학습에 사용하지 않은 자체 데이터·주간 건조·20 km/h 이하·정상 장착 프로파일을 기본 조건으로 하고 다른 조건은 별도 표로 보고한다.

<table header-row="true">
<colgroup>
<col width="130" />
<col width="520" />
<col width="80" />
<col width="620" />
</colgroup>
<tr>
<td>
ID
</td>
<td>
지표·합격 기준
</td>
<td>
우선순위
</td>
<td>
측정 조건
</td>
</tr>
<tr>
<td>
NFR-01
</td>
<td>
포트홀 박스 mAP@0.5 ≥ 0.80
</td>
<td>
M
</td>
<td>
고정 테스트 프레임, 정답과 IoU ≥ 0.5 일대일 매칭
</td>
</tr>
<tr>
<td>
NFR-04
</td>
<td>
남은 도막 클래스 IoU ≥ 0.75
</td>
<td>
M
</td>
<td>
배경 포함 mIoU로 대체하지 않고 도막 IoU를 보고
</td>
</tr>
<tr>
<td>
NFR-05
</td>
<td>
시각적 잔존율 MAE ≤ 10%p
</td>
<td>
M
</td>
<td>
합의 마스크의 면적비와 비교, 판정 가능한 고유 구간 기준
</td>
</tr>
<tr>
<td>
NFR-06
</td>
<td>
10 FPS 입력 전체 파이프라인 처리율 ≥ 10 FPS
</td>
<td>
M
</td>
<td>
RTX 3060 12 GB·RAM 16 GB급 기준. 두 인식·추적·비식별화·로컬 저장, 10분·워밍업 제외. 취득 30 FPS와 구별
</td>
</tr>
<tr>
<td>
NFR-07
</td>
<td>
기본 주행 속도 20 km/h 이하
</td>
<td>
M
</td>
<td>
정지·10·20 km/h 구간별 결과 보고. 30 km/h는 S 평가
</td>
</tr>
<tr>
<td>
NFR-08
</td>
<td>
실시간 모드 분석 완료→서버 ACK P95 ≤ 60초
</td>
<td>
S
</td>
<td>
정상 통신 10 Mbps 이상, 관측 ≥ 100건. 배치 지연과 구분
</td>
</tr>
<tr>
<td>
NFR-09
</td>
<td>
관측 위치 오차 중앙값 ≤ 8 m, P95 ≤ 15 m
</td>
<td>
M
</td>
<td>
개활지 기준점 ≥ 20개. 기준 좌표 출처·추정 오차 함께 보고, 정밀 객체 좌표 목표 아님
</td>
</tr>
<tr>
<td>
NFR-10
</td>
<td>
동일 주행 이벤트 중복률 ≤ 5%
</td>
<td>
S
</td>
<td>
동일 정답 객체에 연결된 초과 이벤트 수 / 해당 객체에 연결된 전체 이벤트 수. FP와 별도
</td>
</tr>
<tr>
<td>
NFR-11
</td>
<td>
로컬 마스킹 대상 탐지 Recall ≥ 95%
</td>
<td>
M
</td>
<td>
얼굴·번호판 각 ≥ 50개, 합계 ≥ 200개. 종류별 결과 보고. 공유 전 검수 생략 불가
</td>
</tr>
<tr>
<td>
NFR-12
</td>
<td>
60분 취득 연속 동작, 영상·GNSS 로그 손상 0건
</td>
<td>
M
</td>
<td>
실차/동일 구성 시험, 누락 프레임 비율 ≤ 1%
</td>
</tr>
<tr>
<td>
NFR-13
</td>
<td>
장착·시야·렌즈·전원 사전 점검 전 항목 통과
</td>
<td>
M
</td>
<td>
IP66은 본체 사양만 표기. 오염·흔들림 시 중지·미점검 처리
</td>
</tr>
<tr>
<td>
NFR-14
</td>
<td>
HTTPS·비밀번호 해시·서버 권한·비공개 저장소
</td>
<td>
M
</td>
<td>
다른 역할/기관·익명 접근·원본 전송 거부. Argon2id, 비밀키 소스 미포함
</td>
</tr>
<tr>
<td>
NFR-15
</td>
<td>
주요 사용자 과제 성공률 ≥ 80%
</td>
<td>
M
</td>
<td>
5명 × 5개 과제, 사전 설명 10분 이내. 검색·사진 확인·기각·조치·보고서
</td>
</tr>
<tr>
<td>
NFR-16
</td>
<td>
환경·가중치·임계값·보정 해시 기록
</td>
<td>
M
</td>
<td>
잠근 버전의 같은 입력 결과 재현, 버전 없는 결과 저장 거부
</td>
</tr>
<tr>
<td>
NFR-17
</td>
<td>
독립 포트홀 이벤트 Recall ≥ 0.85
</td>
<td>
M
</td>
<td>
실제 포트홀 ≥ 30개, 보류·누락은 FN. 매칭 기준 9장
</td>
</tr>
<tr>
<td>
NFR-18
</td>
<td>
마모 후보 Recall ≥ 0.85, Precision ≥ 0.75
</td>
<td>
M
</td>
<td>
독립 차선 ≥ 60구간, 각 군 ≥ 20. 판정 불가인 실제 마모도 FN
</td>
</tr>
<tr>
<td>
NFR-19
</td>
<td>
평가 대상 차선 구간의 판정 가능 비율 ≥ 80%
</td>
<td>
M
</td>
<td>
판정 불가·미검출 포함 분모. 강제 R 생성 금지
</td>
</tr>
<tr>
<td>
NFR-20
</td>
<td>
포트홀 후보 오탐 ≤ 2건/km
</td>
<td>
M
</td>
<td>
독립 연속 주행 ≥ 5 km 전수 확인. 단발·미확인·보류 후보도 검토 부하 집계
</td>
</tr>
<tr>
<td>
NFR-21
</td>
<td>
유효 GNSS의 영상-위치 시각 차이 P95 ≤ 0.5초
</td>
<td>
M
</td>
<td>
GNSS UTC·수신 지연·PC 시계 보정값 기록. 2초 초과 공백 보간 금지
</td>
</tr>
<tr>
<td>
NFR-22
</td>
<td>
2차종 장착·촬영, 차종당 장착·설정 ≤ 30분
</td>
<td>
M
</td>
<td>
최초 브래킷 제작시간 제외. 2차종 시야·저속 촬영, 동일 정확도 보장 아님
</td>
</tr>
<tr>
<td>
NFR-23
</td>
<td>
ACK 관측 유실·이중 생성 0건
</td>
<td>
M
</td>
<td>
100건에 동일 요청 3회·10분 단절·앱 재시작 조합
</td>
</tr>
<tr>
<td>
NFR-24
</td>
<td>
저장 10 GB 미만 경고·신규 원본 녹화 중지
</td>
<td>
M
</td>
<td>
미전송 자료 덮어쓰기 금지, 재시작 안내
</td>
</tr>
<tr>
<td>
NFR-25
</td>
<td>
미검수 이미지·원본 외부 공유 0건
</td>
<td>
M
</td>
<td>
상태 위조·직접 링크 접근 시험, 공유 자료 전수 확인
</td>
</tr>
</table>

기존 NFR-02(깊이 오차)·NFR-03(면적 오차)은 v0.4 검수에서 제외한다. 일부 기존 ID의 조건·우선순위는 변경되었으므로 v0.3 수치와 혼용하지 않는다.

# 6. 데이터 요구사항

## 6.1 라벨·분할·수집

| 데이터 | 라벨·필드 | 기준 |
| --- | --- | --- |
| 포트홀 | pothole 박스, 실물 객체 ID, 물/가림/불명확 속성 | 맨홀·패치·그림자·웅덩이 음성 태그 |
| 차선 | 예상 도색 외곽, 남은 도막, 의도적 공백, 유효 영역, 구간 ID | 백색/황색·실선/점선, 외곽 복원 불가도 라벨링 |
| 주행 | run\_id, 차종·프로파일, 시각, 속도, 위치 품질, 날씨·노면 | 3개 이상 구역·6개 이상 촬영 묶음 |
| 출처 | URL·버전·사용조건·동의/운영 근거·분할 | 승인 없는 자료를 사용 가능으로 표기 금지 |
| 프라이버시 | 얼굴·번호판 정답 박스, 마스킹/사람 검수 상태 | 공개 자료·동의받은 연출 표본, 원본 접근 최소화 |

공개 노면 ≥ 3,000장, 자체 RGB ≥ 600프레임, 자체 차선 마스크 ≥ 300프레임을 계획한다. 독립 테스트는 실제 포트홀 ≥ 30개, 차선 ≥ 60구간(양호/관찰/마모 후보 각각 ≥ 20), 연속 시험 ≥ 5 km다. 장소/노선/동일 객체 단위로 학습·검증·시험 약 70/15/15에 묶되 독립 시험 표본을 우선 확보한다. 테스트·검증 라벨은 2인 교차 검수하고 충돌은 A·B가 합의한다.

AI Hub 도로파임 박스와 차선 segmentation은 사전학습 자료다. 공개 스키마에 없는 “마모 잔존율 정답”을 있다고 가정하지 않는다. [AI Hub 원자료·라벨 설명](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71781&topMenu=100) RDD2022는 포트홀 보조 학습용이며 차선 마모 정답으로 사용하지 않는다. [RDD2022 논문](https://arxiv.org/abs/2209.08538)

## 6.2 관측 데이터 스키마

type은 pothole 또는 lane\_segment다. 종류에 맞는 하위 필드만 채우고 반대쪽은 null로 둔다. 아래 값은 시험 실적이 아닌 비식별 인터페이스 예시다.

```json
{
  "schema_version": "0.4",
  "observation_id": "UUID",
  "event_id": "UUID",
  "run_id": "run-demo-01",
  "device_id": "device-01",
  "type": "pothole",
  "captured_at": "2026-11-03T10:15:30+09:00",
  "frame_id": 1234,
  "location": {
    "lat": 37,
    "lon": 127,
    "source": "vehicle_observation",
    "quality": "valid",
    "hdop": 1.2,
    "age_ms": 100,
    "accuracy_m": null
  },
  "profile_id": "vehicle-A-v1",
  "model_version": "pothole-v1",
  "config_hash": "sha256",
  "pothole": {
    "bbox_xywh_px": [
      100,
      200,
      160,
      80
    ],
    "confidence": 0.91,
    "track_observations": 5,
    "candidate_state": "multi_frame",
    "occluded": false
  },
  "lane": null,
  "quality": {
    "usable": true,
    "reasons": []
  },
  "image": {
    "file_id": "UUID",
    "sha256": "sha256",
    "redacted": true,
    "reviewed_for_sharing": true
  },
  "review": {
    "status": "unreviewed",
    "priority": "normal"
  }
}
```

차선 하위 객체 예시:

```json
{
  "lane": {
    "segment_id": "lane-segment-UUID",
    "estimated_length_m": 10,
    "retention_pct": 55,
    "grade": "worn_candidate",
    "color": "white",
    "line_type": "dashed",
    "valid_area_ratio": 0.9,
    "outline_supported": true,
    "patch_ids": [
      "patch-01",
      "patch-02"
    ],
    "unassessable_reason": null
  },
  "pothole": null
}
```

- 식별자는 UUID 등 충돌하지 않는 값으로 정한다. observation\_id는 재전송해도 바꾸지 않는다.
- lat/lon은 둘 다 유효하거나 둘 다 null이다. WGS84를 사용하며 수신기가 정확도를 제공하지 않으면 accuracy\_m=null이다. HDOP를 m 정확도로 변환해 넣지 않는다.
- retention\_pct는 산출 가능한 경우 유한한 0\~100, valid\_area\_ratio는 산출 가능한 경우 0\~1이다. 외곽 분모가 정의되지 않으면 valid\_area\_ratio=null을 허용한다. 구간 도색 예정 길이 분모가 없으면 segment\_support\_ratio=null이다. 필요한 비율이 미정의이거나 판정 불가이면 grade=unassessable, R=null, 사유 필수다. 미정의 비율을 측정값 0으로 대신하지 않는다.
- 타입·버전·이미지 상태 불일치, 원본·미검수 이미지는 서버에서 거부한다. 실측하지 않은 깊이·면적·부피 필드를 생성하지 않는다.
- priority는 담당자 검토용 긴급도다. 출처·수정 이력을 보존하고 confidence와 혼동하지 않는다.

## 6.3 보관·이력·삭제

원본 좌우영상/선택 SVO·GNSS 로그는 암호화 로컬 저장에서 7일 이내 삭제한다. 서버의 마스킹 사진·관측 상세·상세 이동경로는 90일, 감사로그는 180일을 기본으로 한다. 식별 영상·상세 경로가 포함된 파생·캐시·백업도 같은 정책을 따른다. 삭제 후에는 식별 영상·경로 없이 익명 집계와 삭제 기록만 유지할 수 있으며 익명성·원자료 이용조건을 검토한 경우로 한정한다.

captured\_at과 별도로 수신·검토·조치·삭제 시각을 UTC로 저장하고 UI는 Asia/Seoul로 표시한다. 사진·관측·이력 ID의 참조 무결성을 보장하고 수동 합병은 원관측을 보존한 연결 관계로 구현한다.

# 7. 외부 인터페이스 요구사항

<table header-row="true">
<colgroup>
<col width="150" />
<col width="470" />
<col width="580" />
</colgroup>
<tr>
<td>
구분
</td>
<td>
인터페이스
</td>
<td>
동작·오류 조건
</td>
</tr>
<tr>
<td>
카메라
</td>
<td>
ZED 2i ↔ PC USB 3 / UVC
</td>
<td>
좌우 결합 프레임·시각 분리 저장. 선택 SDK/SVO도 동일 주행 구조
</td>
</tr>
<tr>
<td>
위치
</td>
<td>
GNSS ↔ PC USB / NMEA
</td>
<td>
UTC·fix·HDOP·좌표·속도. 시간 누락·속도 없음·fix 없음 구분
</td>
</tr>
<tr>
<td>
업로드
</td>
<td>
POST /api/v1/images, POST /api/v1/observations
</td>
<td>
장치 인증, 마스킹·검수 확인, 해시·ID 중복 검사. ACK에 서버 ID·상태
</td>
</tr>
<tr>
<td>
조회
</td>
<td>
GET /api/v1/observations
</td>
<td>
bbox·from·to·type·review\_status·quality, 페이지 나눔, 관할 권한 서버 검사
</td>
</tr>
<tr>
<td>
검토·중복
</td>
<td>
PATCH /api/v1/observations/\{id\}/review, POST /api/v1/duplicate-links
</td>
<td>
역할·변경 사유·버전 검사. 승인 없는 병합 거부
</td>
</tr>
<tr>
<td>
보고서
</td>
<td>
GET /api/v1/reports?format=csv 또는 pdf
</td>
<td>
조회 조건·기관 권한 적용, 관측 위치·시각적 지표 한계 출력
</td>
</tr>
<tr>
<td>
지도
</td>
<td>
Leaflet + 브이월드 어댑터
</td>
<td>
키·출처·이용정책 관리, 공급 실패에도 목록·좌표·자체 경로 사용
</td>
</tr>
</table>

공통 응답: 필드 오류 422, 인증 실패 401, 권한 부족 403, 버전 충돌 409, 일시 장애 5xx. 재시도는 동일 observation\_id로 5초·15초·60초 후 실행하고 이후 5분 간격으로 이어간다. 이미지 5 MB/건, JSON 64 KB/건 상한을 적용하며 초과 시 로컬 오류 목록에 남긴다.

# 8. 판정 및 검토 기준

## 8.1 포트홀

confidence는 모델 인식 점수다. 기본 이벤트는 1초 내 3회 관측이며 고신뢰 단발 후보를 별도 검토로 기록한다. 등록 후보는 담당자 확인 전 “포트홀 의심”이다. 깊이·지름이 없으므로 ASTM D6433 심각도를 자동 부여하지 않는다. 담당자가 현장 확인한 긴급도를 입력할 수 있다.

## 8.2 차선 마모

BEV의 고유 패치마다 <b>R = 남은 도막 면적 / 예상 도색 면적 × 100</b>으로 산출한다. 차선 밖 배경·의도적 점선 공백·무효 픽셀은 분모에서 제외한다. 유효 면적 비율은 관측 대상 예상 도색 면적 대비 유효 픽셀 면적이며, 70% 미만이면 패치 판정 불가다. 외곽 자체가 복원되지 않으면 비율도 계산하지 않는다.

| 내부 등급 | 구간 대표 R | 표기·업무 의미 |
| --- | --- | --- |
| visible\_good | 80% 이상 | 시각상 양호, 반사성능 적합 아님 |
| monitor | 60% 이상 80% 미만 | 관찰, 다음 점검 비교 |
| worn\_candidate | 60% 미만 | 마모 후보, 사진·현장 확인 |
| unassessable | R 없음 | 판정 불가 사유·재촬영 |

구간 R은 중복 제거한 유효 패치의 면적 가중 평균이다. 예상 구간의 80% 이상에 외곽·유효 패치 근거가 있을 때만 구간 등급을 내리고 나머지는 구간 판정 불가로 남긴다. 60%/80%는 초기 설계 임계값이며 법정 재도색·안전 기준이 아니다. 재귀반사성능은 별도 측정 절차로 판단한다. [국토교통부 매뉴얼 발간 정보](https://www.codil.or.kr/viewDtlConRpt.do?gubun=rpt&pMetaCode=OTKCMA240494)

## 8.3 상태 전이·검토 순서

미확인에서 확인됨/오탐/재촬영으로 변경한다. 확인됨에서 조치 예정, 조치 예정에서 조치 완료로 이동한다. 단계 건너뛰기·완료 취소는 검토 권한과 사유를 요구하고 이전 이력을 보존한다. 완료 사건 근처에서 새 관측이 생기면 재발 후보를 만들고 과거 기록을 덮어쓰지 않는다.

대기열은 미확인 포트홀→마모 후보→관찰 순이며 같은 군은 최초 관측 시각 오름차순이다. 수동 긴급 건은 맨 앞에 표시한다. 판정 불가/위치 누락은 별도 경고 목록에 남기고 숨기지 않는다.

# 9. 검수 및 평가 기준

## 9.1 시험 원칙

- W6 말 학습/검증/시험 목록, 모델·보정·임계값을 잠그고 W7에 독립 시험한다. 정답자는 모델 결과를 보지 않고 표기하고 2인이 검수한다.
- 필수 운용 범위·가시 ROI에 들어온 정답만 성능 모집단으로 정의하고, 바깥 범위·가림·부적합 환경의 제외 수와 사유도 별도 보고한다.
- 테스트 정답을 본 뒤 임계값을 고쳐 같은 시험 성적을 최종값으로 재사용하지 않는다. 수정 시 버전·추가 시험셋·변경 사유를 남긴다.
- 성능별 표본 수·TP/FP/FN·신뢰구간 또는 표본 한계를 명시한다. 작은 표본으로 전국·고속도로·모든 차종 성능을 일반화하지 않는다.

## 9.2 시험·요구사항 추적표

<table header-row="true">
<colgroup>
<col width="180" />
<col width="270" />
<col width="760" />
</colgroup>
<tr>
<td>
시험
</td>
<td>
대상 요구사항
</td>
<td>
방법·합격 기준
</td>
</tr>
<tr>
<td>
TC-01 취득·동기·품질
</td>
<td>
ACQ-01\~06, NFR-12/13/21/24
</td>
<td>
좌우 10분 재생, 60분 연속 취득, GNSS 공백·오염·저장 부족 주입. 각 NFR 기준과 경고 동작 충족
</td>
</tr>
<tr>
<td>
TC-02 포트홀 박스
</td>
<td>
PH-01/10, NFR-01
</td>
<td>
고정 프레임 정답과 IoU ≥ 0.5 일대일 매칭, mAP@0.5 ≥ 0.80. 실제 원영상 좌표 기준
</td>
</tr>
<tr>
<td>
TC-03 포트홀 이벤트·오탐
</td>
<td>
PH-11/12, LOC-03, NFR-17/20
</td>
<td>
실제 ≥ 30개·연속 ≥ 5 km. 정답 실물 ID의 가시 프레임 중 박스 IoU ≥ 0.5 관측이 있고 이벤트가 생성되어야 TP. 보류·미검출은 FN, 단발은 별도 검토 후보로 생성된 경우 TP 인정. Recall ≥ 0.85, FP ≤ 2건/km
</td>
</tr>
<tr>
<td>
TC-04 차선·마모
</td>
<td>
LN-01\~06/09/10, NFR-04/05/18/19
</td>
<td>
독립 ≥ 60구간, 군별 ≥ 20. 도막 IoU ≥ 0.75, R MAE ≤ 10%p, 마모 Recall ≥ 0.85·Precision ≥ 0.75, 판정 가능 ≥ 80%. 판정 불가 실제 마모는 FN
</td>
</tr>
<tr>
<td>
TC-05 위치·중복
</td>
<td>
LOC-01/04/06, NFR-09/10
</td>
<td>
기준점 ≥ 20개로 차량 촬영위치 오차 평가, 개별 좌표와 기준 오차 함께 기록. 가까운 두 포트홀·반복 통과에서 자동 합병 없음
</td>
</tr>
<tr>
<td>
TC-06 분석 성능
</td>
<td>
NFR-06/07/08
</td>
<td>
동일 기준 PC, 10분 이상 두 인식·추적·마스킹·로컬 저장 처리율 ≥ 10 FPS. 30 FPS 취득·10 FPS 입력을 구분. 실시간 지연은 S로 분리
</td>
</tr>
<tr>
<td>
TC-07 개인정보·보안·삭제
</td>
<td>
PRV-01\~05, SRV-04, NFR-11/14/25
</td>
<td>
얼굴·번호판 합계 ≥ 200개, 종류별 Recall ≥ 95%. 미검수/원본 차단·기관/역할 격리·TTL/백업 삭제 전 항목 통과
</td>
</tr>
<tr>
<td>
TC-08 수신·복구·상태
</td>
<td>
SRV-01\~03, WEB-02\~04, NFR-23
</td>
<td>
100건 전송, 3회 중복·10분 단절·재시작. 유실·이중 생성 0건, 허용 상태 전이·수정 이력 확인
</td>
</tr>
<tr>
<td>
TC-09 보고서·사용성
</td>
<td>
SRV-05, WEB-01\~05, NFR-15
</td>
<td>
5명 × 5과제 성공 ≥ 80%. 필터·행수·좌표 품질·단위·사진·검토 상태가 CSV/PDF와 일치
</td>
</tr>
<tr>
<td>
TC-10 장착 범용성
</td>
<td>
ACQ-06, NFR-22
</td>
<td>
2차종 장착 30분 이내, 프로파일·가시 노면·저속 영상 확인. 주 차량의 정량 성능과 구분
</td>
</tr>
</table>

## 9.3 지표 계산·미달 처리

포트홀 이벤트 Recall = TP/(TP+FN), 마모 후보 Precision = TP/(TP+FP), Recall = TP/(TP+FN)이다. 이벤트별 매칭은 일대일이며, 같은 실물의 추가 이벤트는 중복 지표로 분리한다. 포트홀 FP/km는 정답과 매칭되지 않은 모든 검토 후보 이벤트 수/실제 평가 주행거리다. 단발·보류가 사람 검토 목록을 차지하면 포함한다.

차선 정답은 마스크 기준 R \< 60%를 양성, 60% 이상을 음성으로 한다. 시험 구간은 정답 구간 ID·가시 길이로 매칭하며 관측 길이 겹침 50% 이상인 예측을 연결한다. 일대일 매칭이 안 되면 해당 정답은 FN/판정 불가로 남긴다. R MAE만 낮고 판정 가능 비율이 낮으면 합격이 아니다. 유형·속도별 혼동행렬과 누락 원인도 기록한다.

M 항목은 모두 충족해야 “시제품 필수 검수 합격”이다. 표본 부족·장비 미확보·기관 미협약·성능 미달은 그대로 명시하고 소프트웨어 시연 성공과 구분한다. 2차종 동일 성능·고속 운용·실제 인건비 절감·법정 검사 대체는 본 검수 합격의 의미에 포함되지 않는다.

# 10. 결정 사항 및 실행 전제

<table header-row="true">
<colgroup>
<col width="180" />
<col width="430" />
<col width="540" />
</colgroup>
<tr>
<td>
항목
</td>
<td>
이번 문서의 결정
</td>
<td>
근거·확인 방식
</td>
</tr>
<tr>
<td>
제품 범위
</td>
<td>
포트홀·마모차선 인식과 담당자 검토, 도색·자동 보수 제외
</td>
<td>
사용자 주제 변경 반영
</td>
</tr>
<tr>
<td>
카메라
</td>
<td>
ZED 2i 그대로 사용
</td>
<td>
사용자 추가 확인 반영, 렌즈 사양은 실장비 프로파일로 기록
</td>
</tr>
<tr>
<td>
수행 조건
</td>
<td>
4명·8주, A\~D 각 1명 책임
</td>
<td>
기존 기간·인원 유지, 개인 실명 미제공이므로 역할 기준
</td>
</tr>
<tr>
<td>
연산·전송
</td>
<td>
기존/대여 GPU PC·노트북, 배치 필수·실시간 S
</td>
<td>
추가 구매·최적화 부담 절감. W1 대여·성능 확인 필요
</td>
</tr>
<tr>
<td>
GNSS·장착
</td>
<td>
일반 5 Hz NMEA, 전방 하단 탈부착
</td>
<td>
RTK·고가 GNSS 필수 제외, 차종별 시야·간섭 검증
</td>
</tr>
<tr>
<td>
모델·라이선스
</td>
<td>
YOLOX-Nano·DeepLabV3-MobileNetV3
</td>
<td>
구현 라이선스와 데이터·가중치 조건을 분리해 목록 관리
</td>
</tr>
<tr>
<td>
마모 임계값
</td>
<td>
60%·80%, 품질 70%, 구간 근거 80%
</td>
<td>
팀 초기 설계값. W6 검증셋 기준 동결, 법정 수치로 해석 금지
</td>
</tr>
<tr>
<td>
검토 순서
</td>
<td>
포트홀→마모 후보→관찰, 사람 지정 긴급도 우선
</td>
<td>
측정하지 않은 깊이·도로명 기반 가중치로 위험도를 꾸미지 않음
</td>
</tr>
<tr>
<td>
개인정보
</td>
<td>
원본 7일·점검 상세 90일·감사로그 180일, 공유 전 사람 확인
</td>
<td>
목적 최소 보관을 위한 설계값, 운영 방침에 맞춰 변경 가능
</td>
</tr>
<tr>
<td>
추가 예산
</td>
<td>
제안 한도 800,000원, 계획 736,000원
</td>
<td>
실제 배정액·시장 견적 아님. 세부 배정은 기획서 10장
</td>
</tr>
<tr>
<td>
지도
</td>
<td>
브이월드 우선, 키 없으면 좌표·목록·자체 경로
</td>
<td>
이용 약관의 무제한 공공·상업 이용을 전제하지 않음
</td>
</tr>
<tr>
<td>
미확보·미달
</td>
<td>
조건 확인은 W1, 통합 W4/W6, 검수 W7
</td>
<td>
확인 전 장비 확보·수요기관 동의·목표 달성을 사실로 기록하지 않음
</td>
</tr>
</table>

내부 설계 선택은 위와 같이 채웠다. 실제 예산 배정·장비 대여·촬영 장소 사용·데이터 이용 승인·기관 검증은 문서 작성만으로 확정할 수 없는 실행 전제다. 전제가 충족되지 않으면 대체 자료의 범위와 미달 항목을 명시하며 허위 완료 처리하지 않는다.

# 11. 변경 이력 및 근거 자료

| 버전 | 날짜 | 내용 | 작성 구분 |
| --- | --- | --- | --- |
| v0.1 | <date start="2026-10-27" /> | 초안 작성 | 기존 기록 |
| v0.2 | <date start="2026-10-27" /> | 기존 법령·ASTM·편람 관련 요구 추가 | 과거 버전 기록, v0.4 적용 요건과 구별 |
| v0.3 | <date start="2026-10-27" /> | 8주·장비·정량 측정 목표 설정 | 과거 버전 기록 |
| v0.4 | <date start="2026-11-03" /> | ZED 2i 유지, 포트홀·마모차선 인식·B2G 중심으로 개정. 필수 치수 측정/도색 제외, 데이터·품질·장착·위치·개인정보·검수 기준 재정의 | 사용자 방향 반영 및 작성자 설계 기준 |

연계 문서: <mention url="https://app.notion.com/p/40d8ac7c2dde815b87b4e35b991407dc" />

자료 확인일: <date start="2026-11-03" />. 요구사항 수치는 출처의 제품 성능을 베낀 값이 아닌 본 시제품의 시험 목표다.

- [ZED 2i 공식 비교 사양](https://support.stereolabs.com/articles/9573039616-where-can-i-find-a-comparison-table-for-the-zed-stereo-cameras), [UVC/SDK 안내](https://support.stereolabs.com/articles/7166665108-can-i-use-the-zed-without-cuda)
- [AI Hub 고해상도 도로노면 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71781&topMenu=100), [RDD2022 원논문](https://arxiv.org/abs/2209.08538)
- [YOLOX 공식 구현](https://github.com/Megvii-BaseDetection/YOLOX), [라이선스](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)
- [Torchvision DeepLabV3](https://docs.pytorch.org/vision/stable/models/deeplabv3.html), [라이선스](https://github.com/pytorch/vision/blob/main/LICENSE)
- [개인정보 보호법 제25조의2](https://www.law.go.kr/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1029334871), [개인정보위 안내서](https://pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS217&mCode=D010030000&nttId=10679)
- [국토교통부 차선 도색 유지·관리 매뉴얼 발간 정보](https://www.codil.or.kr/viewDtlConRpt.do?gubun=rpt&pMetaCode=OTKCMA240494) — 별도 노면표시 성능검사와 관리 업무의 참고 자료.
- [브이월드 개발자 페이지](https://www.vworld.kr/dev/v4api.do) — 상세 약관 본문 확인이 제한되어 W1 서비스 도입 시 이용조건을 확인할 항목.
</content>
</page>
