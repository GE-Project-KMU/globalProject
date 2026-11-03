Here is the result of "fetch" for the Page with URL https://app.notion.com/p/40d8ac7c2dde81b5af5cc462aa535923 as of 2026-11-03T09:38:18.878Z:
<page url="https://app.notion.com/p/40d8ac7c2dde81b5af5cc462aa535923" icon="🏷️">
<ancestor-path>
<parent-page url="https://app.notion.com/p/40d8ac7c2dde801095cdca9e832444f8" title="글로벌공학설계프로젝트"/>
</ancestor-path>
<properties>
{"title":"포트홀·마모차선 라벨링 가이드"}
</properties>
<iconMetadata>{"type":"emoji","emoji":"🏷️"}</iconMetadata>
<content>
<callout icon="🏷️" color="blue_bg">
포트홀·마모차선 라벨링 가이드 v1.0 · 작성일: <date start="2026-11-03" /><br />대상 장비 ZED 2i, 기준 SRS v0.4·SDD v1.0. 라벨 스키마 dataset\_schema\_version=1.0, 서버 관측 schema\_version=0.4는 서로 다른 버전이다.<br />정답은 영상에서 확인 가능한 사실과 근거로 작성한다. AI confidence·예측 박스·법정 위험등급을 정답으로 복사하지 않는다.
</callout>

<table-of-contents />

# 1. 적용·작업 순서

필수 대상은 주간·건조 아스팔트의 포트홀, 백색/황색 실선·점선의 시각적 도막 손실이다. 좌영상을 기본 라벨 영상으로 사용하고 같은 시각의 우영상·인접 프레임은 경계/실물 확인의 보조 근거로만 쓴다. 좌우/연속 영상은 같은 분할 그룹이다.

작업 순서: 권한·기한·마스킹 상태 확인→원영상/ROI/보정 확인→실물·차선 ID 부여→박스·마스크·무효 영역 표기→라벨 사유→자동 검사→교차 검수→합의→release 승격. 시험/검증 GT는 모델 예측을 보지 않고 작성한다. 학습용 자동 초안은 사람 검수 후에만 채택하고 초안 생성 모델 버전을 기록한다.

참조: <mention url="https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf" />, <mention url="https://app.notion.com/p/40d8ac7c2dde81f3b7a4f3ae95edb804" />.

# 2. 좌표·식별자·형식

원영상 좌표는 좌상단 (0,0), x는 오른쪽·y는 아래 방향이다. 이미지 크기는 width,height 순서로 기록하고 COCO RLE size만 \[height,width\] 순서다. 박스는 원본 좌영상의 \[x,y,width,height\] 실수 픽셀이며 x/y≥0, width/height>0, x+width≤영상 폭·y+height≤영상 높이다. resize/letterbox 이미지 좌표를 원영상으로 오인하지 않는다.

박스 경계는 연속 좌표/반열림 범위 \[x,x+w),\[y,y+h)로 정의한다. 마스크는 원영상과 같은 해상도로 저장하고 resize/BEV를 쓰면 coord\_space와 변환·보정 버전을 필수 기록한다. 폴리곤은 자기 교차·이미지 밖 좌표·중복 점·면적 0을 허용하지 않는다.

sample\_id·annotation\_id·physical\_object\_id·lane\_segment\_id는 UUID다. 같은 실물의 반복 프레임은 physical\_object\_id, 같은 차선/노면 패치는 lane\_track\_id·patch\_id를 재사용한다. 새로운 가림/화질 변화가 새 실물 ID를 의미하지 않는다. 주행 이벤트·서버 case ID는 GT ID와 별도 매칭표로 연결한다.

# 3. 클래스·학습 내보내기

| 내부 라벨 | 표기 형식 | 학습·평가 사용 |
| --- | --- | --- |
| pothole | tight bbox + 실물 ID + 속성 | 포트홀 탐지. COCO category\_id=1, 모델 내부 class\_index=0으로 별도 매핑 |
| hard\_negative | 그림자/맨홀/패치/웅덩이 등 유형 태그·선택 영역 | 포트홀 아님이 확인된 어려운 음성. 실제 포트홀과 함께 있을 수 있음 |
| lane\_outline | 예상 차선 띠 O 마스크 | 잔존율 분모 복원 근거. 설계상 공백 G와 함께 해석 |
| lane\_paint | 남은 도막 P 마스크 | 도막 분할. 픽셀 학습 라벨 paint=1/background=0, 무효/범위 밖=255 ignore |
| intentional\_gap | 정상 점선 공백 G 마스크 | 도색이 예정되지 않은 영역. 마모 면적에 넣지 않음 |
| valid\_region | 판독 가능한 영역 V 마스크 | 가림·판독 불가·ROI 밖 제외 |
| face / license\_plate | 정답 bbox + 인스턴스 ID | 별도 비식별화 검출기 학습/평가. 포트홀 클래스와 혼합하지 않음 |
| uncertain / out\_of\_scope | 영역·사유·대상 종류 | 학습 기본 음성으로 변환 금지. 평가 제외/불가 정책 별도 기록 |

hard\_negative는 모델 클래스 매핑에서 자동으로 새로운 탐지 클래스가 되는 것이 아니다. 이 MVP의 포트홀 출력 클래스는 1개다. ignore 영역을 지원하지 않는 탐지 학습기에는 불확실 물체가 포함된 이미지를 “빈 annotation=정상 음성”으로 내보내지 않는다. 안전한 crop으로 분리하거나 해당 이미지를 학습 보류한다.

# 4. 포트홀 라벨 기준

## 4.1 양성·음성·보류

<table header-row="true">
<colgroup>
<col width="350" />
<col width="260" />
<col width="650" />
</colgroup>
<tr>
<td>
사례
</td>
<td>
판정
</td>
<td>
표기 방법·근거
</td>
</tr>
<tr>
<td>
포장재가 떨어져 나가 오목한 결손이 영상으로 확인됨
</td>
<td>
pothole confirmed
</td>
<td>
보이는 결손의 실제 경계를 tight bbox로 감쌈. 단순 색 얼룩·그림자까지 포함하지 않음
</td>
</tr>
<tr>
<td>
맨홀·덮개·균열만 있는 노면·평탄한 보수 흔적
</td>
<td>
hard\_negative
</td>
<td>
각 유형 태그. 경계에 별도의 결손이 있으면 그 부분만 별도 pothole
</td>
</tr>
<tr>
<td>
물 고임·강한 그림자로 내부/경계를 확인할 수 없음
</td>
<td>
uncertain
</td>
<td>
possible\_pothole·wet/shadow 사유. 인접 프레임으로 결손이 확인되면 confirmed+wet/occluded 속성
</td>
</tr>
<tr>
<td>
차량/사람에 일부 가려짐
</td>
<td>
근거가 있으면 confirmed, 없으면 uncertain
</td>
<td>
보이는 부분의 tight bbox, occluded=true. 가린 뒤의 경계를 임의로 추정하지 않음
</td>
</tr>
<tr>
<td>
완전히 가려짐·화면 밖·ROI 밖
</td>
<td>
관측 GT로 억지 생성하지 않음
</td>
<td>
visible=false/out\_of\_scope. 실물 ID 연결 근거만 보존하고 보이는 프레임의 정답과 구분
</td>
</tr>
<tr>
<td>
작거나 멀지만 결손이 식별됨
</td>
<td>
confirmed + small/distant 속성
</td>
<td>
크기만으로 삭제하지 않음. 식별 불가면 uncertain, 물리적 깊이/지름 cutoff 적용 안 함
</td>
</tr>
<tr>
<td>
연속 결손 여러 개
</td>
<td>
연결된 한 결손은 1개, 명확한 온전한 노면으로 분리되면 여러 개
</td>
<td>
객체 ID를 일관되게 부여. 동일 실물 반복 프레임 수를 새 객체 수로 세지 않음
</td>
</tr>
</table>

## 4.2 박스·속성 규칙

박스는 눈에 보이는 결손을 빠짐없이 포함하면서 불필요한 주변 노면을 최소화한다. 임의 padding을 넣지 않는다. 잔금·균열이 옆으로 길게 이어져도 단순 균열까지 같은 포트홀 박스에 넣지 않는다. 한 프레임의 모든 확인 가능한 포트홀을 표시하고 누락 양성을 정상 배경으로 남기지 않는다.

필수 속성은 certainty(confirmed/uncertain), wet, occluded, truncated, quality\_reasons, physical\_object\_id, evidence\_frame\_ids다. 영상만으로 알 수 없는 깊이·부피·실제 지름·사고 위험도는 null/미사용이며 추정값을 GT에 넣지 않는다. hard negative 이미지의 빈 박스 목록은 전수 확인 후에만 허용한다.

모델이 보류한 confirmed 실물은 이벤트 Recall의 FN이다. 정답자 자체가 불확실한 표본은 별도 목록·수량·사유로 잠그되 시험 결과를 본 뒤 confirmed 양성을 uncertain으로 바꿔 분모에서 빼지 않는다.

## 4.3 포트홀 라벨 예시

아래는 실제 촬영 자료가 아닌 형식 예시다.

```json
{
  "dataset_schema_version": "1.0",
  "sample_id": "11000000-0000-4000-8000-000000000001",
  "image": {
    "width": 1280,
    "height": 720,
    "coord_space": "left_original"
  },
  "potholes": [
    {
      "annotation_id": "12000000-0000-4000-8000-000000000001",
      "physical_object_id": "13000000-0000-4000-8000-000000000001",
      "category": "pothole",
      "bbox_xywh_px": [
        100,
        200,
        160,
        80
      ],
      "certainty": "confirmed",
      "wet": false,
      "occluded": false,
      "truncated": false,
      "evidence_frame_ids": [
        1234
      ],
      "quality_reasons": []
    }
  ],
  "hard_negatives": [
    {
      "kind": "manhole",
      "bbox_xywh_px": [
        600,
        300,
        60,
        40
      ]
    }
  ],
  "detection_eligible": true,
  "annotation_status": "draft"
}
```

# 5. 차선·마모 라벨 기준

## 5.1 무엇을 칠할 것인가

O는 판단 대상 차선의 예상 띠, G는 그중 설계상 정상 공백, P는 실제로 남아 있는 도막, V는 판독 가능한 영역이다. O를 복원할 근거가 없는 곳은 outline\_supported=false로 두고 외곽을 상상해 채우지 않는다. 모든 마스크는 같은 좌표계·크기다.

실선·점선, 백색·황색을 구분한다. 좌·우 차선과 복선의 두 줄은 각각 lane\_track\_id로 관리한다. 정상 점선 사이의 빈 노면은 G이며 도막 소실이 아니다. 횡단보도·정지선·문자·화살표·청색 차선은 out\_of\_scope로 표시하고 정상 차선 학습 음성으로 무조건 내보내지 않는다.

## 5.2 경계·소실·가림 규칙

<table header-row="true">
<colgroup>
<col width="340" />
<col width="500" />
<col width="480" />
</colgroup>
<tr>
<td>
사례
</td>
<td>
O/G/P/V 표기
</td>
<td>
판정
</td>
</tr>
<tr>
<td>
온전한 실선
</td>
<td>
보이는 경계로 O, G=0, 도막 P, 판독 영역 V
</td>
<td>
유효 비율·구간 근거가 충족되면 R 계산
</td>
</tr>
<tr>
<td>
정상 점선 공백
</td>
<td>
차선 종류·반복 패턴 근거를 기록하고 G 표시
</td>
<td>
G를 마모로 세지 않음. 단순히 검은 부분이라는 이유만으로 G 지정 금지
</td>
</tr>
<tr>
<td>
도막 안의 작은 벗겨짐·구멍
</td>
<td>
O는 유지, 빠진 구멍은 P=0
</td>
<td>
도막 마스크의 내부 구멍을 polygon 채움으로 메우지 않음
</td>
</tr>
<tr>
<td>
경계 일부만 남은 마모
</td>
<td>
남은 양쪽 경계/인접 프레임 등 근거로 O 복원
</td>
<td>
근거 프레임·복원 사유 저장. 외곽 불확실이면 held/불가
</td>
</tr>
<tr>
<td>
완전히 지워져 위치·폭/종류를 알 수 없음
</td>
<td>
O 미확정, outline\_supported=false
</td>
<td>
R=null·unassessable. 잔존율 0%로 강제하지 않음
</td>
</tr>
<tr>
<td>
차량 가림·렌즈 오염·심한 반사/그림자
</td>
<td>
영향 영역 V=0, 원인 태그
</td>
<td>
보이는 도막이 없다고 마모로 라벨하지 않음
</td>
</tr>
<tr>
<td>
그늘 속에서도 도막 경계를 분명히 볼 수 있음
</td>
<td>
실제 남은 P를 표기, V=1
</td>
<td>
밝기만으로 도막 삭제 금지. 재귀반사성능을 추정하지 않음
</td>
</tr>
<tr>
<td>
페인트 오염·낡은 선/새 선 중첩
</td>
<td>
판독 가능한 목표 선만 별도 ID, 모호하면 V=0/불가
</td>
<td>
도색 이력이 불명확하면 임의의 원래 선으로 마모를 계산하지 않음
</td>
</tr>
</table>

# 6. 잔존율·유효성·구간 합산

## 6.1 픽셀 집합과 계산

D = O에서 G를 뺀 도색 예정 영역, E = D와 V의 교집합으로 정의한다. D가 0이거나 외곽이 미확정이면 q와 R을 null로 두고 계산하지 않는다. 유효 면적 비율 q = area(E)/area(D)다. q가 0.70 이상일 때만 R = 100×area(P∩E)/area(E)를 계산한다. 가림을 제외한 <b>시각적</b> 잔존율임을 표시한다.

도막 P는 도색 예정 영역 안에 있어야 하고 범위 밖 흩어진 색은 차선 P에 포함하지 않는다. 원영상 픽셀을 단순 합산하지 않고 프로파일의 BEV 동일 격자에서 면적을 비교한다. 이 면적비가 법정 반사성능이나 실제 측량 면적을 뜻하지 않는다.

| R·조건 | grade | 판정 |
| --- | --- | --- |
| 80% 이상 | visible\_good | 시각상 양호 |
| 60% 이상 80% 미만 | monitor | 관찰 |
| 60% 미만 | worn\_candidate | 마모 후보 |
| 외곽 근거 없음/q\<0.70/구간 근거\<0.80/보정 불가 | unassessable | R=null, 사유 필수 |

등급은 반올림 전 값으로 결정한다. R=59.999는 worn\_candidate, 60은 monitor, 79.999는 monitor, 80은 visible\_good다. q=0.70와 segment\_support\_ratio=0.80은 경계값을 충족한다. 유효 비율과 등급 임계값은 서로 다른 기준이다.

## 6.2 고유 패치·구간

한 차선의 동일 노면 패치는 여러 프레임에서 같은 patch\_id로 연결한다. 대표 패치는 선명도·가림·정면성 근거로 고르고 좌우/반복 촬영의 같은 면적을 합산하지 않는다. 같은 패치가 train/test에 동시에 들어가지 않도록 DMP의 그룹에 연결한다.

기본 구간은 추정 주행거리 약 10 m다. 구간 R = 유효 고유 패치의 남은 도막 면적 합/그 패치 유효 예상 도색 면적 합×100이다. 임의의 프레임별 R 단순 평균은 쓰지 않는다.

segment\_support\_ratio는 해당 구간에서 <b>판정 가능한 고유 패치가 덮은 도색 예정 길이/대상 구간의 전체 도색 예정 길이</b>다. 점선의 의도적 공백은 양쪽에서 제외하고 겹친 길이는 한 번만 센다. 분모·거리·구간 근거가 불명확하면 segment\_support\_ratio=null·구간 판정 불가로 둔다. 0.80 미만이면 구간 R/grade는 null/unassessable이며 유효 패치 R은 근거로 별도 보존할 수 있다.

## 6.3 마스크 형식·검산 예시

내부 편집본은 polygon/이진 PNG로 관리할 수 있지만 마모 구멍·다중 조각이 손실 없이 복원되어야 한다. 교환 표준은 COCO 호환 RLE 또는 동일 해상도의 이진 마스크다. RLE의 size=\[H,W\], counts는 0영역부터 시작하는 열 우선 순서다. [COCO 공식 mask API](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/mask.py)

아래 4×4 예시는 실차 라벨이 아닌 면적 계산용 toy BEV다. O=16, G=0, V=16, P=8픽셀로 q=1, R=50%다. segmentation 포맷 예시와 실제 구간 길이/coverage는 분리한다.

```txt
O 및 V:       도막 P:       정상 공백 G:
1 1 1 1       1 1 0 0       0 0 0 0
1 1 1 1       1 1 0 0       0 0 0 0
1 1 1 1       1 1 0 0       0 0 0 0
1 1 1 1       1 1 0 0       0 0 0 0
예상 도색 16칸 · 유효 16칸 · 남은 도막 8칸 → 잔존율 50%
```

```json
{
  "mask_example": "toy_bev_4x4",
  "coord_space": "bev",
  "outline_rle": {
    "size": [
      4,
      4
    ],
    "counts": [
      0,
      16
    ]
  },
  "intentional_gap_rle": {
    "size": [
      4,
      4
    ],
    "counts": [
      16
    ]
  },
  "valid_region_rle": {
    "size": [
      4,
      4
    ],
    "counts": [
      0,
      16
    ]
  },
  "paint_rle": {
    "size": [
      4,
      4
    ],
    "counts": [
      0,
      8,
      8
    ]
  },
  "outline_supported": true,
  "valid_area_ratio": 1,
  "retention_pct": 50,
  "patch_grade": "worn_candidate"
}
```

학습용 도막 마스크는 0=배경, 1=도막, 255=ignore다. 예상 차선 외곽 라벨을 도막 클래스 정답으로 복사하지 않는다. PNG 색 팔레트·값·RLE 해독 결과와 면적 합을 자동 대조한다. COCO iscrowd를 모든 모호 영역의 범용 ignore 대용으로 임의 사용하지 않고 모델/평가기의 무효 영역 규칙을 명시한다.

# 7. 라벨 속성·저장 스키마

| 그룹 | 필수 속성 | 규칙 |
| --- | --- | --- |
| 샘플 | sample\_id,source\_sample\_id,run\_id,frame\_id,image\_size,coord\_space,captured\_at,profile\_id | 원영상·보정 버전과 연결, 해시·기한·분할은 manifest 참조 |
| 포트홀 | annotation\_id,physical\_object\_id,bbox,certainty,wet,occluded,truncated,evidence\_frame\_ids | 판정 불확실 사유 필수, 하나의 real object에 연결 |
| 차선 | lane\_track\_id,lane\_segment\_id,patch\_id,color,line\_type,O/G/P/V,outline\_supported,evidence\_frame\_ids | 다른 차선/서로 다른 노면을 같은 ID로 합치지 않음 |
| 패치 지표 | valid\_area\_ratio,retention\_pct,unassessable\_reason | q\<0.70 또는 외곽 불가 시 R=null |
| 구간 지표 | estimated\_length\_m,segment\_support\_ratio,retention\_pct,grade,patch\_ids | 길이 불명확은 null, 지원 부족이면 구간 R=null |
| 작업 이력 | annotator\_id,reviewer\_id,annotation\_version,status,change\_reason | 본인 단독 최종 승인 금지, QC·마스킹 승인 구분 |

annotation\_status는 draft/held/qc\_passed/rejected다. API의 review.status는 담당자의 점검 상태이므로 이 라벨 작업 상태를 전송하지 않는다. GT certainty와 모델 confidence, GT 실물 ID와 case\_id도 분리한다.

# 8. 개인정보 라벨·마스킹 작업

얼굴·번호판은 보이는 식별 영역 전체를 tight bbox로 표기하고 잘린 대상도 truncated로 기록한다. 학습/검증 표본은 사용 조건·동의·기한이 확인된 자료만 쓴다. 마스킹은 필요 여백을 추가해 가장자리·일부 반사/왜곡된 번호판·작은 얼굴도 확인한다. 공개 이미지에서 얼굴/번호판을 찾았다는 사실만으로 재배포 권한이 생기지는 않는다.

원본을 외부 라벨링 SaaS에 자동 업로드하지 않는다. 로컬 편집 또는 조건·권한이 승인된 환경에서 처리한다. 공유 직전 사람이 전량 확인하고 수동 가림으로 누락을 정정한다. 사진 변경 시 해시·승인을 갱신한다. 라벨 파일·화면 캡처·썸네일·메타데이터도 같은 확인 대상이다.

마스킹 모델 평가용 원본은 내부 7일 보관 안에 학습/시험을 완료한다. W7 시험용 얼굴·번호판 표본은 시험 직전에 동의받아 새로 확보하고 기한 내 평가·삭제한다. 장기 보관이 필요한 얼굴/번호판 원본을 마스킹 자료인 것처럼 90일 폴더에 넣지 않는다.

# 9. 검수·합의·불량 처리

## 9.1 파일럿과 전량 자동 검사

최초 50프레임 이상에 포트홀·차선 3군·정상 점선·그림자/맨홀·가림·개인정보 사례를 섞어 작업자 간 기준을 맞춘다. 합의한 gold 예시는 사용조건·기한을 적용하고 공개 문서에는 실제 식별 자료를 넣지 않는다.

모든 산출물에서 해시·필수 필드·버전·좌표/마스크 크기·폴리곤 면적·P/D 관계·R/grade 경계·q/support·ID 중복·분할 누수·마스킹 상태·만료를 검사한다. 오류 자료는 내보내기/릴리스에서 차단한다.

## 9.2 사람 검수

학습 배치는 n개 중 min(n,max(30,ceil(0.2n)))를 유형별 층화 표본으로 교차 검수한다. 검증/시험은 전량 2인 독립 표기·비교를 원칙으로 하고, 합의 전 held 상태를 유지한다. 초기 라벨 QC 목표는 클래스 일치≥95%, 박스 IoU≥0.80, 도막 IoU≥0.85, 외곽 IoU≥0.90, 잔존율 차이≤5%p다. 모델 합격 수치가 아니다.

개인정보 누락·양성 누락·틀린 좌표계·분할 누수는 중대 오류다. 발견 시 해당 배치 반려·유형 전수 재검수, 원인·정정·재검수자·영향 release를 기록한다. 일반 경계 오차는 정정·재표본 검수하며 목표 미달이면 전수 재검수한다.

라벨 불일치는 인접 프레임/우영상·프로파일로 다시 확인하고 최종 사유를 남긴다. 모델이 맞게 보인다는 이유로 GT를 바꾸지 않는다. 시험 후 사실 오류 정정은 A/B 독립 검토와 이전/수정 지표를 함께 보존한다.

# 10. 평가·서버 매핑

| 라벨 산출물 | 사용 지표·서버 필드 | 주의 |
| --- | --- | --- |
| 포트홀 bbox | 모델 mAP@0.5 | QC IoU 0.80 기준과 혼용하지 않음 |
| confirmed 실물 ID | 이벤트 Recall, 중복·FP/km | 실물≥30·연속≥5km, 단발/보류의 검토 부담 포함 |
| 도막 P와 유효 영역 | 도막 IoU≥0.75 | 배경 포함 mIoU로 바꾸지 않음, ignore 제외 정책 고정 |
| 구간 GT R·grade | R MAE≤10%p, 마모 Recall≥0.85/Precision≥0.75 | 실제 마모에서 모델 판정 불가이면 FN, 판정 가능 비율≥80%도 함께 보고 |
| O/G/P/V·패치 | lane.retention\_pct,grade,valid\_area\_ratio,outline\_supported,patch\_ids,segment\_support\_ratio | 원본 라벨과 추론 산출물 구분, 서버에는 승인된 비식별 근거만 전송 |

라벨 마스크와 GT ID가 API에 그대로 업로드되는 것은 아니다. API 관측은 실제 모델 출력·관측 위치·승인 사진을 사용한다. 학습 정답에 맞춘 confidence를 조작해 전송하지 않는다.

# 11. 릴리스 체크리스트·변경 이력

릴리스 전 포트홀 양성 누락/중복, 차선의 정상 공백·내부 구멍·완전 소실, 좌표계·BEV/마스크 버전, 판정 불가와 경계값, 실물/구간 ID·분할, 2인 합의·사용조건·기한·개인정보 승인을 확인한다. 세부 삭제·권한·출처·모델 lineage는 데이터 관리 계획서를 따른다.

- [COCO 공식 bbox/RLE 구현 설명](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/mask.py)
- [AI Hub 원자료·어노테이션](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71781&topMenu=100)
- <mention url="https://app.notion.com/p/40d8ac7c2dde818c91dff83bae17dcbf" />, <mention url="https://app.notion.com/p/40d8ac7c2dde81f3b7a4f3ae95edb804" />

| 버전 | 날짜 | 변경 |
| --- | --- | --- |
| v1.0 | <date start="2026-11-03" /> | 클래스·포트홀 박스·차선 O/G/P/V·잔존율/패치·개인정보·QC/서버 매핑 기준 신규 작성 |

연계 문서: <mention url="https://app.notion.com/p/40d8ac7c2dde818aa89ed3a2bc9b24a7" />, <mention url="https://app.notion.com/p/40d8ac7c2dde812ab381cd1b2d45bea2" />.
</content>
</page>
