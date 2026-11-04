# 실험용 MCAP 2D 재생

`agent.recorded_mcap.RecordedMcapReader`는 SDK가 없는 환경에서, 이번에 관측한
특정 MCAP 영상 메시지를 PyAV로 메모리 디코딩한다. 공식 범용 SVO/SVO2
디코더가 아니며 `RecordedZedReader` SDK 어댑터와 분리되어 있다. SDK 설치,
calibration 다운로드, neural depth, 서버 업로드, 자동 학습을 수행하지 않는다.

## 지원 범위

이번에 읽은 파일의 MCAP library는 `libmcap stereolabs`, header의
`ZED_SDK_version`은 `5.5.0`, header version은 `2.0.5`였다. video channel과
schema는 `zed_sdk_encoded`이고 topic은 `Camera_SN<digits>/side_by_side`다.
다른 버전의 전용 메시지가 같은 구조라고 일반화하지 않는다.

관측된 video message는 little-endian uint32 두 개 뒤에 Annex-B H264가 있고,
끝에 의미를 해석하지 않는 56바이트가 있었다. 첫 길이는 전체 메시지 길이-4,
두 번째 길이는 전체 길이-64와 일치해야 한다. 시작 코드와 첫 NAL header도
검사한다. 이 전용 envelope는 관측 기반 계약이며 제조사 공개 규격이 아니다.
불일치는 다른 decoder로 조용히 우회하지 않고 오류로 처리한다.

MCAP은 [공식 규격](https://mcap.dev/spec)에 따라 순차적으로 읽는다. chunk
압축 NONE/LZ4/ZSTD와 존재하는 chunk CRC, 끝까지 읽었을 때 DataEnd의 CRC를
검사한다. CRC가 0이면 검증 값이 제공되지 않은 것이며 인증을 의미하지 않는다.
일부 프레임만 읽으면 전체 파일 무결성을 확인했다고 주장하지 않는다.
[공식 Python reader 설명](https://mcap.dev/docs/python/_modules/mcap/reader).
압축 전후 record/chunk는 최대64MiB, video packet은16MiB로 제한한다.

## 시각과 영상 계약

각 메시지의 원본 `log_time`을 `av.Packet.pts/dts`에 그대로 넣고 time_base를
1/1,000,000,000으로 명시한다. 디코딩한 frame PTS가 아직 대응되지 않은 원본
메시지와 정확히 연결되어야 한다. 이는 SDK IMAGE clock 또는 파일에 저장된
H264 presentation clock을 검증한 것이 아니다. publish_time도 별도로 보존한다.
FPS나 현재 시각을 이용해 UTC/GPS를 만들지 않는다.
[PyAV time_base 설명](https://pyav.org/docs/stable/api/time.html).

PyAV의 EOF flush는 일부 지연 프레임의 pts를 유지하면서 time_base를 None으로
반환한다. 이때에만, 반환된 정수 pts가 원래 ns 단위로 제출한 pending 메시지와
정확히 일치하면 해당 원본 메시지의 시각을 보존한다. raw time_base=None은
결과에도 그대로 남기고 대응 근거를 `timestamp_match_basis`에 기록한다.
누락 pts, 일치하지 않는 값, 중복 대응은 오류다. 임의 fps로 채우지 않는다.

좌우 분리는 side_by_side topic과 짝수 영상 폭만을 근거로 한다. 출력은 각각
독립된 BGR uint8 배열이며 rectification은 unknown이다. metric depth, GNSS,
UTC는 None이다. 주행 경로·10m 거리·마모율 GT를 영상 시각만으로 추정하지 않는다.
frame corruption, decode 오류, CRC 오류, 길이 오류는 정상 EOF가 아니다.

```python
from agent.recorded_mcap import RecordedMcapReader

# 원본 처리 허용 기간이 확인된 경우에만 실행한다.
with RecordedMcapReader("데이터/REPLACE.svo2") as reader:
    for sample in reader:
        left, right = sample["left"], sample["right"]
        original_log_ns = sample["mcap_log_time_ns"]
        original_publish_ns = sample["mcap_publish_time_ns"]
        # 프레임을 로컬 분석 함수에 전달한다. 자동 저장/전송은 없다.
        break
```

`read()`도 지원하며 정상 EOF는 None, 닫힌 reader나 오류는 예외다. reader의
`metadata`는 관측 지원 프로파일·디코더 버전·제약을 제공하고 `integrity`는 읽은
범위의 CRC 상태와 `fully_consumed`를 분리한다. decoder 지연 대응은 최대32개다.
이미지는 파일로 내보내지 않는다.

## 검증 결과와 재현

합성 비개인 H264/MCAP 계약 테스트23개, SDK 가짜 인터페이스16개, 기존 비전
61개를 합친100개가 통과했다. 합성 결과는 실제 도로 모델 정확도 증거가 아니다.
추가한 인계·메타데이터·재생 테스트를 포함한 전체 Python 회귀 검사는
498개 및 하위 검사189개 통과, 실제 symlink 생성1개는 Windows 계정 권한으로
건너뛰었다. junction 탐지와 경로 이탈 차단은 별도 테스트했고, 의존성 검사도
통과했다. 이 회귀 검사는 제공받은 원본 영상을 추가로 디코딩하지 않는다.
실파일16개의 첫10프레임씩160개에서 PTS/ns 대응과 좌우 디코딩을 확인했다.
좌측 해상도는1920×1080인6파일,1280×720인10파일이었다. 원본16개 전체 SHA256은
인계 inventory와 일치했다. 실제 EOF flush나 전체 구간 디코딩을 검증한 것은 아니다.
SDK IMAGE time·보정·물리 거리·GPS와 교차검증한 것도 아니다.

최종 소스의 동일 prefix 재검증은 기간 대기 지시를 수신하기 전에 종료되었다.
이후 원본 읽기와 픽셀 처리를 중지했다. 촬영 시계는 외부 검증되지 않았고 사용자의
날짜 추정은 검증된 epoch가 아니다. 별도 허용 기간 확인 없이 추가 원본 처리,
기한 연장, 자동 삭제나 retention 등록을 수행하지 않는다.

```powershell
# 합성 테스트는 개인 원본을 읽지 않는다.
.\.venv\Scripts\python.exe -m pytest tests/test_recorded_mcap.py tests/test_recorded_zed.py tests/test_vision.py -q

# 추가 원본 처리가 허용된 기간에만 수행한다. 기본 한도는 파일당 첫10프레임이다.
.\.venv\Scripts\python.exe scripts/probe_recorded_mcap.py 데이터 --all-files --max-frames 10 --source-inventory data/PRIVATE-INVENTORY.json --output data/agent/mcap-private/probe.json
```

`mcap==1.6.0`, `lz4==4.4.5`, `zstandard==0.25.0`, 프로젝트의 PyAV19.0.2를 사용한다.
실제 확인 범위는 현재 Windows 로컬 환경이다. 다른 OS/라이브러리 버전에서의
지원은 미검증이다. 공개 집계는 `docs/reports/RECORDED-MCAP-BOUNDED-DECODE.json`,
개별 시각·원본 해시·경로·메타데이터는 `data/recorded-drive-intake`의 비공개
probe 결과에 남긴다. 원본 촬영 자료를 Git이나 서버로 전송하지 않는다.
