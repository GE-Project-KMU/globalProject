# SVO/SVO2 기록 재생 어댑터

`agent/recorded_zed.py`의 `RecordedZedReader`는 기존 `equipment.ZedSDK` 실시간
연결과 분리된 선택 기능이다. 원본은 읽기만 하며, 영상 저장·외부 업로드·학습은
수행하지 않는다. SDK가 없는 상태를 성공으로 해석하지 않는다.

## 현재 실제 실행 제약

공식 Windows 문서는 SDK 설치 프로그램과 CUDA 설정을 안내하며 환경 변수
갱신을 위해 재부팅도 설명한다. 프로젝트 내부 추출만으로 실행하는 공식 지원
portable 절차는 이번 조사에서 확인하지 못했다. 설치 프로그램 실행, 전역
디렉터리 접근·변경, 드라이버/레지스트리 변경은 수행하지 않았다.
[공식 Windows 설치 문서](https://docs.stereolabs.com/docs/development/zed-sdk/windows).

Python `pyzed` 래퍼만 준비한다고 native SDK 의존성이 대체되지는 않는다.
현재 프로젝트의 두 환경에는 pyzed가 없으므로 **실제 SDK 재생은 미검증/차단**이다.
프로젝트 내부에 설치된 의존성이 있더라도 SDK 내부의 파일 쓰기와 네트워크를
Python 경로 검사만으로 샌드박스할 수 없다. 별도의 범위 검증 또는 권한 정책
변경 없이 프로젝트 내부 실행을 보장했다고 주장하지 않는다.

또한 파일 확장자가 `.svo2`여도 실제 컨테이너는 별도로 확인해야 한다. SDK
어댑터의 준비 상태는 일반 컨테이너의 2D 디코딩 가능성과 별개다.

## 입력과 보존 계약

SVO2는 SDK4.1에서 도입되어 기본 기록 형식이 되었다. SVO에는 재생용 serial과
원시 영상·센서 관련 정보가 있지만 보정 파라미터는 담기지 않는다. 재생 장비의
settings에 해당 serial의 calibration이 필요하고, 없으면 SDK가 자동 취득을
시도할 수 있다.
[공식 기록 형식 설명](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording).

따라서 reader는 신뢰할 수 있는 원본 메타데이터에서 확인한 `expected_serial`과
프로젝트 내부의 비어 있지 않은 `SN<serial>.conf`를 먼저 요구한다. 임의 serial을
추측하거나 보정 파일을 위조하면 안 된다. `optional_settings_path`에는 끝 경로
구분자를 붙인다. SDK 로그도 명시한 프로젝트 작업 폴더로 지정한다.
[공식 offline 설정 경로 설명](https://docs.stereolabs.com/docs/development/zed-sdk/offline-deployment),
[끝 구분자 관련 공식 릴리스 수정 이력](https://github.com/stereolabs/zed-sdk/releases).

이 선검사는 SDK 내부 I/O 보장이 아니다. 잘못된 expected serial이면 SDK open
단계에서 다른 보정 파일 자동 다운로드를 시도할 여지가 있으며, 실제 SDK 환경을
감사하지 않은 상태에서 reader를 실행하면 안 된다. open 후 실제 serial이 다르면
즉시 오류로 종료한다.

`read()`는 좌우 rectified BGR 배열의 독립 복사본, 원본 SDK IMAGE timestamp의
정수 nanoseconds, SDK 원본 frame position을 반환한다. 이 시각을 현재 시각으로
바꾸거나 검증되지 않은 UTC epoch로 해석하지 않는다. `utc`와 `gnss`는 None이다.
영상 순서로 GPS/주행 속도/이동 거리/10m 구간을 만들어내지 않는다.

기본 depth는 NONE이다. 선택한 경우에만 고전 ULTRA를 사용하고 neural 모드나
추적/객체/바디 AI를 활성화하지 않는다. depth 단위는 meter, 무효 값은 NaN이며
실물 정확도나 물리적 포트홀 심각도 검증을 의미하지 않는다. 끝은
END_OF_SVOFILE_REACHED로만 판정한다. grab/이미지/depth 오류는 EOF로 숨기지
않고 구분해 발생시키며 카메라를 정리한다. 감소한 timestamp도 오류다.

## 범위가 검증된 실제 SDK 준비 후 사용

```powershell
.\.venv\Scripts\python.exe scripts/probe_recorded_zed.py 데이터/REPLACE.svo2 --settings data/agent/sdk-settings --work-dir data/agent/sdk-probe --serial REPLACE_VERIFIED_SERIAL --max-frames 3 --output data/agent/sdk-probe/result.json
.\.venv\Scripts\python.exe -m pytest tests/test_recorded_zed.py -q
```

첫 명령은 실행 예시이며 현재 성공한 명령이 아니다. 실제 보정 파일·검증된
serial·프로젝트 범위를 만족하는 SDK 실행 환경이 필요하다. probe는 최대100개의
프레임만 읽고 영상 배열을 저장하지 않는다. 원본 파일과 같은 폴더를 작업
폴더로 사용하는 것도 거부한다. 결과의 serial/timestamp/frame 정보와 SDK 로그는
사용자 영상에 연결되는 비공개 메타데이터이며 원본의 보관 계획에 따라 관리한다.
테스트의 가짜 영상/SDK는 인터페이스·실패 처리 증거이며 실제 디코딩 증거가 아니다.
