# SimpleX Chat

[SimpleX Chat](https://simplex.chat/)는 사용자가 자신의 연락처와 그룹을 소유하는 비공개 분산형 메시징 플랫폼입니다. 다른 플랫폼과 달리 SimpleX는 영구적인 사용자 ID를 할당하지 않습니다. 각 연락처는 연결 시 생성되는 불투명한 내부 ID로 식별되므로, 가장 높은 수준의 개인정보 보호를 제공하는 메신저 중 하나입니다.

> `hermes gateway setup`을 실행하고 **SimpleX**를 선택하면 안내에 따라 설정할 수 있습니다.

## 사전 요구 사항

- 설치되어 데몬으로 실행 중인 **simplex-chat** CLI
- Python 패키지 **websockets** (`pip install websockets`)

## simplex-chat 설치

[simplex-chat GitHub 릴리스](https://github.com/simplex-chat/simplex-chat/releases) 페이지에서 최신 릴리스를 다운로드하세요.

```bash
# Linux / macOS binary
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86_64 -o simplex-chat
chmod +x simplex-chat
```

SimpleX Chat 프로젝트는 채팅 클라이언트용 사전 빌드 Docker 이미지를 제공하지 않습니다. Docker에서 실행하려면 [simplex-chat 저장소](https://github.com/simplex-chat/simplex-chat)에서 소스 코드를 빌드하세요.

## 데몬 시작

```bash
simplex-chat -p 5225
```

데몬은 기본적으로 WebSocket에서 `ws://127.0.0.1:5225`를 수신합니다.

## Hermes 구성

### 설정 마법사 사용

```bash
hermes gateway setup
```

**SimpleX Chat**을 선택하고 안내에 따르세요.

### 환경 변수 사용

다음 항목을 `~/.hermes/.env`에 추가하세요.

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| 변수 | 필수 여부 | 설명 |
|---|---|---|
| `SIMPLEX_WS_URL` | 예 | simplex-chat 데몬의 WebSocket URL |
| `SIMPLEX_ALLOWED_USERS` | 권장 | 쉼표로 구분한 허용 목록. 각 항목은 숫자 `contactId` **또는** 표시 이름일 수 있으며, 두 형식 모두 작동합니다. |
| `SIMPLEX_ALLOW_ALL_USERS` | 선택 사항 | 모든 연락처를 허용하려면 `true`로 설정하세요 (주의해서 사용). |
| `SIMPLEX_AUTO_ACCEPT` | 선택 사항 | 수신 연락처 요청을 자동으로 수락합니다 (기본값: `true`). |
| `SIMPLEX_GROUP_ALLOWED` | 선택 사항 | 봇이 참여하는 그룹 ID를 쉼표로 구분해 지정하거나, 모든 그룹에 허용하려면 `*`를 사용하세요. 생략하면 그룹 메시지를 모두 무시합니다. |
| `SIMPLEX_HOME_CHANNEL` | 선택 사항 | cron 작업을 전달할 기본 연락처/그룹 ID |
| `SIMPLEX_HOME_CHANNEL_NAME` | 선택 사항 | 홈 채널의 사람이 읽을 수 있는 레이블 |
| `HERMES_SIMPLEX_TEXT_BATCH_DELAY` | 선택 사항 | 빠르게 연속으로 들어오는 텍스트 메시지를 하나의 이벤트로 합칠 때 사용하는 정숙 기간(초, 기본값: `0.8`) |

## 연락처 ID 또는 표시 이름 찾기

데몬을 시작한 후 에이전트 연락처와 대화를 여세요. 숫자 `contactId`는 세션 로그에 표시됩니다. SimpleX UI에 표시되는 이름을 사용하고 싶다면 그렇게 해도 됩니다 — `SIMPLEX_ALLOWED_USERS`는 두 형식 모두 허용합니다.

## 권한 부여

기본적으로 **모든 연락처가 거부됩니다**. 다음 중 하나를 반드시 수행해야 합니다.

1. `SIMPLEX_ALLOWED_USERS`를 `contactId` 및/또는 표시 이름의 쉼표 구분 목록으로 설정합니다 (예: `SIMPLEX_ALLOWED_USERS=4,alice`는 `contactId`가 4인 연락처 또는 표시 이름이 "alice"인 연락처와 일치합니다).
2. **DM 페어링**을 사용합니다 — 봇에 아무 메시지나 보내면 봇이 페어링 코드를 답장합니다. `hermes pairing approve simplex <CODE>`로 해당 코드를 입력하세요.

## 그룹 채팅

기본적으로 어댑터는 그룹 메시지를 무시합니다 — 그렇지 않으면 그룹에 있는 봇이 모든 구성원의 메시지를 처리하게 됩니다. 명시적으로 선택해야 합니다.

```
SIMPLEX_GROUP_ALLOWED=12,34          # specific group IDs
# or
SIMPLEX_GROUP_ALLOWED=*              # any group the bot is in
```

채팅 ID 앞에 `group:`을 붙여 그룹을 지정합니다. 예를 들어 cron의 `deliver=` 대상이나 `hermes send` 호출에서는 `simplex:group:12`처럼 사용합니다.

## `hermes send`로 전송

SimpleX는 독립적인 전송 대상으로 작동합니다 — 데몬은 실행 중이어야 하지만, 일반 텍스트를 보낼 때 실행 중인 gateway는 필요하지 않습니다.

```bash
hermes send --to simplex:alice "hello"          # DM by contact display name
hermes send --to simplex:group:12 "hello"       # group by numeric ID
hermes send --to simplex "hello"                # SIMPLEX_HOME_CHANNEL
```

gateway가 실행 중이면 어댑터가 연락처와 허용된 그룹을 채널 디렉터리에 열거합니다(5분마다 새로 고침). 따라서 `hermes send --list`에서 이름으로 확인할 수 있습니다. gateway를 처음 실행하기 전에는 플랫폼이 `--list`에 "아직 검색된 채널이 없음"이라는 힌트와 함께 표시되지만, 위와 같은 직접 대상은 언제나 작동합니다.

## 첨부 파일

어댑터는 양방향의 기본 SimpleX 첨부 파일을 지원합니다.

- **수신** — 데몬의 XFTP 흐름(`rcvFileDescrReady` → `/freceive` → `rcvFileComplete` 대기)을 통해 수신한 이미지, 음성 메모, 파일을 허용하며, 적절한 `MessageType`(`PHOTO`, `VOICE`, `TEXT` + 문서)와 함께 `MessageEvent.media_urls`로 표시합니다.
- **발신** — `send_image_file`, `send_voice`, `send_document`, `send_video`는 모두 `filePath`를 포함한 구조화된 `/_send` 형식을 사용하므로, 수신하는 SimpleX 클라이언트가 이미지를 인라인으로 렌더링하고 음성 메모를 다운로드 항목이 아니라 인라인으로 재생합니다.

에이전트의 답변은 일반 텍스트에 `MEDIA:/path/to/file` 태그를 포함할 수도 있습니다 — 어댑터가 본문에서 태그를 제거하고 파일 확장자에 따라 음성 메모(오디오 확장자) 또는 문서로 파일을 보냅니다.

## cron 작업에서 SimpleX 사용

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # uses SIMPLEX_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

또는 cron 작업의 `deliver:` 필드에서 특정 연락처를 지정하거나, [hermes send CLI](/guides/pipe-script-output)를 사용하는 셸 스크립트에서 지정할 수 있습니다.

```bash
hermes send simplex:<contact-id> "Done!"
```

## 개인정보 보호 참고 사항

- SimpleX는 전화번호나 이메일 주소를 절대 노출하지 않습니다 — 연락처는 불투명한 ID를 사용합니다.
- Hermes와 데몬의 연결은 로컬 WebSocket(`ws://127.0.0.1:5225`)입니다 — 어떤 데이터도 컴퓨터 밖으로 나가지 않습니다.
- 메시지는 데몬에 도달하기 전에 SimpleX 프로토콜에 의해 종단 간 암호화됩니다.

## 문제 해결

**"Cannot reach daemon"** — `simplex-chat -p 5225`가 실행 중이고 포트가 `SIMPLEX_WS_URL`과 일치하는지 확인하세요.

**"websockets not installed"** — `pip install websockets`를 실행하세요.

**메시지가 수신되지 않음** — 연락처의 ID가 `SIMPLEX_ALLOWED_USERS`에 있는지 확인하거나 DM 페어링으로 승인하세요.
