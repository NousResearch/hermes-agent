# SimpleX Chat

[SimpleX Chat](https://simplex.chat/)는 사용자가 자신의 연락처와 그룹을 소유하는 개인 정보 보호 중심의 탈중앙화 메시징 플랫폼입니다. 다른 플랫폼과 달리 SimpleX는 영구적인 사용자 ID를 할당하지 않으며, 모든 연락처는 연결 시 생성되는 불투명한 내부 ID로 식별되므로 사용 가능한 메신저 중 가장 프라이버시가 강력한 메신저 중 하나입니다.

> `hermes gateway setup`을 실행하고 **SimpleX**를 선택하면 안내에 따라 설정할 수 있습니다.

## 사전 요구 사항

- **simplex-chat** CLI가 설치되어 데몬으로 실행 중이어야 합니다.
- Python 패키지 **websockets** (`pip install websockets`)

## simplex-chat 설치

[simplex-chat GitHub 릴리스](https://github.com/simplex-chat/simplex-chat/releases) 페이지에서 최신 릴리스를 다운로드합니다:

```bash
# Linux / macOS 바이너리
curl -L https://github.com/simplex-chat/simplex-chat/releases/latest/download/simplex-chat-ubuntu-22_04-x86_64 -o simplex-chat
chmod +x simplex-chat
```

SimpleX Chat 프로젝트는 채팅 클라이언트를 위한 사전 빌드된 Docker 이미지를 게시하지 않습니다. Docker 환경에서 실행하려면 [simplex-chat 저장소](https://github.com/simplex-chat/simplex-chat)에서 소스를 빌드하세요.

## 데몬 시작

```bash
simplex-chat -p 5225
```

데몬은 기본적으로 `ws://127.0.0.1:5225`에서 WebSocket을 수신 대기합니다.

## Hermes 구성

### 설정 마법사 사용

```bash
hermes setup gateway
```

**SimpleX Chat**을 선택하고 프롬프트의 지시를 따릅니다.

### 환경 변수 사용

`~/.hermes/.env` 파일에 다음 내용을 추가합니다:

```
SIMPLEX_WS_URL=ws://127.0.0.1:5225
SIMPLEX_ALLOWED_USERS=<contact-id-1>,<contact-id-2>
SIMPLEX_HOME_CHANNEL=<contact-id>
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `SIMPLEX_WS_URL` | Yes | simplex-chat 데몬의 WebSocket URL |
| `SIMPLEX_ALLOWED_USERS` | 권장 | 에이전트 사용이 허용된 연락처 ID를 쉼표로 구분하여 기재 |
| `SIMPLEX_ALLOW_ALL_USERS` | 선택 사항 | `true`로 설정하면 모든 연락처 허용 (주의해서 사용) |
| `SIMPLEX_HOME_CHANNEL` | 선택 사항 | cron 작업 결과를 전송할 기본 연락처 ID |
| `SIMPLEX_HOME_CHANNEL_NAME` | 선택 사항 | 홈 채널을 식별하기 위한 사람이 읽을 수 있는 라벨 |

## 연락처 ID 찾기

데몬을 시작한 후, 에이전트 연락처와 대화를 엽니다. 연락처 ID는 세션 로그에 나타나거나 `hermes send_message action=list`를 통해 확인할 수 있습니다.

## 권한 부여

기본적으로 **모든 연락처는 차단됩니다**. 두 가지 방법 중 하나를 선택해야 합니다:

1. `SIMPLEX_ALLOWED_USERS`에 쉼표로 구분된 연락처 ID 목록 설정
2. **DM 페어링** 사용 — 봇에게 메시지를 보내면 봇이 페어링 코드로 응답합니다. `hermes gateway pair` 명령어를 통해 해당 코드를 입력하세요.

## SimpleX를 cron 작업과 함께 사용

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="simplex",          # SIMPLEX_HOME_CHANNEL 사용
    prompt="경고를 확인하고 요약합니다."
)
```

또는 특정 연락처를 대상으로 지정할 수 있습니다:

```python
send_message(target="simplex:<contact-id>", message="Done!")
```

## 개인 정보 보호 참고 사항

- SimpleX는 전화번호나 이메일 주소를 절대 노출하지 않습니다 — 연락처는 불투명한 식별자(ID)를 사용합니다.
- Hermes와 데몬 간의 연결은 로컬 WebSocket(`ws://127.0.0.1:5225`)입니다 — 데이터가 내 컴퓨터를 벗어나지 않습니다.
- 메시지는 데몬에 도달하기 전에 SimpleX 프로토콜에 의해 종단간 암호화(end-to-end encrypted)됩니다.

## 문제 해결

**"Cannot reach daemon"** — `simplex-chat -p 5225`가 실행 중인지, 포트가 `SIMPLEX_WS_URL`과 일치하는지 확인하세요.

**"websockets not installed"** — `pip install websockets`를 실행하세요.

**메시지를 받지 못함** — 연락처 ID가 `SIMPLEX_ALLOWED_USERS`에 있는지 확인하거나 DM 페어링을 통해 승인하세요.
