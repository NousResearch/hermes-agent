# IRC

IRC 어댑터는 Hermes를 모든 IRC 서버에 연결하고 IRC 채널(또는 다이렉트 메시지)과 에이전트 사이에서 메시지를 전달합니다. Python 표준 라이브러리의 `asyncio`를 통해 IRC 프로토콜을 사용하므로 **외부 종속성, SDK, 데몬이 필요하지 않습니다**. [Libera.Chat](https://libera.chat/) 같은 공개 네트워크와 자체 호스팅 ircd에서 모두 작동합니다.

IRC는 일반 텍스트 방식입니다. 음성, 이미지, 파일, 스레드, 리액션, 입력 중 표시, 스트리밍은 지원되지 않으며, 긴 메시지는 IRC 라인 제한에 맞도록 나누어 `PRIVMSG` 라인으로 전송됩니다.

> `hermes gateway setup`을 실행하고 안내에 따라 **IRC**를 선택하세요.

## 사전 요구 사항

- 연결할 IRC 서버(예: `irc.libera.chat`)
- 참여할 채널(예: `#hermes`) — 여러 채널에 참여하려면 쉼표로 구분
- 봇의 닉네임(기본값: `hermes-bot`)
- 선택 사항: 네트워크에서 식별을 요구하는 경우 등록된 닉네임과 NickServ 비밀번호

## Hermes 구성

IRC는 두 가지 방법으로 구성할 수 있습니다. 빠른 환경 변수만을 사용하는 설정 또는 `~/.hermes/gateway-config.yaml`의 `gateway` 블록을 사용할 수 있습니다.

### 옵션 A — gateway-config.yaml

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: irc.libera.chat
        port: 6697
        nickname: hermes-bot
        channel: "#hermes"
        use_tls: true
        server_password: ""       # optional server password
        nickserv_password: ""     # optional NickServ identification
        allowed_users: []         # empty = allow all, or list of nicks
        max_message_length: 450   # IRC line limit (safe default)
```

### 옵션 B — 환경 변수

| 변수 | 필수 | 설명 |
|----------|:--------:|-------------|
| `IRC_SERVER` | ✅ | IRC 서버 호스트 이름(예: `irc.libera.chat`) |
| `IRC_CHANNEL` | ✅ | 참여할 채널 — 여러 채널은 쉼표로 구분 |
| `IRC_NICKNAME` | ✅ | 봇 닉네임(기본값: `hermes-bot`) |
| `IRC_PORT` | — | 서버 포트(기본값: TLS 사용 시 `6697`, 미사용 시 `6667`) |
| `IRC_USE_TLS` | — | TLS 사용 여부(`true`/`false`; 포트 6697에서는 기본값 `true`) |
| `IRC_SERVER_PASSWORD` | — | `PASS` 명령에 사용할 서버 비밀번호 |
| `IRC_NICKSERV_PASSWORD` | — | 연결 시 자동 IDENTIFY에 사용할 NickServ 비밀번호 |
| `IRC_ALLOWED_USERS` | — | 봇과 대화할 수 있는 닉네임 목록(쉼표로 구분) |
| `IRC_ALLOW_ALL_USERS` | — | 채널의 누구나 봇과 대화하도록 허용(개발 전용) |
| `IRC_HOME_CHANNEL` | — | cron/알림을 전달할 채널(기본값은 `IRC_CHANNEL`) |

## 접근 제어

기본적으로 `allowed_users`(또는 `IRC_ALLOWED_USERS`)에 나열된 닉네임만 봇과 대화할 수 있습니다. 목록을 비워 두고 **`IRC_ALLOW_ALL_USERS=true`로 설정하면** 채널의 누구나 Hermes와 대화할 수 있습니다. 테스트에는 유용하지만, 네트워크에서 NickServ를 강제하지 않는 한 IRC 닉네임은 인증되지 않으므로 공개 네트워크에서는 권장하지 않습니다.

네트워크가 닉네임을 등록하는 경우 `IRC_NICKSERV_PASSWORD`(또는 `nickserv_password`)를 설정하세요. 그러면 봇이 연결 시 NickServ에 식별하고 등록된 닉네임을 유지합니다.

## 채널과 DM

- 참여한 채널의 메시지는 **그룹** 대화로 처리됩니다.
- 봇에게 보내는 비공개 메시지는 **다이렉트 메시지**로 처리됩니다.

Cron 작업과 알림은 **홈 채널**로 전달됩니다. `IRC_HOME_CHANNEL`이 설정되어 있으면 해당 채널을 사용하고, 그렇지 않으면 `IRC_CHANNEL`의 첫 번째 채널을 사용합니다.

## 게이트웨이 실행

```bash
hermes gateway start
```

`hermes gateway status`로 상태를 확인하세요. 환경 변수만 사용하는 설정을 포함하여 IRC 연결 상태가 표시됩니다.

## 참고

- 긴 에이전트 응답은 IRC 라인 제한(`max_message_length`, 프로토콜 오버헤드 이후 기본 450바이트) 안에 들어가도록 여러 `PRIVMSG` 라인으로 자동 분할됩니다.
- 어댑터는 서버+닉네임별 범위가 지정된 자격 증명 잠금을 획득하므로 두 Hermes 프로필이 동일한 IRC ID를 두고 충돌하지 않습니다.
