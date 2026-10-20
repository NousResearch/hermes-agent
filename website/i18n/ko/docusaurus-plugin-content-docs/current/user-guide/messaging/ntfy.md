# ntfy

[ntfy](https://ntfy.sh/)는 간단한 HTTP 기반 퍼블리시-구독 알림 서비스입니다. 무료 공개 서버인 `ntfy.sh` 또는 자체 호스팅 인스턴스에서 작동하며, HTTP 요청을 보낼 수 있는 모든 클라이언트(휴대폰, 브라우저, 스크립트, 시계)를 지원합니다.

ntfy는 Hermes를 위한 가볍고 훌륭한 푸시 채널입니다. [ntfy 모바일 앱](https://ntfy.sh/docs/subscribe/phone/)에서 토픽을 구독하고, 토픽으로 메시지를 보내 에이전트와 대화하며, 휴대폰으로 응답을 받을 수 있습니다.

> `hermes gateway setup`을 실행하고 **ntfy**를 선택하면 안내에 따라 설정할 수 있습니다.

## 사전 요구 사항

- 토픽 이름(고유한 문자열이면 무엇이든 가능 — `hermes-myname-2026`도 충분함)
- [ntfy 모바일 앱](https://ntfy.sh/docs/subscribe/phone/)을 설치하고 해당 토픽을 구독
- 선택 사항: 자체 호스팅 ntfy 서버 또는 비공개/예약 토픽용 `ntfy.sh` 계정 토큰

이것으로 충분합니다. SDK, 데몬, Node.js는 필요하지 않습니다. 어댑터는 이미 Hermes의 종속성으로 포함된 `httpx`를 사용합니다.

## Hermes 설정

### 설정 마법사 사용

```bash
hermes gateway setup
```

**ntfy**를 선택하고 안내에 따릅니다.

### 환경 변수 사용

다음 항목을 `~/.hermes/.env`에 추가합니다.

```
NTFY_TOPIC=hermes-myname-2026
NTFY_ALLOWED_USERS=hermes-myname-2026
NTFY_HOME_CHANNEL=hermes-myname-2026
```

| 변수 | 필수 여부 | 설명 |
|---|---|---|
| `NTFY_TOPIC` | 예 | 구독할 토픽(수신 메시지) |
| `NTFY_SERVER_URL` | 선택 사항 | 서버 URL(기본값: `https://ntfy.sh`) — 개인정보 보호를 위해 자체 호스팅 ntfy를 지정 |
| `NTFY_TOKEN` | 선택 사항 | Bearer 토큰(예: `tk_xyz`) 또는 Basic 인증용 `user:pass` |
| `NTFY_PUBLISH_TOPIC` | 선택 사항 | 발신 응답에 사용할 다른 토픽(기본값은 `NTFY_TOPIC`) |
| `NTFY_MARKDOWN` | 선택 사항 | `true`로 설정하면 `X-Markdown: true` 헤더와 함께 응답 전송 |
| `NTFY_ALLOWED_USERS` | 권장 | 허용할 토픽 이름을 쉼표로 구분(사용자 ID로 취급; 아래 참조) |
| `NTFY_ALLOW_ALL_USERS` | 선택 사항 | `true`로 설정하면 모든 게시자를 허용 — 읽기 토큰이 있는 비공개 토픽에서만 안전 |
| `NTFY_HOME_CHANNEL` | 선택 사항 | cron/알림 전달에 사용할 기본 토픽 |
| `NTFY_HOME_CHANNEL_NAME` | 선택 사항 | 홈 채널의 사람이 읽을 수 있는 레이블 |

## 배포 전에 반드시 읽어야 할 ID 모델

ntfy에는 기본 제공 인증 사용자 ID가 없습니다. 게시된 메시지의 `title` 필드는 게시자가 제어하며, 발신자가 원하는 어떤 값이든 될 수 있습니다. Hermes 어댑터는 인증에 `title`을 사용하지 않습니다 — 토픽을 아는 모든 게시자가 허용된 사용자를 사칭할 수 있기 때문입니다.

대신 **토픽 이름 자체가 ID**입니다. 토픽에 게시된 모든 메시지는 동일한 논리적 사용자(토픽)가 보낸 것으로 취급됩니다. 따라서 `NTFY_ALLOWED_USERS`는 일반적으로 토픽 이름 자체이며, 전체 채널을 제어하는 단일 항목 허용 목록입니다.

이는 **토픽을 아는 누구나 에이전트와 대화할 수 있음**을 의미합니다. 이를 실제 신뢰 경계로 만들려면 다음 방법을 사용하세요.

- **ntfy를 자체 호스팅**하고 [액세스 제어](https://docs.ntfy.sh/config/#access-control)로 토픽을 잠급니다. 읽기/쓰기 토큰이 있는 승인된 클라이언트만 게시할 수 있습니다.
- 또는 **ntfy.sh에서 비공개 토픽**을 사용하고([예약 토픽](https://docs.ntfy.sh/publish/#reserved-topics)은 계정 필요) `NTFY_TOKEN`으로 보호합니다.
- 또는 **길고 추측하기 어려운 토픽 이름**(`hermes-7d4f9c8b-2026`)을 선택하고 공유 비밀로 취급합니다. 가장 간단한 설정이지만 로그나 스크린샷을 통해 토픽 이름이 유출됩니다.

어떤 경우에도 기반 토픽이 액세스 제어되지 않는다면 ntfy를 통해 민감한 데이터를 보내지 마세요.

## 빠른 시작 — 휴대폰으로 에이전트와 대화하기

1. 토픽 이름을 정합니다: `hermes-myname-2026`
2. 휴대폰에서 [ntfy 앱](https://ntfy.sh/docs/subscribe/phone/)을 설치하고 **+**를 누른 다음 `hermes-myname-2026`을 입력합니다.
3. 호스트에서:
   ```bash
   echo 'NTFY_TOPIC=hermes-myname-2026' >> ~/.hermes/.env
   echo 'NTFY_ALLOWED_USERS=hermes-myname-2026' >> ~/.hermes/.env
   hermes gateway restart
   ```
4. ntfy 앱에서 토픽으로 메시지를 보냅니다. 에이전트의 응답이 푸시 알림으로 도착합니다.

## cron 작업에서 ntfy 사용

`NTFY_HOME_CHANNEL`을 설정하면 cron 작업이 ntfy로 전달할 수 있습니다.

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="ntfy",          # uses NTFY_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

cron 작업의 `deliver:` 필드로 특정 토픽을 명시적으로 지정하거나, [`hermes send` CLI](/guides/pipe-script-output)를 사용하는 셸 스크립트에서 지정할 수도 있습니다.

```bash
hermes send ntfy:alerts-channel "Done!"
```

이 기능은 cron이 게이트웨이와 별도의 프로세스로 실행될 때도 작동합니다 — 플러그인이 자체 HTTP 연결을 여는 `standalone_sender_fn`을 등록하기 때문입니다.

## ntfy 자체 호스팅

완전한 제어가 필요하다면 다음과 같이 설정합니다.

```bash
# Docker
docker run -p 80:80 -it binwiederhier/ntfy serve

# Native
go install heckel.io/ntfy/v2@latest
ntfy serve
```

그런 다음 Hermes가 해당 서버를 사용하도록 지정합니다.

```
NTFY_SERVER_URL=https://ntfy.mydomain.com
NTFY_TOPIC=hermes
NTFY_TOKEN=tk_abc123  # if you've set up access control
```

자체 호스팅을 사용하면 토픽 액세스 제어, 메시지 보존 정책, 첨부 파일 및 이모지 태그를 이용할 수 있습니다. [ntfy 서버 문서](https://docs.ntfy.sh/install/)를 참조하세요.

## Markdown 서식

게시자가 `X-Markdown: true` 헤더를 설정하면 ntfy 클라이언트가 Markdown을 렌더링합니다. 발신되는 Hermes 응답에 사용하려면 다음과 같이 활성화합니다.

```
NTFY_MARKDOWN=true
```

또는 `config.yaml`에서 다음과 같이 설정합니다.

```yaml
platforms:
  ntfy:
    extra:
      markdown: true
```

모바일 앱은 CommonMark의 일부 기능인 굵게, 기울임, 목록, 링크, 펜스 코드 블록을 지원합니다. 정확한 지원 목록은 [ntfy의 Markdown 문서](https://docs.ntfy.sh/publish/#markdown-formatting)를 참조하세요.

## 발신 전용 설정(수신 없이 알림만)

Hermes가 ntfy로 알림(cron 요약, 경고)만 *푸시*하고 메시지는 다시 받지 않게 하려면 `NTFY_TOPIC`과 `NTFY_PUBLISH_TOPIC`을 같은 값으로 설정하고 `NTFY_ALLOWED_USERS`는 완전히 생략합니다. 허용 목록이 없으면 에이전트는 수신 메시지에 절대 응답하지 않습니다 — 휴대폰은 푸시를 받지만 대화는 단방향입니다.

## 제한 사항

- **메시지 크기**: ntfy는 메시지 본문을 4096자로 제한합니다. 이 제한을 초과하면 Hermes가 경고와 함께 잘라냅니다.
- **입력 중 표시 없음**: 프로토콜에서 이를 제공하지 않으므로 `send_typing`은 아무 작업도 하지 않습니다.
- **스레드 또는 첨부 파일 없음**: ntfy는 일반 푸시 알림입니다. 긴 응답도 메시지 본문에 유지되며 스레드로 분산되지 않습니다.
- **기본 제공 사용자 ID 없음**: 위의 ID 모델 섹션을 참조하세요.

## 문제 해결

**인증 실패 / 401** — `NTFY_TOKEN`이 잘못되었거나 토픽에 대한 게시/구독 권한이 토큰에 없습니다. 어댑터는 401에서 재연결 루프를 중단하며 게이트웨이 런타임 상태에 `fatal: ntfy_unauthorized`가 표시됩니다. 토큰을 수정하고 게이트웨이를 다시 시작하세요.

**토픽을 찾을 수 없음 / 404** — 구성된 서버에 `NTFY_TOPIC`이 존재하지 않습니다. ntfy.sh에서는 최초 게시 시 토픽이 자동으로 생성되므로, 404는 토픽이 준비되지 않은 자체 호스팅 서버를 가리키고 있다는 뜻입니다. 어댑터는 `fatal: ntfy_topic_not_found`와 함께 재연결 루프를 중단합니다.

**연결되었지만 메시지가 없음** — `NTFY_ALLOWED_USERS`에 토픽 이름 자체가 포함되어 있는지 확인하세요. ntfy의 ID 모델에서는 토픽이 곧 사용자이므로 허용 목록을 비워 두면 모든 메시지가 거부됩니다.

**60초마다 재연결** — 스트림의 keepalive 기본값은 55초이며 ntfy에 간헐적인 네트워크 문제가 있을 수 있습니다. 어댑터는 지수 백오프(2 → 5 → 10 → 30 → 60초)를 적용하고 스트림이 60초 이상 유지되면 0으로 재설정합니다.
