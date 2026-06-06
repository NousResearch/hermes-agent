# ntfy

[ntfy](https://ntfy.sh/)는 간단한 HTTP 기반 pub-sub 알림 서비스입니다. 무료 공개 서버인 `ntfy.sh` 또는 자체 호스팅(self-hosted) 인스턴스와 작동하며, 휴대폰, 브라우저, 스크립트, 스마트워치 등 HTTP 요청을 할 수 있는 모든 클라이언트를 지원합니다.

ntfy는 Hermes를 위한 훌륭하고 가벼운 푸시 채널이 될 수 있습니다. [ntfy 모바일 앱](https://ntfy.sh/docs/subscribe/phone/)에서 주제(topic)를 구독하고, 주제에 메시지를 보내 에이전트와 대화하며, 휴대폰에서 답변을 푸시 알림으로 받아보세요.

> `hermes gateway setup`을 실행하고 플랫폼에서 **ntfy**를 선택하여 가이드를 따르세요.

## 전제 조건

- 주제 이름(고유한 문자열이면 무엇이든 가능 — `hermes-myname-2026`과 같은 형식)
- 해당 주제를 구독하고 [ntfy 모바일 앱](https://ntfy.sh/docs/subscribe/phone/)을 설치한 기기
- 선택 사항: 자체 호스팅 ntfy 서버, 또는 프라이빗/예약 주제를 위한 `ntfy.sh` 계정 토큰

이게 전부입니다. SDK, 데몬, Node.js는 필요하지 않습니다. 어댑터는 이미 Hermes 종속성에 포함된 `httpx`를 사용합니다.

## Hermes 구성

### 설정 마법사를 통한 설정

```bash
hermes setup gateway
```

**ntfy**를 선택하고 프롬프트를 따릅니다.

### 환경 변수를 통한 설정

`~/.hermes/.env`에 다음을 추가합니다:

```
NTFY_TOPIC=hermes-myname-2026
NTFY_ALLOWED_USERS=hermes-myname-2026
NTFY_HOME_CHANNEL=hermes-myname-2026
```

| 변수 | 필수 여부 | 설명 |
|---|---|---|
| `NTFY_TOPIC` | 예 | 구독할 주제 (수신 메시지) |
| `NTFY_SERVER_URL` | 선택 | 서버 URL (기본값: `https://ntfy.sh`) — 개인 정보를 위해 자체 호스팅 ntfy를 가리킬 수 있습니다 |
| `NTFY_TOKEN` | 선택 | Bearer 토큰 (예: `tk_xyz`) 또는 기본 인증(Basic auth)을 위한 `user:pass` |
| `NTFY_PUBLISH_TOPIC` | 선택 | 발신 응답을 위한 다른 주제 (기본적으로 `NTFY_TOPIC`과 동일) |
| `NTFY_MARKDOWN` | 선택 | `X-Markdown: true` 헤더와 함께 응답을 보내려면 `true`로 설정 |
| `NTFY_ALLOWED_USERS` | 권장 | 허용되는 쉼표로 구분된 주제 이름 (사용자 ID로 취급됨; 아래 참조) |
| `NTFY_ALLOW_ALL_USERS` | 선택 | 모든 퍼블리셔(publisher)를 허용하려면 `true`로 설정 — 읽기 토큰이 있는 프라이빗 주제에서만 안전 |
| `NTFY_HOME_CHANNEL` | 선택 | cron / 알림 전송을 위한 기본 주제 |
| `NTFY_HOME_CHANNEL_NAME` | 선택 | 홈 채널을 식별하기 위한 사람이 읽을 수 있는 이름 |

## 자격 증명 모델 (Identity model) — 배포 전 필독

ntfy에는 네이티브 인증 사용자 ID(authenticated user identity)가 없습니다. 퍼블리시된 메시지의 `title` 필드는 **발신자가 제어**하며 발신자가 원하는 대로 설정할 수 있습니다. Hermes 어댑터는 이 `title`을 권한 부여(authorization) 용도로 사용하지 **않습니다**. (주제를 아는 다른 발신자가 허용된 사용자를 스푸핑할 수 있기 때문입니다.)

대신, **주제 이름 자체가 ID(신원)가 됩니다**. 해당 주제에 게시된 모든 메시지는 동일한 논리적 사용자(주제)로부터 온 것으로 간주됩니다. 따라서 `NTFY_ALLOWED_USERS`는 일반적으로 주제 이름 자체입니다 — 즉, 채널 전체를 통제하는 단일 항목 허용 목록(allowlist)입니다.

이는 **주제를 아는 누구나 에이전트와 대화할 수 있음**을 의미합니다. 이를 진정한 신뢰 경계(trust boundary)로 만들기 위한 방법은 다음과 같습니다:

- **ntfy 자체 호스팅** 및 [접근 제어(Access Control)](https://docs.ntfy.sh/config/#access-control)로 주제를 잠급니다. 읽기/쓰기 토큰이 있는 승인된 클라이언트만 게시할 수 있습니다.
- 또는 **ntfy.sh의 프라이빗 주제**를 사용하고 ([예약된 주제(reserved topics)](https://docs.ntfy.sh/publish/#reserved-topics)에는 계정이 필요) `NTFY_TOKEN`으로 보호합니다.
- 또는 **길고 추측하기 어려운 주제 이름**을 선택하고 (`hermes-7d4f9c8b-2026`) 이를 공유 비밀키처럼 취급합니다. 이 방식이 가장 가볍지만 주제 이름이 로그나 스크린샷을 통해 유출될 수 있습니다.

모든 경우에 있어서 기본 주제가 접근 제어(access-controlled)되지 않는 한, 민감한 데이터를 ntfy를 통해 보내지 마십시오.

## 빠른 시작 — 휴대폰으로 에이전트와 대화하기

1. 주제 이름 선택: `hermes-myname-2026`
2. 휴대폰에서: [ntfy 앱](https://ntfy.sh/docs/subscribe/phone/)을 설치하고 **+**를 탭한 뒤 `hermes-myname-2026`을 입력합니다.
3. 호스트에서:
   ```bash
   echo 'NTFY_TOPIC=hermes-myname-2026' >> ~/.hermes/.env
   echo 'NTFY_ALLOWED_USERS=hermes-myname-2026' >> ~/.hermes/.env
   hermes gateway restart
   ```
4. ntfy 앱에서 해당 주제로 메시지를 보냅니다. 에이전트의 응답이 푸시 알림으로 도착합니다.

## cron 작업에 ntfy 사용하기

`NTFY_HOME_CHANNEL`이 설정되면, cron 작업은 ntfy로 메시지를 전송할 수 있습니다:

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="ntfy",          # NTFY_HOME_CHANNEL 사용
    prompt="Check for alerts and summarise."
)
```

또는 특정 주제를 명시적으로 타겟팅할 수 있습니다:

```python
send_message(target="ntfy:alerts-channel", message="Done!")
```

이는 cron이 게이트웨이와 별도의 프로세스(out-of-process)로 실행될 때도 작동합니다 — 플러그인은 자체 HTTP 연결을 여는 `standalone_sender_fn`을 등록합니다.

## ntfy 자체 호스팅

완전한 제어를 원할 경우:

```bash
# Docker
docker run -p 80:80 -it binwiederhier/ntfy serve

# Native
go install heckel.io/ntfy/v2@latest
ntfy serve
```

그런 다음 Hermes를 해당 서버로 가리킵니다:

```
NTFY_SERVER_URL=https://ntfy.mydomain.com
NTFY_TOPIC=hermes
NTFY_TOKEN=tk_abc123  # 접근 제어를 설정한 경우
```

자체 호스팅은 주제에 대한 접근 제어, 메시지 지속성 정책, 첨부 파일, 그리고 이모지 태그를 제공합니다. [ntfy 서버 문서](https://docs.ntfy.sh/install/)를 참고하세요.

## 마크다운 포맷팅

퍼블리셔가 `X-Markdown: true` 헤더를 설정하면 ntfy 클라이언트는 마크다운을 렌더링합니다. 발신 Hermes 응답에 이를 활성화하려면 다음을 설정하세요:

```
NTFY_MARKDOWN=true
```

또는 `config.yaml`에:

```yaml
platforms:
  ntfy:
    extra:
      markdown: true
```

모바일 앱은 굵게, 기울임꼴, 목록, 링크, 코드 블록(fenced code block) 등 CommonMark의 하위 집합을 지원합니다. 정확한 집합은 [ntfy 마크다운 문서](https://docs.ntfy.sh/publish/#markdown-formatting)를 참고하세요.

## 단방향 알림 전용 설정 (수신 없이 발신만)

Hermes가 ntfy에 알림(cron 요약, 알림 등)을 푸시(push)하기만 하고 메시지를 수신하지 않도록 하려면, `NTFY_TOPIC`과 `NTFY_PUBLISH_TOPIC`을 같은 값으로 설정하고 `NTFY_ALLOWED_USERS`를 아예 비워두세요. 허용 목록(allowlist)이 없으면 에이전트는 인바운드 메시지에 응답하지 않습니다. 휴대폰으로 푸시는 받지만, 대화는 단방향이 됩니다.

## 제한 사항

- **메시지 크기**: ntfy는 메시지 본문을 4096자로 제한합니다. 이 한도를 초과하면 Hermes는 경고와 함께 메시지를 잘라냅니다.
- **타이핑 표시 없음(No typing indicators)**: 프로토콜이 이를 노출하지 않으므로 `send_typing`은 아무 동작도 하지 않습니다.
- **스레드나 첨부 파일 없음**: ntfy는 단순한 푸시 알림 서비스입니다. 긴 응답은 스레드로 나뉘지 않고 메시지 본문 안에 남아 있습니다.
- **네이티브 사용자 식별(ID) 없음**: 위의 자격 증명 모델(identity-model) 섹션을 참조하세요.

## 문제 해결

**인증 실패 / 401** — `NTFY_TOKEN`이 잘못되었거나 해당 토큰이 이 주제에 대해 게시(publish)/구독(subscribe) 권한이 없습니다. 어댑터는 401 오류 시 재연결 루프를 중지하며, 게이트웨이 런타임 상태는 `fatal: ntfy_unauthorized`로 표시됩니다. 토큰을 수정하고 게이트웨이를 다시 시작하세요.

**주제를 찾을 수 없음 / 404** — 설정된 서버에 `NTFY_TOPIC`이 존재하지 않습니다. ntfy.sh의 경우, 처음 게시할 때 주제가 자동으로 생성되므로 404는 주제가 미리 프로비저닝되지 않은 자체 호스팅 서버를 가리키고 있음을 의미합니다. 어댑터는 `fatal: ntfy_topic_not_found`와 함께 재연결 루프를 중지합니다.

**연결되었지만 메시지가 없음** — `NTFY_ALLOWED_USERS`에 주제 이름 자체가 포함되어 있는지 확인하세요. ntfy의 ID 모델에서는 주제가 곧 사용자입니다. 허용 목록을 비워두면 모든 메시지가 거부됩니다.

**60초마다 재연결됨** — 스트림 유지(keepalive) 기본값은 55초입니다. ntfy 네트워크에 간헐적인 문제가 있을 수 있습니다. 어댑터는 지수 백오프(2 → 5 → 10 → 30 → 60초)를 적용하며 스트림이 60초 이상 유지되면 0으로 초기화됩니다.
