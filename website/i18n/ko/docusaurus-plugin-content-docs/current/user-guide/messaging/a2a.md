# A2A (에이전트 간 통신)

[A2A](https://a2a-protocol.org)는 독립적인 AI 에이전트 간 통신을 위한 공개 Agent2Agent 프로토콜(v1.0, Linux Foundation 관리)입니다. Hermes A2A 플러그인은 **양방향**으로 작동합니다. 에이전트가 다른 A2A 에이전트를 도구로 호출할 수 있고, 다른 에이전트가 HTTP를 통해 Hermes에 작업을 보낼 수도 있습니다.

다른 Hermes, LangChain, CrewAI, Google ADK 에이전트 또는 공식 `a2a-sdk`로 구축된 무엇이든 A2A 호환 피어라면 상호 운용할 수 있습니다.

## A2A를 사용해야 하는 경우

- **머신 간 Hermes ↔ Hermes** — 데스크톱 에이전트가 서버의 Hermes에 작업을 맡기거나 그 반대로 할 수 있습니다. 각 에이전트는 자체 메모리, 도구, 자격 증명을 사용합니다.
- **전문 에이전트에 위임** — Agent Card에서 `web_search`/`research`/`coding` 스킬을 제공한다고 알리는 피어를 대화 중에 검색하고 호출할 수 있습니다.
- **호출 가능한 서비스로 사용** — 다른 프레임워크의 에이전트가 작업을 보낼 수 있도록 Hermes를 노출합니다.

같은 머신에서 여러 에이전트를 사용하려면 [위임](../features/delegation.md)(프로세스 내 서브에이전트) 또는 [칸반 보드](../features/kanban.md)(영구적인 다중 프로필 작업 큐)를 우선 사용하세요. A2A는 프로세스/머신/프레임워크 경계를 넘을 때 사용합니다.

## 활성화

```bash
hermes gateway setup      # pick A2A
```

또는 `~/.hermes/config.yaml`에서 설정합니다.

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900
```

아웃바운드 클라이언트 도구는 `a2a` 도구 세트로 제공되며 **기본적으로 꺼져 있습니다**. 플랫폼별로 활성화하세요.

```bash
hermes tools enable a2a --platform cli        # CLI/TUI sessions
hermes tools enable a2a --platform telegram   # or any messaging platform
hermes tools enable a2a --platform a2a        # let inbound A2A tasks call peers (agent chaining)
```

인바운드 플랫폼을 활성화하지 않아도 CLI, TUI, 게이트웨이, cron 등 모든 프로세스 유형에서 도구를 사용할 수 있습니다.

## 아웃바운드: 다른 에이전트 호출

`a2a` 도구 세트를 활성화하면 에이전트가 다음 도구를 사용할 수 있습니다.

| 도구 | 기능 |
|---|---|
| `a2a_discover(url)` | 피어의 Agent Card를 가져와 요약합니다. |
| `a2a_call(agent, message, context_id?)` | 작업을 보내고 답변을 받습니다. `context_id`를 사용하면 여러 턴으로 이어갈 수 있습니다. |
| `a2a_list()` | 설정된 피어, 저장된 대화, 메트릭을 표시합니다. |
| `a2a_history(context_id)` | 저장된 A2A 대화를 불러옵니다. |
| `a2a_orchestrate(capability, message, mode?)` | 해당 기능을 제공한다고 알린 모든 피어에 작업을 분산합니다(`all` / `first` / `best`). |

`config.yaml`에서 알려진 피어를 설정합니다.

```yaml
a2a_agents:
  researcher:
    url: "http://research-box.local:9900"
    auth: { type: bearer, token: "..." }
    timeout: 120
    capabilities: [web_search, research]
```

그런 다음 이렇게 요청하면 됩니다. *"researcher 에이전트에게 오늘의 arXiv 게시물을 요약해 달라고 요청해."* 직접 URL도 사용할 수 있습니다. `a2a_call`은 모든 A2A 엔드포인트를 허용합니다.

## 인바운드: 호출 가능한 에이전트로 사용

플랫폼을 활성화하면 Hermes가 다음을 제공합니다.

- `GET /.well-known/agent-card.json`의 **Agent Card**(표준 v1.0 경로이며 레거시 `agent.json`에도 응답) — 에이전트의 이름, 스킬(활성화된 도구 세트에서 파생), 인증 요구 사항을 알립니다.
- `POST /`의 **JSON-RPC 2.0** — 표준 v1.0 메서드(`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, 푸시 알림 설정 CRUD)와 v1.0 이전의 경로 스타일 별칭(`message/send`, …)을 지원합니다.
- `SendStreamingMessage`를 위한 **SSE 스트리밍** — 사양에 맞는 JSON-RPC 래퍼 프레임을 사용합니다.
- 장시간 실행되는 작업을 위한 **푸시 알림**(웹훅) — HMAC-SHA256으로 서명합니다.

인바운드 작업은 **실행 중인 게이트웨이 세션**에 주입됩니다. 즉, 다른 채널을 제공하는 동일한 에이전트, 메모리, 도구를 사용하며 최종 답변은 작업 결과로 호출자에게 반환됩니다. 대화는 A2A `contextId`를 키로 사용하므로 피어가 여러 턴의 대화를 이어갈 수 있습니다.

공식 Python `a2a-sdk`를 사용해 상호 운용성을 검증했습니다(카드 확인, `SendMessage`, 스트리밍).

## 보안 모델

기본적으로 안전하며, 범위를 넓히는 각 단계는 명시적으로 설정해야 합니다.

- **토큰이 없으면 localhost만 허용됩니다.** 서버는 `127.0.0.1`에 바인딩됩니다. 원격으로 노출하려면 bearer token **및** 명시적인 `A2A_HOST`가 필요합니다.
- **피어별 토큰** — `A2A_PEER_TOKENS="alice:tok1,bob:tok2"`를 사용하면 각 피어에 자체 자격 증명을 부여할 수 있습니다. 인증된 이름이 속도 제한, 신뢰, 감사에 사용됩니다.
- **프롬프트 인젝션 필터링** — 인바운드 텍스트를 필터링하고 신뢰할 수 없는 피어 입력으로 표시합니다. 원격 피어는 운영자 슬래시 명령을 호출할 수 없습니다.
- **아웃바운드 정보 삭제** — 자격 증명 형태의 문자열(API 키, JWT, 토큰)을 답변에서 제거합니다.
- **감사 로그** — 모든 교환 내용을 `~/.hermes/a2a_audit.jsonl`에 추가합니다.
- **루프 방지** — 컨텍스트별 턴 제한으로 두 에이전트가 영원히 핑퐁하는 것을 막습니다.

## 설정 참고

| 환경 변수 | 기본값 | 의미 |
|---|---|---|
| `A2A_PEER_TOKENS` | `_(unset)_` | 피어별 자격 증명 `name:token,…`(권장) |
| `A2A_BEARER_TOKEN` | `_(unset)_` | 공유 토큰이며, ID는 호출자 IP로 대체됩니다. |
| `A2A_HOST` | `127.0.0.1` | 바인딩할 호스트이며, 토큰이 설정된 경우에만 범위를 넓힐 수 있습니다. |
| `A2A_PORT` | `9900` | 인바운드 포트 |
| `A2A_AGENT_NAME` | 호스트 이름에서 파생 | Agent Card에 표시할 이름 |
| `A2A_PUBLIC_URL` | `_(unset)_` | 카드에 알릴 라우팅 가능한 URL(리버스 프록시 / k8s) |
| `A2A_TRUSTED_PEERS` | `_(unset)_` | 인증된 ID의 허용 목록 |
| `A2A_ALLOW_ALL_USERS` | `false` | 인증된 모든 피어 허용(개발 전용) |
| `A2A_RATE_LIMIT` | `60` | ID별 분당 요청 수 |
| `A2A_MAX_PINGPONG_TURNS` | `5` | 컨텍스트별 루프 방지 턴 제한(최대 20) |
| `A2A_REPLY_TIMEOUT` | `300` | 에이전트의 답변을 기다리는 시간(초) |
| `A2A_PUSH_SECRET` | bearer token | 푸시 알림 서명을 위한 HMAC 시크릿 |
| `A2A_ADVERTISED_TOOLSETS` | 등록된 모든 도구 세트 | Agent Card에 표시할 스킬 제한 |

리버스 프록시 또는 Kubernetes Service 뒤에서 실행하는 경우 `A2A_PUBLIC_URL`을 설정하세요(또는 `X-Forwarded-Host`/`X-Forwarded-Proto`에 의존할 수 있습니다). 그래야 Agent Card가 피어가 실제로 다시 호출할 수 있는 URL을 알립니다.

## 빠른 테스트

```bash
# From another machine / agent:
curl http://your-host:9900/.well-known/agent-card.json

curl -X POST http://your-host:9900/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
       "params":{"message":{"messageId":"m1","role":"ROLE_USER",
                 "parts":[{"text":"What tools do you have?"}]}}}'
```

## 문제 해결

- **피어가 카드 URL에 연결할 수 없음** — 카드에 바인딩 주소가 표시되고 있습니다. 외부에서 라우팅 가능한 URL로 `A2A_PUBLIC_URL`을 설정하세요.
- **`401 Unauthorized`** — 토큰이 일치하지 않습니다. 서버의 `A2A_PEER_TOKENS`/`A2A_BEARER_TOKEN`과 피어의 `auth:` 블록을 확인하세요.
- **서버가 localhost가 아닌 주소에 바인딩되지 않음** — 의도된 동작입니다. 먼저 bearer token을 설정한 다음 `A2A_HOST=0.0.0.0`으로 설정하세요.
- **장시간 작업에서 답변 시간 초과** — `A2A_REPLY_TIMEOUT`을 늘리거나 호출자가 푸시 알림 설정을 등록한 후 `GetTask`를 폴링하도록 하세요.
