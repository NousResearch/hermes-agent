---
sidebar_position: 23
title: "Microsoft Graph Webhook Listener"
description: "Receive Microsoft Graph change notifications (meetings, calendar, chat, etc.) in Hermes"
---

# Microsoft Graph 웹훅 리스너

`msgraph_webhook` 게이트웨이 플랫폼은 인바운드 이벤트 리스너입니다. 이는 Hermes가 Microsoft Graph로부터 **변경 알림(change notifications)**(예: "Teams 회의가 종료됨", "이 채팅에 새 메시지가 도착함", "이 캘린더 이벤트가 업데이트됨")을 받는 방법입니다. (사용자가 직접 타이핑하는 채팅 봇인 `teams` 플랫폼과는 다릅니다. 이 플랫폼은 사용자가 아닌 M365가 Hermes에게 어떤 일이 일어났다고 알려주는 역할을 합니다.)

현재 주요 사용처는 Teams 회의 요약 파이프라인(Teams meeting summary pipeline)입니다: Graph가 회의 스크립트 생성을 알리면 파이프라인이 이를 가져오고, Hermes가 요약 내용을 다시 Teams에 게시합니다. 다른 Graph 리소스(`/chats/.../messages`, `/users/.../events`)도 동일한 리스너를 사용합니다. 파이프라인 소비자(consumer)들은 각자의 PR을 통해 등록됩니다.

## 전제 조건

- Microsoft Graph 애플리케이션 자격 증명 — [Microsoft Graph 애플리케이션 등록](/guides/microsoft-graph-app-registration)
- Microsoft Graph가 연결할 수 있는 **공개 HTTPS URL** (Graph는 비공개 엔드포인트를 호출하지 않습니다). 테스트용으로는 dev tunnel이 적합하지만, 프로덕션용으로는 유효한 인증서가 있는 실제 도메인이 필요합니다.
- `clientState` 값으로 사용할 강력한 공유 비밀키. `openssl rand -hex 32`로 생성하고 `~/.hermes/.env`에 `MSGRAPH_WEBHOOK_CLIENT_STATE`로 추가하세요.

## 빠른 시작

최소 `~/.hermes/config.yaml` 설정:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8646
      client_state: "replace-with-a-strong-secret"
      accepted_resources:
        - "communications/onlineMeetings"
```

또는 `~/.hermes/.env`에 환경 변수를 통해 설정할 수도 있습니다 (시작 시 자동 병합됨):

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

참고: 바인딩 호스트는 `config.yaml`의 `extra.host`에서 읽어옵니다 (위 예제 참조). 이를 재정의할 수 있는 `MSGRAPH_WEBHOOK_HOST` 환경 변수는 없습니다.

게이트웨이 시작: `hermes gateway run`. 리스너는 다음을 노출합니다:

- `POST /msgraph/webhook` — Graph로부터의 변경 알림
- `GET /msgraph/webhook?validationToken=...` — Graph 구독 유효성 검사 핸드셰이크
- `GET /health` — 허용됨/중복된 수(accepted/duplicate) 카운터가 포함된 준비성(readiness) 프로브

리스너를 공개적으로 노출(리버스 프록시, dev tunnel, ingress)하세요. Graph 구독을 위한 알림 URL은 공개 HTTPS origin 뒤에 `/msgraph/webhook`이 붙은 형태입니다:

```
https://ops.example.com/msgraph/webhook
```

## 구성

모든 설정은 `platforms.msgraph_webhook.extra` 아래에 위치합니다:

| 설정 | 기본값 | 설명 |
|---------|---------|-------------|
| `host` | `0.0.0.0` | HTTP 리스너의 바인딩 주소. 루프백(loopback)이 아닌 바인딩의 경우 `allowed_source_cidrs`가 필요합니다. 루프백(`127.0.0.1` / `::1`)은 dev-tunnel / 리버스 프록시 설정에 가장 쉽습니다. |
| `port` | `8646` | 바인드 포트. |
| `webhook_path` | `/msgraph/webhook` | Graph가 POST 요청을 보내는 URL 경로. |
| `health_path` | `/health` | 준비성(Readiness) 엔드포인트. |
| `client_state` | — | Graph가 모든 알림에 에코(echo)하는 공유 비밀키. `hmac.compare_digest`와 비교됩니다 — `openssl rand -hex 32`로 생성하세요. |
| `accepted_resources` | `[]` (모두 허용) | Graph 리소스 경로/패턴의 허용 목록(allowlist). 후행 `*`는 접두사 일치(prefix match)로 작동합니다. 선행 `/`는 허용됩니다. 예: `["communications/onlineMeetings", "chats/*/messages"]`. |
| `max_seen_receipts` | `5000` | 알림 ID에 대한 중복 제거 캐시 크기. 상한에 도달하면 가장 오래된 항목이 제거됩니다. |
| `allowed_source_cidrs` | `[]` | 루프백이 아닌 바인딩에 필요합니다. 리스너가 루프백에 바인딩되고 로컬 터널 / 리버스 프록시가 앞단에 있는 경우에만 비워 두세요. |

각 설정에는 게이트웨이 시작 시 구성에 병합되는 해당하는 환경 변수(`MSGRAPH_WEBHOOK_*`)도 있습니다 — [환경 변수 참조](/reference/environment-variables#microsoft-graph-teams-meetings)를 참고하세요.

## 보안 강화 (Security Hardening)

### clientState는 기본 인증 검사입니다

모든 Graph 알림에는 구독 시 등록한 `clientState` 문자열이 포함됩니다. 리스너는 타이밍에 안전한(timing-safe) 비교를 사용하여 `clientState`가 일치하지 않는 알림을 모두 거부합니다. 이것은 Microsoft에서 문서화한 메커니즘입니다 — 이 값을 강력한 공유 비밀키로 취급하세요.

`client_state`가 설정되지 않으면 리스너는 시작을 거부합니다.

### 소스 IP 허용 목록 설정 (프로덕션 배포)

프로덕션의 경우, 리스너를 Microsoft가 게시한 Graph 웹훅 소스 IP 범위로 제한하세요. Microsoft는 [Office 365 IP 주소 및 URL 웹 서비스](https://learn.microsoft.com/ko-kr/microsoft-365/enterprise/urls-and-ip-address-ranges) 아래에 송신(egress) 범위를 문서화합니다. 이를 다음과 같이 구성하세요:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 0.0.0.0
      client_state: "..."
      allowed_source_cidrs:
        - "52.96.0.0/14"
        - "52.104.0.0/14"
        # ...현재 Microsoft 365 "Common" + "Teams" 범주 송신 범위를 추가하세요.
```

또는 환경 변수로 설정:

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

`allowed_source_cidrs` 없이 `0.0.0.0`, `::` 또는 LAN IP와 같은 비루프백 호스트를 바인딩하면 시작 시 거부됩니다. 동일한 시스템에서 dev tunnel이나 리버스 프록시를 사용하는 경우 Hermes를 `127.0.0.1` 또는 `::1`에 바인딩하고 거기에서는 허용 목록을 비워 두세요. 유효하지 않은 CIDR 문자열은 경고를 기록하고 무시됩니다. **Microsoft IP 목록은 변경되므로 분기별로 검토하세요**.

### HTTPS 종료(termination)

리스너는 일반 HTTP 통신을 합니다. 리버스 프록시(Caddy, Nginx, Cloudflare Tunnel, AWS ALB)에서 TLS를 종료(terminate)하고 로컬 네트워크를 통해 리스너로 프록시하세요. Graph는 HTTPS가 아닌 엔드포인트로의 전달을 거부하므로 Graph 자체에서 암호화되지 않은 트래픽이 사용자에게 도달할 경로는 없습니다.

### 응답 위생(hygiene)

성공 시 리스너는 본문이 없는 `202 Accepted`를 반환합니다. — 내부 카운터는 유선 응답(wire response)에 포함되지 않습니다. 운영자는 웹훅 경로와 동일한 소스 IP 규칙의 보호를 받는 `/health`를 통해 카운트를 관찰할 수 있습니다.

상태 코드 표:

| 결과 | 상태 |
|---------|--------|
| 알림이 허용되거나 중복 제거됨 | 202 |
| 유효성 검사 핸드셰이크 (`validationToken`을 포함한 GET) | 200 (토큰을 에코함) |
| 배치의 모든 항목이 clientState 실패 | 403 |
| 잘못된 JSON / `value` 배열 누락 / 알 수 없는 리소스 | 400 |
| 소스 IP가 허용 목록에 없음 | 403 |
| `validationToken` 없는 일반 GET | 400 |

## 문제 해결

| 문제 | 확인 사항 |
|---------|---------------|
| Graph 구독 유효성 검사 실패 | 공개 URL에 접속할 수 있는지, `/msgraph/webhook` 경로가 일치하는지, `validationToken`을 사용한 GET 요청이 10초 이내에 토큰을 그대로 `text/plain`으로 에코하는지 확인하세요. |
| 알림은 POST되지만 수집(ingest)되는 것이 없음 | `client_state`가 구독을 등록할 때 사용한 값과 일치하는지 확인하세요. 값이 변경된 경우 `openssl rand -hex 32`를 다시 실행하고 새 구독을 만드세요. `accepted_resources`에 Graph가 보내는 리소스 경로가 포함되어 있는지 확인하세요. |
| 모든 알림이 403 반환 | `clientState` 불일치 (위조되었거나 다른 값으로 등록된 구독). `hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...` (파이프라인 런타임 PR과 함께 제공됨)을 사용하여 구독을 다시 만드세요. |
| 리스너가 `0.0.0.0`에서 시작을 거부함 | `allowed_source_cidrs`를 현재 Microsoft의 웹훅 송신 범위로 설정하거나 터널이나 리버스 프록시 뒤의 `127.0.0.1` / `::1`에 Hermes를 바인딩하세요. |
| 리스너는 시작되지만 `curl http://localhost:8646/health`가 지연(hang)됨 | 포트 바인딩 충돌. `ss -tlnp \| grep 8646`을 확인하고 필요하면 `port:`를 변경하세요. |
| Microsoft에서 오는 실제 Graph 요청이 403으로 차단됨 | 소스 IP 허용 목록이 너무 좁습니다. 목록을 확장하여 현재 Microsoft 송신 범위를 포함하세요. 아직 터널 경로를 검증 중인 경우 Hermes를 루프백에 바인딩하고 터널이 공용 노출을 처리하도록 하세요. |

## 관련 문서

- [Microsoft Graph 애플리케이션 등록](/guides/microsoft-graph-app-registration) — Azure 앱 등록 전제 조건
- [환경 변수 → Microsoft Graph](/reference/environment-variables#microsoft-graph-teams-meetings) — 전체 환경 변수 목록
- [Microsoft Teams 봇 설정](/user-guide/messaging/teams) — 사용자가 Teams에서 Hermes와 채팅할 수 있게 해주는 다른 플랫폼
