---
sidebar_position: 23
title: "Microsoft Graph 웹훅 리스너"
description: "Microsoft Graph 변경 알림(회의, 캘린더, 채팅 등)을 Hermes에서 수신합니다"
---

# Microsoft Graph 웹훅 리스너

`msgraph_webhook` 게이트웨이 플랫폼은 인바운드 이벤트 리스너입니다. Hermes가 Microsoft Graph에서 보내는 **변경 알림**을 수신하는 방식으로, 예를 들면 "Teams 회의가 종료됨", "이 채팅에 새 메시지가 도착함", "이 캘린더 이벤트가 업데이트됨"과 같은 알림입니다. 사용자가 입력하는 챗봇인 `teams` 플랫폼과는 다릅니다. 이 플랫폼은 사람이 아니라 M365가 Hermes에 무언가 발생했다고 알리는 방식입니다.

현재 주요 사용처는 Teams 회의 요약 파이프라인입니다. Graph가 회의에서 트랜스크립트가 생성되면 알리고, 파이프라인이 이를 가져온 다음 Hermes가 Teams에 요약을 다시 게시합니다. 다른 Graph 리소스(`/chats/.../messages`, `/users/.../events`)도 동일한 리스너를 사용하며, 해당 파이프라인 소비자는 각자의 PR로 추가됩니다.

## 사전 요구 사항

- Microsoft Graph 애플리케이션 자격 증명 — [Microsoft Graph 애플리케이션 등록](/guides/microsoft-graph-app-registration)
- Microsoft Graph에서 접근할 수 있는 **공개 HTTPS URL**(Graph는 비공개 엔드포인트를 호출하지 않습니다). 테스트에는 개발 터널을 사용할 수 있지만, 프로덕션에는 유효한 인증서가 있는 실제 도메인이 필요합니다.
- `clientState` 값으로 사용할 강력한 공유 시크릿. `openssl rand -hex 32`로 생성하고 `~/.hermes/.env`에 `MSGRAPH_WEBHOOK_CLIENT_STATE`로 저장합니다.

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

또는 `~/.hermes/.env`의 환경 변수로 설정합니다(시작 시 자동 병합).

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<generate-with-openssl-rand-hex-32>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

참고: 바인드 호스트는 `config.yaml`의 `extra.host`에서 읽습니다(위 예시 참조). `MSGRAPH_WEBHOOK_HOST` 환경 변수 재정의는 없습니다.

게이트웨이를 시작합니다: `hermes gateway run`. 리스너가 제공하는 엔드포인트는 다음과 같습니다.

- `POST /msgraph/webhook` — Graph의 변경 알림
- `GET /msgraph/webhook?validationToken=...` — Graph 구독 검증 핸드셰이크
- `GET /health` — 수락 및 중복 카운터가 포함된 준비 상태 프로브

리스너를 공개적으로 노출합니다(리버스 프록시, 개발 터널, 인그레스). Graph 구독에 사용할 알림 URL은 공개 HTTPS 오리진 뒤에 `/msgraph/webhook`을 붙인 주소입니다.

```
https://ops.example.com/msgraph/webhook
```

## 구성

모든 설정은 `platforms.msgraph_webhook.extra` 아래에 둡니다.

| 설정 | 기본값 | 설명 |
|---------|---------|-------------|
| `host` | 설정되지 않음(듀얼 스택: 모든 인터페이스, IPv4+IPv6) | HTTP 리스너의 바인드 주소입니다. 루프백이 아닌 바인드에는 `allowed_source_cidrs`가 필요합니다. 루프백(`127.0.0.1` / `::1`)은 개발 터널 또는 리버스 프록시 구성에 가장 간단합니다. |
| `port` | `8646` | 바인드 포트입니다. |
| `webhook_path` | `/msgraph/webhook` | Graph가 POST를 전송하는 URL 경로입니다. |
| `health_path` | `/health` | 준비 상태 엔드포인트입니다. |
| `client_state` | — | Graph가 모든 알림에 되돌려 보내는 공유 시크릿입니다. `hmac.compare_digest`로 비교하며 `openssl rand -hex 32`로 생성합니다. |
| `accepted_resources` | `[]`(모두 허용) | Graph 리소스 경로/패턴의 허용 목록입니다. 뒤의 `*`는 접두사 일치로 동작합니다. 앞의 `/`는 허용됩니다. 예: `["communications/onlineMeetings", "chats/*/messages"]`. |
| `max_seen_receipts` | `5000` | 알림 ID 중복 제거 캐시의 크기입니다. 한도에 도달하면 가장 오래된 항목이 제거됩니다. |
| `allowed_source_cidrs` | `[]` | 루프백이 아닌 바인드에 필요합니다. 리스너가 루프백에 바인드되고 로컬 터널 또는 리버스 프록시 앞에 있을 때만 비워 두세요. |

대부분의 설정에는 동일한 환경 변수(`MSGRAPH_WEBHOOK_*`)도 있으며, 게이트웨이 시작 시 구성에 병합됩니다(`host`는 예외로 구성 파일에서만 설정합니다. 위 참고 사항 참조). [환경 변수 참조](/reference/environment-variables#microsoft-graph-teams-meetings)를 확인하세요.

## 보안 강화

### clientState가 기본 인증 검사입니다

모든 Graph 알림에는 구독 등록 시 사용한 `clientState` 문자열이 포함됩니다. 리스너는 타이밍 안전 비교를 사용해 `clientState`가 일치하지 않는 알림을 거부합니다. 이는 Microsoft가 문서화한 메커니즘이므로 해당 값을 강력한 공유 시크릿으로 취급하세요.

`client_state`가 설정되지 않으면 리스너는 시작을 거부합니다.

### 소스 IP 허용 목록(프로덕션 배포)

프로덕션에서는 리스너를 Microsoft가 공개한 Graph 웹훅 소스 IP 범위로 제한하세요. Microsoft는 [Office 365 IP 주소 및 URL 웹 서비스](https://learn.microsoft.com/en-us/microsoft-365/enterprise/urls-and-ip-address-ranges)에서 송신 범위를 문서화합니다. 다음과 같이 구성합니다.

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
        # ...add the current Microsoft 365 "Common" + "Teams" category egress ranges
```

또는 환경 변수로 설정합니다.

```bash
MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS="52.96.0.0/14,52.104.0.0/14"
```

`allowed_source_cidrs` 없이 `0.0.0.0`, `::`, LAN IP와 같은 루프백이 아닌 호스트에 바인드하면 시작 시 거부됩니다. 같은 컴퓨터에서 개발 터널 또는 리버스 프록시를 사용하는 경우 Hermes를 `127.0.0.1` 또는 `::1`에 바인드하고 허용 목록을 비워 두세요. 잘못된 CIDR 문자열은 경고를 기록하고 무시합니다. **Microsoft IP 목록을 분기마다 검토하세요** — 목록은 변경됩니다.

### HTTPS 종료

리스너는 일반 HTTP를 사용합니다. 리버스 프록시(Caddy, Nginx, Cloudflare Tunnel, AWS ALB)에서 TLS를 종료하고 로컬 네트워크를 통해 리스너로 프록시하세요. Graph는 HTTPS가 아닌 엔드포인트로의 전달을 거부하므로 Graph 자체에서 암호화되지 않은 트래픽이 사용자에게 도달할 경로는 없습니다.

### 응답 위생

성공 시 리스너는 빈 본문과 함께 `202 Accepted`를 반환하며, 내부 카운터는 유선 응답에 포함되지 않습니다. 운영자는 `/health`를 통해 카운트를 확인할 수 있고, 이 엔드포인트에도 웹훅 경로와 동일한 소스 IP 규칙이 적용됩니다.

상태 코드 표:

| 결과 | 상태 |
|---------|--------|
| 알림 수락 또는 중복 제거 | 202 |
| 검증 핸드셰이크(`validationToken`이 포함된 GET) | 200(토큰을 그대로 반환) |
| 배치의 모든 항목이 clientState 검증에 실패 | 403 |
| 잘못된 JSON / `value` 배열 누락 / 알 수 없는 리소스 | 400 |
| 소스 IP가 허용 목록에 없음 | 403 |
| `validationToken`이 없는 단순 GET | 400 |

## 문제 해결

| 문제 | 확인할 사항 |
|---------|---------------|
| Graph 구독 검증 실패 | 공개 URL에 접근할 수 있는지, `/msgraph/webhook` 경로가 일치하는지, `validationToken`이 포함된 GET이 10초 이내에 토큰을 `text/plain`으로 그대로 반환하는지 확인합니다. |
| 알림 POST는 들어오지만 아무것도 수집되지 않음 | `client_state`가 구독 등록 시 사용한 값과 일치하는지 확인합니다. 값이 달라졌다면 `openssl rand -hex 32`를 다시 실행하고 새 구독을 만드세요. `accepted_resources`에 Graph가 보내는 리소스 경로가 포함되어 있는지 확인합니다. |
| 모든 알림이 403을 반환함 | `clientState` 불일치(위조되었거나 다른 값으로 구독이 등록됨)입니다. `hermes teams-pipeline subscribe --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE" ...`로 구독을 다시 만드세요(파이프라인 런타임 PR에 포함됨). |
| `0.0.0.0`에서 리스너 시작이 거부됨 | Microsoft의 최신 웹훅 송신 범위로 `allowed_source_cidrs`를 설정하거나, 터널 또는 리버스 프록시 뒤에서 Hermes를 `127.0.0.1` / `::1`에 바인드합니다. |
| 리스너는 시작되지만 `curl http://localhost:8646/health`가 멈춤 | 포트 바인딩 충돌입니다. `ss -tlnp \| grep 8646`을 확인하고 필요하면 `port:`를 변경합니다. |
| Microsoft의 실제 Graph 요청이 403으로 거부됨 | 소스 IP 허용 목록이 너무 좁습니다. 현재 Microsoft 송신 범위를 포함하도록 목록을 넓히세요. 터널 경로를 아직 검증하는 중이라면 Hermes를 루프백에 바인드하고 터널이 공개 노출을 처리하게 하세요. |

## 관련 문서

- [Microsoft Graph 애플리케이션 등록](/guides/microsoft-graph-app-registration) — Azure 앱 등록 사전 요구 사항
- [환경 변수 → Microsoft Graph](/reference/environment-variables#microsoft-graph-teams-meetings) — 전체 환경 변수 목록
- [Microsoft Teams 봇 설정](/user-guide/messaging/teams) — 사용자가 Teams에서 Hermes와 채팅할 수 있도록 하는 다른 플랫폼
