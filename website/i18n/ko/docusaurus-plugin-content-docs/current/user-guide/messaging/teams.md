---
sidebar_position: 5
title: "Microsoft Teams"
description: "Hermes Agent를 Microsoft Teams 봇으로 설정"
---

# Microsoft Teams 설정

Hermes Agent를 Microsoft Teams에 봇으로 연결합니다. Slack의 Socket Mode와 달리 Teams는 **공개 HTTPS 웹훅**을 호출해 메시지를 전달하므로, 인스턴스에 공개적으로 접근할 수 있는 엔드포인트가 필요합니다. 개발 환경에서는 개발용 터널(로컬 개발)을, 프로덕션에서는 실제 도메인을 사용하세요.

일반적인 봇 대화가 아니라 Microsoft Graph 이벤트에서 회의 요약을 받아야 하나요? 전용 설정 페이지인 [Teams 회의](/user-guide/messaging/teams-meetings)를 사용하세요.

> `hermes gateway setup`을 실행하고 안내에 따라 **Microsoft Teams**를 선택하세요.

## 봇의 응답 방식

| 컨텍스트 | 동작 |
|---------|----------|
| **개인 채팅(DM)** | 봇이 모든 메시지에 응답합니다. @멘션이 필요하지 않습니다. |
| **그룹 채팅** | @멘션된 경우에만 봇이 응답합니다. |
| **채널** | @멘션된 경우에만 봇이 응답합니다. |

Teams는 `<at>BotName</at>` 태그가 포함된 일반 메시지로 @멘션을 전달하며, Hermes는 처리 전에 이를 자동으로 제거합니다.

---

소스에서 설치하거나 로컬 설치를 사용하는 경우, 번들된 어댑터가 Microsoft Teams SDK를 가져올 수 있도록 Teams extra를 포함하세요.

```bash
uv sync --extra teams
# or, for editable installs:
uv pip install -e ".[teams]"
```

## 1단계: Teams CLI 설치

`@microsoft/teams.cli`는 Azure 포털 없이 봇 등록을 자동화합니다.

```bash
npm install -g @microsoft/teams.cli@preview
teams login
```

로그인을 확인하고 `TEAMS_ALLOWED_USERS`에 필요한 자신의 AAD 객체 ID를 찾으려면 다음을 실행하세요.

```bash
teams status --verbose
```

---

## 2단계: 웹훅 포트 공개

Teams는 `localhost`로 메시지를 전달할 수 없습니다. 로컬 개발에서는 터널 도구를 사용해 공개 HTTPS URL을 얻으세요. 기본 포트는 `3978`이며, 필요한 경우 `TEAMS_PORT`로 변경할 수 있습니다.

```bash
# devtunnel (Microsoft)
devtunnel create hermes-bot --allow-anonymous
devtunnel port create hermes-bot -p 3978 --protocol http  # replace 3978 with TEAMS_PORT if changed
devtunnel host hermes-bot

# ngrok
ngrok http 3978  # replace 3978 with TEAMS_PORT if changed

# cloudflared
cloudflared tunnel --url http://localhost:3978  # replace 3978 with TEAMS_PORT if changed
```

출력에서 `https://` URL을 복사하세요. 다음 단계에서 사용합니다. 개발하는 동안 터널을 실행한 상태로 두세요.

공개 터널 URL은 HTTPS를 사용하지만 Hermes의 로컬 웹훅 리스너는 일반 HTTP를 사용합니다. 터널이 TLS를 종료하고 HTTP 요청을 `3978` 포트로 전달하므로, 로컬 터널 포트를 HTTPS로 설정하지 마세요.

프로덕션에서는 대신 봇의 엔드포인트를 서버의 공개 도메인으로 지정하세요([프로덕션 배포](#production-deployment) 참고).

---

## 3단계: 봇 생성

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://<your-tunnel-url>/api/messages"
```

CLI는 `CLIENT_ID`, `CLIENT_SECRET`, `TENANT_ID`와 6단계에서 사용할 설치 링크를 출력합니다. 클라이언트 시크릿은 다시 표시되지 않으므로 저장해 두세요.

---

## 4단계: 환경 변수 설정

`~/.hermes/.env`에 다음을 추가하세요.

```bash
# Required
TEAMS_CLIENT_ID=<your-client-id>
TEAMS_CLIENT_SECRET=<your-client-secret>
TEAMS_TENANT_ID=<your-tenant-id>

# Restrict access to specific users (recommended)
# Use AAD object IDs from `teams status --verbose`
TEAMS_ALLOWED_USERS=<your-aad-object-id>
```

---

## 5단계: 게이트웨이 시작

**Docker** (`docker-compose.yml`이 포함된 디렉터리에서 실행해야 합니다. 일반적으로 `~`가 아니라 복제한 `hermes-agent` 저장소입니다):

```bash
cd /path/to/hermes-agent
HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d gateway
```

**Native / systemd install** (일반적인 `~/.hermes/hermes-agent` 경로의 `hermes` 한 줄 설치 프로그램):

```bash
hermes gateway restart
# or foreground: hermes gateway run
```

Teams SDK는 선택 사항입니다. Teams가 활성화되면 게이트웨이는 처음 시작할 때 Hermes 자체 venv에 SDK를 지연 설치합니다(Ubuntu 24.04에서는 시스템 `pip install`을 사용하지 마세요. PEP 668의 `externally-managed-environment` 오류가 발생합니다). Hermes venv에 수동으로 설치하려면 다음을 실행하세요.

```bash
~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp
# or from a clone of the agent: uv sync --extra teams
```

기본 웹훅 포트는 `3978`입니다(`TEAMS_PORT`로 재정의 가능). 실행 중인지 확인하세요.

```bash
curl http://localhost:3978/health   # should return: ok
# Docker:
docker logs -f hermes
# Native:
hermes gateway status -l
```

다음 로그를 확인하세요.

```
[teams] Webhook server listening on * (all interfaces, IPv4+IPv6):3978/api/messages
```

---

## 6단계: Teams에 앱 설치

```bash
teams app get <teamsAppId> --install-link
```

출력된 링크를 브라우저에서 여세요. Teams 클라이언트에서 바로 열립니다. 설치한 후 봇에 직접 메시지를 보내면 사용할 준비가 끝납니다.

---

## 설정 레퍼런스

### 환경 변수

| 변수 | 설명 |
|----------|-------------|
| `TEAMS_CLIENT_ID` | Azure AD 앱(클라이언트) ID |
| `TEAMS_CLIENT_SECRET` | Azure AD 클라이언트 시크릿 |
| `TEAMS_TENANT_ID` | Azure AD 테넌트 ID |
| `TEAMS_ALLOWED_USERS` | 봇 사용이 허용된 AAD 객체 ID를 쉼표로 구분한 목록 |
| `TEAMS_ALLOW_ALL_USERS` | `true`로 설정하면 허용 목록을 건너뛰고 누구나 허용 |
| `TEAMS_HOME_CHANNEL` | cron/사전 알림 메시지를 전달할 대화 ID |
| `TEAMS_HOME_CHANNEL_NAME` | 홈 채널의 표시 이름 |
| `TEAMS_PORT` | 웹훅 포트(기본값: `3978`) |

### config.yaml

또는 `~/.hermes/config.yaml`을 통해 설정할 수 있습니다.

```yaml
platforms:
  teams:
    enabled: true
    extra:
      client_id: "your-client-id"
      client_secret: "your-secret"
      tenant_id: "your-tenant-id"
      port: 3978
```

---

## 기능

### 대화형 승인 카드

에이전트가 잠재적으로 위험한 명령을 실행해야 할 때 `/approve`를 입력하도록 요청하는 대신 네 개의 버튼이 있는 Adaptive Card를 보냅니다.

- **한 번 허용** — 이 특정 명령을 승인
- **세션 동안 허용** — 세션이 끝날 때까지 이 패턴을 승인
- **항상 허용** — 이 패턴을 영구적으로 승인
- **거부** — 명령 거부

버튼을 클릭하면 승인 결과가 인라인으로 처리되고 카드가 결정 내용으로 바뀝니다.

### 회의 요약 전달(Teams 회의 파이프라인)

[Teams 회의 파이프라인 플러그인](/user-guide/messaging/msgraph-webhook)이 활성화되면 이 어댑터는 회의 요약의 아웃바운드 전달도 처리합니다. Teams 통합 표면은 둘이 아니라 하나입니다. 회의 기록이 요약된 후 작성기가 선택한 Teams 대상에 요약을 게시합니다.

파이프라인 요약 전달은 봇 설정과 함께 `teams` 플랫폼 항목에서 설정합니다.

```yaml
platforms:
  teams:
    enabled: true
    extra:
      # existing bot config (client_id, client_secret, tenant_id, port) ...

      # Meeting summary delivery (only used when the teams_pipeline plugin is enabled)
      delivery_mode: "graph"       # or "incoming_webhook"
      # For delivery_mode: graph — pick ONE of:
      chat_id: "19:meeting_..."    # post into a Teams chat
      # team_id: "..."             # OR post into a channel
      # channel_id: "..."
      # access_token: "..."        # optional; falls back to MSGRAPH_* app credentials
      # For delivery_mode: incoming_webhook:
      # incoming_webhook_url: "https://outlook.office.com/webhook/..."
```

| 모드 | 다음 경우에 사용 | 절충점 |
|------|----------|-----------|
| `incoming_webhook` | 정적으로 생성된 Teams URL을 사용해 "이 채널에 요약 게시"를 간단히 처리할 때 | 답글 스레드와 리액션을 지원하지 않으며, 웹훅에 설정된 ID로 표시됨 |
| `graph` | Microsoft Graph를 통해 봇 ID로 스레드가 있는 채널 게시물이나 1:1/그룹 채팅 게시물을 작성할 때 | 채널에는 `ChannelMessage.Send`, 채팅에는 `Chat.ReadWrite.All` 애플리케이션 권한이 포함된 [Graph 앱 등록](/guides/microsoft-graph-app-registration)이 필요함 |

`teams_pipeline` 플러그인이 활성화되어 있지 않으면 이러한 설정은 적용되지 않습니다. 파이프라인 런타임이 Graph 웹훅 수신부에 연결될 때만 작동합니다.

---

## 프로덕션 배포

영구 서버에서는 리버스 프록시에서 TLS를 종료하고 일반적으로 `http://127.0.0.1:3978`인 일반 HTTP Hermes 리스너로 요청을 전달하세요. 프록시의 공개 HTTPS 엔드포인트를 Teams에 등록합니다.

```bash
teams app create \
  --name "Hermes" \
  --endpoint "https://your-domain.com/api/messages"
```

이미 봇을 만들었고 엔드포인트만 업데이트하면 되는 경우:

```bash
teams app update --id <teamsAppId> --endpoint "https://your-domain.com/api/messages"
```

공개 HTTPS 엔드포인트가 인터넷에서 접근 가능하고 유효한 TLS 인증서를 사용하는지 확인하세요. Teams는 자체 서명 인증서를 거부합니다. Hermes 리스너는 프록시 뒤에 두세요. `3978` 포트 자체는 HTTPS를 제공하지 않습니다.

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| `docker compose`에서 `Can't find a suitable configuration file` 표시 | `docker-compose.yml`이 있는 저장소에 있지 않거나 네이티브 설치를 사용 중입니다. 대신 `hermes gateway restart`를 사용하거나 먼저 복제본 디렉터리로 `cd`하세요 |
| `requirements not met` / `Teams SDK missing` / `No adapter available for teams` | 게이트웨이를 다시 시작해 지연 설치가 실행되도록 하거나 **Hermes venv**에 `~/.hermes/hermes-agent/venv/bin/pip install microsoft-teams-apps aiohttp`를 설치하세요. Ubuntu 24.04에서는 시스템 `pip`가 PEP 668로 실패하며 서비스에도 영향을 주지 않습니다 |
| `health` 엔드포인트는 작동하지만 봇이 응답하지 않음 | 터널이 계속 실행 중인지, 봇의 메시징 엔드포인트가 터널 URL과 일치하는지 확인하세요 |
| Teams가 메시지를 보낼 때 로그에 `"UNKNOWN / HTTP/1.0" 400` 표시 | 터널 또는 리버스 프록시가 HTTPS를 Hermes의 일반 HTTP 리스너로 전달하고 있습니다. 프록시에서 TLS를 종료하고 HTTP를 `3978` 포트로 전달하세요 |
| 로그에 `KeyError: 'teams'` 표시 | 컨테이너를 다시 시작하세요. 현재 버전에서 수정되었습니다 |
| 봇이 인증 오류로 응답함 | `TEAMS_CLIENT_ID`, `TEAMS_CLIENT_SECRET`, `TEAMS_TENANT_ID`가 모두 올바르게 설정되었는지 확인하세요 |
| `No inference provider configured` | `~/.hermes/.env`에 `ANTHROPIC_API_KEY`(또는 다른 프로바이더 키)가 설정되어 있는지 확인하세요 |
| 봇이 메시지를 받지만 무시함 | AAD 객체 ID가 `TEAMS_ALLOWED_USERS`에 없을 수 있습니다. `teams status --verbose`를 실행해 확인하세요 |
| 다시 시작할 때 터널 URL이 변경됨 | 이름이 지정된 터널(`devtunnel create hermes-bot`)을 사용하면 devtunnel URL은 유지됩니다. 유료 플랜이 없는 경우 ngrok과 cloudflared는 실행할 때마다 새 URL을 생성하므로, 변경되면 `teams app update`로 봇 엔드포인트를 업데이트하세요 |
| Teams에 "This bot is not responding" 표시 | 웹훅에서 오류를 반환했습니다. `docker logs hermes` / `hermes gateway status -l`에서 트레이스백을 확인하세요 |
| 로그에 `[teams] Failed to connect` 표시 | SDK 인증에 실패했습니다. 자격 증명과 테넌트 ID가 `teams login`에 사용한 계정과 일치하는지 다시 확인하세요 |

---

## 보안

:::warning
**항상 `TEAMS_ALLOWED_USERS`를 설정하세요.** 권한이 있는 사용자의 AAD 객체 ID를 지정해야 합니다. 이 설정이 없으면 봇을 찾거나 설치할 수 있는 누구나 봇과 상호작용할 수 있습니다.

`TEAMS_CLIENT_SECRET`은 비밀번호처럼 취급하고 Azure 포털 또는 Teams CLI를 통해 정기적으로 교체하세요.
:::

- 자격 증명은 권한을 `600`으로 설정한 `~/.hermes/.env`에 저장하세요(`chmod 600 ~/.hermes/.env`).
- 봇은 `TEAMS_ALLOWED_USERS`에 포함된 사용자의 메시지만 수락하며, 권한이 없는 메시지는 조용히 삭제됩니다.
- 공개 엔드포인트(`/api/messages`)는 Teams Bot Framework가 인증합니다. 유효한 JWT가 없는 요청은 거부됩니다.

## 관련 문서

- [Teams 회의](/user-guide/messaging/teams-meetings)
- [Teams 회의 파이프라인 운영](/guides/operate-teams-meeting-pipeline)
