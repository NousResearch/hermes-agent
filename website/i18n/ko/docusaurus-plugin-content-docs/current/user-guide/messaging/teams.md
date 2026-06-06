# Microsoft Teams

Hermes는 Microsoft Teams 플랫폼과 통합되어 조직의 M365 환경에서 강력한 에이전트를 제공합니다.

> **참고**: 이 가이드는 사용자가 타이핑하고 에이전트와 대화하는 채팅 봇 인터페이스를 다룹니다. 시스템 이벤트(예: 회의 요약 파이프라인(meeting summary pipeline)을 위한 "온라인 회의 종료됨" 알림)를 구성하려면 [Microsoft Graph 웹훅 리스너(Webhook Listener)](/user-guide/messaging/msgraph-webhook)를 참조하세요.

## 전제 조건

- **Microsoft Entra ID (Azure AD) 앱 등록**: Azure 포털에서 생성된 봇을 나타냅니다.
- **Azure Bot 리소스**: 이 앱 등록에 연결되고 Teams 채널이 활성화되어 있어야 합니다.
- **공개 HTTPS 웹훅(Webhook)**: Azure Bot 서비스가 메시지 이벤트를 `POST`할 수 있는 엔드포인트입니다.

자세한 단계별 설정 과정은 [Microsoft Teams 설정 튜토리얼](/guides/microsoft-graph-app-registration)을 참조하세요.

## 구성

### 대화형 설정

```bash
hermes gateway setup
```

**Microsoft Teams**를 선택하고 제공되는 지침을 따르세요.

### 수동 구성

`~/.hermes/.env`에 자격 증명을 설정하세요:

```bash
# Entra ID 애플리케이션 ID
TEAMS_APP_ID=your-client-id
# 해당 앱에 대해 생성된 클라이언트 암호(Client Secret)
TEAMS_APP_PASSWORD=your-client-secret

# 웹훅을 수신할 URL (Bot 프레임워크에 구성된 것과 동일)
TEAMS_WEBHOOK_URL=https://your-public-domain.com/api/messages
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `TEAMS_APP_ID` | 예 | Azure 앱 클라이언트 ID |
| `TEAMS_APP_PASSWORD` | 예 | Azure 앱 클라이언트 암호 (Client secret) |
| `TEAMS_WEBHOOK_URL` | 예 | 수신 알림을 위한 공개 콜백 URL |
| `TEAMS_WEBHOOK_PORT` | 아니요 | 인바운드 웹훅을 위해 바인딩할 로컬 포트 (기본값: 3978) |
| `TEAMS_ALLOWED_USERS` | 아니요 | 액세스가 허용된 사용자의 이메일 (UPN) 쉼표 구분 목록 |

:::tip 사용자 허용 목록 (User Allowlist)
`TEAMS_ALLOWED_USERS`는 개별 Microsoft 계정 이메일(User Principal Names) 또는 도메인 접미사(예: `*@example.com`)를 매칭할 수 있습니다. 비워두면 Teams 환경 내의 모든 사용자가 봇과 상호작용할 수 있습니다.
:::

## 작동 방식

게이트웨이가 시작되면 `TEAMS_WEBHOOK_PORT`에서 HTTP 웹훅 서버를 호스팅합니다:

1. **인바운드(Inbound)**: Microsoft의 Bot Framework 서비스가 구성된 `TEAMS_WEBHOOK_URL`로 JSON 페이로드(`Activity` 객체)를 HTTP `POST`합니다. 어댑터는 들어오는 요청의 JWT 서명을 검증합니다.
2. **아웃바운드(Outbound)**: 어댑터는 Hermes 에이전트의 응답을 `MicrosoftAppCredentials`를 사용하여 인증된 Bot Framework REST API에 대한 아웃바운드 HTTP 요청으로 변환합니다.

```
[Microsoft Teams 클라이언트] <--> [Azure Bot Service] <--(HTTP POST)--> [Hermes Gateway]
```

원활한 작동을 위해서는 Nginx 호스트, ngrok 또는 Cloudflare Tunnel 등을 통해 트래픽이 웹훅 포트로 올바르게 라우팅되는지 확인해야 합니다.
