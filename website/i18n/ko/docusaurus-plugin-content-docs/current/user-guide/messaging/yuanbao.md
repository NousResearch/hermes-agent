# 텐센트 위안바오 (Tencent Yuanbao)

[Tencent Yuanbao (위안바오)](https://yuanbao.tencent.com/)는 기업용 메시징 플랫폼입니다. Hermes는 공식 **Yuanbao Bot API**와 연결하여 에이전트 기능을 사내 채팅에 도입합니다.

> **참고**: 기업의 내부 Yuanbao 환경을 위해 설계되었습니다. 퍼블릭 QQ나 WeChat용이 아닙니다.

## 전제 조건

- **Yuanbao 앱 자격 증명(App Credentials)**
  - App ID
  - App Secret
  - 봇 토큰 (웹훅용)
  - 인코딩 AES 키 (메시지 복호화용)
- **공개 엔드포인트**: 인터넷에서 포트 80 또는 443으로 Hermes에 연결할 수 있는 주소 (웹훅 콜백용).

## 구성

### 대화형 설정

```bash
hermes gateway setup
```

플랫폼 목록에서 **Yuanbao**를 선택하고 제공되는 지침을 따르세요.

### 수동 구성

`~/.hermes/.env`에 다음 환경 변수들을 설정하세요:

```bash
YUANBAO_APP_ID=your-app-id
YUANBAO_APP_SECRET=your-app-secret
YUANBAO_TOKEN=your-webhook-token
YUANBAO_ENCODING_AES_KEY=your-encoding-aes-key
YUANBAO_WEBHOOK_URL=https://your-public-domain.com/yuanbao
```

| 변수 | 필수 | 설명 |
|---|---|---|
| `YUANBAO_APP_ID` | 예 | 애플리케이션 ID |
| `YUANBAO_APP_SECRET` | 예 | API 인증을 위한 App Secret |
| `YUANBAO_TOKEN` | 예 | 웹훅 서명(signature)을 검증하기 위한 토큰 |
| `YUANBAO_ENCODING_AES_KEY` | 예 | 수신되는 페이로드를 복호화하기 위한 AES 키 |
| `YUANBAO_WEBHOOK_URL` | 예 | 봇이 이벤트를 수신할 전체 공개 URL |
| `YUANBAO_WEBHOOK_PORT` | 아니요 | 인바운드 웹훅을 위해 바인딩할 로컬 포트 (기본값: 8645) |
| `YUANBAO_ALLOWED_USERS` | 아니요 | 허용된 사용자 ID의 쉼표로 구분된 목록. 비워두면 모두 허용됩니다. |

:::info 사용자 식별 (Identity)
Yuanbao 어댑터는 플랫폼의 네이티브 사용자 ID를 사용합니다. `YUANBAO_ALLOWED_USERS`에서 허용 목록(allowlist)을 지정할 때, Yuanbao 시스템에서 제공하는 사용자 ID(user ID) 문자열을 사용하세요.
:::

## 시작하기

환경이 구성된 상태에서 게이트웨이를 시작합니다:

```bash
hermes gateway
```

어댑터가 로컬 포트(`YUANBAO_WEBHOOK_PORT`, 기본값 8645)를 바인딩하고 HTTP 수신 대기(listen)를 시작합니다.

1. **리버스 프록시**: Nginx 또는 Cloudflare Tunnel을 구성하여 `YUANBAO_WEBHOOK_URL`에서 이 로컬 포트로 트래픽을 라우팅(프록시)합니다.
2. **웹훅 검증**: Yuanbao 개발자 포털에서 웹훅 URL을 구성할 때, 플랫폼은 `GET` 요청을 보내 서명을 확인합니다. 어댑터는 이 검증을 자동으로 처리합니다.
3. **메시징**: 웹훅이 검증되면 일반 메시지(`POST` 요청)가 에이전트로 라우팅되고 에이전트는 Yuanbao API를 통해 응답합니다.
