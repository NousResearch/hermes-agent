---
sidebar_position: 6
title: "WhatsApp Business (Cloud API)"
description: "Meta의 공식 Business Cloud API를 통해 Hermes Agent를 WhatsApp 봇으로 설정합니다"
---

# WhatsApp Business Cloud API 설정

Hermes는 Meta의 **공식** WhatsApp Business Cloud API를 통해 WhatsApp에 연결할 수 있습니다. 이는 프로덕션에 적합한 방식입니다. Node.js 브리지 하위 프로세스가 필요 없고, QR 코드도 없으며, 계정이 차단될 위험도 없습니다.

대신 다음이 필요합니다.

- **Meta Business 계정** (개인 WhatsApp 계정이 아님)
- 개인 번호가 아닌, 봇 전용 비즈니스 전화번호
- Meta가 웹훅을 통해 수신 메시지를 전달할 수 있도록 Hermes 게이트웨이에 **공개 HTTPS URL**
- 사용자의 마지막 메시지로부터 24시간이 지난 뒤에 답장하려면 사전 승인을 받은 **템플릿** (이는 Hermes의 제한이 아니라 Meta의 "고객 서비스 창" 규칙입니다.)

이 제약이 사용 사례에 맞지 않는다면 [Baileys 브리지 통합](./whatsapp.md)을 대안으로 사용할 수 있습니다. 개인 계정으로 운영하고 공개 URL이 필요 없지만, 비공식 방식이므로 차단될 위험이 있습니다.

:::tip 어떤 방식을 사용해야 하나요?
- **Cloud API (이 가이드)** — 실제 비즈니스 봇을 운영하고 안정성을 원하며, Meta 인증과 템플릿 관련 서류 작업을 감수할 수 있는 경우
- **[Baileys 브리지](./whatsapp.md)** — 개인 프로젝트, 빠른 데모, 단일 사용자 설정에 적합하며 봇 전화번호 계정이 차단될 위험을 감수할 수 있는 경우
:::

---

## 빠른 시작

```bash
hermes whatsapp-cloud
```

마법사가 모든 자격 증명 입력을 안내하고, 붙여 넣는 즉시 각각을 검증합니다 (전화번호를 Phone Number ID 필드에 붙여 넣는 가장 흔한 설정 실수도 잡아냅니다). 또한 마법사 외부에서 진행해야 하는 작업(예: cloudflared 시작, Meta 웹훅 대시보드 설정)에 대한 정확한 후속 안내를 출력합니다.

이 페이지의 나머지 부분은 수동 설정을 위한 참고 자료입니다.

---

## 사전 요구 사항

1. **Meta Business 계정**. [business.facebook.com](https://business.facebook.com/)에서 만드세요.
2. **WhatsApp이 활성화된 Meta 앱**. 아래의 "Meta 앱 만들기"를 참고하세요.
3. **로컬 포트를 HTTPS를 사용하는 공개 인터넷에 노출할 방법**. Cloudflare Tunnel(`cloudflared`)을 권장합니다. 무료이고 포트 포워딩이나 도메인이 필요하지 않습니다. ngrok, 리버스 프록시 + TLS를 사용하는 자체 도메인, 또는 게이트웨이를 공개 IP에 직접 바인딩한 VPS도 사용할 수 있습니다.
4. **선택 사항이지만 권장**: `PATH`에 ffmpeg를 설치하면 발신 음성 메시지가 MP3 오디오 첨부 파일 대신 기본 WhatsApp 음성 메모 말풍선(녹색 파형)으로 표시됩니다. 없어도 Hermes는 정상적으로 대체 동작합니다.

---

## Meta 앱 만들기

1. [developers.facebook.com/apps](https://developers.facebook.com/apps)로 이동 → **Create App**을 클릭합니다.
2. 사용 사례로 **"Connect with customers through WhatsApp"**를 선택 → **Next**를 클릭합니다.
3. 비즈니스 포트폴리오를 선택하거나 만듭니다. 게시 요구 사항을 검토합니다. Confirm → **Create app**을 클릭합니다.
4. 앱을 만든 뒤 **Customize use case → Connect on WhatsApp → Quickstart** 화면으로 이동합니다. **Start using the API**를 클릭하면 **API Setup** 페이지로 이동합니다.
5. WhatsApp Business Account(WABA)가 연결되어 있는지 확인합니다. 3단계에서 새 포트폴리오를 만들었다면 하나가 자동으로 생성됩니다. API Setup 페이지에서 확인하세요.

대시보드에서 다음 값을 확인해야 합니다. 마법사는 아래 순서대로 입력을 요청합니다.

| 값 | 대시보드 위치 | 필드 형식 | 참고 |
|---|---|---|---|
| **Phone Number ID** | App Dashboard → WhatsApp → API Setup → "From" 드롭다운 아래 | 숫자 15~17자리 | **전화번호 자체가 아닙니다.** 실제 전화번호를 여기에 붙여 넣는 것이 가장 흔한 설정 실수입니다. |
| **Access Token** | App Dashboard → WhatsApp → API Setup → "Generate access token" | `EAA`로 시작하며 100자 이상 | 임시 토큰은 24시간 동안 유효합니다. 프로덕션에서는 아래의 "영구 토큰"을 참고하세요. |
| **App Secret** | App Dashboard → Settings → Basic → App secret 옆의 "Show" 클릭 | 32자의 소문자 16진수 | 수신 웹훅 서명을 검증하는 데 사용됩니다. 없으면 수신 전달이 503으로 거부됩니다. |
| **App ID** (선택 사항) | App Dashboard → Settings → Basic | 숫자 15~16자리 | 메시징에는 필요하지 않지만 분석에 유용합니다. |
| **WABA ID** (선택 사항) | App Dashboard → WhatsApp → API Setup → 상단 부근 | 숫자 15자리 이상 | 메시징에는 필요하지 않지만 분석에 유용합니다. |

---

## 영구 토큰 (프로덕션)

임시 액세스 토큰은 **24시간** 후 만료되므로, 오늘 생성한 토큰은 내일 작동하지 않습니다. 프로덕션 배포에서는 **System User 영구 토큰**을 사용하세요.

1. [business.facebook.com/latest/settings](https://business.facebook.com/latest/settings)로 이동 → 왼쪽 사이드바에서 **System users**를 선택합니다.
2. **Add** → 이름(예: `hermes-bot`) → 역할: **Admin**을 선택합니다.
3. 새 사용자를 선택 → **Assign Assets**:
   - 앱을 선택 → Full control 아래에서 **Manage app**을 켭니다.
   - WhatsApp 계정을 선택 → Full control 아래에서 **Manage WhatsApp Business Accounts**를 켭니다.
   - **Assign assets**를 클릭합니다.
4. 다음 권한으로 **Generate token**을 실행합니다.
   - `business_management`
   - `whatsapp_business_messaging`
   - `whatsapp_business_management`
5. **token expiration: Never**를 선택합니다.
6. 토큰을 복사 → `~/.hermes/.env`의 `WHATSAPP_CLOUD_ACCESS_TOKEN`을 업데이트 → 게이트웨이를 다시 시작합니다.

System User 토큰은 명시적으로 폐기하지 않는 한 만료되지 않습니다.

---

## Hermes를 인터넷에 노출하기

Cloud API는 웹훅 URL로 HTTPS POST를 보내 수신 메시지를 전달합니다. 따라서 Hermes 게이트웨이는 Meta 서버에서 접근할 수 있어야 합니다. 일반적인 방법은 세 가지입니다.

### Cloudflare Tunnel (권장)

무료이고 포트 포워딩이 필요 없으며 Windows / macOS / Linux에서 작동합니다. 게이트웨이와 함께 별도의 프로세스로 실행합니다.

**설치:**

```bash
# Windows
winget install Cloudflare.cloudflared

# macOS
brew install cloudflared

# Linux
# Download the binary from https://github.com/cloudflare/cloudflared/releases
```

**빠른 터널 실행** (Cloudflare 계정이 필요 없으며 `https://<random>.trycloudflare.com` URL이 제공됩니다):

```bash
cloudflared tunnel --url http://localhost:8090
```

출력된 URL을 기록하세요. 이 URL을 Meta에 제공하게 됩니다.

:::warning 빠른 터널 URL은 변경됩니다
무료 빠른 터널 URL은 `cloudflared`를 다시 시작할 때마다 변경됩니다. 안정적인 URL을 사용하려면 `cloudflared tunnel login`으로 로그인하고 이름이 지정된 터널을 만드세요. 무료 Cloudflare 계정은 이름이 지정된 터널을 무제한으로 사용할 수 있습니다. 이름이 지정된 터널을 만드는 방법은 [Cloudflare 문서](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)의 안내를 참고하세요.
:::

### ngrok

```bash
ngrok http 8090
```

무료 요금제에서는 다시 시작할 때마다 다른 URL이 표시됩니다. 유료 요금제에서는 안정적인 하위 도메인을 사용할 수 있습니다.

### 자체 도메인 + 리버스 프록시

TLS 인증서가 있는 서버(Caddy, nginx 등)를 이미 보유하고 있다면 `localhost:8090`으로 경로를 지정하세요. 프로덕션에서 가장 안정적인 방법이지만 기존 인프라가 필요합니다.

---

## Meta 측에서 웹훅 설정하기

터널이 실행되면 다음 단계를 진행합니다.

1. 터널이 출력한 공개 URL을 기록합니다. 예를 들어 `https://abc123.trycloudflare.com`입니다.
2. **Verify Token**을 생성합니다. 마법사가 `secrets.token_urlsafe(32)`를 사용해 대신 생성해 줍니다. 수동으로 설정한다면 다음을 실행하세요.
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   `WHATSAPP_CLOUD_VERIFY_TOKEN`이라는 이름으로 `~/.hermes/.env`에 저장합니다.
3. Hermes 게이트웨이를 시작합니다: `hermes gateway`.
4. Meta App Dashboard에서 **WhatsApp → Configuration**(또는 UI 버전에 따라 **Use cases → Customize → Configuration**)으로 이동 → Webhook 섹션에서 **Edit**를 클릭합니다.
5. 다음 값을 입력합니다.
   - **Callback URL**: `https://abc123.trycloudflare.com/whatsapp/webhook`
   - **Verify Token**: 2단계의 문자열 (정확히 일치해야 함)
6. **Verify and save**를 클릭합니다. Meta가 GET 요청으로 URL을 호출하면 게이트웨이가 challenge를 그대로 응답하고, Meta는 웹훅을 인증된 것으로 표시합니다.
7. **Webhook fields**에서 **Manage**를 클릭 → **messages** 필드를 구독합니다. 이를 통해 Meta가 실제로 웹훅으로 수신 메시지를 전달합니다.

**루프를 수동으로 확인하려면** (세 번째 터미널에서):

```bash
TUNNEL="https://abc123.trycloudflare.com"
VERIFY="<your verify token>"

# Should print HTTP 200 with body "hello"
curl -i "$TUNNEL/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY&hub.challenge=hello"

# Health endpoint — should show verify_token_configured: true and app_secret_configured: true
curl "$TUNNEL/health"
```

---

## 수신자 허용 목록 (Meta 측)

개발 모드(앱이 App Review를 통과하기 전)에서는 Meta가 봇이 메시지를 보낼 수 있는 번호를 제한합니다.

1. App Dashboard → WhatsApp → API Setup → **To** 드롭다운으로 이동합니다.
2. **Manage phone number list**를 클릭합니다.
3. 메시지를 보낼 전화번호(본인, 팀원, 신뢰할 수 있는 테스터)를 추가합니다. Meta가 각 번호로 SMS 또는 WhatsApp을 통해 6자리 인증 코드를 보냅니다.

개발 모드에서는 최대 5개의 번호를 사용할 수 있습니다. App Review를 진행하면 이 제한이 사라집니다.

---

## 허용 목록 (Hermes 측)

Meta의 수신자 허용 목록과 별개로, Hermes에는 에이전트가 **처리할 수신 메시지**를 제어하는 플랫폼별 허용 목록이 있습니다. `~/.hermes/.env`에 다음을 추가하세요.

```bash
# Comma-separated phone numbers, country code, no '+' / spaces / dashes
WHATSAPP_CLOUD_ALLOWED_USERS=15551234567,15557654321

# Or allow everyone (only safe in combination with Meta's recipient whitelist)
# WHATSAPP_CLOUD_ALLOW_ALL_USERS=true
```

마법사의 6단계에서 이를 설정합니다. 허용 목록이 없으면 **모든 수신 메시지가 거부됩니다**. 이는 수신자 허용 목록이 나중에 느슨해지더라도 임의의 번호로 봇이 호출되지 않도록 의도된 동작입니다.

---

## 봇의 WhatsApp 프로필 꾸미기

WhatsApp은 채팅 헤더와 연락처 목록에 봇의 **이름과 프로필 사진**을 표시합니다. Cloud API로는 이를 설정할 수 없으며 Meta Business Manager에서 관리합니다.

봇이 작동하면 [business.facebook.com/wa/manage/phone-numbers](https://business.facebook.com/wa/manage/phone-numbers/)로 이동해 전화번호를 클릭하세요. 다음 항목을 찾을 수 있습니다.

| 항목 | 위치 | 참고 |
|---|---|---|
| **표시 이름** | 전화번호 페이지 상단 | Meta의 이름 검토 절차를 거칩니다(약 24~48시간). |
| **프로필 사진** | 전화번호 페이지 상단 | 정사각형 이미지, 640×640px 이상 권장. 즉시 업데이트됩니다. |
| **소개 / 설명 / 웹사이트 / 이메일 / 운영 시간 / 카테고리** | "Edit profile" 버튼 | 사용자가 봇 이름을 탭하면 정보 창에 표시됩니다. 외관을 위한 항목입니다. |
| **인증 배지** (녹색 체크 표시) | Business Manager → Security Center → Start Verification | Meta의 별도 비즈니스 인증 절차가 필요합니다. |

`hermes whatsapp-cloud` 마법사는 설정 마지막에 이 링크들을 출력합니다. 봇이 작동하는 데 필요한 항목은 없으며, 봇이 사용자에게 표시되는 방식을 꾸미기 위한 기능입니다.

---

## 설정 참고

모든 설정은 `~/.hermes/.env`에 저장됩니다. 필수 값은 **굵게** 표시되어 있습니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| **`WHATSAPP_CLOUD_PHONE_NUMBER_ID`** | — | API Setup의 15~17자리 ID. **전화번호가 아닙니다.** |
| **`WHATSAPP_CLOUD_ACCESS_TOKEN`** | — | Meta 액세스 토큰(`EAA`로 시작). 24시간 임시 토큰 또는 영구 System User 토큰입니다. |
| **`WHATSAPP_CLOUD_APP_SECRET`** | — | Settings → Basic의 32자 16진수. 없으면 수신이 503으로 거부됩니다. |
| **`WHATSAPP_CLOUD_VERIFY_TOKEN`** | — | GET 핸드셰이크용 공유 비밀 값. 마법사가 자동으로 생성합니다. |
| **`WHATSAPP_CLOUD_ALLOWED_USERS`** | — | 봇에 메시지를 보낼 수 있는 쉼표로 구분된 wa_id 목록입니다. |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | `false` | `true`로 설정하면 허용 목록을 우회합니다. |
| `WHATSAPP_CLOUD_APP_ID` | — | 선택 사항, 향후 분석 통합용입니다. |
| `WHATSAPP_CLOUD_WABA_ID` | — | 선택 사항, 향후 분석 통합용입니다. |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | 설정되지 않음 (듀얼 스택: 모든 인터페이스, IPv4+IPv6) | 웹훅 서버가 바인딩할 인터페이스입니다. |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | `8090` | 웹훅 서버가 바인딩할 포트입니다. 터널이 전달하는 포트와 일치해야 합니다. |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | `/whatsapp/webhook` | Meta가 POST를 보내는 URL 경로입니다. |
| `WHATSAPP_CLOUD_API_VERSION` | `v20.0` | Meta Graph API 버전입니다. Meta 문서에서 새 버전을 권장하는 경우에만 재정의하세요. |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | — | 봇의 홈 채널(예: cron 작업)에 사용할 wa_id입니다. |

Baileys(`whatsapp`)와 Cloud(`whatsapp_cloud`) 어댑터를 모두 활성화해 서로 다른 전화번호를 대상으로 동시에 사용할 수 있습니다.

---

## 기능

### 수신

- **텍스트 메시지** — 에이전트에 바로 전달됩니다.
- **이미지** — 자동으로 다운로드되어 에이전트 입력에 첨부됩니다. 네이티브 비전 기능이 있는 모델(Claude, GPT-4o, Gemini 등)은 이미지를 직접 읽고, 비전 기능이 없는 모델은 자동 생성된 텍스트 설명을 받습니다.
- **음성 메모** — `.ogg`로 자동 다운로드되고, 설정한 STT 제공자(로컬 faster-whisper, OpenAI/Nous, Groq 등)를 통해 전사된 뒤 텍스트로 에이전트에 전달됩니다.
- **문서** — 자동으로 다운로드됩니다. 작은 텍스트 파일(`.txt`, `.md`, `.json`, `.py`, `.csv` 등)은 최대 100KB까지 에이전트 입력에 인라인으로 포함되므로 도구 호출 없이 읽을 수 있습니다. 더 큰 파일은 에이전트의 다른 도구가 접근할 수 있도록 로컬에 캐시됩니다.
- **버튼 탭** — 사용자가 봇이 앞서 보낸 버튼(확인 질문, 명령 승인, 슬래시 명령 확인)을 탭하면 해당 탭이 올바른 핸들러로 직접 전달됩니다. 오래된 탭은 일반 텍스트 입력으로 처리됩니다.
- **답장 컨텍스트** — 사용자가 이전 봇 메시지에 답장하면 에이전트가 원본 메시지를 컨텍스트로 확인합니다.

### 발신

- **텍스트** — Markdown이 WhatsApp의 자체 문법으로 자동 변환됩니다(`**굵게**` → `*굵게*`, `~~취소선~~` → `~취소선~`, 헤더 → 굵게, `[링크](url)` → `링크 (url)`). 긴 메시지는 청크당 4096자로 나뉩니다.
- **이미지** — 에이전트가 생성한 이미지와 로컬 이미지 파일을 모두 지원하며, 네이티브 사진 첨부 파일로 전송됩니다.
- **음성 메시지** — 텍스트 음성 변환 출력이 ffmpeg를 통해 네이티브 WhatsApp 음성 메모 말풍선(녹색 파형)으로 변환됩니다. ffmpeg가 설치되지 않았으면 MP3 오디오 첨부 파일로 대체됩니다. 자세한 내용은 아래의 "음성 메시지"를 참고하세요.
- **동영상 / 문서** — 모두 지원하며 네이티브 첨부 파일로 전송됩니다.

### 인터랙티브 UX

에이전트가 다음 흐름 중 하나를 실행하면 Hermes는 WhatsApp의 네이티브 인터랙티브 메시지를 사용합니다. "번호로 답장"하라는 안내 대신 탭해서 답할 수 있는 버튼이 표시됩니다.

- **`clarify` 도구** — 다중 선택 질문은 빠른 답장 버튼(1~3개) 또는 탭해서 여는 목록 시트(4개 이상)로 표시됩니다. "✏️ Other"를 선택하면 자유 형식 답변을 입력할 수 있고, 에이전트는 이를 해결 결과로 받습니다.
- **위험한 명령 승인** — 에이전트의 터미널/코드 실행이 제한된 명령에 도달하면 `/approve` 또는 `/deny`를 입력하는 대신 `✅ Approve` / `❌ Deny` 버튼이 표시됩니다.
- **슬래시 명령 확인** — `/reload-mcp`와 같은 권한 있는 명령에는 `✅ Approve Once` / `🔒 Always` / `❌ Cancel` 버튼이 표시됩니다.

버튼을 표시할 수 없는 경우(예: 레거시 WhatsApp 클라이언트) 모든 인터랙티브 프롬프트는 일반 텍스트로 자연스럽게 대체됩니다.

### 읽음 확인 및 입력 중 표시

Hermes는 수신 메시지를 즉시 확인합니다.

- 게이트웨이가 메시지를 받는 즉시 사용자의 메시지에 **파란색 이중 체크 표시**가 나타납니다.
- 에이전트가 답장을 준비하는 동안 WhatsApp 채팅의 봇 이름 아래에 **"입력 중…"**이 표시됩니다.
- 봇의 첫 응답 메시지가 도착하면 입력 중 표시가 자동으로 사라집니다.

이를 통해 봇이 메시지를 확인했는지, 아니면 아직 답변을 준비 중인지 분명히 알 수 있습니다.

### 음성 메시지

WhatsApp은 "음성 메모"(녹색 파형 말풍선)와 일반 오디오 파일 첨부 파일을 구분합니다. 차이는 순전히 코덱에 있습니다. 음성 메모에는 `opus` 인코딩을 사용하는 `audio/ogg`가 필요합니다.

Hermes TTS는 MP3를 생성합니다. 방법은 두 가지입니다.

- **`PATH`에 ffmpeg가 있는 경우** (권장) — 발신 TTS가 변환되어 올바른 음성 메모로 도착합니다. 설치 방법:
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: 패키지 관리자
- **ffmpeg가 없는 경우** — 발신 TTS가 MP3 오디오 첨부 파일로 도착합니다. 재생에는 문제가 없지만 음성 메모처럼 보이지 않습니다. 이를 알 수 있도록 게이트웨이 로그에 한 번 경고가 기록됩니다.

게이트웨이가 ffmpeg를 찾았는지는 health 엔드포인트에서 확인할 수 있습니다.

```bash
curl http://localhost:8090/health
# look for "ffmpeg_present": true
```

---

## 알려진 제한 사항

### 24시간 대화 창

Meta는 사용자의 마지막 수신 메시지 이후 24시간 동안만 **자유 형식 메시지**를 허용합니다. 이 시간 창이 지나면 Meta API가 받아들이는 것은 사전 승인된 **메시지 템플릿**뿐입니다.

**실제로는 다음과 같습니다.**

- 반응형 채팅(사용자가 DM → 봇이 24시간 이내에 답장 → 사용자가 답장 → ...)은 영구적으로 작동합니다. 일반적인 봇 사용의 95% 이상이 여기에 해당합니다.
- **24시간이 넘는 공백 후 WhatsApp으로 전달되는 cron 작업**은 Graph 오류 코드 `131047`("재참여 메시지")와 함께 실패합니다.
- **24시간보다 오래 걸리는 장기 실행 `delegate_task` 비동기 결과**도 같은 방식으로 실패합니다.
- **외부 이벤트를 WhatsApp으로 라우팅하는 웹훅 구독자**는 사용자가 최근에 봇에 DM을 보내지 않았다면 실패합니다.

Hermes는 시스템 프롬프트에서 이 창을 에이전트에 알려 모델이 지연된 메시지를 예약할 때 이를 언급할 수 있도록 합니다.

창 밖으로 보내기 위한 메시지 템플릿 지원(이 문제의 해결 방법)은 아직 Hermes에 구현되지 않았습니다. 필요하다면 [이슈를 열어 주세요](https://github.com/NousResearch/hermes-agent/issues). 이 기능은 계획되어 있지만 명확한 수요 신호를 기다리고 있습니다.

### 그룹 채팅

Cloud API는 그룹 지원이 제한적입니다(Meta의 기능 등급에 따라 달라짐). Hermes의 `whatsapp_cloud` 어댑터는 현재 v1에서 **1:1 메시지만** 처리합니다. 그룹 채팅이 필요하다면 Baileys 브리지를 사용하세요.

### 발신 속도 제한

Meta의 기본 처리량은 **비즈니스 전화번호당 초당 80개 메시지**이며, 상향할 수 있습니다. Hermes는 현재 클라이언트 측에서 이 제한을 적용하지 않으므로, 매우 많은 양을 전송하면 Meta의 제한에 걸릴 수 있습니다.

---

## 문제 해결

### Meta 대시보드에서 설정 확인 실패("URL couldn't be validated")

대부분 다음 중 하나입니다.

- **터널 URL이 잘못되었거나 오래됨** — cloudflared 빠른 터널은 URL이 변경됩니다. 새 URL을 받아 `.env`와 Meta 대시보드를 모두 업데이트하세요.
- **Verify Token 불일치** — `~/.hermes/.env`의 `WHATSAPP_CLOUD_VERIFY_TOKEN` 값은 Meta 대시보드에 입력한 값과 정확히 일치해야 합니다. 위의 curl 프로브를 실행해 먼저 로컬에서 게이트웨이의 verify 핸드셰이크가 작동하는지 확인하세요.
- **게이트웨이가 실행 중이 아님** — `hermes gateway`가 실행 중인지 확인하세요.
- **App Secret이 설정되지 않음** — 없으면 Hermes가 수신 POST를 503으로 거부합니다. Meta는 이를 "검증할 수 없음"으로 해석합니다.

### `graph error 100`: ID가 '...'인 객체가 존재하지 않음

Phone Number ID 대신 전화번호(10~11자리)를 `WHATSAPP_CLOUD_PHONE_NUMBER_ID`에 붙여 넣었습니다. API Setup 페이지를 다시 확인하세요. Phone Number ID는 "From" 드롭다운 *아래*에 표시되는 15~17자리 내부 ID입니다.

마법사가 이제 검증기로 이를 잡아내지만, 수동으로 설정할 때 알아 두면 좋습니다.

### `graph error 190`: 인증 오류

액세스 토큰이 유효하지 않습니다. 하위 코드는 다음과 같습니다.

- `subcode 463` — 토큰이 만료되었습니다. 임시 토큰은 24시간 동안 유효합니다. 토큰을 새로 생성하거나 위에서 설명한 System User 영구 토큰으로 전환하세요.
- `subcode 467` — 토큰이 무효화되었습니다(폐기되었거나 비밀번호가 변경됨).
- 그 밖의 190 — 토큰을 생성할 때 필요한 권한이 없었습니다. 세 권한(`business_management`, `whatsapp_business_messaging`, `whatsapp_business_management`)을 모두 선택했는지 확인하세요.

### `graph error 131047`: 재참여 메시지

24시간 대화 창이 만료되었습니다(“알려진 제한 사항” 참고). 다음 중 하나를 수행하세요.

- 사용자에게 먼저 봇에 DM을 보내 대화 창을 다시 열도록 요청합니다.
- Hermes에 템플릿 지원이 추가될 때까지 기다립니다.

### 수신 메시지: `media metadata fetch failed (status=401)`

발신(`graph error 190`)과 동일한 401 원인입니다. 액세스 토큰이 유효하지 않거나 만료되었습니다. 토큰을 수정하세요.

### 봇 답장이 원시 JSON / 도구 호출 정보 노출로 표시됨

일반적인 원인은 `whatsapp_cloud`에 설정된 도구 모음에 에이전트가 호출하려는 도구가 빠져 있는 것입니다. `hermes tools list`를 확인하고 플랫폼이 `hermes-whatsapp`(Baileys와 동일한 기본 Cloud 어댑터 도구 모음)을 사용하는지 검증하세요.

모델이 구조화된 호출 대신 도구 호출 형태의 텍스트를 출력한다면, 도구 모음이 사실상 비어 있는 경우가 많습니다. 플랫폼과 기본 도구 모음의 매핑은 `hermes_cli/platforms.py`를 참고하세요.

### STT(음성 메모 전사)가 비어 있거나 "could not transcribe"를 반환함

기본 `stt.provider: local`은 `pip install faster-whisper`를 필요로 합니다. Nous 구독자라면 Meta의 관리형 오디오 게이트웨이를 통해 STT를 라우팅할 수도 있습니다.

```bash
hermes config set stt.provider openai
hermes config set stt.use_gateway true
hermes gateway restart
```

이 방식은 별도의 OpenAI 키 대신 Nous Portal 액세스 토큰을 사용합니다.

---

## 보안 참고 사항

- **App Secret을 비밀번호처럼 취급하세요** — 이 값을 가진 사람은 Hermes가 진짜로 받아들일 위조 웹훅 페이로드를 만들 수 있습니다.
- **Verify Token은 공유 비밀 값입니다** — 유출되어도 위험도는 낮습니다(최악의 경우 누군가 Meta의 웹훅을 자신의 다른 URL로 다시 구독할 수 있음). 그래도 커밋하지 않도록 주의하세요.
- **액세스 토큰은 봇의 신원입니다** — System User 토큰은 장기간 유효한 API 키와 같습니다. 배포가 침해되었다면 즉시 교체하세요.
- **`WHATSAPP_CLOUD_APP_SECRET`이 설정되면 웹훅 엔드포인트는 서명된 요청만 허용합니다** — 개발 환경에서도 설정해 두세요. 없으면 게이트웨이가 수신 전달을 HTTP 503으로 거부합니다.
- **`/health` 엔드포인트는 인증되지 않습니다** — 설정 값 자체가 아니라 설정 여부를 나타내는 불리언만 보고하므로 공개해도 안전합니다. 그래도 노출하고 싶지 않다면 리버스 프록시 / 터널 계층에서 접근을 제한하세요.

---

## Baileys 브리지와 비교

| | Baileys (`hermes whatsapp`) | Cloud API (`hermes whatsapp-cloud`) |
|---|---|---|
| 계정 유형 | 개인 | 비즈니스 |
| 설정 | QR 코드 스캔 | Meta 앱 + WABA + 토큰 |
| 종속성 | Node.js + npm | 순수 Python (httpx + aiohttp) |
| 프로세스 | 관리되는 Node 하위 프로세스 | aiohttp 웹훅 서버 |
| 공개 URL 필요 여부 | 아니요 | 예 |
| 계정 차단 위험 | 있음 (비공식 API) | 없음 (공식 지원) |
| 수신 | Node 브리지 폴링 | Meta의 웹훅 POST |
| 발신 | 로컬 브리지 → Baileys | graph.facebook.com으로 HTTPS |
| 그룹 | 완전 지원 | DM만 지원(v1) |
| 24시간 창 | 제한 없음 | 엄격한 규칙 — 이후 템플릿 필요 |
| 음성 메모 (발신) | 네이티브 | ffmpeg 사용 시 네이티브, 아니면 MP3 대체 |
| 읽음 확인 | 아니요 | 예 (파란색 이중 체크 표시) |
| 입력 중 표시 | 아니요 | 예 (응답 시 자동 해제) |
| 인터랙티브 버튼 | 텍스트로만 대체 | 네이티브 (`clarify`, 승인, 슬래시 확인) |
| 프로덕션 사용 | 위험함 (Meta가 차단할 수 있음) | 프로덕션 사용을 위해 설계됨 |

개인 프로젝트에 Hermes를 사용하는 대부분의 사용자는 Baileys를 선호합니다. 고객 대상 봇을 운영하는 대부분의 사용자는 Cloud API를 선호합니다.

---

## 함께 보기

- [Meta 공식 WhatsApp Business Cloud API 문서](https://developers.facebook.com/documentation/business-messaging/whatsapp/) — 기반 플랫폼, 요금, App Review, Meta 측 속도 제한에 대한 권위 있는 참고 자료입니다.
- [WhatsApp (Baileys 브리지) 설정](whatsapp.md) — 개인 프로젝트를 위한 대체 통합입니다.
- [메시징 플랫폼 개요](index.md) — 모든 메시징 통합을 한눈에 볼 수 있습니다.
