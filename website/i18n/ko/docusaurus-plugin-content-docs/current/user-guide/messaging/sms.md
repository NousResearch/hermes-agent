---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "Twilio를 통해 Hermes Agent를 SMS 챗봇으로 설정"
---

# SMS 설정(Twilio)

Hermes는 [Twilio](https://www.twilio.com/) API를 통해 SMS에 연결됩니다. Twilio 전화번호로 문자를 보내면 Telegram이나 Discord와 같은 대화 경험을 표준 문자 메시지로 받을 수 있습니다.

:::info 공유 자격 증명
SMS 게이트웨이는 선택 사항인 [전화 통신 스킬](/reference/skills-catalog)과 자격 증명을 공유합니다. 음성 통화나 일회성 SMS를 위해 이미 Twilio를 설정했다면 게이트웨이는 동일한 `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`를 사용합니다.
:::

---

## 사전 요구 사항

- **Twilio 계정** — [twilio.com에서 가입](https://www.twilio.com/try-twilio)(무료 체험 제공)
- **SMS 기능이 있는 Twilio 전화번호**
- **공개적으로 접근 가능한 서버** — SMS가 도착하면 Twilio가 서버로 웹훅을 전송합니다
- **aiohttp** — `cd ~/.hermes/hermes-agent && uv pip install -e ".[sms]"`

---

## 1단계: Twilio 자격 증명 가져오기

1. [Twilio Console](https://console.twilio.com/)로 이동합니다
2. 대시보드에서 **Account SID**와 **Auth Token**을 복사합니다
3. **Phone Numbers → Manage → Active Numbers**로 이동해 E.164 형식의 전화번호를 확인합니다(예: `+15551234567`)

---

## 2단계: Hermes 구성

### 대화형 설정(권장)

```bash
hermes gateway setup
```

플랫폼 목록에서 **SMS (Twilio)**를 선택합니다. 마법사가 자격 증명을 요청합니다.

### 수동 설정

`~/.hermes/.env`에 추가합니다:

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# Security: restrict to specific phone numbers (recommended)
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Optional: set a home channel for cron job delivery
SMS_HOME_CHANNEL=+15559876543
```

---

## 3단계: Twilio 웹훅 구성

Twilio는 수신 메시지를 어디로 보낼지 알아야 합니다. [Twilio Console](https://console.twilio.com/)에서 다음을 수행합니다:

1. **Phone Numbers → Manage → Active Numbers**로 이동합니다
2. 전화번호를 클릭합니다
3. **Messaging → A MESSAGE COMES IN**에서 다음을 설정합니다:
   - **Webhook**: `https://your-server:8080/webhooks/twilio`
   - **HTTP Method**: `POST`

:::tip 웹훅 공개
Hermes를 로컬에서 실행하는 경우 터널을 사용해 웹훅을 공개하세요:

```bash
# Using cloudflared
cloudflared tunnel --url http://localhost:8080

# Using ngrok
ngrok http 8080
```

생성된 공개 URL을 Twilio 웹훅으로 설정합니다.
:::

**`SMS_WEBHOOK_URL`을 Twilio에서 구성한 동일한 URL로 설정하세요.** Twilio 서명 검증에 필요하며, 이 값이 없으면 어댑터가 시작을 거부합니다:

```bash
# Must match the webhook URL in your Twilio Console
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

웹훅 포트 기본값은 `8080`입니다. 다음과 같이 재정의합니다:

```bash
SMS_WEBHOOK_PORT=3000
```

---

## 4단계: 게이트웨이 시작

```bash
hermes gateway
```

다음과 같은 메시지가 표시됩니다:

```
[sms] Twilio webhook server listening on 127.0.0.1:8080, from: +1555***4567
```

`Refusing to start: SMS_WEBHOOK_URL is required`가 표시되면 Twilio Console에서 구성한 공개 URL로 `SMS_WEBHOOK_URL`을 설정하세요(3단계 참조).

Twilio 번호로 문자를 보내면 Hermes가 SMS로 응답합니다.

---

## 환경 변수

| 변수 | 필수 여부 | 설명 |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | 예 | Twilio Account SID(`AC`로 시작) |
| `TWILIO_AUTH_TOKEN` | 예 | Twilio Auth Token(웹훅 서명 검증에도 사용) |
| `TWILIO_PHONE_NUMBER` | 예 | Twilio 전화번호(E.164 형식) |
| `SMS_WEBHOOK_URL` | 예 | Twilio 서명 검증을 위한 공개 URL — Twilio Console의 웹훅 URL과 일치해야 함 |
| `SMS_WEBHOOK_PORT` | 아니요 | 웹훅 리스너 포트(기본값: `8080`) |
| `SMS_WEBHOOK_HOST` | 아니요 | 웹훅 바인드 주소(기본값: `127.0.0.1`) |
| `SMS_INSECURE_NO_SIGNATURE` | 아니요 | 서명 검증을 비활성화하려면 `true`로 설정(로컬 개발 전용 — **프로덕션에서는 사용하지 마세요**) |
| `SMS_ALLOWED_USERS` | 아니요 | 채팅을 허용할 E.164 전화번호(쉼표로 구분) |
| `SMS_ALLOW_ALL_USERS` | 아니요 | 누구나 허용하려면 `true`로 설정(권장하지 않음) |
| `SMS_HOME_CHANNEL` | 아니요 | cron 작업/알림을 전달할 전화번호 |
| `SMS_HOME_CHANNEL_NAME` | 아니요 | 홈 채널 표시 이름(기본값: `Home`) |

---

## SMS 전용 동작

- **일반 텍스트만 지원** — SMS에서 Markdown이 리터럴 문자로 표시되므로 자동으로 제거됩니다
- **1600자 제한** — 더 긴 응답은 자연스러운 경계(줄바꿈 우선, 그다음 공백)에서 여러 메시지로 나뉩니다
- **에코 방지** — 루프를 방지하기 위해 자신의 Twilio 번호에서 온 메시지는 무시됩니다
- **전화번호 비식별화** — 개인정보 보호를 위해 로그에서 전화번호가 비식별화됩니다

---

## 보안

### 웹훅 서명 검증

Hermes는 `X-Twilio-Signature` 헤더를 검증(HMAC-SHA1)해 수신 웹훅이 실제로 Twilio에서 온 것인지 확인합니다. 이를 통해 공격자가 위조 메시지를 주입하는 것을 방지합니다.

**`SMS_WEBHOOK_URL`은 필수입니다.** Twilio Console에서 구성한 공개 URL로 설정하세요. 이 값이 없으면 어댑터가 시작을 거부합니다.

공개 URL이 없는 로컬 개발 환경에서는 검증을 비활성화할 수 있습니다:

```bash
# Local dev only — NOT for production
SMS_INSECURE_NO_SIGNATURE=true
```

### 사용자 허용 목록

**게이트웨이는 기본적으로 모든 사용자를 거부합니다.** 허용 목록을 구성하세요:

```bash
# Recommended: restrict to specific phone numbers
SMS_ALLOWED_USERS=+15559876543,+15551112222

# Or allow all (NOT recommended for bots with terminal access)
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS에는 기본 제공 암호화가 없습니다. 보안 영향을 이해하지 못한 상태에서는 민감한 작업에 SMS를 사용하지 마세요. 민감한 사용 사례에는 Signal 또는 Telegram을 권장합니다.
:::

---

## 문제 해결

### 메시지가 도착하지 않음

1. Twilio 웹훅 URL이 정확하고 공개적으로 접근 가능한지 확인합니다
2. `TWILIO_ACCOUNT_SID`와 `TWILIO_AUTH_TOKEN`이 올바른지 확인합니다
3. Twilio Console → **Monitor → Logs → Messaging**에서 전송 오류를 확인합니다
4. 전화번호가 `SMS_ALLOWED_USERS`에 포함되어 있는지 확인합니다(또는 `SMS_ALLOW_ALL_USERS=true`)

### 답장이 전송되지 않음

1. `TWILIO_PHONE_NUMBER`가 올바르게 설정되었는지 확인합니다(`+`가 포함된 E.164 형식)
2. Twilio 계정에 SMS 기능이 있는 번호가 있는지 확인합니다
3. Hermes 게이트웨이 로그에서 Twilio API 오류를 확인합니다

### 웹훅 포트 충돌

포트 8080이 이미 사용 중이면 변경합니다:

```bash
SMS_WEBHOOK_PORT=3001
```

Twilio Console의 웹훅 URL도 일치하도록 업데이트합니다.
