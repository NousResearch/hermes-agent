---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "Twilio를 통해 Hermes Agent를 SMS 챗봇으로 설정하기"
---

# SMS 설정 (Twilio)

Hermes는 [Twilio](https://www.twilio.com/) API를 통해 SMS와 연결됩니다. 사람들이 여러분의 Twilio 전화번호로 문자를 보내면 AI 응답을 받게 됩니다 — Telegram이나 Discord와 동일한 대화 경험을 표준 텍스트 메시지를 통해 제공합니다.

:::info 자격 증명 공유
SMS 게이트웨이는 선택적 [전화 기능 스킬(telephony skill)](/reference/skills-catalog)과 자격 증명을 공유합니다. 음성 통화 또는 일회성 SMS를 위해 이미 Twilio를 설정했다면 게이트웨이는 동일한 `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`로 작동합니다.
:::

---

## 전제 조건

- **Twilio 계정** — [twilio.com에서 가입](https://www.twilio.com/try-twilio) (무료 평가판 제공)
- SMS 기능이 있는 **Twilio 전화번호**
- **공개적으로 접근 가능한 서버** — SMS가 도착하면 Twilio가 서버로 웹훅을 보냅니다
- **aiohttp** — `pip install 'hermes-agent[sms]'`

---

## 1단계: Twilio 자격 증명 가져오기

1. [Twilio Console](https://console.twilio.com/)로 이동
2. 대시보드에서 **Account SID**와 **Auth Token**을 복사
3. **Phone Numbers → Manage → Active Numbers**로 이동 — E.164 형식의 전화번호(예: `+15551234567`)를 기록해 둡니다.

---

## 2단계: Hermes 구성

### 대화형 설정 (권장)

```bash
hermes gateway setup
```

플랫폼 목록에서 **SMS (Twilio)**를 선택합니다. 마법사가 자격 증명을 묻는 프롬프트를 표시합니다.

### 수동 설정

`~/.hermes/.env`에 추가:

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# 보안: 특정 전화번호로 제한 (권장)
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 선택 사항: 크론 작업 전달을 위한 홈 채널 설정
SMS_HOME_CHANNEL=+15559876543
```

---

## 3단계: Twilio 웹훅 구성

Twilio는 수신 메시지를 어디로 보낼지 알아야 합니다. [Twilio Console](https://console.twilio.com/)에서:

1. **Phone Numbers → Manage → Active Numbers**로 이동
2. 여러분의 전화번호 클릭
3. **Messaging → A MESSAGE COMES IN** 아래에서 다음 설정:
   - **Webhook**: `https://your-server:8080/webhooks/twilio`
   - **HTTP Method**: `POST`

:::tip 웹훅 노출하기
Hermes를 로컬에서 실행하는 경우, 터널을 사용하여 웹훅을 노출하세요:

```bash
# cloudflared 사용
cloudflared tunnel --url http://localhost:8080

# ngrok 사용
ngrok http 8080
```

결과로 얻은 공개 URL을 Twilio 웹훅으로 설정하세요.
:::

**`SMS_WEBHOOK_URL`을 Twilio에 설정한 URL과 동일하게 설정하세요.** 이는 Twilio 서명 검증을 위해 필수입니다 — 어댑터는 이 설정 없이는 시작을 거부합니다:

```bash
# Twilio Console의 웹훅 URL과 반드시 일치해야 함
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

웹훅 포트 기본값은 `8080`입니다. 다음과 같이 재정의할 수 있습니다:

```bash
SMS_WEBHOOK_PORT=3000
```

---

## 4단계: 게이트웨이 시작

```bash
hermes gateway
```

다음과 같은 메시지가 표시되어야 합니다:

```
[sms] Twilio webhook server listening on 127.0.0.1:8080, from: +1555***4567
```

`Refusing to start: SMS_WEBHOOK_URL is required`가 표시된다면 `SMS_WEBHOOK_URL`을 Twilio Console에 설정한 공개 URL로 설정하세요(3단계 참조).

Twilio 번호로 문자를 보내보세요 — Hermes가 SMS를 통해 응답할 것입니다.

---

## 환경 변수

| 변수 | 필수 | 설명 |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | 예 | Twilio 계정 SID (`AC`로 시작) |
| `TWILIO_AUTH_TOKEN` | 예 | Twilio Auth Token (웹훅 서명 검증에도 사용됨) |
| `TWILIO_PHONE_NUMBER` | 예 | Twilio 전화번호 (E.164 형식) |
| `SMS_WEBHOOK_URL` | 예 | Twilio 서명 검증을 위한 공개 URL — Twilio Console의 웹훅 URL과 반드시 일치해야 함 |
| `SMS_WEBHOOK_PORT` | 아니오 | 웹훅 리스너 포트 (기본값: `8080`) |
| `SMS_WEBHOOK_HOST` | 아니오 | 웹훅 바인드 주소 (기본값: `0.0.0.0`) |
| `SMS_INSECURE_NO_SIGNATURE` | 아니오 | 서명 검증 비활성화를 위해 `true` 설정 (로컬 개발 전용 — **프로덕션용 아님**) |
| `SMS_ALLOWED_USERS` | 아니오 | 채팅이 허용된 쉼표로 구분된 E.164 전화번호 |
| `SMS_ALLOW_ALL_USERS` | 아니오 | 누구나 허용하려면 `true`로 설정 (권장하지 않음) |
| `SMS_HOME_CHANNEL` | 아니오 | 크론 작업 / 알림 전달을 위한 전화번호 |
| `SMS_HOME_CHANNEL_NAME` | 아니오 | 홈 채널의 표시 이름 (기본값: `Home`) |

---

## SMS 고유의 동작

- **일반 텍스트 전용** — SMS는 이를 문자 그대로 렌더링하기 때문에 마크다운은 자동으로 제거됩니다
- **1600자 제한** — 더 긴 응답은 자연스러운 경계(개행, 그 다음 공백)에서 여러 메시지로 분할됩니다
- **에코 방지** — 무한 루프를 방지하기 위해 여러분의 Twilio 번호에서 보낸 메시지는 무시됩니다
- **전화번호 교정** — 개인정보 보호를 위해 로그에서 전화번호의 일부를 숨깁니다

---

## 보안

### 웹훅 서명 검증

Hermes는 `X-Twilio-Signature` 헤더(HMAC-SHA1)를 검증하여 인바운드 웹훅이 실제로 Twilio에서 보낸 것인지 확인합니다. 이것은 공격자가 조작된 메시지를 주입하는 것을 방지합니다.

**`SMS_WEBHOOK_URL`은 필수입니다.** Twilio Console에 설정한 공개 URL로 이를 설정하세요. 어댑터는 이 설정 없이는 시작을 거부합니다.

공개 URL 없이 로컬 개발을 수행할 경우, 검증을 비활성화할 수 있습니다:

```bash
# 로컬 개발 전용 — 프로덕션용 아님
SMS_INSECURE_NO_SIGNATURE=true
```

### 사용자 허용 목록

**게이트웨이는 기본적으로 모든 사용자를 거부합니다.** 허용 목록을 구성하세요:

```bash
# 권장: 특정 전화번호로 제한
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 또는 모두 허용 (터미널 접근 권한이 있는 봇에는 권장하지 않음)
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS는 기본적으로 암호화가 지원되지 않습니다. 보안의 위험성을 명확히 이해하지 못했다면 민감한 작업에 SMS를 사용하지 마십시오. 민감한 사용 사례의 경우 Signal이나 Telegram을 선호하세요.
:::

---

## 문제 해결

### 메시지가 도착하지 않음

1. Twilio 웹훅 URL이 올바르고 외부에 공개되어 접근 가능한지 확인하세요
2. `TWILIO_ACCOUNT_SID`와 `TWILIO_AUTH_TOKEN`이 맞는지 확인하세요
3. Twilio Console → **Monitor → Logs → Messaging**에서 전송 오류를 확인하세요
4. 여러분의 전화번호가 `SMS_ALLOWED_USERS`에 포함되어 있는지(또는 `SMS_ALLOW_ALL_USERS=true`인지) 확인하세요

### 답글이 발송되지 않음

1. `TWILIO_PHONE_NUMBER`가 올바르게 설정되었는지 확인하세요(`+`가 포함된 E.164 형식)
2. Twilio 계정에 SMS가 지원되는 번호가 있는지 확인하세요
3. Twilio API 오류는 Hermes 게이트웨이 로그를 확인하세요

### 웹훅 포트 충돌

8080 포트가 이미 사용 중이라면, 이를 변경하세요:

```bash
SMS_WEBHOOK_PORT=3001
```

Twilio Console의 웹훅 URL도 일치하도록 업데이트하세요.
