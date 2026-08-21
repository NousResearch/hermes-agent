---
sidebar_position: 7
title: "이메일"
description: "IMAP/SMTP를 통해 Hermes Agent를 이메일 어시스턴트로 설정"
---

# 이메일 설정

Hermes는 표준 IMAP 및 SMTP 프로토콜을 사용해 이메일을 수신하고 답장할 수 있습니다. 에이전트의 주소로 이메일을 보내면 별도의 클라이언트나 봇 API 없이 같은 스레드로 답장합니다. Gmail, Outlook, Yahoo, Fastmail 또는 IMAP/SMTP를 지원하는 모든 제공업체에서 작동합니다.

:::info 게이트웨이 어댑터 전용: 외부 종속성 없음
이 페이지에서는 Python의 기본 제공 `imaplib`, `smtplib`, `email` 모듈을 사용하는 이메일 게이트웨이 어댑터를 다룹니다. 이 게이트웨이 경로에는 추가 패키지나 외부 서비스가 필요하지 않습니다.
:::

이는 번들로 제공되는 [Himalaya 이메일 스킬](/docs/user-guide/skills/bundled/email/email-himalaya)과는 별개입니다. Himalaya 이메일 스킬을 사용하면 터미널 명령으로 이메일을 관리할 수 있으며, 외부 `himalaya` CLI와 Himalaya 설정 파일이 필요합니다.

| 사용 사례 | 구성할 항목 | 외부 종속성 |
|---|---|---|
| 사람들이 Hermes 에이전트에 이메일을 보내고 답장을 받도록 하기 | 이 페이지의 이메일 게이트웨이 어댑터 | IMAP/SMTP 이메일 계정 외에는 없음 |
| 에이전트가 터미널 도구에서 메일함 메시지를 검사하고, 작성하고, 이동하고, 관리하도록 하기 | Himalaya 이메일 스킬 | `himalaya` CLI 및 `~/.config/himalaya/config.toml` |

---

## 사전 요구 사항

- **Hermes 에이전트 전용 이메일 계정**(개인 이메일은 사용하지 마세요)
- 이메일 계정에서 **IMAP 활성화**
- Gmail 또는 2FA를 사용하는 다른 제공업체의 경우 **앱 비밀번호**

### Gmail 설정

1. Google 계정에서 2단계 인증 활성화
2. [앱 비밀번호](https://myaccount.google.com/apppasswords)로 이동
3. 새 앱 비밀번호 생성("Mail" 또는 "Other" 선택)
4. 16자리 비밀번호를 복사 — 일반 비밀번호 대신 사용합니다

### Outlook / Microsoft 365

1. [보안 설정](https://account.microsoft.com/security)으로 이동
2. 아직 활성화하지 않았다면 2FA 활성화
3. "Additional security options"에서 앱 비밀번호 생성
4. IMAP 호스트: `outlook.office365.com`, SMTP 호스트: `smtp.office365.com`

### 기타 제공업체

대부분의 이메일 제공업체는 IMAP/SMTP를 지원합니다. 다음 항목은 제공업체의 문서에서 확인하세요.
- IMAP 호스트 및 포트(일반적으로 SSL을 사용하는 포트 993)
- SMTP 호스트 및 포트(일반적으로 STARTTLS를 사용하는 포트 587)
- 앱 비밀번호가 필요한지 여부

---

## 1단계: Hermes 구성

가장 쉬운 방법은 다음과 같습니다.

```bash
hermes gateway setup
```

플랫폼 메뉴에서 **Email**을 선택합니다. 마법사가 이메일 주소, 비밀번호, IMAP/SMTP 호스트 및 허용된 발신자를 묻습니다.

### 수동 구성

`~/.hermes/.env`에 다음을 추가합니다.

```bash
# Required
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # App password (not your regular password)
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# Security (recommended)
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# Optional
EMAIL_IMAP_PORT=993                    # Default: 993 (IMAP SSL)
EMAIL_SMTP_PORT=587                    # Default: 587 (SMTP STARTTLS)
EMAIL_POLL_INTERVAL=15                 # Seconds between inbox checks (default: 15)
EMAIL_HOME_ADDRESS=your@email.com      # Default delivery target for cron jobs
```

---

## 2단계: 게이트웨이 시작

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

시작할 때 어댑터는 다음 작업을 수행합니다.
1. IMAP 및 SMTP 연결 테스트
2. 기존 받은 편지함 메시지를 모두 "읽음"으로 표시(새 이메일만 처리)
3. 새 메시지 폴링 시작

---

## 작동 방식

### 메시지 수신

어댑터는 구성 가능한 간격(기본값: 15초)으로 IMAP 받은 편지함에서 읽지 않은 메시지를 폴링합니다. 새 이메일마다 다음이 적용됩니다.

- **제목 줄**이 컨텍스트에 포함됩니다(예: `[Subject: Deploy to production]`)
- **답장 이메일**(제목이 `Re:`로 시작)은 제목 접두사를 건너뜁니다 — 스레드 컨텍스트가 이미 설정되어 있기 때문입니다
- **첨부 파일**은 로컬에 캐시됩니다.
  - 이미지(JPEG, PNG, GIF, WebP) → 비전 도구에서 사용 가능
  - 문서(PDF, ZIP 등) → 파일 액세스에서 사용 가능
- **HTML 전용 이메일**은 일반 텍스트 추출을 위해 태그가 제거됩니다
- **자기 자신에게 보낸 메시지**는 답장 루프를 방지하기 위해 필터링됩니다
- **자동화된/noreply 발신자**는 조용히 무시됩니다 — `noreply@`, `mailer-daemon@`, `bounce@`, `no-reply@` 및 `Auto-Submitted`, `Precedence: bulk`, `List-Unsubscribe` 헤더가 있는 이메일

### 답장 전송

답장은 올바른 이메일 스레딩을 적용해 SMTP로 전송됩니다.

- **In-Reply-To** 및 **References** 헤더가 스레드를 유지합니다
- **제목 줄**은 `Re:` 접두사와 함께 유지됩니다(중복된 `Re: Re:` 없음)
- **Message-ID**는 에이전트의 도메인으로 생성됩니다
- 응답은 일반 텍스트(UTF-8)로 전송됩니다

### 파일 첨부

에이전트는 답장에 파일 첨부를 포함할 수 있습니다. 응답에 `MEDIA:/path/to/file`을 포함하면 파일이 발신 이메일에 첨부됩니다.

### 첨부 파일 건너뛰기

모든 수신 첨부 파일을 무시하려면(멀웨어 방지 또는 대역폭 절약) `config.yaml`에 다음을 추가합니다.

```yaml
platforms:
  email:
    skip_attachments: true
```

활성화하면 페이로드 디코딩 전에 첨부 파일과 인라인 파트를 건너뜁니다. 이메일 본문 텍스트는 정상적으로 처리됩니다.

---

## 액세스 제어

이메일 액세스는 기본적으로 채팅형 플랫폼보다 더 엄격합니다.

1. **`EMAIL_ALLOWED_USERS`가 설정됨** → 해당 주소에서 온 이메일만 처리
2. **허용 목록이 설정되지 않음** → 알 수 없는 발신자는 조용히 무시
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 모든 발신자 허용(주의해서 사용)
4. **`platforms.email.unauthorized_dm_behavior: pair`** → 알 수 없는 발신자에게 페어링 코드 전송

:::warning
**전용 받은 편지함을 사용하고 일반적인 운영에서는 `EMAIL_ALLOWED_USERS`를 구성하세요.** 공유 받은 편지함에는 관련 없는 읽지 않은 메시지가 포함되는 경우가 많으므로 이메일 페어링은 선택 사항이며, Hermes는 기본적으로 해당 연락처에 답장하지 않아야 합니다.
:::

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| 시작 시 **"IMAP connection failed"** | `EMAIL_IMAP_HOST` 및 `EMAIL_IMAP_PORT`를 확인합니다. 계정에서 IMAP이 활성화되어 있는지 확인합니다. Gmail의 경우 Settings → Forwarding and POP/IMAP에서 활성화합니다. |
| 시작 시 **"SMTP connection failed"** | `EMAIL_SMTP_HOST` 및 `EMAIL_SMTP_PORT`를 확인합니다. 비밀번호가 올바른지 확인합니다(Gmail에는 앱 비밀번호 사용). |
| **메시지를 받지 못함** | `EMAIL_ALLOWED_USERS`에 발신자의 이메일이 포함되어 있는지 확인합니다. 일부 제공업체는 자동 답장을 스팸으로 분류하므로 스팸 폴더를 확인합니다. |
| **"Authentication failed"** | Gmail에서는 일반 비밀번호가 아니라 앱 비밀번호를 사용해야 합니다. 먼저 2FA가 활성화되어 있는지 확인합니다. |
| **중복 답장** | 실행 중인 게이트웨이 인스턴스가 하나뿐인지 확인합니다. `hermes gateway status`를 확인합니다. |
| **응답이 느림** | 기본 폴링 간격은 15초입니다. 더 빠른 응답을 위해 `EMAIL_POLL_INTERVAL=5`로 줄일 수 있습니다(단, IMAP 연결이 늘어남). |
| **답장이 스레드로 묶이지 않음** | 어댑터는 In-Reply-To 헤더를 사용합니다. 일부 이메일 클라이언트(특히 웹 기반)는 자동 메시지를 올바르게 스레드로 묶지 못할 수 있습니다. |

---

## 보안

:::warning
**전용 이메일 계정을 사용하세요.** 개인 이메일은 사용하지 마세요 — 에이전트는 비밀번호를 `.env`에 저장하며 IMAP을 통해 받은 편지함에 대한 전체 액세스 권한을 가집니다.
:::

- 기본 비밀번호 대신 **앱 비밀번호** 사용(Gmail에서 2FA를 사용하는 경우 필수)
- 에이전트와 상호작용할 수 있는 사용자를 제한하려면 `EMAIL_ALLOWED_USERS` 설정
- 비밀번호는 `~/.hermes/.env`에 저장되므로 이 파일을 보호하세요(`chmod 600`)
- IMAP은 기본적으로 SSL(포트 993)을 사용하고 SMTP는 STARTTLS(포트 587)를 사용하므로 연결이 암호화됩니다

---

## 환경 변수 참고

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | 예 | — | 에이전트의 이메일 주소 |
| `EMAIL_PASSWORD` | 예 | — | 이메일 비밀번호 또는 앱 비밀번호 |
| `EMAIL_IMAP_HOST` | 예 | — | IMAP 서버 호스트(예: `imap.gmail.com`) |
| `EMAIL_SMTP_HOST` | 예 | — | SMTP 서버 호스트(예: `smtp.gmail.com`) |
| `EMAIL_IMAP_PORT` | 아니요 | `993` | IMAP 서버 포트 |
| `EMAIL_SMTP_PORT` | 아니요 | `587` | SMTP 서버 포트 |
| `EMAIL_POLL_INTERVAL` | 아니요 | `15` | 받은 편지함 확인 간격(초) |
| `EMAIL_ALLOWED_USERS` | 아니요 | — | 쉼표로 구분한 허용 발신자 주소 |
| `EMAIL_HOME_ADDRESS` | 아니요 | — | cron 작업의 기본 전달 대상 |
| `EMAIL_ALLOW_ALL_USERS` | 아니요 | `false` | 모든 발신자 허용(권장하지 않음) |
