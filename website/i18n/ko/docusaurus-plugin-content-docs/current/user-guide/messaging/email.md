---
sidebar_position: 7
title: "이메일 (Email)"
description: "IMAP/SMTP를 통해 Hermes Agent를 이메일 비서로 설정하기"
---

# 이메일 설정

Hermes는 표준 IMAP 및 SMTP 프로토콜을 사용하여 이메일을 수신하고 회신할 수 있습니다. 에이전트의 이메일 주소로 메일을 보내면 동일한 스레드에서 답장합니다. 특별한 클라이언트나 봇 API가 필요하지 않습니다. Gmail, Outlook, Yahoo, Fastmail 또는 IMAP/SMTP를 지원하는 모든 제공업체와 연동됩니다.

:::info 게이트웨이 어댑터 전용: 외부 종속성 없음
이 페이지는 Python에 내장된 `imaplib`, `smtplib`, `email` 모듈을 사용하는 이메일 게이트웨이 어댑터를 다룹니다. 이 게이트웨이 경로를 위해 추가 패키지나 외부 서비스는 필요하지 않습니다.
:::

이 설정은 에이전트가 터미널 명령을 통해 이메일을 관리할 수 있도록 해주는 번들로 제공된 [Himalaya 이메일 스킬](/docs/user-guide/skills/bundled/email/email-himalaya)과는 별개입니다. Himalaya 스킬은 외부의 `himalaya` CLI와 Himalaya 구성 파일을 요구합니다.

| 사용 사례 | 설정 대상 | 외부 종속성 |
|---|---|---|
| 사람들이 Hermes 에이전트에게 이메일을 보내고 회신을 받을 수 있도록 허용 | 이 페이지의 이메일 게이트웨이 어댑터 | IMAP/SMTP 이메일 계정 외에는 없음 |
| 에이전트가 터미널 도구를 통해 편지함 메시지를 검사, 작성, 이동 및 관리할 수 있도록 허용 | Himalaya 이메일 스킬 | `himalaya` CLI 및 `~/.config/himalaya/config.toml` |

---

## 사전 요구 사항

- Hermes 에이전트를 위한 **전용 이메일 계정** (개인 이메일은 사용하지 마세요)
- 이메일 계정에서 **IMAP 활성화**
- 2단계 인증(2FA)을 사용하는 Gmail이나 다른 제공업체의 경우 **앱 비밀번호(App password)**

### Gmail 설정

1. Google 계정에서 2단계 인증을 활성화합니다.
2. [앱 비밀번호](https://myaccount.google.com/apppasswords)로 이동합니다.
3. 새 앱 비밀번호를 생성합니다 ("메일" 또는 "기타" 선택).
4. 생성된 16자리 비밀번호를 복사합니다. 일반 비밀번호 대신 이것을 사용하게 됩니다.

### Outlook / Microsoft 365

1. [보안 설정](https://account.microsoft.com/security)으로 이동합니다.
2. 2단계 인증이 활성화되어 있지 않다면 활성화합니다.
3. "추가 보안 옵션" 아래에서 앱 비밀번호를 생성합니다.
4. IMAP 호스트: `outlook.office365.com`, SMTP 호스트: `smtp.office365.com`

### 기타 제공업체

대부분의 이메일 제공업체는 IMAP/SMTP를 지원합니다. 제공업체의 문서에서 다음 사항을 확인하세요:
- IMAP 호스트 및 포트 (일반적으로 SSL이 적용된 포트 993)
- SMTP 호스트 및 포트 (일반적으로 STARTTLS가 적용된 포트 587)
- 앱 비밀번호 필요 여부

---

## 1단계: Hermes 구성하기

가장 쉬운 방법은 다음과 같습니다:

```bash
hermes gateway setup
```

플랫폼 메뉴에서 **Email**을 선택하세요. 마법사가 이메일 주소, 비밀번호, IMAP/SMTP 호스트, 그리고 허용된 발신자를 입력하라는 프롬프트를 띄웁니다.

### 수동 구성

`~/.hermes/.env` 파일에 다음 내용을 추가합니다:

```bash
# 필수
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # 앱 비밀번호 (일반 비밀번호가 아님)
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# 보안 (권장)
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# 선택 사항
EMAIL_IMAP_PORT=993                    # 기본값: 993 (IMAP SSL)
EMAIL_SMTP_PORT=587                    # 기본값: 587 (SMTP STARTTLS)
EMAIL_POLL_INTERVAL=15                 # 받은 편지함 확인 간격(초) (기본값: 15)
EMAIL_HOME_ADDRESS=your@email.com      # cron 작업의 기본 전송 대상
```

---

## 2단계: 게이트웨이 시작하기

```bash
hermes gateway              # 포그라운드에서 실행
hermes gateway install      # 사용자 서비스로 설치
sudo hermes gateway install --system   # Linux 전용: 부팅 시 실행되는 시스템 서비스
```

시작 시, 어댑터는 다음을 수행합니다:
1. IMAP 및 SMTP 연결을 테스트합니다.
2. 기존의 수신함 메시지를 모두 "읽음" 상태로 표시합니다 (새 이메일만 처리).
3. 새 메시지에 대한 폴링을 시작합니다.

---

## 작동 방식

### 메시지 수신

어댑터는 설정된 간격(기본값: 15초)마다 IMAP 받은 편지함의 UNSEEN(읽지 않은) 메시지를 폴링합니다. 각각의 새 이메일에 대해:

- **제목(Subject line)**은 컨텍스트에 포함됩니다 (예: `[Subject: Deploy to production]`).
- **답장 이메일** (`Re:`로 시작하는 제목)은 제목 접두사를 생략합니다 — 스레드 컨텍스트가 이미 설정되어 있습니다.
- **첨부 파일**은 로컬에 캐시됩니다:
  - 이미지 (JPEG, PNG, GIF, WebP) → 비전 툴에서 사용 가능
  - 문서 (PDF, ZIP 등) → 파일 접근용으로 사용 가능
- **HTML 전용 이메일**은 일반 텍스트 추출을 위해 태그가 제거됩니다.
- 응답 루프를 방지하기 위해 **자신이 보낸 메시지(Self-messages)**는 필터링됩니다.
- **자동 발송/수신 불가 발신자(Automated/noreply senders)**는 조용히 무시됩니다 — `noreply@`, `mailer-daemon@`, `bounce@`, `no-reply@`와 헤더에 `Auto-Submitted`, `Precedence: bulk` 또는 `List-Unsubscribe`가 포함된 이메일.

### 회신 보내기

답장은 적절한 이메일 스레드와 함께 SMTP를 통해 전송됩니다:

- **In-Reply-To** 및 **References** 헤더가 스레드를 유지합니다.
- **제목(Subject line)**은 `Re:` 접두사와 함께 유지됩니다 (`Re: Re:` 중복 방지).
- 에이전트의 도메인으로 생성된 **Message-ID**가 포함됩니다.
- 응답은 일반 텍스트(UTF-8)로 전송됩니다.

### 파일 첨부

에이전트는 답장에 파일을 첨부하여 보낼 수 있습니다. 응답에 `MEDIA:/path/to/file`을 포함하면 전송되는 이메일에 파일이 첨부됩니다.

### 첨부 파일 건너뛰기

악성 소프트웨어 방지나 대역폭 절약을 위해 들어오는 모든 첨부 파일을 무시하려면, `config.yaml`에 다음을 추가하세요:

```yaml
platforms:
  email:
    skip_attachments: true
```

이 옵션을 활성화하면 페이로드 디코딩 전에 첨부 파일과 인라인(inline) 파트를 모두 건너뜁니다. 단, 이메일 본문 텍스트는 정상적으로 처리됩니다.

---

## 접근 제어

이메일 접근 권한은 다른 모든 Hermes 플랫폼과 동일한 패턴을 따릅니다:

1. **`EMAIL_ALLOWED_USERS`가 설정된 경우** → 해당 주소의 이메일만 처리됩니다.
2. **허용 목록(allowlist)이 설정되지 않은 경우** → 알 수 없는 발신자에게는 페어링 코드가 전송됩니다.
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 모든 발신자가 허용됩니다 (주의해서 사용하세요).

:::warning
**항상 `EMAIL_ALLOWED_USERS`를 구성하세요.** 이 옵션이 설정되어 있지 않으면 에이전트의 이메일 주소를 아는 사람은 누구나 명령을 보낼 수 있습니다. 에이전트는 기본적으로 터미널 액세스 권한을 가집니다.
:::

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| 시작 시 **"IMAP connection failed"** 발생 | `EMAIL_IMAP_HOST`와 `EMAIL_IMAP_PORT`를 확인하세요. 계정에서 IMAP이 활성화되어 있는지 확인하세요. Gmail의 경우 설정 → 전달 및 POP/IMAP에서 활성화합니다. |
| 시작 시 **"SMTP connection failed"** 발생 | `EMAIL_SMTP_HOST`와 `EMAIL_SMTP_PORT`를 확인하세요. 비밀번호가 올바른지 확인하세요 (Gmail의 경우 앱 비밀번호 사용). |
| **메시지를 받지 못함** | `EMAIL_ALLOWED_USERS`에 발신자의 이메일이 포함되어 있는지 확인하세요. 스팸 폴더를 확인하세요 — 일부 제공업체는 자동 응답을 스팸으로 분류합니다. |
| **"Authentication failed"** | Gmail의 경우 일반 비밀번호가 아닌 앱 비밀번호를 사용해야 합니다. 먼저 2단계 인증이 활성화되어 있는지 확인하세요. |
| **중복된 답장** | 게이트웨이 인스턴스가 단 하나만 실행 중인지 확인하세요. `hermes gateway status`로 확인할 수 있습니다. |
| **느린 응답 속도** | 기본 폴링 간격은 15초입니다. 더 빠른 응답을 원한다면 `EMAIL_POLL_INTERVAL=5`로 줄이세요 (단, IMAP 연결 빈도가 증가합니다). |
| **답장 스레딩 오류** | 어댑터는 In-Reply-To 헤더를 사용합니다. 일부 이메일 클라이언트(특히 웹 기반)에서는 자동 생성 메시지를 스레드로 묶지 못할 수 있습니다. |

---

## 보안

:::warning
**전용 이메일 계정을 사용하세요.** 개인 이메일은 사용하지 마세요. 에이전트는 비밀번호를 `.env`에 저장하고 IMAP을 통해 수신함에 완전히 접근할 수 있습니다.
:::

- 메인 비밀번호 대신 **앱 비밀번호**를 사용하세요 (2FA가 적용된 Gmail의 경우 필수).
- `EMAIL_ALLOWED_USERS`를 설정하여 에이전트와 상호 작용할 수 있는 사람을 제한하세요.
- 비밀번호는 `~/.hermes/.env`에 저장됩니다 — 이 파일을 보호하세요 (`chmod 600`).
- IMAP은 기본적으로 SSL(포트 993)을 사용하고 SMTP는 STARTTLS(포트 587)를 사용합니다 — 연결은 암호화됩니다.

---

## 환경 변수 참조

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | Yes | — | 에이전트의 이메일 주소 |
| `EMAIL_PASSWORD` | Yes | — | 이메일 비밀번호 또는 앱 비밀번호 |
| `EMAIL_IMAP_HOST` | Yes | — | IMAP 서버 호스트 (예: `imap.gmail.com`) |
| `EMAIL_SMTP_HOST` | Yes | — | SMTP 서버 호스트 (예: `smtp.gmail.com`) |
| `EMAIL_IMAP_PORT` | No | `993` | IMAP 서버 포트 |
| `EMAIL_SMTP_PORT` | No | `587` | SMTP 서버 포트 |
| `EMAIL_POLL_INTERVAL` | No | `15` | 수신함 확인 간격(초) |
| `EMAIL_ALLOWED_USERS` | No | — | 쉼표로 구분된 허용 발신자 주소 |
| `EMAIL_HOME_ADDRESS` | No | — | cron 작업의 기본 전송 대상 |
| `EMAIL_ALLOW_ALL_USERS` | No | `false` | 모든 발신자 허용 (권장하지 않음) |
