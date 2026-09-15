---
sidebar_position: 14
title: "WeCom(기업용 WeChat)"
description: "AI Bot WebSocket 게이트웨이를 통해 Hermes Agent를 WeCom에 연결"
---

# WeCom(기업용 WeChat)

Hermes를 Tencent의 기업용 메시징 플랫폼인 [WeCom](https://work.weixin.qq.com/)(企业微信)에 연결합니다. 이 어댑터는 실시간 양방향 통신을 위해 WeCom의 AI Bot WebSocket 게이트웨이를 사용하므로, 공개 엔드포인트나 웹훅이 필요하지 않습니다.

참고: 수신 웹훅 설정은 [WeCom 콜백](./wecom-callback.md)을 참조하세요.

## 사전 요구 사항

- WeCom 조직 계정
- WeCom 관리자 콘솔에서 생성한 AI Bot
- 봇 자격 증명 페이지의 Bot ID와 Secret
- Python 패키지: `aiohttp` 및 `httpx`

## 설정

### 1단계: AI Bot 생성

#### 권장: 스캔하여 생성(명령 하나)

```bash
hermes gateway setup
```

**WeCom**을 선택하고 WeCom 모바일 앱으로 QR 코드를 스캔하세요. Hermes가 올바른 권한으로 봇 애플리케이션을 자동으로 생성하고 자격 증명을 저장합니다.

설정 마법사는 다음을 수행합니다.
1. 터미널에 QR 코드를 표시합니다.
2. WeCom 모바일 앱으로 스캔할 때까지 기다립니다.
3. Bot ID와 Secret을 자동으로 가져옵니다.
4. 액세스 제어 구성을 안내합니다.

#### 대안: 수동 설정

스캔하여 생성하는 기능을 사용할 수 없으면 마법사가 수동 입력으로 전환됩니다.

1. [WeCom 관리자 콘솔](https://work.weixin.qq.com/wework_admin/frame)에 로그인합니다.
2. **애플리케이션** → **애플리케이션 생성** → **AI Bot**으로 이동합니다.
3. 봇 이름과 설명을 구성합니다.
4. 자격 증명 페이지에서 **Bot ID**와 **Secret**을 복사합니다.
5. `hermes gateway setup`을 실행하고 **WeCom**을 선택한 다음, 안내가 나오면 자격 증명을 입력합니다.

:::warning
Bot Secret을 안전하게 보호하세요. 이 값을 가진 사람은 누구나 봇을 사칭할 수 있습니다.
:::

### 2단계: Hermes 구성

#### 옵션 A: 대화형 설정(권장)

```bash
hermes gateway setup
```

**WeCom**을 선택하고 안내를 따르세요. 마법사가 다음 항목을 안내합니다.
- 봇 자격 증명(QR 스캔 또는 수동 입력)
- 액세스 제어 설정(허용 목록, 페어링 모드 또는 공개 액세스)
- 알림용 홈 채널

#### 옵션 B: 수동 구성

다음 내용을 `~/.hermes/.env`에 추가합니다.

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# Optional: restrict access
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# Optional: home channel for cron/notifications
WECOM_HOME_CHANNEL=chat_id
```

### 3단계: 게이트웨이 시작

```bash
hermes gateway
```

## 기능

- **WebSocket 전송** — 지속적인 연결을 사용하며 공개 엔드포인트가 필요하지 않음
- **DM 및 그룹 메시징** — 구성 가능한 액세스 정책
- **그룹별 발신자 허용 목록** — 각 그룹에서 상호 작용할 수 있는 사용자를 세밀하게 제어
- **미디어 지원** — 이미지, 파일, 음성, 동영상 업로드 및 다운로드
- **AES 암호화 미디어** — 수신 첨부 파일 자동 복호화
- **인용 컨텍스트** — 답장 스레드 보존
- **Markdown 렌더링** — 서식 있는 텍스트 응답
- **답장 상관 관계** — 수신 메시지 컨텍스트에 응답 연결
- **자동 재연결** — 연결이 끊어지면 지수 백오프로 재연결

:::note 스트리밍 및 입력 중 표시
WeCom 어댑터는 각 응답을 하나의 완전한 메시지로 전달합니다. 토큰 단위로 응답을 **스트리밍하지 않으며**, 입력 중 표시도 **보여 주지 않습니다**. 아래의 "답장 상관 관계"는 응답을 수신 요청에 연결하는 기능일 뿐, 실시간 스트리밍이 아닙니다.
:::

## 구성 옵션

`config.yaml`의 `platforms.wecom.extra` 아래에 설정합니다.

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID(필수) |
| `secret` | — | WeCom AI Bot Secret(필수) |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocket 게이트웨이 URL |
| `dm_policy` | `open` | DM 액세스: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `open` | 그룹 액세스: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | DM이 허용된 사용자 ID(`dm_policy=allowlist`인 경우) |
| `group_allow_from` | `[]` | 허용된 그룹 ID(`group_policy=allowlist`인 경우) |
| `groups` | `{}` | 그룹별 구성(아래 참조) |

## 액세스 정책

### DM 정책

봇에 직접 메시지를 보낼 수 있는 사용자를 제어합니다.

| 값 | 동작 |
|-------|----------|
| `open` | 누구나 봇에 DM을 보낼 수 있음(기본값) |
| `allowlist` | `allow_from`의 사용자 ID만 DM을 보낼 수 있음 |
| `disabled` | 모든 DM을 무시함 |
| `pairing` | 페어링 모드(초기 설정용) |

```bash
WECOM_DM_POLICY=allowlist
```

### 그룹 정책

봇이 응답할 그룹을 제어합니다.

| 값 | 동작 |
|-------|----------|
| `open` | 봇이 모든 그룹에서 응답함(기본값) |
| `allowlist` | `group_allow_from`에 나열된 그룹 ID에서만 봇이 응답함 |
| `disabled` | 모든 그룹 메시지를 무시함 |

```bash
WECOM_GROUP_POLICY=allowlist
```

### 그룹별 발신자 허용 목록

세밀하게 제어하려면 특정 그룹 안에서 봇과 상호 작용할 수 있는 사용자를 제한할 수 있습니다. 이는 `config.yaml`에서 구성합니다.

```yaml
platforms:
  wecom:
    enabled: true
    extra:
      bot_id: "your-bot-id"
      secret: "your-secret"
      group_policy: "allowlist"
      group_allow_from:
        - "group_id_1"
        - "group_id_2"
      groups:
        group_id_1:
          allow_from:
            - "user_alice"
            - "user_bob"
        group_id_2:
          allow_from:
            - "user_charlie"
        "*":
          allow_from:
            - "user_admin"
```

**작동 방식:**

1. `group_policy`와 `group_allow_from`이 해당 그룹이 아예 허용되는지 결정합니다.
2. 그룹이 최상위 검사를 통과하면 `groups.<group_id>.allow_from` 목록이 있는 경우 해당 그룹에서 상호 작용할 수 있는 발신자를 추가로 제한합니다.
3. 와일드카드 `"*"` 그룹 항목은 명시적으로 나열되지 않은 그룹의 기본값으로 사용됩니다.
4. 허용 목록 항목은 모든 사용자를 허용하는 `*` 와일드카드를 지원하며, 대소문자를 구분하지 않습니다.
5. 항목에는 선택적으로 `wecom:user:` 또는 `wecom:group:` 접두사 형식을 사용할 수 있습니다. 접두사는 자동으로 제거됩니다.

그룹에 `allow_from`이 구성되지 않으면 해당 그룹이 최상위 정책 검사를 통과하는 것을 전제로 그 그룹의 모든 사용자가 허용됩니다.

## 미디어 지원

### 수신

어댑터는 사용자로부터 미디어 첨부 파일을 받아 에이전트가 처리할 수 있도록 로컬에 캐시합니다.

| 유형 | 처리 방식 |
|------|-----------------|
| **이미지** | 다운로드하여 로컬에 캐시합니다. URL 기반 이미지와 base64 인코딩 이미지를 모두 지원합니다. |
| **파일** | 다운로드하여 캐시합니다. 원본 메시지의 파일 이름을 유지합니다. |
| **음성** | 가능한 경우 음성 메시지 텍스트의 전사를 추출합니다. |
| **혼합 메시지** | WeCom 혼합 유형 메시지(텍스트 + 이미지)를 파싱하고 모든 구성 요소를 추출합니다. |

**인용 메시지:** 인용된(답장된) 메시지의 미디어도 추출되므로 에이전트가 사용자가 답장하는 대상의 컨텍스트를 파악할 수 있습니다.

### AES 암호화 미디어 복호화

WeCom은 일부 수신 미디어 첨부 파일을 AES-256-CBC로 암호화합니다. 어댑터가 이를 자동으로 처리합니다.

- 수신 미디어 항목에 `aeskey` 필드가 포함되면 어댑터가 암호화된 바이트를 다운로드하고 PKCS#7 패딩을 사용하는 AES-256-CBC로 복호화합니다.
- AES 키는 `aeskey` 필드 값을 base64 디코딩한 값이며, 정확히 32바이트여야 합니다.
- IV는 키의 처음 16바이트에서 파생됩니다.
- `cryptography` Python 패키지가 필요합니다(`pip install cryptography`).

구성은 필요하지 않습니다. 암호화된 미디어를 받으면 복호화가 투명하게 수행됩니다.

### 발신

| 메서드 | 전송 내용 | 크기 제한 |
|--------|--------------|------------|
| `send` | Markdown 텍스트 메시지 | 4000자 |
| `send_image` / `send_image_file` | 기본 이미지 메시지 | 10MB |
| `send_document` | 파일 첨부 | 20MB |
| `send_voice` | 음성 메시지(기본 음성은 AMR 형식만 가능) | 2MB |
| `send_video` | 동영상 메시지 | 10MB |

**분할 업로드:** 파일은 512KB 단위로 3단계 프로토콜(init → chunks → finish)을 통해 업로드됩니다. 어댑터가 이를 자동으로 처리합니다.

**자동 하향 전환:** 미디어가 기본 유형의 크기 제한을 초과하지만 절대 파일 제한인 20MB 이내인 경우, 자동으로 일반 파일 첨부로 전송됩니다.

- 이미지 10MB 초과 → 파일로 전송
- 동영상 10MB 초과 → 파일로 전송
- 음성 2MB 초과 → 파일로 전송
- AMR이 아닌 오디오 → 파일로 전송(WeCom은 기본 음성에 AMR만 지원)

절대 제한인 20MB를 초과하는 파일은 거부되며 채팅에 안내 메시지가 전송됩니다.

## 답장 모드 응답

봇이 WeCom 콜백을 통해 메시지를 받으면 어댑터가 수신 요청 ID를 기억합니다. 요청 컨텍스트가 아직 활성화된 동안 응답이 전송되면 어댑터는 WeCom의 답장 모드(`aibot_respond_msg`)를 사용해 응답을 수신 메시지에 직접 연결합니다. 이를 통해 WeCom 클라이언트에서 더 자연스러운 대화 경험을 제공합니다.

전체 응답은 하나의 메시지로 전달되며, 어댑터는 토큰을 점진적으로 스트리밍하지 않습니다. 수신 요청 컨텍스트가 만료되었거나 사용할 수 없으면 어댑터는 사전 메시지 전송(`aibot_send_msg`)으로 대체합니다.

답장 모드는 미디어에도 작동합니다. 업로드된 미디어를 원본 메시지에 대한 답장으로 보낼 수 있습니다.

## 연결 및 재연결

어댑터는 `wss://openws.work.weixin.qq.com`의 WeCom 게이트웨이에 지속적인 WebSocket 연결을 유지합니다.

### 연결 수명 주기

1. **연결:** WebSocket 연결을 열고 bot_id와 secret이 포함된 `aibot_subscribe` 인증 프레임을 전송합니다.
2. **하트비트:** 연결을 유지하기 위해 30초마다 애플리케이션 수준의 ping 프레임을 전송합니다.
3. **수신:** 수신 프레임을 계속 읽고 메시지 콜백으로 디스패치합니다.

### 재연결 동작

연결이 끊기면 어댑터는 지수 백오프로 재연결합니다.

| 시도 | 지연 |
|---------|-------|
| 첫 번째 재시도 | 2초 |
| 두 번째 재시도 | 5초 |
| 세 번째 재시도 | 10초 |
| 네 번째 재시도 | 30초 |
| 다섯 번째 이후 재시도 | 60초 |

재연결에 성공할 때마다 백오프 카운터가 0으로 초기화됩니다. 연결이 끊기면 대기 중인 모든 요청 future가 실패 처리되므로 호출자가 무기한 대기하지 않습니다.

### 중복 제거

수신 메시지는 5분의 시간 창과 최대 1000개 항목의 캐시를 사용해 메시지 ID로 중복 제거됩니다. 이를 통해 재연결이나 네트워크 불안정 중 메시지가 중복 처리되는 것을 방지합니다.

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | WeCom AI Bot ID |
| `WECOM_SECRET` | ✅ | — | WeCom AI Bot Secret |
| `WECOM_ALLOWED_USERS` | — | _(비어 있음)_ | 게이트웨이 수준 허용 목록에 사용할 쉼표로 구분된 사용자 ID |
| `WECOM_HOME_CHANNEL` | — | — | cron/알림 출력을 위한 채팅 ID |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | WebSocket 게이트웨이 URL |
| `WECOM_DM_POLICY` | — | `open` | DM 액세스 정책 |
| `WECOM_GROUP_POLICY` | — | `open` | 그룹 액세스 정책 |

## 문제 해결

| 문제 | 해결 방법 |
|-----|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | 두 환경 변수를 모두 설정하거나 설정 마법사에서 구성합니다. |
| `WeCom startup failed: aiohttp not installed` | aiohttp를 설치합니다: `pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | httpx를 설치합니다: `pip install httpx` |
| `invalid secret (errcode=40013)` | Secret이 봇의 자격 증명과 일치하는지 확인합니다. |
| `Timed out waiting for subscribe acknowledgement` | `openws.work.weixin.qq.com`에 연결할 수 있는지 확인합니다. |
| 봇이 그룹에서 응답하지 않음 | `group_policy` 설정을 확인하고 그룹 ID가 `group_allow_from`에 있는지 확인합니다. |
| 봇이 그룹의 특정 사용자를 무시함 | `groups` 구성 섹션의 그룹별 `allow_from` 목록을 확인합니다. |
| 미디어 복호화 실패 | `cryptography`를 설치합니다: `pip install cryptography` |
| `cryptography is required for WeCom media decryption` | 수신 미디어가 AES로 암호화되어 있습니다. 설치하세요: `pip install cryptography` |
| 음성 메시지가 파일로 전송됨 | WeCom은 기본 음성에 AMR 형식만 지원합니다. 다른 형식은 자동으로 파일로 하향 전환됩니다. |
| `File too large` 오류 | WeCom은 모든 파일 업로드에 20MB 절대 제한을 적용합니다. 파일을 압축하거나 분할하세요. |
| 이미지가 파일로 전송됨 | 10MB를 초과하는 이미지는 기본 이미지 제한을 넘으므로 파일 첨부로 자동 하향 전환됩니다. |
| `Timeout sending message to WeCom` | WebSocket 연결이 끊겼을 수 있습니다. 로그에서 재연결 메시지를 확인합니다. |
| `WeCom websocket closed during authentication` | 네트워크 문제이거나 자격 증명이 올바르지 않습니다. bot_id와 secret을 확인합니다. |
