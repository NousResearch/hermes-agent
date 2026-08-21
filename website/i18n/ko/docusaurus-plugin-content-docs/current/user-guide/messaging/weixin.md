---
sidebar_position: 15
title: "Weixin (WeChat)"
description: "iLink Bot API를 통해 개인 WeChat 계정을 Hermes Agent에 연결"
---

# Weixin (WeChat)

[WeChat](https://weixin.qq.com/) (微信)은 Tencent의 개인 메시징 플랫폼입니다. Hermes를 WeChat에 연결하세요. 이 어댑터는 개인 WeChat 계정에 **iLink Bot API**를 사용하며, WeCom(Enterprise WeChat)과는 별개입니다. 메시지는 롱 폴링으로 전달되므로 공개 엔드포인트나 웹훅이 필요하지 않습니다.

:::info
이 어댑터는 **개인 WeChat 계정**(微信)용입니다. 기업용 WeChat이 필요하다면 대신 [WeCom 어댑터](./wecom.md)를 참조하세요.
:::

:::warning iLink 봇 정체성 — 일반 WeChat 그룹은 작동하지 않을 수 있음
QR 로그인은 Hermes를 완전히 스크립팅할 수 있는 일반 개인 WeChat 계정이 아니라 **iLink 봇 정체성**(예: `a5ace6fd482e@im.bot`)에 연결합니다. 이에 따른 결과는 다음과 같습니다.

- iLink 봇 정체성은 일반 연락처처럼 **일반 WeChat 그룹에 초대될 수 없는** 경우가 많습니다.
- 대부분의 봇 유형 계정에서 iLink는 일반적으로 **일반 WeChat 그룹 이벤트**(QR 로그인에 사용한 개인 계정에 대한 `@` 멘션 포함)를 게이트웨이에 전달하지 않습니다.
- QR 코드를 스캔하는 데 사용한 개인 WeChat 계정을 `@` 멘션하는 것은 iLink 봇을 `@` 멘션하는 것과 **같지 않습니다**. 봇은 별도의 정체성입니다.
- 아래의 `WEIXIN_GROUP_POLICY` / `WEIXIN_GROUP_ALLOWED_USERS` 설정은 iLink가 실제로 계정 유형에 대한 그룹 이벤트를 반환할 때만 적용됩니다. 그렇지 않으면 정책과 관계없이 그룹 메시지가 Hermes에 도달하지 않습니다.

실제로는 대부분의 배포에서 iLink 봇으로의 DM만 안정적으로 작동합니다. 구성 후에도 그룹 전달이 작동하지 않는다면 제한의 원인은 Hermes가 아니라 iLink 측입니다. `WEIXIN_GROUP_POLICY`가 `disabled` 이외의 값으로 설정된 경우 게이트웨이는 시작 시 `WARNING`을 기록합니다.
:::

## 사전 요구 사항

- 개인 WeChat 계정
- Python 패키지: `aiohttp` 및 `cryptography`
- Hermes를 `messaging` extra와 함께 설치하면 터미널 QR 렌더링이 포함됩니다.

필요한 종속성을 설치합니다.

```bash
pip install aiohttp cryptography
# Optional: for terminal QR code display
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

## 설정

### 1. 설정 마법사 실행

WeChat 계정을 연결하는 가장 쉬운 방법은 대화형 설정을 이용하는 것입니다.

```bash
hermes gateway setup
```

메시지가 표시되면 **Weixin**을 선택하세요. 마법사는 다음 작업을 수행합니다.

1. iLink Bot API에서 QR 코드를 요청합니다.
2. 터미널에 QR 코드를 표시하거나 URL을 제공합니다.
3. WeChat 모바일 앱으로 QR 코드를 스캔할 때까지 기다립니다.
4. 휴대폰에서 로그인을 확인하도록 요청합니다.
5. 계정 자격 증명을 `~/.hermes/weixin/accounts/`에 자동으로 저장합니다.

확인하면 다음과 비슷한 메시지가 표시됩니다.

```
微信连接成功，account_id=your-account-id
```

마법사는 `account_id`, `token`, `base_url`을 저장하므로 직접 구성할 필요가 없습니다.

### 2. 환경 변수 구성

최초 QR 로그인 후 최소한 `~/.hermes/.env`에 계정 ID를 설정합니다.

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# Optional: override the token (normally auto-saved from QR login)
# WEIXIN_TOKEN=your-bot-token

# Optional: restrict access
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# Optional: restore legacy multiline splitting behavior
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# Optional: home channel for cron/notifications
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. 게이트웨이 시작

```bash
hermes gateway
```

어댑터가 저장된 자격 증명을 복원하고 iLink API에 연결한 다음 메시지를 롱 폴링하기 시작합니다.

## 기능

- **롱 폴링 전송** — 공개 엔드포인트, 웹훅 또는 WebSocket이 필요하지 않습니다.
- **QR 코드 로그인** — `hermes gateway setup`을 통한 스캔-연결 설정
- **DM 메시징** — 액세스 정책을 구성할 수 있습니다. 그룹 메시징은 iLink가 연결된 정체성에 대한 그룹 이벤트를 실제로 전달하는지에 따라 달라지며(대개 iLink 봇 계정에서는 그렇지 않음), 위의 경고를 참조하세요.
- **미디어 지원** — 이미지, 동영상, 파일 및 음성 메시지
- **AES-128-ECB 암호화 CDN** — 모든 미디어 전송을 자동으로 암호화/복호화
- **컨텍스트 토큰 지속성** — 디스크 기반으로 재시작 후에도 답장 연속성 유지
- **Markdown 형식** — 헤더, 표, 코드 블록을 포함한 Markdown을 보존하므로 Markdown을 지원하는 WeChat 클라이언트에서 기본으로 렌더링할 수 있습니다.
- **스마트 메시지 청크 분할** — 제한보다 짧은 메시지는 하나의 말풍선으로 유지하고, 크기가 초과된 페이로드만 논리적 경계에서 분할합니다.
- **입력 중 표시** — 에이전트가 처리하는 동안 WeChat 클라이언트에 `typing…` 상태를 표시합니다.
- **SSRF 보호** — 다운로드 전에 외부 미디어 URL을 검증합니다.
- **메시지 중복 제거** — 5분 슬라이딩 윈도로 중복 처리를 방지합니다.
- **백오프를 적용한 자동 재시도** — 일시적인 API 오류에서 복구합니다.

## 구성 옵션

`config.yaml`의 `platforms.weixin.extra` 아래에 설정합니다.

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `account_id` | — | iLink Bot 계정 ID(필수) |
| `token` | — | iLink Bot 토큰(필수, QR 로그인에서 자동 저장) |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API 기본 URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | 미디어 전송용 CDN 기본 URL |
| `dm_policy` | `open` | DM 액세스: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `disabled` | 그룹 액세스: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | DM을 허용할 사용자 ID(`dm_policy=allowlist`인 경우) |
| `group_allow_from` | `[]` | 허용할 그룹 ID(`group_policy=allowlist`인 경우) |
| `split_multiline_messages` | `false` | `true`이면 여러 줄 답장을 여러 채팅 메시지로 분할합니다(레거시 동작). `false`이면 길이 제한을 초과하지 않는 한 여러 줄 답장을 하나의 메시지로 유지합니다. |
| `text_batch_delay_seconds` | `3.0` | 빠르게 도착한 텍스트 메시지 버퍼를 하나의 결합된 요청으로 전송하기 전의 대기 시간(초)입니다. iLink는 메시지를 개별적으로 전달하므로 이 디바운스는 조각마다 에이전트를 한 번씩 호출하는 것을 방지합니다. 각 메시지를 즉시 전송하려면 `0`으로 설정합니다. |
| `text_batch_split_delay_seconds` | `5.0` | 최신 조각이 분할 임계값에 가까울 때 사용하는 연장된 플러시 지연 시간입니다(iLink가 분할했을 수 있는 긴 메시지). |

## 액세스 정책

### DM 정책

봇에 직접 메시지를 보낼 수 있는 사람을 제어합니다.

| 값 | 동작 |
|-------|----------|
| `open` | 누구나 봇에 DM을 보낼 수 있음(기본값) |
| `allowlist` | `allow_from`에 있는 사용자 ID만 DM을 보낼 수 있음 |
| `disabled` | 모든 DM을 무시함 |
| `pairing` | 페어링 모드(초기 설정용) |

```bash
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

`WEIXIN_ALLOWED_USERS`는 **인바운드 필터**이지 초대 시스템이 아닙니다. QR 로그인은 하나의 iLink 봇 정체성을 Hermes에 연결합니다. 다른 사람들은 자신의 계정으로 Hermes QR 코드를 스캔하지 않습니다. 대신 WeChat을 통해 연결된 iLink 봇/연락처로 메시지를 보내야 하며, Hermes는 발신자의 Weixin 사용자 ID가 `WEIXIN_ALLOWED_USERS`에 있는 경우에만 DM을 처리합니다.

실용적인 설정 흐름은 다음과 같습니다.

1. `hermes gateway setup`으로 Hermes를 한 번 페어링하고 연결된 iLink 봇 계정을 기록합니다.
2. 허용된 각 사용자가 해당 봇/연락처로 직접 메시지를 보내도록 합니다.
3. 게이트웨이 로그 또는 인바운드 이벤트 페이로드에서 발신자/사용자 ID를 확인합니다.
4. 해당 ID를 `WEIXIN_ALLOWED_USERS`에 추가한 다음 게이트웨이를 재시작합니다.

QR 코드를 스캔한 계정만 Hermes와 대화할 수 있다면 다른 사용자들이 QR 로그인을 수행한 개인 WeChat 계정이 아니라 iLink 봇 정체성 자체에 메시지를 보내고 있는지 확인하세요. iLink 봇은 별도의 정체성이며 일반 WeChat 연락처/그룹 라우팅은 Tencent의 iLink 동작에 의해 제한될 수 있습니다.

### 그룹 정책

**iLink가 연결된 정체성에 대한 그룹 이벤트를 전달하는 경우** 봇이 응답할 그룹을 제어합니다. QR 로그인 iLink 봇 정체성(예: `...@im.bot`)에서는 그룹 이벤트가 일반적으로 전혀 전달되지 않으므로 이 정책이 효과가 없을 수 있습니다. 페이지 상단의 iLink 봇 제한 경고를 참조하세요.

| 값 | 동작 |
|-------|----------|
| `open` | 이벤트가 전달되는 경우 봇이 모든 그룹에 응답함 |
| `allowlist` | 이벤트가 전달되는 경우 `group_allow_from`에 나열된 그룹 ID에만 봇이 응답함 |
| `disabled` | 모든 그룹 메시지를 무시함(기본값) |

```bash
WEIXIN_GROUP_POLICY=allowlist
# NOTE: this is a comma-separated list of group chat IDs, NOT member user IDs,
# despite the variable name containing "USERS". Keep this in mind when configuring.
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
Weixin의 기본 그룹 정책은 `disabled`입니다(기본값이 `open`인 WeCom과 다름). 이는 의도된 동작입니다. 개인 WeChat 계정은 많은 그룹에 속해 있을 수 있으며 iLink 봇 정체성은 일반적으로 일반 WeChat 그룹 메시지를 전혀 받을 수 없습니다. `WEIXIN_GROUP_POLICY`를 `disabled` 이외의 값으로 설정하면 게이트웨이는 시작 시 `WARNING`을 기록합니다.
:::

## 미디어 지원

### 인바운드(수신)

어댑터는 사용자로부터 미디어 첨부 파일을 받고, WeChat CDN에서 다운로드하고, 복호화한 다음 에이전트 처리를 위해 로컬에 캐시합니다.

| 유형 | 처리 방식 |
|------|-----------------| 
| **Images** | 다운로드하고 AES로 복호화한 뒤 JPEG로 캐시합니다. |
| **Video** | 다운로드하고 AES로 복호화한 뒤 MP4로 캐시합니다. |
| **Files** | 다운로드하고 AES로 복호화한 뒤 캐시합니다. 원래 파일 이름은 보존됩니다. |
| **Voice** | 텍스트 변환을 사용할 수 있으면 텍스트로 추출합니다. 그렇지 않으면 오디오(SILK 형식)를 다운로드하고 캐시합니다. |

**인용된 메시지:** 인용된(답장 대상) 메시지의 미디어도 추출하므로 에이전트가 사용자가 답장하는 대상에 대한 맥락을 파악할 수 있습니다.

### AES-128-ECB 암호화 CDN

WeChat 미디어 파일은 암호화된 CDN을 통해 전송됩니다. 어댑터가 이를 투명하게 처리합니다.

- **인바운드:** `encrypted_query_param` URL을 사용하여 CDN에서 암호화된 미디어를 다운로드한 다음 메시지 페이로드에 제공된 파일별 키로 AES-128-ECB를 사용해 복호화합니다.
- **아웃바운드:** 파일을 임의의 AES-128-ECB 키로 로컬에서 암호화하고 CDN에 업로드한 다음 암호화된 참조를 아웃바운드 메시지에 포함합니다.
- AES 키는 16바이트(128비트)입니다. 키는 원시 base64 또는 16진수 인코딩으로 전달될 수 있으며 어댑터가 두 형식을 모두 처리합니다.
- 이를 위해 `cryptography` Python 패키지가 필요합니다.

구성은 필요하지 않습니다. 암호화와 복호화가 자동으로 수행됩니다.

### 아웃바운드(전송)

| 방식 | 전송 내용 |
|--------|--------------|
| `send` | Markdown 형식의 텍스트 메시지 |
| `send_image` / `send_image_file` | 네이티브 이미지 메시지(CDN 업로드 사용) |
| `send_document` | 파일 첨부(CDN 업로드 사용) |
| `send_video` | 동영상 메시지(CDN 업로드 사용) |

모든 아웃바운드 미디어는 암호화된 CDN 업로드 흐름을 거칩니다.

1. 임의의 AES-128 키를 생성합니다.
2. AES-128-ECB + PKCS#7 패딩으로 파일을 암호화합니다.
3. iLink API에서 업로드 URL(`getuploadurl`)을 요청합니다.
4. 암호문을 CDN에 업로드합니다.
5. 암호화된 미디어 참조를 포함하여 메시지를 전송합니다.

## 컨텍스트 토큰 지속성

iLink Bot API는 지정된 피어에 대한 각 아웃바운드 메시지에 `context_token`을 다시 포함할 것을 요구합니다. 어댑터는 디스크 기반 컨텍스트 토큰 저장소를 유지합니다.

- 토큰은 계정+피어별로 `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`에 저장됩니다.
- 시작 시 이전에 저장된 토큰이 복원됩니다.
- 모든 인바운드 메시지는 해당 발신자의 저장된 토큰을 업데이트합니다.
- 아웃바운드 메시지는 최신 컨텍스트 토큰을 자동으로 포함합니다.

이를 통해 게이트웨이를 재시작한 후에도 답장 연속성이 유지됩니다.

## Markdown 형식

iLink Bot API를 통해 연결된 WeChat 클라이언트는 Markdown을 직접 렌더링할 수 있으므로 어댑터는 Markdown을 다시 작성하지 않고 보존합니다.

- **헤더**는 Markdown 제목(`#`, `##`, ...)으로 유지됩니다.
- **표**는 Markdown 표로 유지됩니다.
- **코드 펜스**는 펜스로 감싼 코드 블록으로 유지됩니다.
- **과도한 빈 줄**은 펜스로 감싼 코드 블록 바깥에서 두 개의 줄바꿈으로 축소됩니다.

## 메시지 청크 분할

메시지는 플랫폼 제한에 맞는 경우 하나의 채팅 메시지로 전달됩니다. 크기가 초과된 페이로드만 전송을 위해 분할됩니다.

- 최대 메시지 길이: **4000자**
- 제한보다 짧은 메시지는 여러 단락이나 줄바꿈을 포함하더라도 그대로 유지됩니다.
- 크기가 초과된 메시지는 논리적 경계(단락, 빈 줄, 코드 펜스)에서 분할됩니다.
- 가능한 경우 코드 펜스는 그대로 유지됩니다(펜스 자체가 제한을 초과하지 않는 한 블록 중간에서 분할하지 않음).
- 크기가 초과된 개별 블록은 기본 어댑터의 잘라내기 로직으로 대체됩니다.
- 청크 사이의 0.3초 지연은 여러 청크를 보낼 때 WeChat의 속도 제한으로 인한 삭제를 방지합니다.

## 입력 중 표시

어댑터는 WeChat 클라이언트에 입력 중 상태를 표시합니다.

1. 메시지가 도착하면 어댑터가 `getconfig` API를 통해 `typing_ticket`을 가져옵니다.
2. 입력 중 티켓은 사용자별로 10분간 캐시됩니다.
3. `send_typing`은 입력 시작 신호를 보내고 `stop_typing`은 입력 종료 신호를 보냅니다.
4. 에이전트가 메시지를 처리하는 동안 게이트웨이가 입력 중 표시를 자동으로 트리거합니다.

## 롱 폴링 연결

어댑터는 메시지를 수신하는 데 HTTP 롱 폴링(WebSocket 아님)을 사용합니다.

### 작동 방식

1. **연결:** 자격 증명을 검증하고 폴링 루프를 시작합니다.
2. **폴링:** 35초 타임아웃으로 `getupdates`를 호출합니다. 서버는 메시지가 도착하거나 타임아웃이 만료될 때까지 요청을 유지합니다.
3. **디스패치:** 인바운드 메시지는 `asyncio.create_task`를 통해 동시에 디스패치됩니다.
4. **동기화 버퍼:** 영구 동기화 커서(`get_updates_buf`)를 디스크에 저장하므로 재시작 후 올바른 위치에서 어댑터를 재개합니다.

### 재시도 동작

API 오류가 발생하면 어댑터는 간단한 재시도 전략을 사용합니다.

| 조건 | 동작 |
|-----------|----------|
| 일시적 오류(1~2회) | 2초 후 재시도 |
| 반복 오류(3회 이상) | 30초간 백오프한 다음 카운터 재설정 |
| 세션 만료(`errcode=-14`) | 10분간 일시 중지(재로그인이 필요할 수 있음) |
| 타임아웃 | 즉시 다시 폴링(정상적인 롱 폴링 동작) |

### 중복 제거

인바운드 메시지는 5분 윈도의 메시지 ID를 사용해 중복 제거됩니다. 이를 통해 네트워크 장애나 겹치는 폴링 응답 중 이중 처리를 방지합니다.

### 토큰 잠금

하나의 Weixin 게이트웨이 인스턴스만 특정 토큰을 사용할 수 있습니다. 어댑터는 시작 시 범위가 지정된 잠금을 획득하고 종료 시 해제합니다. 다른 게이트웨이가 이미 같은 토큰을 사용 중이면 유용한 오류 메시지와 함께 시작에 실패합니다.

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot 계정 ID(QR 로그인에서 가져옴) |
| `WEIXIN_TOKEN` | ✅ | — | iLink Bot 토큰(QR 로그인에서 자동 저장) |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API 기본 URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | 미디어 전송용 CDN 기본 URL |
| `WEIXIN_DM_POLICY` | — | `open` | DM 액세스 정책: `open`, `allowlist`, `disabled`, `pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | 그룹 액세스 정책: `open`, `allowlist`, `disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _(empty)_ | DM 허용 목록을 위한 쉼표로 구분된 사용자 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _(empty)_ | 그룹 허용 목록을 위한 쉼표로 구분된 **그룹 채팅 ID**(멤버 사용자 ID가 아님)입니다. 변수 이름은 레거시이며 사용자 ID가 아니라 그룹 ID를 사용합니다. |
| `WEIXIN_HOME_CHANNEL` | — | — | cron/알림 출력을 위한 채팅 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | 홈 채널의 표시 이름 |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | 모든 사용자를 허용하는 게이트웨이 수준 플래그(설정 마법사에서 사용) |

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | 둘 다 설치합니다: `pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | `hermes gateway setup`을 실행하여 QR 로그인을 완료하거나 `WEIXIN_TOKEN`을 수동으로 설정합니다. |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | `.env`에 `WEIXIN_ACCOUNT_ID`를 설정하거나 `hermes gateway setup`을 실행합니다. |
| `Another local Hermes gateway is already using this Weixin token` | 먼저 다른 게이트웨이 인스턴스를 중지합니다. 토큰당 하나의 폴러만 허용됩니다. |
| 세션 만료(`errcode=-14`) | 로그인 세션이 만료되었습니다. `hermes gateway setup`을 다시 실행하여 새 QR 코드를 스캔합니다. |
| 설정 중 QR 코드 만료 | QR 코드는 최대 3회 자동으로 새로 고쳐집니다. 계속 만료된다면 네트워크 연결을 확인합니다. |
| 봇이 DM에 응답하지 않음 | `WEIXIN_DM_POLICY`를 확인합니다. `allowlist`로 설정된 경우 발신자가 `WEIXIN_ALLOWED_USERS`에 있어야 합니다. |
| 봇이 그룹 메시지를 무시함 | 그룹 정책의 기본값은 `disabled`입니다. `WEIXIN_GROUP_POLICY=open` 또는 `allowlist`로 설정하세요. 단, QR 로그인 iLink 봇 정체성(`...@im.bot`)은 일반적으로 일반 WeChat 그룹 메시지를 전혀 받을 수 없습니다. 게이트웨이 로그에 그룹 메시지에 대한 원시 인바운드 이벤트가 없다면 제한의 원인은 Hermes가 아니라 iLink 측입니다. |
| 미디어 다운로드/업로드 실패 | `cryptography`가 설치되어 있는지 확인합니다. `novac2c.cdn.weixin.qq.com`에 대한 네트워크 액세스를 확인합니다. |
| `Blocked unsafe URL (SSRF protection)` | 외부 미디어 URL이 비공개/내부 주소를 가리킵니다. 공개 URL만 허용됩니다. |
| 음성 메시지가 텍스트로 표시됨 | WeChat이 텍스트 변환을 제공하면 어댑터가 텍스트를 사용합니다. 이는 정상적인 동작입니다. |
| 메시지가 중복되어 표시됨 | 어댑터는 메시지 ID로 중복을 제거합니다. 중복이 보이면 여러 게이트웨이 인스턴스가 실행 중인지 확인합니다. |
| `iLink POST ... HTTP 4xx/5xx` | iLink 서비스의 API 오류입니다. 토큰의 유효성과 네트워크 연결을 확인합니다. |
| 터미널 QR 코드가 렌더링되지 않음 | messaging extra와 함께 다시 설치합니다: `cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"`. 또는 QR 코드 위에 출력된 URL을 엽니다. |
