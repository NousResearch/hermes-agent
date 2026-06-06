---
sidebar_position: 15
title: "Weixin (WeChat)"
description: "iLink Bot API를 통해 개인 WeChat 계정에 Hermes Agent 연결하기"
---

# Weixin (WeChat)

Hermes를 Tencent의 개인 메시징 플랫폼인 [WeChat](https://weixin.qq.com/)(위챗, 微信)에 연결하세요. 어댑터는 개인 WeChat 계정을 위한 Tencent의 **iLink Bot API**를 사용합니다 — 이는 WeCom(기업용 WeChat)과는 다릅니다. 메시지는 롱 폴링(long-polling)을 통해 전달되므로 공개 엔드포인트나 웹훅이 필요하지 않습니다.

:::info
이 어댑터는 **개인 WeChat 계정(微信)**을 위한 것입니다. 기업/회사용 WeChat이 필요하다면 대신 [WeCom 어댑터](./wecom.md) 문서를 참고하세요.
:::

:::warning iLink 봇 아이덴티티 — 일반 WeChat 그룹이 작동하지 않을 수 있습니다
QR 로그인은 Hermes를 정상적으로 스크립팅 가능한 일반 개인 WeChat 계정이 **아닌**, **iLink 봇 아이덴티티**(예: `a5ace6fd482e@im.bot`)에 연결합니다. 결과:

- iLink 봇 아이덴티티는 일반적으로 일반 연락처처럼 **일반 WeChat 그룹에 초대될 수 없습니다**.
- 대부분의 봇 유형 계정에 대해 iLink는 일반적으로 **일반 WeChat 그룹 이벤트**(QR 로그인에 사용된 개인 계정의 `@`-멘션 포함)를 게이트웨이에 전달하지 않습니다.
- QR 코드를 스캔하는 데 사용된 개인 WeChat 계정을 `@`-멘션하는 것은 iLink 봇을 `@`-멘션하는 것과 **같지 않습니다** — 봇은 별개의 아이덴티티입니다.
- 아래의 `WEIXIN_GROUP_POLICY` / `WEIXIN_GROUP_ALLOWED_USERS` 설정은 iLink가 계정 유형에 대해 그룹 이벤트를 실제로 반환할 때만 효과가 있습니다. 그렇지 않다면 정책과 관계없이 그룹 메시지가 절대 Hermes에 도달하지 않습니다.

실제로는 대부분의 배포가 iLink 봇으로의 DM만 안정적으로 작동합니다. 설정 후에도 그룹 메시지 전송이 되지 않는다면 제한은 Hermes가 아닌 iLink 쪽에 있는 것입니다. 게이트웨이는 시작 시 `WEIXIN_GROUP_POLICY`가 `disabled` 외의 다른 값으로 설정되어 있을 때 항상 `WARNING`을 기록합니다.
:::

## 전제 조건

- 개인 WeChat 계정
- Python 패키지: `aiohttp` 및 `cryptography`
- 터미널 QR 렌더링은 `messaging` 엑스트라와 함께 Hermes를 설치할 때 포함됩니다

필요한 의존성을 설치하세요:

```bash
pip install aiohttp cryptography
# 선택 사항: 터미널 QR 코드 표시를 위한 패키지
pip install hermes-agent[messaging]
```

## 설정

### 1. 설정 마법사 실행

WeChat 계정을 연결하는 가장 쉬운 방법은 대화형 설정을 이용하는 것입니다:

```bash
hermes gateway setup
```

프롬프트가 나타나면 **Weixin**을 선택하세요. 마법사는 다음을 수행합니다:

1. iLink Bot API에서 QR 코드를 요청
2. 터미널에 QR 코드를 표시(또는 URL 제공)
3. WeChat 모바일 앱으로 QR 코드를 스캔할 때까지 대기
4. 전화기에서 로그인을 확인하라는 프롬프트 표시
5. 계정 자격 증명을 `~/.hermes/weixin/accounts/`에 자동으로 저장

확인되면, 다음과 같은 메시지가 나타납니다:

```
微信连接成功，account_id=your-account-id
```

마법사는 `account_id`, `token`, `base_url`을 저장하므로 이를 수동으로 구성할 필요가 없습니다.

### 2. 환경 변수 구성

초기 QR 로그인 이후, 최소한 계정 ID를 `~/.hermes/.env`에 설정합니다:

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# 선택 사항: 토큰 오버라이드 (일반적으로 QR 로그인에서 자동 저장됨)
# WEIXIN_TOKEN=your-bot-token

# 선택 사항: 접근 제한
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# 선택 사항: 레거시 여러 줄 분할 동작 복원
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# 선택 사항: 크론/알림용 홈 채널
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. 게이트웨이 시작

```bash
hermes gateway
```

어댑터는 저장된 자격 증명을 복원하고, iLink API에 연결한 뒤, 메시지에 대해 롱 폴링을 시작합니다.

## 기능

- **롱 폴링 전송** — 공개 엔드포인트, 웹훅, WebSocket 불필요
- **QR 코드 로그인** — `hermes gateway setup`을 통한 스캔하여 연결하는 설정 방식
- **DM 메시징** — 설정 가능한 접근 정책; 그룹 메시징은 iLink가 연결된 아이덴티티에 대해 실제로 그룹 이벤트를 전달하는지에 따라 다름 (iLink 봇 계정의 경우 전달 안 되는 경우가 잦음 — 위 경고 참조)
- **미디어 지원** — 이미지, 동영상, 파일, 음성 메시지 지원
- **AES-128-ECB 암호화 CDN** — 모든 미디어 전송 시 자동 암호화/복호화
- **컨텍스트 토큰 유지** — 재시작 후에도 디스크 기반 답글 연속성 유지
- **마크다운 포맷 유지** — 헤더, 표, 코드 블록을 포함한 마크다운을 보존하므로, 마크다운을 지원하는 WeChat 클라이언트가 이를 기본적으로 렌더링할 수 있음
- **스마트 메시지 청킹(분할)** — 제한을 초과하지 않을 경우 단일 말풍선으로 유지; 오직 초과되는 페이로드만 논리적인 경계에서 분할
- **타이핑 인디케이터** — 에이전트가 처리하는 동안 WeChat 클라이언트에 "입력 중..." 상태를 보여줌
- **SSRF 방지** — 아웃바운드 미디어 URL은 다운로드 전에 검증됨
- **메시지 중복 제거** — 5분 슬라이딩 윈도우 방식으로 이중 처리 방지
- **자동 백오프(재시도)** — 일시적인 API 오류를 자체 복구

## 구성 옵션

`config.yaml`의 `platforms.weixin.extra` 아래에서 설정:

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `account_id` | — | iLink Bot 계정 ID (필수) |
| `token` | — | iLink Bot 토큰 (필수, QR 로그인에서 자동 저장됨) |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API 기본 URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | 미디어 전송을 위한 CDN 기본 URL |
| `dm_policy` | `open` | DM 접근: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `disabled` | 그룹 접근: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | DM이 허용된 사용자 ID (dm_policy=allowlist일 때) |
| `group_allow_from` | `[]` | 허용된 그룹 ID (group_policy=allowlist일 때) |
| `split_multiline_messages` | `false` | `true`일 경우 레거시 동작인 다중 라인 답글을 여러 채팅 메시지로 분할. `false`일 경우 길이 제한을 넘지 않는 이상 다중 라인을 하나의 메시지로 유지. |
| `text_batch_delay_seconds` | `3.0` | 버퍼링 된 빠르게 전송되는 텍스트 조각들을 하나의 응답으로 모아 보내기 전 대기하는 지연 시간(초). iLink는 메시지를 개별적으로 전달하므로, 이를 통해 파편별로 에이전트가 호출되는 것을 방지. `0`으로 설정하면 각 메시지를 즉시 보냄. |
| `text_batch_split_delay_seconds` | `5.0` | 최신 조각이 iLink가 청킹(분할)했을 가능성이 높은 제한에 가까운 длинных 메시지일 때 사용하는 긴 대기 지연. |

## 접근 정책

### DM 정책

봇에게 다이렉트 메시지를 보낼 수 있는 권한을 제어합니다:

| 값 | 동작 |
|-------|----------|
| `open` | 누구나 봇에게 DM을 보낼 수 있음 (기본값) |
| `allowlist` | `allow_from`에 지정된 사용자 ID만 DM 가능 |
| `disabled` | 모든 DM 무시 |
| `pairing` | 페어링 모드 (초기 설정용) |

```bash
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

`WEIXIN_ALLOWED_USERS`는 인바운드 필터이며, 초대를 위한 시스템이 **아닙니다**. QR 로그인은 하나의 iLink 봇 아이덴티티를 Hermes에 연결합니다. 다른 사용자들은 자신의 계정으로 Hermes QR 코드를 스캔하지 않으며; 연결된 iLink 봇/연락처에 WeChat을 통해 메시지를 보내야 하고, 보낸 사람의 Weixin 사용자 ID가 `WEIXIN_ALLOWED_USERS`에 존재할 때만 Hermes가 DM을 처리합니다.

실용적인 설정 흐름은 다음과 같습니다:

1. `hermes gateway setup`을 통해 Hermes를 한 번 페어링하고 연결된 iLink 봇 계정을 확인합니다.
2. 허용된 각 사용자가 해당 봇/연락처에 직접 메시지를 보내도록 합니다.
3. 게이트웨이 로그 또는 인바운드 이벤트 페이로드에서 보낸 사람/사용자 ID를 읽습니다.
4. 해당 ID들을 `WEIXIN_ALLOWED_USERS`에 추가하고 게이트웨이를 다시 시작합니다.

QR 코드를 스캔한 계정만 Hermes와 대화할 수 있는 경우, 다른 사용자들이 QR 로그인을 수행한 개인 WeChat 계정이 아니라 iLink 봇 아이덴티티 자체에 메시지를 보내고 있는지 확인하세요. iLink 봇은 별도의 아이덴티티이며, 일반 WeChat 연락처/그룹 라우팅은 Tencent의 iLink 동작 방식에 의해 제한될 수 있습니다.

### 그룹 정책

봇이 그룹에서 어떻게 반응할지를 제어합니다 **(iLink가 연결된 계정에 대해 그룹 이벤트를 전송할 경우에만)**. QR로 로그인한 iLink 봇 계정(예: `...@im.bot`)의 경우 일반적인 그룹 메시지는 전달되지 않으므로, 이 정책은 영향을 미치지 않을 수 있습니다 — 페이지 상단의 iLink 봇 제한 사항 경고를 확인하세요.

| 값 | 동작 |
|-------|----------|
| `open` | 봇은 모든 그룹에서 반응함 (이벤트가 전달될 경우) |
| `allowlist` | 봇은 `group_allow_from`에 나열된 그룹 ID에서만 응답함 (이벤트가 전달될 경우) |
| `disabled` | 모든 그룹 메시지 무시 (기본값) |

```bash
WEIXIN_GROUP_POLICY=allowlist
# 참고: 이 값은 쉼표로 구분된 그룹 채팅 ID 목록이며,
# 변수 이름에 "USERS"가 포함되어 있더라도 멤버 사용자 ID가 아닙니다.
# 이를 유념하여 설정하세요.
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
Weixin의 기본 그룹 정책은 `disabled`입니다 (WeCom의 기본값인 `open`과는 다름). 이는 의도적인 것입니다 — 개인 WeChat 계정은 많은 그룹에 속할 수 있고, iLink 봇 아이덴티티는 일반적으로 일반 WeChat 그룹 메시지를 받을 수 없습니다. `WEIXIN_GROUP_POLICY`를 `disabled` 외의 값으로 설정하면 게이트웨이가 시작될 때 `WARNING`을 기록합니다.
:::

## 미디어 지원

### 인바운드 (수신)

어댑터는 사용자의 미디어 첨부 파일을 수신하고 WeChat CDN에서 다운로드하며, 복호화한 후 에이전트 처리를 위해 로컬에 캐시합니다:

| 타입 | 처리 방법 |
|------|-----------------|
| **이미지** | 다운로드, AES 복호화 후 JPEG로 캐시됨. |
| **비디오** | 다운로드, AES 복호화 후 MP4로 캐시됨. |
| **파일** | 다운로드, AES 복호화 및 캐시됨. 원본 파일명이 보존됨. |
| **음성** | 텍스트 변환(transcription)을 사용할 수 있는 경우 텍스트로 추출. 그렇지 않은 경우 오디오(SILK 포맷)가 다운로드되고 캐시됨. |

**인용된 메시지(Quoted messages):** 인용된(답글을 달고 있는) 메시지의 미디어 역시 추출되므로, 에이전트는 사용자가 무엇에 답변하고 있는지 맥락을 파악할 수 있습니다.

### AES-128-ECB 암호화 CDN

WeChat 미디어 파일은 암호화된 CDN을 통해 전송됩니다. 어댑터는 이를 투명하게 처리합니다:

- **인바운드:** 암호화된 미디어는 `encrypted_query_param` URL을 사용하여 CDN에서 다운로드된 다음, 메시지 페이로드에 포함된 파일별 키를 사용해 AES-128-ECB로 복호화됩니다.
- **아웃바운드:** 파일은 임의의 AES-128-ECB 키로 로컬에서 암호화되고 CDN에 업로드되며, 발신 메시지에 암호화된 참조가 포함됩니다.
- AES 키는 16바이트(128비트)입니다. 키는 순수 base64나 16진수 인코딩 형식으로 수신될 수 있으며 — 어댑터는 두 형식을 모두 처리합니다.
- 이를 위해서는 `cryptography` Python 패키지가 필수적입니다.

추가적인 설정은 필요하지 않습니다 — 암호화 및 복호화는 자동으로 수행됩니다.

### 아웃바운드 (발신)

| 메서드 | 발신되는 것 |
|--------|--------------|
| `send` | 마크다운 형식이 보존된 텍스트 메시지 | 
| `send_image` / `send_image_file` | 네이티브 이미지 메시지 (CDN 업로드를 통해) |
| `send_document` | 파일 첨부 (CDN 업로드를 통해) |
| `send_video` | 비디오 메시지 (CDN 업로드를 통해) |

모든 아웃바운드 미디어는 암호화된 CDN 업로드 흐름을 따릅니다:

1. 랜덤 AES-128 키 생성
2. AES-128-ECB + PKCS#7 패딩을 사용한 파일 암호화
3. iLink API에서 업로드 URL 요청 (`getuploadurl`)
4. CDN에 암호문(ciphertext) 업로드
5. 암호화된 미디어 참조를 포함하여 메시지 전송

## 컨텍스트 토큰 영구 저장

iLink Bot API는 주어진 사용자를 위한 발신 메시지마다 `context_token`이 동반될 것을 요구합니다. 어댑터는 디스크 기반의 컨텍스트 토큰 스토리지를 유지 관리합니다:

- 토큰은 계정+상대방(peer)별로 `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`에 저장됩니다.
- 시작 시 이전에 저장된 토큰이 복원됩니다.
- 들어오는 메시지마다 해당 발신자의 저장된 토큰이 업데이트됩니다.
- 아웃바운드 메시지에는 최신 컨텍스트 토큰이 자동으로 포함됩니다.

이를 통해 게이트웨이를 다시 시작한 후에도 답글이 지속적으로 유지되게 됩니다.

## 마크다운 포맷

iLink Bot API를 통해 연결된 WeChat 클라이언트는 마크다운을 직접 렌더링할 수 있으므로, 어댑터는 이를 다시 작성하는 대신 마크다운 형식을 유지합니다:

- **헤더**는 마크다운 헤딩으로 유지됨 (`#`, `##`, ...)
- **표**는 마크다운 표 구조를 그대로 유지함
- **코드 영역**은 코드 블록으로 유지됨
- **과도한 빈 줄**은 코드 블록 외에서 이중 줄바꿈으로 축소됨

## 메시지 분할(Chunking)

플랫폼 제한 크기에 맞는 경우 하나의 채팅 메시지로 전송됩니다. 크기가 초과된 페이로드만이 전송을 위해 분할됩니다:

- 최대 메시지 길이: **4000자**
- 여러 단락이나 줄바꿈을 포함하더라도 제한 미만이면 단일 메시지로 유지됨
- 너무 큰 메시지는 논리적인 경계(단락, 빈 줄, 코드 영역)를 기준으로 분할됨
- 코드 영역은 가능한 한 유지됨 (영역 자체가 제한을 초과하지 않는 한 블록 중간을 자르지 않음)
- 개별 블록 자체가 초과될 경우 기본 어댑터의 잘라내기(truncation) 로직으로 폴백
- 여러 청크가 전송될 때 WeChat 속도 제한 문제 방지를 위해 0.3초의 청크 간 지연 시간이 주어짐

## 타이핑 상태 인디케이터

어댑터는 WeChat 클라이언트에 입력 중 상태를 표시합니다:

1. 메시지가 도착하면 어댑터가 `getconfig` API를 통해 `typing_ticket`을 가져옴
2. 타이핑 티켓은 사용자당 10분간 캐시됨
3. `send_typing`이 입력 시작 신호를 전송함; `stop_typing`이 입력 중지 신호를 전송함
4. 에이전트가 메시지를 처리하는 동안 게이트웨이는 타이핑 상태를 자동으로 표시함

## 롱 폴링(Long-Poll) 연결

어댑터는 (WebSocket 대신) 메시지를 수신하기 위해 HTTP 롱 폴링을 사용합니다:

### 작동 방식

1. **연결:** 자격 증명을 검증하고 폴링 루프를 시작
2. **폴(Poll):** 35초의 타임아웃으로 `getupdates`를 호출; 서버는 메시지가 도착하거나 타임아웃 만료 시까지 요청을 쥐고 있음
3. **디스패치:** 인바운드 메시지는 `asyncio.create_task`를 통해 병행 처리됨
4. **동기화 버퍼:** 지속적인 동기화 커서(`get_updates_buf`)가 디스크에 저장되어, 재시작 시 어댑터가 올바른 위치에서부터 수신을 재개함

### 재시도 동작

API 오류 발생 시, 어댑터는 단순 재시도 전략을 사용합니다:

| 상황 | 동작 |
|-----------|----------|
| 일시적 오류 (1–2회) | 2초 후 재시도 |
| 반복된 오류 (3회 이상) | 30초 대기(백오프) 후 카운터 리셋 |
| 세션 만료 (`errcode=-14`) | 10분 일시 정지 (새로운 로그인이 필요할 수 있음) |
| 타임아웃 | 즉시 다시 폴링 (정상 롱 폴링 동작) |

### 중복 제거(Deduplication)

인바운드 메시지는 메시지 ID를 사용해 5분 윈도우 내에서 중복이 제거됩니다. 이는 네트워크 문제나 폴 응답이 겹칠 때의 이중 처리를 막아줍니다.

### 토큰 잠금(Token Lock)

주어진 토큰에 대해 한 번에 하나의 Weixin 게이트웨이 인스턴스만 사용할 수 있습니다. 어댑터는 시작 시 범위 잠금(scoped lock)을 획득하고 종료 시 이를 해제합니다. 다른 게이트웨이가 이미 동일한 토큰을 사용하고 있다면, 시작은 정보 제공 오류 메시지와 함께 실패합니다.

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot 계정 ID (QR 로그인에서 획득) |
| `WEIXIN_TOKEN` | ✅ | — | iLink Bot 토큰 (QR 로그인에서 자동 저장됨) |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API 기본 URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | 미디어 전송을 위한 CDN 기본 URL |
| `WEIXIN_DM_POLICY` | — | `open` | DM 접근 정책: `open`, `allowlist`, `disabled`, `pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | 그룹 접근 정책: `open`, `allowlist`, `disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _(비어 있음)_ | DM 허용 목록을 위한 쉼표로 구분된 사용자 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _(비어 있음)_ | 그룹 허용 목록을 위한 쉼표로 구분된 **그룹 채팅 ID** (멤버 사용자 ID 아님). 변수 이름은 과거 호환성을 위한 것으로 — 그룹 ID를 지정해야 함. |
| `WEIXIN_HOME_CHANNEL` | — | — | 크론/알림 출력을 위한 채팅 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | 홈 채널의 표시 이름 |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | 모든 사용자를 허용하는 게이트웨이 수준 플래그 (설정 마법사에서 사용됨) |

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | 둘 다 설치하세요: `pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | QR 로그인을 완료하기 위해 `hermes gateway setup`을 실행하거나 `WEIXIN_TOKEN`을 수동으로 설정하세요 |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | `.env`에 `WEIXIN_ACCOUNT_ID`를 설정하거나 `hermes gateway setup`을 실행하세요 |
| `Another local Hermes gateway is already using this Weixin token` | 토큰당 오직 1대의 폴러만 허용되므로, 다른 게이트웨이 인스턴스를 먼저 중지하세요 |
| Session expired (`errcode=-14`) | 로그인 세션이 만료되었습니다. 새 QR 코드를 스캔하기 위해 `hermes gateway setup`을 다시 실행하세요 |
| 설정 중에 QR 코드 만료 | QR 코드는 최대 3번까지 자동 새로 고침됩니다. 계속 만료되면 네트워크 연결을 확인하세요 |
| 봇이 DM에 응답하지 않음 | `WEIXIN_DM_POLICY` 확인 — `allowlist`로 설정된 경우 보낸 사람이 `WEIXIN_ALLOWED_USERS`에 있어야 합니다 |
| 봇이 그룹 메시지를 무시함 | 그룹 정책 기본값은 `disabled`입니다. `WEIXIN_GROUP_POLICY=open` 또는 `allowlist`로 설정하세요 — 하지만 QR 로그인의 iLink 봇 계정(`...@im.bot`)은 일반적으로 일반 WeChat 그룹 메시지를 받을 수 없습니다. 게이트웨이 로그에서 그룹 메시지의 원시 수신 이벤트가 보이지 않는다면, 이는 Hermes가 아닌 iLink 쪽의 제한 때문입니다. |
| 미디어 다운로드/업로드 실패 | `cryptography`가 설치되어 있는지 확인하세요. `novac2c.cdn.weixin.qq.com`에 대한 네트워크 접근을 확인하세요 |
| `Blocked unsafe URL (SSRF protection)` | 아웃바운드 미디어 URL이 사설/내부 주소를 가리키고 있습니다. 오직 공개 URL만 허용됩니다 |
| 음성 메시지가 문자로 표시됨 | WeChat이 전사(transcription)를 제공한다면 어댑터는 해당 텍스트를 사용합니다. 이는 예상된 동작입니다 |
| 메시지가 중복되어 표시됨 | 어댑터는 메시지 ID로 중복을 제거합니다. 중복이 발생한다면 다수의 게이트웨이 인스턴스가 실행 중인지 확인하세요 |
| `iLink POST ... HTTP 4xx/5xx` | iLink 서비스에서 API 오류가 발생했습니다. 토큰의 유효성 및 네트워크 연결을 확인하세요 |
| 터미널 QR 코드가 렌더링되지 않음 | 메시징 엑스트라와 함께 재설치하세요: `pip install hermes-agent[messaging]`. 대안으로, QR 위에 출력된 URL을 브라우저로 여세요 |
