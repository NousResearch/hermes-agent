---
sidebar_position: 14
title: "WeCom (기업용 WeChat)"
description: "AI Bot WebSocket 게이트웨이를 통해 Hermes Agent를 WeCom에 연결하기"
---

# WeCom (기업용 WeChat)

Hermes를 Tencent의 기업용 메시징 플랫폼인 [WeCom](https://work.weixin.qq.com/) (企业微信, 기업 위챗)에 연결합니다. 어댑터는 실시간 양방향 통신을 위해 WeCom의 AI Bot WebSocket 게이트웨이를 사용하므로, 퍼블릭 엔드포인트나 웹훅이 필요하지 않습니다.

인바운드 웹훅 설정과 관련해서는 [WeCom 콜백](./wecom-callback.md)을 참조하세요.

## 사전 요구 사항

- WeCom 조직 계정
- WeCom 관리자 콘솔에서 생성한 AI Bot
- 해당 봇의 자격 증명 페이지에 있는 봇 ID 및 Secret
- Python 패키지: `aiohttp` 및 `httpx`

## 설정

### 1단계: AI Bot 만들기

#### 권장: 스캔하여 생성 (단일 명령어)

```bash
hermes gateway setup
```

**WeCom**을 선택하고 WeCom 모바일 앱으로 QR 코드를 스캔하세요. Hermes가 자동으로 올바른 권한을 가진 봇 애플리케이션을 생성하고 자격 증명을 저장합니다.

설정 마법사는 다음을 수행합니다:
1. 터미널에 QR 코드를 표시합니다
2. 사용자가 WeCom 모바일 앱으로 스캔할 때까지 기다립니다
3. 봇 ID와 Secret을 자동으로 가져옵니다
4. 접근 제어 구성을 안내합니다

#### 대안: 수동 설정

스캔하여 생성 기능을 사용할 수 없는 경우, 마법사가 수동 입력 모드로 전환됩니다:

1. [WeCom 관리자 콘솔](https://work.weixin.qq.com/wework_admin/frame)에 로그인합니다
2. **Applications(앱 관리)** → **Create Application(앱 생성)** → **AI Bot**으로 이동합니다
3. 봇 이름과 설명을 구성합니다
4. 자격 증명 페이지에서 **Bot ID**와 **Secret**을 복사합니다
5. `hermes gateway setup`을 실행하고 **WeCom**을 선택한 후, 프롬프트에 자격 증명을 입력합니다

:::warning
Bot Secret을 비공개로 유지하세요. 이를 가진 사람은 누구나 귀하의 봇을 사칭할 수 있습니다.
:::

### 2단계: Hermes 구성

#### 옵션 A: 대화형 설정 (권장)

```bash
hermes gateway setup
```

**WeCom**을 선택하고 프롬프트를 따르세요. 마법사가 다음 사항을 안내합니다:
- 봇 자격 증명 (QR 스캔 또는 수동 입력)
- 접근 제어 설정 (허용 목록, 페어링 모드 또는 전체 공개 접근)
- 알림을 위한 홈 채널 지정

#### 옵션 B: 수동 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# 선택 사항: 접근 제한
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# 선택 사항: cron/알림을 위한 홈 채널
WECOM_HOME_CHANNEL=chat_id
```

### 3단계: 게이트웨이 시작

```bash
hermes gateway
```

## 주요 기능

- **WebSocket 통신** — 영구적인 연결 유지, 퍼블릭 엔드포인트 불필요
- **다이렉트 메시지(DM) 및 그룹 메시징** — 설정 가능한 접근 정책
- **그룹별 발신자 허용 목록** — 각 그룹에서 봇과 상호작용할 수 있는 사람을 세밀하게 제어
- **미디어 지원** — 이미지, 파일, 음성, 비디오 업로드 및 다운로드 지원
- **AES 암호화 미디어** — 인바운드(수신) 첨부 파일에 대한 자동 암호 해독
- **인용(Quote) 문맥 유지** — 답장 스레드 문맥 보존
- **마크다운 렌더링** — 서식이 있는 텍스트 응답 지원
- **답장 연결성(Correlation)** — 인바운드 메시지의 맥락에 맞춘 직접 응답
- **자동 재연결** — 연결 끊김 시 지수 백오프(exponential backoff) 적용

:::note 스트리밍 및 입력 중 표시 관련
WeCom 어댑터는 각 응답을 완성된 단일 메시지 형태로 전달합니다. 응답을 토큰별로 쪼개어 실시간 스트리밍(streaming) **하지 않으며**, 사용자가 보는 화면에 '입력 중...' 상태 표시를 **띄우지도 않습니다**. 아래에 언급될 "답장 연결성"은 발신자의 요청에 답글(thread) 형태로 대답한다는 의미이지, 실시간 텍스트 출력을 의미하는 것은 아닙니다.
:::

## 구성 옵션

`config.yaml`의 `platforms.wecom.extra` 항목에서 다음 값을 설정하세요:

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `bot_id` | — | WeCom AI Bot ID (필수) |
| `secret` | — | WeCom AI Bot Secret (필수) |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocket 게이트웨이 URL |
| `dm_policy` | `open` | DM 접근 권한: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `open` | 그룹 접근 권한: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | DM을 허용할 사용자 ID (dm_policy=allowlist일 때) |
| `group_allow_from` | `[]` | 봇의 활동이 허용된 그룹 ID (group_policy=allowlist일 때) |
| `groups` | `{}` | 각 그룹에 대한 개별 접근 제한 설정 (아래 내용 참고) |

## 접근 정책

### DM 정책

봇에게 다이렉트 메시지(DM)를 보낼 수 있는 사람을 제어합니다:

| 값 | 동작 |
|-------|----------|
| `open` | 누구나 봇에게 DM을 보낼 수 있음 (기본값) |
| `allowlist` | `allow_from`에 지정된 사용자 ID만 DM을 보낼 수 있음 |
| `disabled` | 모든 DM 무시 |
| `pairing` | 페어링 모드 (초기 설정용) |

```bash
WECOM_DM_POLICY=allowlist
```

### 그룹 정책

봇이 응답할 그룹을 제어합니다:

| 값 | 동작 |
|-------|----------|
| `open` | 봇이 모든 그룹에서 응답함 (기본값) |
| `allowlist` | 봇이 `group_allow_from`에 나열된 그룹 ID에서만 응답함 |
| `disabled` | 모든 그룹 메시지 무시 |

```bash
WECOM_GROUP_POLICY=allowlist
```

### 그룹별 발신자 허용 목록

세분화된 제어를 위해, 특정 그룹 내에서 봇과 상호작용할 수 있는 사용자를 제한할 수 있습니다. 이는 `config.yaml`에 구성됩니다:

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

**동작 방식:**

1. `group_policy`와 `group_allow_from` 설정이 그룹 자체의 허용 여부를 1차적으로 판단합니다.
2. 그룹이 최상위(top-level) 검사를 통과한 경우, 해당 그룹에 `groups.<group_id>.allow_from` 목록이 존재하면 그 목록을 바탕으로 봇과 대화할 수 있는 구체적 발신자를 2차적으로 제한합니다.
3. 와일드카드 `"*"` 그룹 항목은 명시적으로 나열되지 않은 나머지 그룹들에 대한 기본 설정값 역할을 수행합니다.
4. 모든 사용자를 허용하려면 허용 목록에 `*` 와일드카드를 사용할 수 있습니다. 영문은 대소문자를 구분하지 않습니다.
5. `wecom:user:` 또는 `wecom:group:` 와 같은 접두사를 사용할 수 있으며 — 이들 접두사는 내부적으로 자동 필터링 됩니다.

그룹에 대해 `allow_from`이 명시적으로 구성되지 않은 경우, (그룹 자체가 최상위 정책 검사를 통과했다면) 그 그룹 안의 모든 사용자가 허용됩니다.

## 미디어 지원

### 인바운드 (수신)

어댑터는 사용자가 보낸 미디어 첨부 파일을 수신하고 에이전트 처리를 위해 로컬에 캐싱합니다:

| 유형 | 처리 방식 |
|------|-----------------|
| **이미지** | 다운로드되어 로컬에 캐시됨. URL 기반 및 base64 인코딩 이미지 모두 지원. |
| **파일** | 다운로드되어 로컬에 캐시됨. 원본 메시지의 파일 이름을 보존. |
| **음성** | 변환 가능한 텍스트 내용이 있다면 별도로 추출. |
| **혼합 메시지** | 텍스트와 이미지가 뒤섞인 WeCom 혼합 메시지 형식의 경우, 파싱(parsing) 작업을 통해 각 요소를 모두 추출. |

**인용 메시지 (Quoted messages):** 사용자가 답장한 원본 메시지의 미디어 또한 추출되므로 에이전트는 사용자가 어떤 내용에 답하고 있는지 컨텍스트를 파악할 수 있습니다.

### AES-Encrypted Media Decryption

WeCom은 들어오는 일부 미디어 첨부 파일을 AES-256-CBC 방식으로 암호화합니다. 어댑터는 이 작업을 자동으로 처리합니다:

- 수신된 미디어 항목에 `aeskey` 필드가 존재하면 어댑터는 암호화된 바이트를 다운로드한 후 PKCS#7 패딩이 적용된 AES-256-CBC를 사용하여 복호화합니다.
- 복호화에 쓰일 AES 키는 `aeskey` 필드값을 Base64 디코딩하여 얻습니다 (반드시 32바이트 크기여야 합니다).
- IV(초기화 벡터) 값은 키의 처음 16바이트 값을 추출해 도출합니다.
- 이 기능을 위해 `cryptography` Python 패키지가 필요합니다 (`pip install cryptography`).

별도의 설정은 필요 없으며, 암호화된 미디어가 들어올 때 자동으로 복호화가 진행됩니다.

### 아웃바운드 (발신)

| 메서드 | 발신 내용 | 크기 제한 |
|--------|--------------|------------|
| `send` | 마크다운 텍스트 메시지 | 4000자 |
| `send_image` / `send_image_file` | 네이티브 이미지 메시지 | 10 MB |
| `send_document` | 파일 첨부물 | 20 MB |
| `send_voice` | 음성 메시지 (네이티브 음성은 AMR 형식만 가능) | 2 MB |
| `send_video` | 비디오 메시지 | 10 MB |

**청크 단위 업로드 (Chunked upload):** 파일은 3단계 프로토콜(init → chunks → finish)을 통해 512KB 단위의 청크로 업로드됩니다. 이 과정은 어댑터가 자동으로 처리합니다.

**자동 다운그레이드:** 미디어가 네이티브 유형의 크기 제한을 초과하지만 절대 파일 제한(20MB)을 넘지 않을 경우, 자동으로 일반 첨부 파일 형식으로 전환되어 전송됩니다:

- 이미지 > 10 MB → 파일로 전송
- 비디오 > 10 MB → 파일로 전송
- 음성 > 2 MB → 파일로 전송
- 비 AMR 오디오 → 파일로 전송 (WeCom은 네이티브 음성에 대해 AMR만 지원)

절대 제한인 20MB를 초과하는 파일은 전송이 거부되며, 이를 알리는 메시지가 채팅창에 발송됩니다.

## 응답 모드 (Reply-Mode Responses)

WeCom 콜백을 통해 봇이 메시지를 수신하면 어댑터는 인바운드 요청 ID를 기억합니다. 요청 컨텍스트가 아직 활성 상태인 동안 응답이 전송되면 어댑터는 WeCom의 응답 모드(`aibot_respond_msg`)를 사용하여 해당 답변을 받은 원본 메시지에 바로 답글 형태로 연결해 줍니다. 이는 WeCom 클라이언트에서 훨씬 더 자연스러운 대화 환경을 제공합니다.

전체 응답은 한 번에 완성된 단일 메시지로 전달되며 — 토큰을 점진적으로 스트리밍하지 않습니다. 인바운드 요청 컨텍스트가 만료되었거나 사용할 수 없는 상태가 되면, 어댑터는 능동적 메시지 발송(`aibot_send_msg`) 방식으로 예외처리(fallback)를 진행합니다.

응답 모드는 미디어 전송 시에도 사용 가능합니다: 업로드된 미디어를 원본 메시지의 답글로 보낼 수 있습니다.

## 연결 및 재연결

어댑터는 `wss://openws.work.weixin.qq.com`에 위치한 WeCom의 게이트웨이와 지속적인 WebSocket 통신을 유지합니다.

### 연결 수명 주기

1. **Connect:** WebSocket 연결을 열고 bot_id와 secret을 포함한 `aibot_subscribe` 인증 프레임을 보냅니다.
2. **Heartbeat:** 연결 유지를 위해 애플리케이션 레벨의 핑(ping) 프레임을 30초마다 전송합니다.
3. **Listen:** 들어오는 프레임을 지속해서 읽고 메시지 콜백을 배분(dispatch)합니다.

### 재연결 동작

연결이 끊어졌을 때 어댑터는 지수 백오프(exponential backoff) 방식으로 재연결을 시도합니다:

| 시도 횟수 | 대기 시간 |
|---------|-------|
| 1번째 재시도 | 2초 |
| 2번째 재시도 | 5초 |
| 3번째 재시도 | 10초 |
| 4번째 재시도 | 30초 |
| 5번째 이상 재시도 | 60초 |

재연결에 성공할 때마다 백오프 카운터는 0으로 초기화됩니다. 계속 대기 상태에 빠지지 않게 하려고 연결 해제 시 처리 중이던 퓨처(Futures) 요청들은 일괄적으로 실패 처리합니다.

### 중복 제거 (Deduplication)

메시지 ID를 사용해 5분 이내에 도착한 메시지의 중복을 제거하며, 최대 1000개의 항목을 캐시에 저장합니다. 이를 통해 재연결 중이거나 일시적인 네트워크 장애 상황에서 메시지가 중복 처리되는 것을 방지합니다.

## 모든 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | WeCom AI Bot ID |
| `WECOM_SECRET` | ✅ | — | WeCom AI Bot Secret |
| `WECOM_ALLOWED_USERS` | — | _(비어있음)_ | 게이트웨이 레벨의 허용 목록을 위한 사용자 ID들의 쉼표 구분자 목록 |
| `WECOM_HOME_CHANNEL` | — | — | cron/알림 출력을 위한 채팅 ID |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | WebSocket 게이트웨이 URL |
| `WECOM_DM_POLICY` | — | `open` | DM 접근 정책 |
| `WECOM_GROUP_POLICY` | — | `open` | 그룹 접근 정책 |

## 문제 해결

| 문제 | 해결 방법 |
|---------|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | 2개의 환경 변수를 등록하거나 setup wizard를 이용해 설정하세요 |
| `WeCom startup failed: aiohttp not installed` | aiohttp 설치: `pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | httpx 설치: `pip install httpx` |
| `invalid secret (errcode=40013)` | 사용 중인 비밀키(secret)가 관리자 화면의 봇 자격 증명과 일치하는지 점검하세요 |
| `Timed out waiting for subscribe acknowledgement` | 컴퓨터가 `openws.work.weixin.qq.com` 으로 정상적인 네트워크 접속을 할 수 있는지 체크하세요 |
| 그룹 메시지에 봇이 묵묵부답입니다 | `group_policy` 세팅을 확인하고 대상 그룹의 ID가 `group_allow_from` 목록 안에 속해 있는지 점검하세요 |
| 특정 그룹 안에서 유독 몇몇 사용자의 메시지만 봇이 씹습니다 | `config.yaml`의 `groups` 섹션 안에 지정된 각 그룹별 `allow_from` 명단을 확인하세요 |
| 미디어 복호화에 실패합니다 | `cryptography` 설치: `pip install cryptography` |
| `cryptography is required for WeCom media decryption` | 인바운드 미디어가 AES로 암호화되어 있습니다. 설치하세요: `pip install cryptography` |
| 음성(Voice) 메시지가 파일로 전송됩니다 | WeCom은 네이티브 음성에 대해 AMR 포맷만을 허용합니다. 그 외의 형식은 시스템이 자동으로 일반 첨부 파일로 강등(downgrade)시켜 보냅니다. |
| `File too large` (파일 크기 초과) 에러 | WeCom에 등록되는 모든 파일의 크기는 20MB를 절대 넘을 수 없습니다. 압축하거나 용량을 분할하세요. |
| 이미지가 첨부 파일 형태로 날아갑니다 | 이미지의 용량이 10MB를 넘어갔으므로 시스템이 첨부 파일 형태로 강등(downgrade)시켰습니다. |
| `Timeout sending message to WeCom` | WebSocket 접속이 잠시 끊어졌을 확률이 큽니다. 재연결 시도 중인지 로그를 확인하세요. |
| `WeCom websocket closed during authentication` | 네트워크 연결 이슈이거나 잘못된 자격 증명을 썼습니다. bot_id와 secret을 점검하세요. |
