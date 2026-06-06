---
sidebar_position: 4
title: "Slack"
description: "Socket Mode를 사용하여 Hermes Agent를 Slack 봇으로 설정하기"
---

# Slack 설정

Hermes Agent를 Socket Mode를 사용하는 봇으로 Slack에 연결합니다. Socket Mode는 공개 HTTP 엔드포인트 대신 WebSocket을 사용하므로 Hermes 인스턴스가 외부에 공개될 필요가 없습니다 — 방화벽 내부, 노트북 또는 개인 서버에서도 작동합니다.

:::warning 클래식 Slack 앱 사용 중단됨
클래식 Slack 앱(RTM API 사용)은 **2025년 3월에 완전히 지원 중단되었습니다**. Hermes는 Socket Mode와 함께 최신 Bolt SDK를 사용합니다. 오래된 클래식 앱이 있다면 아래 단계에 따라 새 앱을 생성해야 합니다.
:::

## 개요

| 구성 요소 | 값 |
|-----------|-------|
| **라이브러리** | Python용 `slack-bolt` / `slack_sdk` (Socket Mode) |
| **연결** | WebSocket — 공개 URL 필요 없음 |
| **필요한 인증 토큰** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **사용자 식별** | Slack Member IDs (예: `U01ABC2DEF3`) |

---

## 1단계: Slack 앱 생성

가장 빠른 방법은 Hermes가 생성한 매니페스트를 붙여넣는 것입니다. 이 매니페스트는 모든 내장 슬래시 명령(`/btw`, `/stop`, `/model`, 등), 필요한 모든 OAuth 범위, 모든 이벤트 구독을 선언하고 Socket Mode를 활성화합니다 — 이 모든 것을 한 번에 처리합니다.

### 옵션 A: Hermes가 생성한 매니페스트에서 생성 (권장)

1. 매니페스트 생성:
   ```bash
   hermes slack manifest --write
   ```
   이 명령은 `~/.hermes/slack-manifest.json`을 작성하고 붙여넣기 안내를 출력합니다.
2. [https://api.slack.com/apps](https://api.slack.com/apps)로 이동 → **Create New App** → **From an app manifest** 클릭
3. 작업 공간을 선택하고, JSON 내용을 붙여넣은 뒤, 검토하고 **Next** → **Create** 클릭
4. **6단계: 작업 공간에 앱 설치**로 건너뛰세요. 매니페스트가 범위, 이벤트, 슬래시 명령을 모두 처리했습니다.

### 옵션 B: 처음부터 생성 (수동)

1. [https://api.slack.com/apps](https://api.slack.com/apps)로 이동
2. **Create New App** 클릭
3. **From scratch** 선택
4. 앱 이름(예: "Hermes Agent")을 입력하고 작업 공간 선택
5. **Create App** 클릭

앱의 **Basic Information** 페이지로 이동하게 됩니다. 아래 2~6단계를 계속 진행하세요.

---

## 2단계: 봇 토큰 범위 구성

사이드바에서 **Features → OAuth & Permissions**로 이동합니다. **Scopes → Bot Token Scopes**로 스크롤하여 다음을 추가합니다:

| 범위 | 목적 |
|-------|---------|
| `chat:write` | 봇으로서 메시지 전송 |
| `app_mentions:read` | 채널에서 @멘션될 때 감지 |
| `channels:history` | 봇이 있는 공개 채널의 메시지 읽기 |
| `channels:read` | 공개 채널 목록 및 정보 가져오기 |
| `groups:history` | 봇이 초대된 비공개 채널의 메시지 읽기 |
| `im:history` | 다이렉트 메시지 기록 읽기 |
| `im:read` | 기본 DM 정보 보기 |
| `im:write` | DM 열기 및 관리 |
| `users:read` | 사용자 정보 조회 |
| `files:read` | 음성 메모/오디오를 포함한 첨부 파일 읽기 및 다운로드 |
| `files:write` | 파일(이미지, 오디오, 문서) 업로드 |

:::caution 누락된 범위 = 누락된 기능
`channels:history` 및 `groups:history`가 없으면 봇은 **채널에서 메시지를 받을 수 없으며** — 오직 DM에서만 작동합니다. `files:read`가 없으면 Hermes는 채팅은 가능하지만 **사용자가 업로드한 첨부 파일을 제대로 읽을 수 없습니다**.
이것들이 가장 흔히 누락되는 범위입니다.
:::

**선택적 범위:**

| 범위 | 목적 |
|-------|---------|
| `groups:read` | 비공개 채널 목록 및 정보 가져오기 |

---

## 3단계: Socket Mode 활성화

Socket Mode를 사용하면 공개 URL을 요구하는 대신 봇이 WebSocket을 통해 연결할 수 있습니다.

1. 사이드바에서 **Settings → Socket Mode**로 이동
2. **Enable Socket Mode**를 ON으로 전환
3. **App-Level Token** 생성 프롬프트가 표시됩니다:
   - 이름을 `hermes-socket`과 같이 지정 (이름은 중요하지 않음)
   - **`connections:write`** 범위를 추가
   - **Generate** 클릭
4. **토큰을 복사합니다** — `xapp-`로 시작합니다. 이것이 당신의 `SLACK_APP_TOKEN`입니다.

:::tip
언제든지 **Settings → Basic Information → App-Level Tokens**에서 앱 레벨 토큰을 찾거나 다시 생성할 수 있습니다.
:::

---

## 4단계: 이벤트 구독

이 단계는 매우 중요합니다 — 봇이 어떤 메시지를 볼 수 있는지를 제어합니다.

1. 사이드바에서 **Features → Event Subscriptions**로 이동
2. **Enable Events**를 ON으로 전환
3. **Subscribe to bot events**를 확장하고 다음을 추가:

| 이벤트 | 필수 여부 | 목적 |
|-------|-----------|---------|
| `message.im` | **예** | 봇이 다이렉트 메시지를 받음 |
| `message.channels` | **예** | 봇이 추가된 **공개** 채널에서 메시지를 받음 |
| `message.groups` | **권장** | 봇이 초대된 **비공개** 채널에서 메시지를 받음 |
| `app_mention` | **예** | 봇이 @멘션될 때 Bolt SDK 오류 방지 |

4. 페이지 하단의 **Save Changes** 클릭

:::danger 이벤트 구독 누락은 가장 흔한 설정 문제입니다
봇이 DM에서는 작동하지만 **채널에서는 작동하지 않는다면**, 거의 확실하게 `message.channels` (공개 채널용) 및/또는 `message.groups` (비공개 채널용)를 추가하는 것을 잊은 것입니다.
이 이벤트들이 없으면 Slack은 채널 메시지를 봇에게 전혀 전달하지 않습니다.
:::

---

## 5단계: Messages 탭 활성화

이 단계는 봇으로의 다이렉트 메시지를 활성화합니다. 이 과정 없이는 사용자가 봇에게 DM을 보내려고 할 때 **"이 앱으로 메시지를 보내는 것이 꺼져 있습니다(Sending messages to this app has been turned off)"**라는 메시지를 보게 됩니다.

1. 사이드바에서 **Features → App Home**으로 이동
2. **Show Tabs**로 스크롤
3. **Messages Tab**을 ON으로 전환
4. **"Allow users to send Slash commands and messages from the messages tab"** 체크

:::danger 이 단계 없이는 DM이 완전히 차단됩니다
모든 올바른 범위와 이벤트 구독을 갖추고 있더라도 Messages 탭이 활성화되지 않으면 Slack은 사용자가 봇에게 다이렉트 메시지를 보내는 것을 허용하지 않습니다. 이는 Hermes 설정 문제가 아니라 Slack 플랫폼 요구사항입니다.
:::

---

## 6단계: 작업 공간에 앱 설치

1. 사이드바에서 **Settings → Install App**으로 이동
2. **Install to Workspace** 클릭
3. 권한을 검토하고 **Allow** 클릭
4. 인증 후, `xoxb-`로 시작하는 **Bot User OAuth Token**을 볼 수 있습니다.
5. **이 토큰을 복사합니다** — 이것이 당신의 `SLACK_BOT_TOKEN`입니다.

:::tip
나중에 범위나 이벤트 구독을 변경할 경우, 변경 사항을 적용하려면 **앱을 다시 설치해야 합니다**. Install App 페이지에 재설치를 유도하는 배너가 표시됩니다.
:::

---

## 7단계: 허용 목록을 위한 사용자 ID 찾기

Hermes는 허용 목록(allowlist)을 위해 사용자 이름이나 표시 이름이 아닌 Slack **Member ID**를 사용합니다.

Member ID를 찾는 방법:

1. Slack에서 사용자의 이름이나 아바타를 클릭
2. **View full profile** 클릭
3. **⋮** (더보기) 버튼 클릭
4. **Copy member ID** 선택

Member ID는 `U01ABC2DEF3`과 같은 형태입니다. 최소한 본인의 Member ID가 필요합니다.

---

## 8단계: Hermes 구성

`~/.hermes/.env` 파일에 다음을 추가합니다:

```bash
# 필수
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # 쉼표로 구분된 Member ID

# 선택 사항
SLACK_HOME_CHANNEL=C01234567890              # 크론/예약 메시지의 기본 채널
SLACK_HOME_CHANNEL_NAME=general              # 홈 채널의 읽기 쉬운 이름 (선택 사항)
```

또는 대화형 설정을 실행합니다:

```bash
hermes gateway setup    # 프롬프트가 나타나면 Slack 선택
```

그런 다음 게이트웨이를 시작합니다:

```bash
hermes gateway              # 포그라운드 실행
hermes gateway install      # 사용자 서비스로 설치
sudo hermes gateway install --system   # Linux 전용: 부팅 시 시스템 서비스로 실행
```

---

## 9단계: 채널에 봇 초대

게이트웨이를 시작한 후, 응답을 원하는 모든 채널에 **봇을 초대해야** 합니다:

```
/invite @Hermes Agent
```

봇은 자동으로 채널에 참여하지 **않습니다**. 각 채널마다 개별적으로 초대해야 합니다.

---

## 슬래시 명령

모든 Hermes 명령(`/btw`, `/stop`, `/new`, `/model`, `/help` 등)은 Telegram 및 Discord에서 작동하는 것과 똑같이 기본 Slack 슬래시 명령으로 작동합니다. Slack에서 `/`를 입력하면 자동 완성 선택기에 설명과 함께 모든 Hermes 명령이 나열됩니다.

내부 동작: Hermes는 [`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py)의 모든 명령을 슬래시 명령으로 선언하는 생성된 Slack 앱 매니페스트(1단계, 옵션 A 참조)와 함께 제공됩니다. Socket Mode에서 Slack은 매니페스트의 `url` 필드에 관계없이 WebSocket을 통해 명령 이벤트를 라우팅합니다.

### 업데이트 후 슬래시 명령 새로 고침

Hermes가 새 명령을 추가할 때(예: `hermes update` 후), 매니페스트를 다시 생성하고 Slack 앱을 업데이트하세요:

```bash
hermes slack manifest --write
```

그런 다음 Slack에서:
1. [https://api.slack.com/apps](https://api.slack.com/apps) 열기 → 내 Hermes 앱 선택
2. **Features → App Manifest → Edit**
3. `~/.hermes/slack-manifest.json`의 새 내용을 붙여넣기
4. **Save** 클릭. 범위나 슬래시 명령이 변경된 경우 Slack이 앱을 다시 설치하라는 메시지를 표시합니다.

### 레거시 `/hermes <subcommand>` 계속 지원

이전 매니페스트와의 역호환성을 위해 계속해서 `/hermes btw run the tests`를 입력할 수 있습니다 — Hermes는 이를 `/btw run the tests`와 동일하게 라우팅합니다. 자유로운 형태의 질문도 작동합니다: `/hermes what's the weather?`는 일반 메시지로 취급됩니다.

### 스레드 내에서 명령 사용 (`!cmd` 접두사)

Slack 자체는 스레드 답글 내에서 기본 슬래시 명령을 차단합니다 — 스레드에서 `/queue`를 시도하면 Slack은 *"/queue is not supported in threads. Sorry!"*라고 응답합니다. 이를 다시 활성화하는 앱 측 설정은 없으며, Slack은 이를 절대 Hermes에 전달하지 않습니다.

해결 방법으로, Hermes는 스레드(및 기타 모든 곳)에서 작동하는 대체 명령 접두사로 선행 `!`를 인식합니다. 일반 스레드 답글로 `!queue`, `!stop`, `!model gpt-5.4` 등을 입력하면 Hermes는 이를 슬래시 형태와 동일하게 취급하고 같은 스레드에 응답합니다.

첫 번째 토큰만 알려진 명령 목록과 대조되므로, `!nice work`와 같은 일상적인 메시지는 변경 없이 에이전트에게 전달됩니다.

### 고급: 슬래시 명령 배열만 내보내기

수동으로 Slack 매니페스트를 유지 관리하고 슬래시 명령 목록만 필요한 경우:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

기존 매니페스트의 `features.slash_commands` 키에 해당 배열을 붙여넣으세요.

---

## 봇의 응답 방식

다양한 상황에서 Hermes가 어떻게 동작하는지 이해하기:

| 컨텍스트 | 동작 |
|---------|----------|
| **DM** | 봇은 모든 메시지에 응답합니다 — @멘션이 필요하지 않습니다 |
| **채널** | 봇은 **@멘션될 때만 응답합니다** (예: `@Hermes Agent 지금 몇 시야?`). 채널에서 Hermes는 해당 메시지에 연결된 스레드로 답글을 단다. |
| **스레드** | 기존 스레드 내에서 Hermes를 @멘션하면, 그 스레드 내에 응답합니다. 일단 봇이 스레드에 활성 세션을 가지면, **해당 스레드 내의 후속 답글은 @멘션이 필요하지 않습니다** — 봇이 자연스럽게 대화를 따라갑니다. |

:::tip
채널에서는 항상 대화를 시작할 때 봇을 @멘션하세요. 일단 봇이 스레드에 활성화되면, 멘션하지 않고도 해당 스레드에서 대답할 수 있습니다. 바쁜 채널에서의 소음을 방지하기 위해 스레드 밖에서는 @멘션이 없는 메시지가 무시됩니다.
:::

---

## 구성 옵션

8단계의 필수 환경 변수 외에도, `~/.hermes/config.yaml`을 통해 Slack 봇 동작을 사용자 정의할 수 있습니다.

### 스레드 및 답글 동작

```yaml
platforms:
  slack:
    # 다중 파트 응답이 스레드로 연결되는 방식을 제어
    # "off"   — 원래 메시지에 스레드로 응답하지 않음
    # "first" — 첫 번째 청크는 사용자의 메시지에 스레드로 연결 (기본값)
    # "all"   — 모든 청크를 사용자의 메시지에 스레드로 연결
    reply_to_mode: "first"

    extra:
      # 스레드로 답글을 달지 여부 (기본값: true).
      # false일 경우, 채널 메시지는 스레드 대신 채널에 직접 답글로 달립니다.
      # 기존 스레드 내부의 메시지는 여전히 스레드 내에서 응답합니다.
      reply_in_thread: true

      # 스레드 답글을 기본 채널에도 게시합니다.
      # (Slack의 "Also send to channel" 기능).
      # 첫 번째 응답의 첫 번째 청크만 방송됩니다.
      reply_broadcast: false
```

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | 다중 파트 메시지의 스레딩 모드: `"off"`, `"first"`, 또는 `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | `false`일 때 채널 메시지는 스레드 대신 직접 답글로 받습니다. 기존 스레드 내 메시지는 여전히 스레드에서 답글을 받습니다. |
| `platforms.slack.extra.reply_broadcast` | `false` | `true`일 때 스레드 답글은 기본 채널에도 게시됩니다. 첫 번째 청크만 방송됩니다. |

### 세션 분리

```yaml
# 전역 설정 — Slack 및 기타 모든 플랫폼에 적용
group_sessions_per_user: true
```

`true`(기본값)로 설정된 경우 공유 채널의 각 사용자는 독립된 대화 세션을 갖게 됩니다. `#general`에서 Hermes와 대화하는 두 명의 사람은 별도의 기록과 컨텍스트를 가집니다.

전체 채널이 단일 대화 세션을 공유하는 협업 모드를 원한다면 `false`로 설정하십시오. 이 경우 사용자들이 컨텍스트 증가와 토큰 비용을 공유하게 되며, 한 사용자의 `/reset`이 모든 사람의 세션을 초기화한다는 점에 유의하세요.

### 멘션 및 트리거 동작

```yaml
slack:
  # 채널에서 @멘션 필수 (이것이 기본 동작입니다;
  # Slack 어댑터는 채널에서 @멘션 차단을 무조건 강제하지만,
  # 다른 플랫폼과의 일관성을 위해 명시적으로 설정할 수 있습니다)
  require_mention: true

  # 스레드 자동 참여 방지: 명시적인 @멘션이 포함된 채널 메시지에만
  # 응답합니다. 이것이 OFF일 경우(기본값), Slack은 "자동 참여"할 수 있습니다 —
  # 스레드에서의 과거 멘션을 기억하고 봇 메시지 답글에 후속 처리를 하며,
  # 새로운 멘션 없이도 활성 세션을 재개합니다. strict_mention을 ON으로 하면
  # 모든 새로운 채널 메시지는 Hermes가 응답하기 전에 반드시 봇을 @멘션해야 합니다.
  strict_mention: false

  # 봇을 트리거하는 사용자 정의 멘션 패턴
  # (기본 @멘션 감지에 추가로 작동)
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # 보내는 모든 메시지 앞에 추가되는 텍스트
  reply_prefix: ""
```

:::tip 언제 `strict_mention`을 사용해야 할까요
Slack의 기본 동작인 "봇이 이 스레드를 기억함"이 사용자를 놀라게 할 수 있는 바쁜 작업 공간에서는 이를 `true`로 설정하세요 — 예를 들어 봇이 처음에 도움을 주었지만 다시 명시적으로 부르기 전까지는 조용히 있기를 바라는 긴 기술 지원 스레드의 경우. DM 및 활성 인터랙티브 세션은 영향을 받지 않습니다.
:::

:::info
Slack은 두 가지 패턴을 모두 지원합니다: 대화를 시작하려면 기본적으로 `@멘션`이 필요하지만, `SLACK_FREE_RESPONSE_CHANNELS`(쉼표로 구분된 채널 ID) 또는 `config.yaml`의 `slack.free_response_channels`를 통해 특정 채널을 예외로 지정할 수 있습니다. 봇이 스레드에 활성 세션을 가지면 후속 스레드 답글에는 멘션이 필요하지 않습니다. DM에서 봇은 멘션 없이 항상 응답합니다.
:::

### 채널 허용 목록 (`allowed_channels`)

봇을 고정된 일련의 Slack 채널로 제한합니다 — 봇이 많은 채널에 초대되었지만 소수에서만 응답해야 할 때 유용합니다. 이 값을 설정하면, 이 목록에 없는 채널에서 온 메시지는 봇이 `@멘션`되더라도 **조용히 무시됩니다**.

**DM은 이 필터에서 제외**되므로 권한이 있는 사용자는 항상 다이렉트 메시지로 봇에 접근할 수 있습니다.

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

또는 환경 변수(쉼표로 구분)를 통해:

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

동작 방식:

- 비어 있음 / 설정되지 않음 → 제한 없음 (완전한 역호환성 유지).
- 비어 있지 않음 → 채널 ID가 반드시 목록에 있어야 하며, 그렇지 않으면 메시지는 다른 필터(멘션 필요 여부, `free_response_channels` 등)가 실행되기 전에 삭제됩니다.
- Slack 채널 ID는 `C` (공개), `G` (비공개) 또는 `D` (DM)로 시작합니다. Slack UI의 "Open channel details" → "About" 패널이나 API를 통해 찾아보세요.

참고: [admin/user slash command split](../../reference/slash-commands.md#permissions-and-adminuser-split).

### 승인되지 않은 사용자 처리

```yaml
slack:
  # 승인되지 않은 사용자(SLACK_ALLOWED_USERS에 없는 사용자)가 봇에 DM을 보낼 때의 동작
  # "pair"   — 페어링 코드를 묻는 프롬프트 표시 (기본값)
  # "ignore" — 메시지를 조용히 무시
  unauthorized_dm_behavior: "pair"
```

모든 플랫폼에 대해 이를 전역으로 설정할 수도 있습니다:

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` 아래의 플랫폼별 설정이 전역 설정보다 우선합니다.

### 음성 기록(Transcription)

```yaml
# 전역 설정 — 수신 음성 메시지의 자동 텍스트 변환 활성화/비활성화
stt_enabled: true
```

`true`(기본값)인 경우, 들어오는 오디오 메시지는 에이전트가 처리하기 전에 구성된 STT 제공자(로컬 `faster-whisper`, Groq Whisper(`GROQ_API_KEY`) 또는 OpenAI Whisper(`VOICE_TOOLS_OPENAI_KEY`))를 사용하여 자동으로 전사(transcribe)됩니다.

### 전체 예제

```yaml
# 전역 게이트웨이 설정
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack 전용 설정
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# 플랫폼 구성
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---

## 홈 채널

Hermes가 예약된 메시지, 크론 작업 결과 및 기타 선제적 알림을 전달할 채널 ID로 `SLACK_HOME_CHANNEL`을 설정하세요. 채널 ID를 찾으려면:

1. Slack에서 채널 이름을 마우스 오른쪽 버튼으로 클릭
2. **View channel details** 클릭
3. 맨 아래로 스크롤 — 거기에 채널 ID가 표시됩니다.

```bash
SLACK_HOME_CHANNEL=C01234567890
```

반드시 봇이 **채널에 초대되어 있어야** 합니다 (`/invite @Hermes Agent`).

---

## 다중 작업 공간 지원

Hermes는 단일 게이트웨이 인스턴스를 사용하여 **여러 Slack 작업 공간**에 동시에 연결할 수 있습니다. 각 작업 공간은 독립적인 봇 사용자 ID로 인증됩니다.

### 구성

`SLACK_BOT_TOKEN`에 여러 봇 토큰을 **쉼표로 구분된 목록**으로 제공합니다:

```bash
# 여러 봇 토큰 — 작업 공간당 하나씩
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# Socket Mode에는 여전히 단일 앱 레벨 토큰이 사용됩니다.
SLACK_APP_TOKEN=xapp-your-app-token
```

또는 `~/.hermes/config.yaml`에서:

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuth 토큰 파일

환경이나 구성의 토큰 외에도 Hermes는 다음 위치의 **OAuth 토큰 파일**에서도 토큰을 로드합니다:

```
~/.hermes/slack_tokens.json
```

이 파일은 팀 ID를 토큰 항목에 매핑하는 JSON 객체입니다:

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

이 파일의 토큰은 `SLACK_BOT_TOKEN`을 통해 지정된 모든 토큰과 병합됩니다. 중복 토큰은 자동으로 제거됩니다.

### 작동 방식

- 목록의 **첫 번째 토큰**이 기본 토큰이며 Socket Mode 연결(AsyncApp)에 사용됩니다.
- 각 토큰은 시작 시 `auth.test`를 통해 인증됩니다. 게이트웨이는 각 `team_id`를 고유한 `WebClient` 및 `bot_user_id`에 매핑합니다.
- 메시지가 도착하면 Hermes는 적절한 작업 공간 전용 클라이언트를 사용하여 응답합니다.
- 첫 번째 토큰의 기본 `bot_user_id`는 단일 봇 아이덴티티를 기대하는 기능들과의 역호환성을 위해 사용됩니다.

---

## 음성 메시지

Hermes는 Slack에서 음성을 지원합니다:

- **수신:** 음성/오디오 메시지는 설정된 STT 제공자(로컬 `faster-whisper`, Groq Whisper(`GROQ_API_KEY`) 또는 OpenAI Whisper(`VOICE_TOOLS_OPENAI_KEY`))를 통해 자동으로 전사됩니다.
- **발신:** TTS 응답은 오디오 파일 첨부로 전송됩니다.

---

## 채널별 프롬프트

특정 Slack 채널에 임시 시스템 프롬프트를 할당하세요. 프롬프트는 모든 턴마다 런타임에 주입되며 — 대화 기록 히스토리에 유지되지 않으므로 — 변경 사항이 즉시 적용됩니다.

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      당신은 연구 보조원입니다. 학술적 출처, 인용, 
      그리고 간결한 통합에 집중하세요.
    "C02ENGINEERING": |
      코드 리뷰 모드입니다. 에지 케이스와 성능 관련
      영향에 대해 정밀하게 다뤄주세요.
```

키는 Slack 채널 ID입니다(채널 세부정보 → "About" → 하단으로 스크롤하여 확인). 일치하는 채널의 모든 메시지는 프롬프트를 임시 시스템 명령으로 주입받습니다.

## 채널별 스킬 바인딩

특정 채널이나 DM에서 새 세션이 시작될 때마다 스킬을 자동 로드합니다. 채널별 프롬프트(모든 턴에 주입됨)와 달리 스킬 바인딩은 **세션 시작 시** 스킬 내용을 사용자 메시지로 주입합니다 — 이는 대화 기록의 일부가 되며 이후 턴에서 다시 로드할 필요가 없습니다.

이 기능은 (플래시 카드, 도메인 특화 Q&A 봇, 지원(support) 분류 채널 등) 짧은 답글마다 모델의 스킬 선택기가 로드 여부를 결정하는 것을 원치 않는 전용 목적의 DM이나 채널에 이상적입니다.

```yaml
slack:
  channel_skill_bindings:
    # DM 채널 — 항상 "german-flashcards" 모드로 실행
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # 리서치 채널 — 여러 스킬을 순서대로 사전 로드
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # 짧은 형태: 단일 스킬을 문자열로
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

참고:
- 바인딩은 채널 ID로 일치시킵니다. 바인딩된 채널 내 스레드 메시지의 경우 스레드는 부모 채널의 바인딩을 상속합니다.
- 스킬은 세션 시작 시(새 세션 또는 자동 초기화 후)에만 로드됩니다. 바인딩을 변경하면 `/new`를 실행하거나 세션이 자동 리셋되기를 기다려야 적용됩니다.
- 스킬 명령 외에 채널별 어조/제약 사항을 원할 경우 `channel_prompts`와 결합하세요.

## 문제 해결

| 문제 | 해결책 |
|---------|----------|
| 봇이 DM에 응답하지 않음 | 이벤트 구독에 `message.im`이 있는지, 앱이 다시 설치되었는지 확인 |
| DM에서는 작동하지만 채널에서는 작동하지 않음 | **가장 흔한 문제.** 이벤트 구독에 `message.channels` 및 `message.groups`를 추가하고 앱을 다시 설치한 다음, `/invite @Hermes Agent`로 봇을 채널에 초대 |
| 채널에서 @멘션에 응답하지 않음 | 1) `message.channels` 이벤트가 구독되었는지 확인. 2) 봇이 채널에 초대되어야 함. 3) `channels:history` 범위가 추가되었는지 확인. 4) 범위/이벤트 변경 후 앱 재설치 |
| 비공개 채널의 메시지 무시됨 | `message.groups` 이벤트 구독과 `groups:history` 범위를 모두 추가한 다음 앱을 재설치하고 봇을 `/invite` |
| DM에서 "이 앱으로 메시지를 보내는 것이 꺼져 있습니다" 표시 | 앱 홈 설정에서 **Messages Tab**을 활성화(5단계 참조) |
| "not_authed" 또는 "invalid_auth" 오류 | Bot Token과 App Token을 다시 생성하고 `.env` 업데이트 |
| 봇이 응답하지만 채널에 글을 쓸 수 없음 | `/invite @Hermes Agent`로 채널에 봇을 초대 |
| 채팅은 가능하지만 업로드된 이미지/파일을 읽을 수 없음 | `files:read`를 추가한 다음 앱을 **재설치**. Slack에서 범위/인증/권한 실패를 반환하면 Hermes가 теперь 채팅 내에서 첨부 파일 접근 진단 정보를 제공합니다. |
| `missing_scope` 오류 | OAuth & Permissions에서 필요한 범위를 추가한 다음 앱을 **재설치** |
| 소켓 연결이 자주 끊김 | 네트워크 확인; Bolt가 자동으로 재연결을 시도하지만 불안정한 연결은 지연을 유발합니다 |
| 범위/이벤트를 변경했지만 아무것도 바뀌지 않음 | 범위나 이벤트 구독을 변경한 후 작업 공간에 앱을 **반드시 다시 설치**해야 합니다 |

### 빠른 체크리스트

봇이 채널에서 작동하지 않는 경우, 다음 사항을 **모두** 확인하세요:

1. ✅ `message.channels` 이벤트 구독됨 (공개 채널용)
2. ✅ `message.groups` 이벤트 구독됨 (비공개 채널용)
3. ✅ `app_mention` 이벤트 구독됨
4. ✅ `channels:history` 범위 추가됨 (공개 채널용)
5. ✅ `groups:history` 범위 추가됨 (비공개 채널용)
6. ✅ 범위/이벤트 추가 후 앱 **재설치**됨
7. ✅ 채널에 봇이 **초대**됨 (`/invite @Hermes Agent`)
8. ✅ 메시지에서 봇을 **@멘션**하고 있음

---

## 보안

:::warning
승인된 사용자의 Member ID로 **항상 `SLACK_ALLOWED_USERS`를 설정하세요**. 이 설정이 없으면 게이트웨이는 보안 조치로서 기본적으로 **모든 메시지를 거부**합니다. 봇 토큰을 절대 공유하지 마세요 — 비밀번호처럼 취급하십시오.
:::

- 토큰은 `~/.hermes/.env` 파일(파일 권한 `600`)에 저장해야 합니다.
- Slack 앱 설정을 통해 주기적으로 토큰을 순환하세요.
- Hermes 구성 디렉터리에 접근할 수 있는 사람을 감사(audit)하세요.
- Socket Mode는 노출된 공개 엔드포인트가 없음을 의미하며 공격 표면을 하나 줄여줍니다.
