---
sidebar_position: 4
title: "Slack"
description: "Socket Mode를 사용해 Hermes Agent를 Slack 봇으로 설정"
---

# Slack 설정

Socket Mode를 사용해 Hermes Agent를 Slack 봇으로 연결합니다. Socket Mode는 공개 HTTP 엔드포인트 대신
WebSocket을 사용하므로 Hermes 인스턴스를 공개적으로 접근 가능하게 만들 필요가 없습니다. 방화벽 뒤,
노트북 또는 비공개 서버에서도 작동합니다.

:::warning 클래식 Slack 앱 지원 중단
RTM API를 사용하는 클래식 Slack 앱은 **2025년 3월에 완전히 지원이 중단되었습니다**. Hermes는
Socket Mode와 최신 Bolt SDK를 사용합니다. 오래된 클래식 앱이 있다면 아래 단계에 따라 새 앱을
만들어야 합니다.
:::

## 개요

| 구성 요소 | 값 |
|-----------|-------|
| **라이브러리** | Python용 `slack-bolt` / `slack_sdk` (Socket Mode) |
| **연결** | WebSocket — 공개 URL 불필요 |
| **필요한 인증 토큰** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **사용자 식별** | Slack Member ID (예: `U01ABC2DEF3`) |

---

## 1단계: Slack 앱 만들기

가장 빠른 방법은 Hermes가 생성한 매니페스트를 붙여 넣는 것입니다. 매니페스트는 모든 기본 슬래시
명령(`/btw`, `/stop`, `/model`, …), 필요한 모든 OAuth 스코프, 모든 이벤트 구독을 선언하고 Socket
Mode를 한 번에 활성화합니다.

### 옵션 A: Hermes가 생성한 매니페스트 사용 (권장)

1. 매니페스트를 생성합니다. 새 Slack 앱은 Agent 보기를 사용해야 합니다:
   ```bash
   hermes slack manifest --agent-view --write
   ```
   이 명령은 `~/.hermes/slack-manifest.json`을 기록하고 붙여 넣기 방법을 출력합니다. Slack의 기존
   Assistant 보기를 계속 사용하는 앱은 마이그레이션할 준비가 될 때까지 `--agent-view`를 생략할 수
   있습니다.

   기존 UTF-8 텍스트 또는 Markdown 파일에서 Slack의 긴 앱 설명을 채우려면 `--long-description-file`을
   추가합니다:

   ```bash
   hermes slack manifest --agent-view \
     --long-description-file AGENTS.md --write
   ```

   파일 내용은 Slack의 175~4,000자 범위 안에서 정확히 보존됩니다. 대신 인라인 텍스트에는
   `--long-description "..."`를 사용합니다. 인라인 옵션과 파일 옵션은 상호 배타적이며
   `--slashes-only`와 함께 사용할 수 없습니다.
2. [https://api.slack.com/apps](https://api.slack.com/apps) → **Create New App** →
   **From an app manifest**로 이동합니다.
3. 워크스페이스를 선택하고 JSON 내용을 붙여 넣은 뒤 검토하고 **Next** → **Create**를 클릭합니다.
4. **6단계: 워크스페이스에 앱 설치**로 건너뜁니다. 매니페스트가 스코프, 이벤트 및 슬래시 명령을
   자동으로 처리했습니다.

### 옵션 B: 처음부터 수동으로 만들기

1. [https://api.slack.com/apps](https://api.slack.com/apps)로 이동합니다.
2. **Create New App**을 클릭합니다.
3. **From scratch**를 선택합니다.
4. 앱 이름(예: "Hermes Agent")을 입력하고 워크스페이스를 선택합니다.
5. **Create App**을 클릭합니다.

앱의 **Basic Information** 페이지가 열립니다. 아래 2~6단계를 계속 진행합니다.

---

## 2단계: Bot Token 스코프 설정

사이드바에서 **Features → OAuth & Permissions**로 이동합니다. **Scopes → Bot Token Scopes**까지
스크롤한 뒤 다음을 추가합니다.

| 스코프 | 용도 |
|-------|---------|
| `chat:write` | 봇으로 메시지 보내기 |
| `app_mentions:read` | 채널에서 @멘션될 때 감지 |
| `channels:history` | 봇이 참여한 공개 채널의 메시지 읽기 |
| `channels:read` | 공개 채널 목록 및 정보 조회 |
| `groups:history` | 초대된 비공개 채널의 메시지 읽기 |
| `im:history` | 다이렉트 메시지 기록 읽기 |
| `im:read` | 기본 DM 정보 보기 |
| `im:write` | DM 열기 및 관리 |
| `mpim:history` | 그룹 다이렉트 메시지 기록 읽기 |
| `mpim:read` | 기본 그룹 DM 정보 보기 |
| `users:read` | 사용자 정보 조회 |
| `files:read` | 음성 메모/오디오를 포함한 첨부 파일 읽기 및 다운로드 |
| `files:write` | 파일(이미지, 오디오, 문서) 업로드 |

:::caution 스코프가 없으면 기능도 없습니다
`channels:history`와 `groups:history`가 없으면 봇은 **채널의 메시지를 받지 못하며** DM에서만
작동합니다. `files:read`가 없으면 Hermes는 대화할 수는 있지만 **사용자가 업로드한 첨부 파일을
안정적으로 읽을 수 없습니다**. 가장 자주 빠뜨리는 스코프입니다.
:::

**선택적 스코프:**

| 스코프 | 용도 |
|-------|---------|
| `groups:read` | 비공개 채널 목록 및 정보 조회 |
| `assistant:write` | 메시지를 처리하는 동안 봇 이름 옆에 작업 상태 표시줄("is thinking…") 표시. 이 스코프가 없으면 `assistant.threads.setStatus` 호출이 조용히 실패하고 Slack이 자체 일반 플레이스홀더("Finding answers…", "Reviewing findings…", …)를 표시합니다. Hermes는 텍스트를 제어하지 않습니다. `typing_status_text`가 실제로 표시되려면 필요합니다. |

---

## 3단계: Socket Mode 활성화

Socket Mode를 사용하면 공개 URL 없이 WebSocket으로 봇을 연결할 수 있습니다.

1. 사이드바에서 **Settings → Socket Mode**로 이동합니다.
2. **Enable Socket Mode**를 ON으로 전환합니다.
3. **App-Level Token**을 만들라는 메시지가 표시됩니다.
   - `hermes-socket`처럼 이름을 지정합니다(이름은 중요하지 않음).
   - **`connections:write`** 스코프를 추가합니다.
   - **Generate**를 클릭합니다.
4. **토큰을 복사합니다** — `xapp-`으로 시작합니다. 이 값이 `SLACK_APP_TOKEN`입니다.

:::tip
앱 수준 토큰은 언제든 **Settings → Basic Information → App-Level Tokens**에서 확인하거나
다시 생성할 수 있습니다.
:::

---

## 4단계: 이벤트 구독

이 단계는 매우 중요합니다. 봇이 볼 수 있는 메시지를 결정합니다.

1. 사이드바에서 **Features → Event Subscriptions**로 이동합니다.
2. **Enable Events**를 ON으로 전환합니다.
3. **Subscribe to bot events**를 펼치고 다음을 추가합니다.

| 이벤트 | 필수 여부 | 용도 |
|---------|-----------|---------|
| `message.im` | **예** | 봇이 다이렉트 메시지 수신 |
| `message.mpim` | **예** | 봇이 추가된 **그룹 DM**의 메시지 수신 |
| `message.channels` | **예** | 봇이 추가된 **공개** 채널의 메시지 수신 |
| `message.groups` | **권장** | 봇이 초대된 **비공개** 채널의 메시지 수신 |
| `app_mention` | **예** | 봇이 @멘션될 때 Bolt SDK 오류 방지 |

4. 페이지 아래쪽에서 **Save Changes**를 클릭합니다.

:::danger 이벤트 구독 누락은 설정 문제 1순위입니다
봇이 DM에서는 작동하지만 **채널에서는 작동하지 않는다면**, `message.channels`(공개 채널용) 및/또는
`message.groups`(비공개 채널용)를 추가하지 않았을 가능성이 거의 확실합니다. 이 이벤트가 없으면
Slack은 채널 메시지를 봇에 전달하지 않습니다.
:::

---

## 5단계: Messages 탭 활성화

이 단계는 봇에 대한 다이렉트 메시지를 활성화합니다. 그렇지 않으면 사용자가 봇에 DM을 보내려 할
때 **"Sending messages to this app has been turned off"**가 표시됩니다.

1. 사이드바에서 **Features → App Home**으로 이동합니다.
2. **Show Tabs**까지 스크롤합니다.
3. **Messages Tab**을 ON으로 전환합니다.
4. **"Allow users to send Slash commands and messages from the messages tab"**을 선택합니다.

:::danger 이 단계를 생략하면 DM이 완전히 차단됩니다
스코프와 이벤트 구독이 모두 올바르더라도 Messages Tab을 활성화하지 않으면 Slack은 사용자가 봇에
다이렉트 메시지를 보내도록 허용하지 않습니다. 이는 Hermes 설정 문제가 아니라 Slack 플랫폼 요구사항입니다.
:::

---

## 6단계: 워크스페이스에 앱 설치

1. 사이드바에서 **Settings → Install App**으로 이동합니다.
2. **Install to Workspace**를 클릭합니다.
3. 권한을 검토하고 **Allow**를 클릭합니다.
4. 인증 후 `xoxb-`로 시작하는 **Bot User OAuth Token**이 표시됩니다.
5. **이 토큰을 복사합니다** — 이것이 `SLACK_BOT_TOKEN`입니다.

:::tip
나중에 스코프나 이벤트 구독을 변경하면 변경사항을 적용하기 위해 **반드시 앱을 다시 설치해야 합니다**.
Install App 페이지에 다시 설치하라는 배너가 표시됩니다.
:::

---

## 7단계: 허용 목록에 사용할 사용자 ID 찾기

Hermes는 허용 목록에 사용자 이름이나 표시 이름이 아닌 Slack **Member ID**를 사용합니다.

Member ID를 찾으려면:

1. Slack에서 사용자의 이름 또는 아바타를 클릭합니다.
2. **View full profile**을 클릭합니다.
3. **⋮**(더 보기) 버튼을 클릭합니다.
4. **Copy member ID**를 선택합니다.

Member ID는 `U01ABC2DEF3`처럼 생겼습니다. 최소한 본인의 Member ID가 필요합니다.

---

## 8단계: Hermes 설정

`~/.hermes/.env` 파일에 다음을 추가합니다.

```bash
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # Comma-separated Member IDs

# Optional
SLACK_HOME_CHANNEL=C01234567890              # Default channel for cron/scheduled messages
SLACK_HOME_CHANNEL_NAME=general              # Human-readable name for the home channel (optional)
```

또는 대화형 설정을 실행합니다.

```bash
hermes gateway setup    # Select Slack when prompted
```

그런 다음 게이트웨이를 시작합니다.

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

:::tip Codex 추론 노력 안전성
Codex 기반 Slack 피어 에이전트 채널에서는 `agent.reasoning_effort: high` 이하를 권장합니다. `xhigh`는
전체 턴을 숨겨진 추론에 소비해 보이는 어시스턴트 텍스트를 생성하지 못할 수 있습니다. Hermes는 이제
이러한 불완전한 턴 경고를 스레드에서 숨기고 진단 정보를 게이트웨이 로그에 유지합니다.
:::

---

## 9단계: 채널에 봇 초대

게이트웨이를 시작한 후 봇이 응답하기를 원하는 모든 채널에 **봇을 초대해야 합니다**:

```
/invite @Hermes Agent
```

봇은 채널에 **자동으로 참여하지 않습니다**. 각 채널에 개별적으로 초대해야 합니다.

---

## 슬래시 명령

모든 Hermes 명령(`/btw`, `/stop`, `/new`, `/model`, `/help`, ... )은 Telegram 및 Discord에서와 정확히
같은 방식으로 동작하는 기본 Slack 슬래시 명령입니다. Slack에서 `/`를 입력하면 자동 완성 선택기에
설명과 함께 모든 Hermes 명령이 표시됩니다.

내부적으로 Hermes는 생성된 Slack 앱 매니페스트(1단계 옵션 A 참조)를 함께 제공합니다. 이 매니페스트는
[`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py)의
모든 명령을 슬래시 명령으로 선언합니다. Socket Mode에서는 매니페스트의 `url` 필드와 관계없이 Slack이
WebSocket을 통해 명령 이벤트를 라우팅합니다.

### Agent 메시징 경험

새 Slack 앱은 Slack의 **Agent** 메시징 경험을 사용합니다. 기존 Hermes Assistant 앱은 `--agent-view`로
매니페스트를 다시 생성해 마이그레이션할 수 있습니다.

```bash
hermes slack manifest --agent-view --write
```

**Features → App Manifest**에서 매니페스트를 업데이트하고 Slack이 요청하면 앱을 다시 설치합니다.
Agent 보기에서 Assistant 보기로 되돌릴 수 없으며 전환 후 Slack을 강제 새로 고침해야 할 수 있습니다.
생성된 Agent 매니페스트는 `message.im`, `app_home_opened`, `app_context_changed`를 구독하므로 Hermes가
Messages 탭 DM을 식별하고 사용자의 현재 Slack 컨텍스트를 턴과 함께 받을 수 있습니다. Hermes는 해당
컨텍스트를 라벨로만 제공하며 보고 있던 채널의 기록은 읽지 않습니다.

### 업데이트 후 슬래시 명령 새로 고침

Hermes가 새 명령을 추가하면(예: `hermes update` 후) 매니페스트를 다시 생성하고 Slack 앱을 업데이트합니다.

```bash
hermes slack manifest --write
```

그런 다음 Slack에서:
1. [https://api.slack.com/apps](https://api.slack.com/apps) → Hermes 앱을 엽니다.
2. **Features → App Manifest → Edit**
3. `~/.hermes/slack-manifest.json`의 새 내용을 붙여 넣습니다.
4. **Save**. 스코프나 슬래시 명령이 변경되면 Slack이 앱 재설치를 요청합니다.

### 기존 `/hermes <subcommand>`도 계속 작동

이전 매니페스트와의 호환성을 위해 `/hermes btw run the tests`를 계속 입력할 수 있습니다. Hermes는 이를
`/btw run the tests`와 동일하게 라우팅합니다. 자유 형식 질문도 작동합니다. `/hermes what's the weather?`는
일반 메시지로 처리됩니다.

### 스레드 안에서 명령 사용 (`!cmd` 접두사)

Slack 자체는 스레드 답글 안에서 기본 슬래시 명령을 차단합니다. 스레드에서 `/queue`를 시도하면 Slack은
*"/queue is not supported in threads. Sorry!"*라고 응답합니다. 이를 다시 활성화하는 앱 설정은 없습니다.
Slack은 해당 명령을 Hermes에 전달하지 않습니다.

대안으로 Hermes는 앞에 `!`를 붙인 명령을 인식합니다. 이는 스레드와 다른 곳에서 모두 작동합니다. 일반
스레드 답글로 `!queue`, `!stop`, `!model gpt-5.4` 등을 입력하면 Hermes는 슬래시 형식과 동일하게 처리하고
같은 스레드에 답합니다.

알려진 명령 목록과 대조하는 것은 첫 번째 토큰뿐이므로 `!nice work` 같은 일반 메시지는 변경 없이 에이전트로
전달됩니다. 멘션 뒤의 `@Hermes !stop` 및 앞쪽 공백이 있는 형식도 작동하며 스레드에서 명령으로 디스패치됩니다.

승인 프롬프트(위험한 명령 또는 `execute_code` 승인)는 보통 대화형 버튼으로 표시됩니다. 버튼을 전달할 수 없어
Hermes가 텍스트 프롬프트로 전환하면 스레드에서 작동하는 형식인 `!approve` / `!deny`로 답하라는 안내가 표시됩니다.

### 슬래시 답글은 에페메럴입니다

기본 슬래시 명령(예: `/status`, `/help`)에 대한 답글은 **에페메럴**, 즉 "나에게만 표시"로 전달되므로 명령
출력이 채널을 도배하지 않습니다. "Running /cmd…" 플레이스홀더는 실제 답글로 교체되며, 긴 답글은 후속 에페메럴
메시지로 나뉩니다. Slack은 답글 흐름을 5개 게시물로 제한하므로 매우 긴 출력은 조용히 버려지지 않고 명시적인
잘림 알림과 함께 종료됩니다. 기본 에페메럴 경로가 실패하면 Hermes는 두 번째 에페메럴 API 경로로 재시도합니다.
슬래시 답글은 대체 경로로 채널에 공개 게시되지 않습니다. 일반 메시지로 입력한 명령(스레드의 `!cmd`, `@Hermes /cmd`)은
대신 일반적으로 보이는 메시지로 답합니다.

### 명확화 프롬프트 (한 번 탭하는 버튼)

에이전트가 객관식 질문(`clarify` 도구)을 해야 하면 Slack은 이를 **Block Kit 버튼**으로 표시합니다. 옵션마다 한
번 탭하고, 자유 입력 모드로 전환하는 **"✏️ Other…"** 버튼이 있습니다. 탭하면 메시지가 제자리에서 업데이트되어
누가 답했고 무엇을 선택했는지 보여 주며, 같은 프롬프트를 다시 클릭해도 무시됩니다. 버튼 클릭에도 메시지와 동일한
사용자 인증이 적용되고, 만료된 프롬프트(게이트웨이 재시작 또는 시간 초과)는 클릭을 조용히 삼키지 않고 다시
질문하라고 안내합니다. 개방형 명확화 질문은 일반 질문으로 표시되고 다음에 입력한 답글을 받습니다. 설정은 필요
없으며 `rich_blocks` 설정과 관계없이 작동합니다.

### 고급: 슬래시 명령 배열만 출력

Slack 매니페스트를 직접 관리하고 슬래시 명령 목록만 필요하다면:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

해당 배열을 기존 매니페스트의 `features.slash_commands` 키에 붙여 넣습니다.

---

## 봇의 응답 방식

Hermes가 여러 컨텍스트에서 동작하는 방식을 이해하려면 다음을 참고하세요.

| 컨텍스트 | 동작 |
|---------|----------|
| **DM** | 봇은 모든 메시지에 응답 — @멘션 불필요 |
| **채널** | 봇은 **@멘션될 때만 응답**(예: `@Hermes Agent what time is it?`). 채널에서 Hermes는 해당 메시지에 연결된 스레드로 답합니다. |
| **스레드** | 기존 스레드 안에서 Hermes를 @멘션하면 같은 스레드에 답합니다. 봇이 스레드에서 활성 세션을 시작하면 **이후 답글에는 @멘션이 필요 없으며**, 봇이 자연스럽게 대화를 따라갑니다. |

:::tip
채널에서는 항상 봇을 @멘션해 대화를 시작하세요. 봇이 스레드에서 활성화된 뒤에는 멘션 없이 답할 수 있습니다.
스레드 밖에서 @멘션이 없는 메시지는 바쁜 채널의 소음을 막기 위해 무시됩니다.
:::

## 설정 옵션

8단계의 필수 환경 변수 외에도 `~/.hermes/config.yaml`을 통해 Slack 봇 동작을 사용자 지정할 수 있습니다.

### 스레드 및 답글 동작

```yaml
platforms:
  slack:
    # Controls how multi-part responses are threaded
    # "off"   — never thread replies to the original message
    # "first" — first chunk threads to user's message (default)
    # "all"   — all chunks thread to user's message
    reply_to_mode: "first"

    extra:
      # Whether to reply in a thread (default: true).
      # When false, channel messages get direct channel replies instead
      # of threads. Messages inside existing threads still reply in-thread.
      reply_in_thread: true

      # Also post thread replies to the main channel
      # (Slack's "Also send to channel" feature).
      # Only the first chunk of the first reply is broadcast.
      reply_broadcast: false

      # Render agent messages as Slack Block Kit blocks (default: false).
      # When true, the final agent message is sent with structured blocks —
      # section headers, dividers, true nested lists (via rich_text), and
      # native Block Kit tables — instead of flat mrkdwn text. A plain-text
      # fallback is always sent alongside for notifications/accessibility.
      # Tables exceeding Slack's limits (100 rows / 20 cols / 10k chars)
      # gracefully fall back to aligned monospace.
      rich_blocks: false

      # Append Slack-native feedback controls to final Block Kit replies.
      # Requires rich_blocks: true. Default: false.
      feedback_buttons: false

      # Render live tool calls as Slack-native plan/task cards. This explicit
      # opt-in activates native progress even when text tool_progress is off.
      # If Slack rejects the native stream, Hermes keeps one editable text
      # fallback current for the rest of the turn.
      native_task_cards: false

      # Suggested prompts pinned at the top of Agent view's Messages tab.
      # Either a list of {title, message} rows, or a titled object:
      # {title: "Start here", prompts: [{title: "Plan", message: "..."}]}
      suggested_prompts: []

      # Title Agent/Assistant DM threads from the first user message.
      # Default: true. Set false to leave Slack's default thread titles.
      assistant_thread_titles: true

      # Accept messages posted by other Slack bots (default: "none").
      # "none" ignores bots, "mentions" accepts a bot message only when
      # that message itself @mentions Hermes, and "all" accepts every
      # other bot. Hermes always ignores its own bot user to prevent
      # self-echoes.
      allow_bots: "none"

      # Continuable-cron delivery surface (default: "thread").
      # "in_channel" delivers a continuable cron job FLAT into the channel
      # (no dedicated thread); pair with reply_in_thread: false (and
      # require_mention: false) so a plain reply continues the job.
      # See the cron guide → "Flat, in-channel continuation".
      cron_continuable_surface: thread
```

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | 여러 부분으로 된 메시지의 스레드 방식: `"off"`, `"first"`, `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | `false`이면 채널 메시지가 스레드 대신 직접 답글을 받습니다. 기존 스레드 안의 메시지는 계속 스레드에 답합니다. |
| `platforms.slack.extra.reply_broadcast` | `false` | `true`이면 스레드 답글도 기본 채널에 게시합니다. 첫 답글의 첫 부분만 전달됩니다. |
| `platforms.slack.extra.rich_blocks` | `false` | `true`이면 에이전트 메시지를 [Block Kit](https://docs.slack.dev/block-kit/) 블록(헤더, 구분선, 중첩 목록, 네이티브 표)으로 표시합니다. 일반 텍스트 대체본도 항상 전송됩니다. Slack 제한을 넘는 표는 정렬된 고정폭 텍스트로 대체됩니다. 앱 재설치는 필요 없습니다. |
| `platforms.slack.extra.feedback_buttons` | `false` | `rich_blocks`와 함께 `true`이면 최종 답글에 Slack 네이티브 피드백 컨트롤을 추가합니다. |
| `platforms.slack.extra.native_task_cards` | `false` | `true`이면 실행 중인 도구 호출을 Slack 네이티브 계획/작업 카드로 표시합니다. Slack 기본값인 `tool_progress: off`와 독립적인 명시적 진행률 설정입니다. 네이티브 API 실패 시 계속 편집되는 텍스트 업데이트 하나로 대체됩니다. |
| `platforms.slack.extra.suggested_prompts` | `[]` | Agent/Assistant DM 진입점에 표시할 최대 4개의 `{title, message}` 프롬프트. 목록 또는 `{title, prompts}` 형식을 받습니다. |
| `platforms.slack.extra.assistant_thread_titles` | `true` | `true`이면 첫 사용자 메시지로 Agent/Assistant DM 스레드 이름을 지정합니다. |
| `platforms.slack.extra.allow_bots` | `"none"` | 다른 Slack 봇의 메시지를 제어합니다. `"none"`은 무시하고, `"mentions"`는 **해당 메시지 자체가** Hermes를 @멘션할 때만 받고, `"all"`은 모두 받습니다. 가장 안전한 봇 간 협업 모드에는 `"mentions"`를 사용하세요. [다른 봇의 메시지 받기](#accepting-messages-from-other-bots-allow_bots)를 참고하세요. |
| `platforms.slack.extra.cron_continuable_surface` | `"thread"` | [계속 가능한 cron 작업](../features/cron.md#flat-in-channel-continuation-slack)의 전달 표면. `"thread"`는 전달마다 전용 스레드를 열고(기본값), `"in_channel"`은 채널 타임라인에 평평하게 전달합니다. 일반 채널 답글로 작업을 계속하려면 `in_channel`을 `reply_in_thread: false`(및 `require_mention: false`)와 함께 사용합니다. |

동등한 환경 변수는 `SLACK_ALLOW_BOTS=none|mentions|all`입니다. 둘 다 설정하면
`platforms.slack.extra.allow_bots`가 우선합니다. 피어 봇이 명시적인 멘션 없이 서로 답할 수 있다면
`all`을 피하세요. 각 봇의 답글 정책이 루프를 만들 수 있습니다.

### 작업 상태 표시줄

에이전트가 메시지를 처리하는 동안 Slack은 스레드의 봇 이름 옆에 상태 표시줄을 보여 줍니다. 기본적으로 Hermes는
`is thinking...`으로 설정합니다. `typing_status_text`로 사용자 지정할 수 있습니다. 예를 들어 Ada라는 고양이
어시스턴트라면:

```yaml
platforms:
  slack:
    # Custom working-state status line (default: "is thinking...").
    typing_status_text: "is pouncing… 🐾"
```

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `platforms.slack.typing_status_text` | `"is thinking..."` | 에이전트가 메시지를 처리하는 동안 표시되는 작업 상태 표시줄의 텍스트. `assistant:write` 스코프가 필요합니다. 없으면 상태 호출이 조용히 실패하고 Slack이 자체 일반 플레이스홀더를 표시합니다. 상태 표시줄을 완전히 비활성화하려면 `typing_indicator: false`를 설정합니다. |

:::note 상태가 표시되는 위치
사용자 지정 상태는 **답글 작성기 아래의 푸터**("*BotName* is thinking…")에 표시되며 메시지 목록 안에 표시되지 않습니다. AI 앱이 작동하는 동안 Slack이 메시지 영역에 표시하는 인라인 "Generating response…" / "Finding answers…" 줄은 **Slack 자체의 순환 표시기**입니다. `assistant.threads.setStatus`는 이를 제어하지 않으며 둘 다 동시에 나타날 수 있습니다.
:::

같은 키로 Google Chat의 표시 작업 상태 메시지도 사용자 지정합니다(`platforms.google_chat.typing_status_text`,
기본값 `"Hermes is thinking…"`). 단, Google Chat에서는 실제 게시된 메시지를 답글에 패치하며 에페메럴 상태가 아닙니다.

### 실시간 상태 (도구별)

기본적으로 상태 표시줄은 에이전트가 작업할 때 **실시간으로 업데이트**됩니다. 정적인 `is thinking...` 대신 현재
수행 중인 작업(예: `is running pytest tests/…`, `is reading docs/api.md…`, `is searching the web for slack api limits…`)을
표시합니다. 도구 호출 사이에는 정적 텍스트로 되돌아갑니다. 기존 상태 새로 고침 주기를 사용하므로 추가 Slack API 호출이
없고, `tool_progress: off`에서도 작동합니다. 진행률 말풍선과 달리 상태 표시줄은 에페메럴하여 채널에 흔적을 남기지 않습니다.

`display.live_status`로 제어합니다(전역 또는 플랫폼별).

```yaml
display:
  platforms:
    slack:
      # full = verb + argument ("is running pytest…")   [default]
      # verb = verb only ("is running…") — hides commands/paths,
      #        useful in shared or customer-facing channels
      # off  = static text (typing_status_text or "is thinking...")
      live_status: full
```

| 키 | 기본값 | 설명 |
|-----|---------|-------------|
| `display.live_status` | `"full"` | 도구별 실시간 상태 표시줄. `full`은 동사와 인자 미리보기를, `verb`는 동사만 표시해 공유 채널에서 파일 경로와 명령을 숨기며, `off`는 정적 텍스트를 복원합니다. 정적 상태 표시줄과 마찬가지로 `assistant:write` 스코프가 필요합니다. |

### 네이티브 스트리밍 (실시간 입력 답글)

Slack의 [Agents & AI Apps](https://docs.slack.dev/ai/) 기능은 답글을 실시간 입력 메시지로 표시하는 네이티브
스트리밍 표면(`chat.startStream` / `chat.appendStream` / `chat.stopStream`)을 제공합니다. 이는 일반적으로 사용하는
편집 기반 점진적 업데이트보다 훨씬 부드럽습니다. `streaming.enabled`가 켜져 있고 전송 방식이 `auto` 또는 `draft`이면
Hermes는 가능한 곳에서 네이티브 스트리밍을 자동으로 사용합니다.

- 스트림은 첫 프레임에서 시작하고 델타만 추가합니다(API는 추가 전용). 스트리밍 메시지가 **최종 메시지**이므로 Hermes는 중복 최종 답글을 게시하는 대신 `chat.stopStream`으로 이를 봉인합니다.
- Slack 앱에서 AI 기능이 활성화되지 않았거나 `assistant:write` 스코프가 없으면 첫 실패가 캐시되고 Hermes는 문제 해결 방법을 알려 주는 로그 경고 하나와 함께 편집 기반 스트리밍으로 대체합니다.
- 선택적 Block Kit(`rich_blocks: true`)은 편집 기반 최종화 경로와 동일하게 봉인된 메시지에 적용됩니다.

스트리밍 활성화 외에는 추가 설정이 필요하지 않습니다.

```yaml
streaming:
  enabled: true       # transport auto/draft lights up Slack native streaming
```

### 네이티브 작업 카드 (실시간 도구 진행률)

`platforms.slack.extra.native_task_cards: true`이면 실행 중인 도구 호출이 텍스트 진행률 말풍선 대신 Slack 네이티브
**계획/작업 카드**로 표시됩니다. 턴마다 카드 하나, 도구 호출마다 행 하나가 생기며 각 작업의 실행 중/완료/오류 상태가
제자리에서 업데이트됩니다.

```yaml
platforms:
  slack:
    extra:
      native_task_cards: true
```

- 명시적인 진행률 선택 기능이며 Slack 기본값이 `tool_progress: off`여도 작동합니다. 텍스트 말풍선은 채널을 도배하지만 네이티브 카드는 그렇지 않습니다.
- 같은 도구에 대한 동시 호출은 실제 도구 호출 ID로 연결되므로 병렬 `web_search` 호출도 각각 올바른 상태의 행을 가집니다.
- 네이티브 스트림을 시작하거나 업데이트할 수 없으면 Hermes는 턴 동안 진행률을 유지하기 위해 계속 편집되는 텍스트 메시지 하나로 대체합니다.
- 인터럽트/연결 해제를 포함해 턴이 최종화될 때 카드 스트림은 정확히 한 번 중지되므로 남아 있는 실시간 표시기가 없습니다.

### 세션 격리

```yaml
# Global setting — applies to Slack and all other platforms
group_sessions_per_user: true
```

`true`(기본값)이면 공유 채널의 각 사용자가 격리된 대화 세션을 가집니다. `#general`에서 두 사람이 Hermes와
대화해도 기록과 컨텍스트는 서로 다릅니다.

채널 전체가 하나의 대화 세션을 공유하는 협업 모드를 원한다면 `false`로 설정합니다. 이 경우 사용자가 컨텍스트 증가와
토큰 비용을 공유하며 한 사용자의 `/reset`이 모두의 세션을 지운다는 점에 유의하세요.

### 멘션 및 트리거 동작

```yaml
slack:
  # Require @mention in channels (this is the default behavior;
  # the Slack adapter enforces @mention gating in channels regardless,
  # but you can set this explicitly for consistency with other platforms)
  require_mention: true

  # Prevent thread auto-engagement: only reply to channel messages that
  # contain an explicit @mention. With this OFF (default), Slack can
  # "auto-engage" — remembering past mentions in a thread and following
  # up on bot-message replies, and resuming active sessions without a
  # fresh mention. With strict_mention ON, every new channel message
  # must @mention the bot before Hermes will respond.
  strict_mention: false

  # Ignore messages addressed to another user: when a channel or thread
  # message *opens* by @mentioning someone other than the bot (e.g.
  # "@rasha can you take this?"), stay silent unless the bot is also
  # mentioned. Only a *leading* mention counts as "addressed to" — a
  # message that references someone mid-sentence ("loop in @rasha")
  # still reaches the bot. Overrides free_response_channels and thread
  # auto-engagement. Opt-in; default off. Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
  ignore_other_user_mentions: false

  # Require an explicit @mention for THREAD replies, while leaving
  # top-level channel messages governed by require_mention /
  # free_response_channels. Narrower than strict_mention: use it when a
  # free-response bot should not join every follow-up in busy threads.
  # Opt-in; default off. Env: SLACK_THREAD_REQUIRE_MENTION.
  thread_require_mention: false

  # Per-channel force-mention override — the opposite direction of
  # free_response_channels. Channels listed here ALWAYS require an
  # explicit @mention, even when require_mention is false globally.
  # Ongoing conversations still auto-follow (mentioned threads, active
  # sessions, bot-authored threads). Comma-separated IDs or a list.
  # Env: SLACK_REQUIRE_MENTION_CHANNELS.
  require_mention_channels: ""

  # Custom mention patterns that trigger the bot
  # (in addition to the default @mention detection)
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # Text prepended to every outgoing message
  reply_prefix: ""
```

:::tip `strict_mention`은 언제 사용하나요?
Slack의 기본 "봇이 이 스레드를 기억" 동작이 사용자를 놀라게 하는 바쁜 워크스페이스에서는 `true`로 설정하세요.
예를 들어 긴 기술 지원 스레드에서 봇이 처음에는 도움을 주었지만 이후에는 다시 명시적으로 호출될 때까지 조용히 있기를
원하는 경우입니다. DM과 활성 대화형 세션에는 영향을 주지 않습니다.
:::

:::tip `ignore_other_user_mentions`는 언제 사용하나요?
봇이 바쁜 스레드를 따라가면서(스레드 자동 참여 또는 `free_response_channels`를 통해) 사람이 서로에게 보낸 메시지에
끼어드는 경우 `true`로 설정하세요. `strict_mention`보다 좁은 도구입니다. 참여 중인 스레드의 일반 후속 메시지는 계속 답하고,
다른 사람을 @멘션하며 시작하는 메시지만 건너뜁니다. **1:1 DM에는 영향을 주지 않습니다**. 그룹 DM(MPIM)과 채널에는 아래
공유 표면 정책과 동일하게 적용됩니다. 브로드캐스트 토큰(`@here`, `@channel`)과 채널 참조는 사람이 아닌 방을 대상으로 하므로
절대 건너뛰지 않습니다.
:::

:::info
Slack은 두 패턴을 모두 지원합니다. 기본적으로 대화를 시작할 때 `@mention`이 필요하지만, `SLACK_FREE_RESPONSE_CHANNELS`(쉼표로
구분한 채널 ID) 또는 `config.yaml`의 `slack.free_response_channels`로 특정 채널은 예외로 지정할 수 있습니다. 봇이 스레드에서
활성 세션을 시작하면 이후 스레드 답글에는 멘션이 필요 없습니다. **1:1 DM**에서는 멘션 없이도 항상 응답합니다.
:::

:::caution 그룹 DM(MPIM)은 1:1 DM이 아니라 공유 표면입니다
**1:1 다이렉트 메시지**는 한 사람과의 비공개 대화이므로 멘션 예외입니다. **그룹 DM(MPIM/다인 DM)**은 여러 사람이 보고
봇을 호출할 수 있는 **공유 표면**이므로 채널과 동일한 운영 제어를 따릅니다. `require_mention`, `strict_mention`,
`free_response_channels`, `allowed_channels`가 모두 적용되고, 봇은 실제로 @멘션된 경우에만 `:eyes:`/`:white_check_mark:`
반응을 추가합니다. 특정 그룹 DM에서 봇이 자유롭게 응답하게 하려면 채널 ID(`G`로 시작)를 `free_response_channels`에 추가하세요.
:::

#### 어떤 멘션 옵션을 사용해야 하나요?

게이팅 옵션은 조합되며 각각 다른 질문에 답합니다.

| 옵션 | 답하는 질문 | 기본값 | 범위 |
|--------|--------------------|---------|-------|
| `require_mention` | **최상위 채널 메시지**에 @멘션이 필요한가? | `true` | 모든 채널 |
| `free_response_channels` | `require_mention`이 면제되는 채널은? | 없음 | 지정 채널 |
| `require_mention_channels` | `require_mention`이 `false`이거나 자유 응답 채널이어도 항상 @멘션이 필요한 채널은? 두 옵션보다 우선함. | 없음 | 지정 채널 |
| `thread_require_mention` | 최상위 메시지에 멘션이 없어도 **스레드 답글**에 @멘션이 필요한가? 멘션된 스레드는 기억하지 않음. | `false` | 스레드만 |
| `strict_mention` | 모든 채널 메시지(최상위 및 스레드)에 새로운 @멘션이 필요한가? 멘션된 스레드 기억, 봇 답글 후속 처리, 활성 세션 재개를 모두 비활성화함. | `false` | 모든 채널 및 스레드 |
| `ignore_other_user_mentions` | **다른 사람을 @멘션하며 시작하는** 메시지(`@rasha can you take this?`)를 건너뛸 것인가? 자유 응답 및 스레드 자동 추적보다 우선하며, 중간 문장의 참조는 봇에 전달됨. | `false` | 채널 및 그룹 DM |

기본 원칙: `strict_mention`은 가장 광범위한 수단이고, `thread_require_mention`은 최상위 게이팅에 영향을 주지 않고 바쁜
스레드를 조용히 하며, `require_mention_channels`는 자유 응답 봇에서 개별 채널을 다시 엄격하게 만들고,
`ignore_other_user_mentions`는 다른 사람에게 명시적으로 보내는 메시지만 건너뜁니다. 1:1 DM은 항상 응답하며 이 모든 옵션의
영향을 받지 않습니다.

### 다른 봇의 메시지 받기 (`allow_bots`)

기본적으로 Hermes는 다른 Slack 봇 또는 앱이 작성한 모든 메시지(Workflow Builder 게시물 포함)를 무시합니다. 여러 Hermes
인스턴스나 피어 봇이 한 채널에서 협업하는 멀티 에이전트 워크스페이스에서는 `allow_bots`로 활성화합니다.

```yaml
platforms:
  slack:
    extra:
      # "none" (default) — ignore all bot/app-authored messages
      # "mentions"       — accept a bot message only when THAT message
      #                    @mentions this bot
      # "all"            — accept every bot message (except the bot's own)
      allow_bots: mentions
```

환경 변수 형태: `SLACK_ALLOW_BOTS=none|mentions|all`(둘 다 지정하면 설정 키가 우선). 알 수 없는 값은 `none`으로 처리됩니다.

`mentions` 모드의 게이팅 방식:

- 피어 봇 메시지는 **메시지 자체가 이 봇을 현재 @멘션할 때만** 허용됩니다. 텍스트 또는 Block Kit 블록에 포함되어야 합니다. 스레드 기록은 해당하지 않습니다. 이전에 스레드에서 봇이 멘션되었거나, 봇 자신의 메시지에 대한 답글이거나, 활성 스레드 세션이 있더라도 이후 멘션 없는 피어 봇 메시지를 허용하지 않습니다. 이는 에이전트 간 확인/상태 루프를 끊기 위한 의도적인 동작입니다.
- 사람의 메시지에는 영향을 주지 않으며 일반 멘션 게이팅이 적용됩니다.
- Hermes는 셀프 에코 루프를 막기 위해 모든 모드에서 자신의 메시지를 무시합니다.

`mentions`는 봇 간 협업에 권장되는 모드입니다. 각 에이전트가 턴마다 다른 에이전트를 명시적으로 호출해야 하기 때문입니다.
모든 피어 봇의 자체 답글 정책이 루프에 안전하다는 확신이 없다면 `all`을 피하세요. 감지는 라벨이 있는 봇 메시지(`bot_id`,
`subtype: bot_message`), 앱에서 생성된 이벤트, 라벨이 없는 봇 *사용자*(`users.info`로 조회)를 포함하므로 피어 Hermes 에이전트가
워크스페이스 전체에서 일관되게 필터링됩니다.

엄격한 멀티 봇 배포에서는 `require_mention: true` 및 `strict_mention: true`와 함께 사용하세요. 아래 스모크 확인을 참고하세요.

### 반응 트리거 (`reaction_triggers`)

기본적으로 이모지 반응은 확인된 뒤 버려집니다. 봇 메시지에 👍를 달아도 아무 일도 일어나지 않습니다. 반응을 에이전트 루프로
라우팅하려면 `slack.reaction_triggers`를 설정합니다(Slack 앱 매니페스트에 `reactions:read` 스코프와 `reaction_added`/
`reaction_removed` 봇 이벤트 구독이 필요하며 `hermes slack manifest`로 다시 생성합니다).

```yaml
slack:
  # Opt-in. false/absent (default) = reactions are acked and dropped.
  # true = any reaction ON THE BOT'S OWN MESSAGES routes to the agent.
  reaction_triggers: true
  # Or an explicit emoji allowlist — only these names route, and they may
  # target ANY message (emoji-handoff workflows, e.g. :task: to capture):
  # reaction_triggers: [white_check_mark, thumbsup, task]
  # Optional handoff target: respond in this channel (top-level) or thread
  # (C123:<thread_ts>) instead of the reacted-to message's thread.
  # reaction_trigger_target: C0123456789
```

환경 변수 형태는 `SLACK_REACTION_TRIGGERS`(`true`/`all` 또는 쉼표로 구분한 목록)와
`SLACK_REACTION_TRIGGER_TARGET`입니다.

동작:

- 반응은 `reaction:added:👍` / `reaction:removed:👍`라는 텍스트의 일반 에이전트 턴으로 도착합니다(일반적인 Slack 이름은 유니코드로 변환되며, `reaction:added:custom-emoji`처럼 알 수 없는 이름은 그대로 전달됩니다). 반응한 메시지 아래 스레드에 들어가므로 에이전트가 반응 대상의 내용을 볼 수 있고 답글과 같은 세션에 턴이 도착합니다.
- 반응한 사용자가 메시지의 사용자가 되므로 사용자 인증과 `allowed_channels` 게이팅이 입력 메시지와 동일하게 적용됩니다. 권한 없는 임의 사용자의 반응은 해당 사용자의 메시지가 트리거할 수 없는 곳에서 에이전트를 트리거할 수 없습니다.
- `reaction_triggers: true`이면 봇 자신의 **메시지에 대한** 반응만 라우팅됩니다. 명시적 이모지 허용 목록이면 나열된 이모지는 어느 메시지에서든 라우팅됩니다.
- 봇 자체의 생명주기 반응(`:eyes:` 등)은 다시 입력되지 않습니다.
- 이 선택 기능과 별개로 모든 사람의 반응은 에이전트 턴이 필요 없는 관찰자를 위해 `reaction:added`/`reaction:removed` [게이트웨이 훅](../features/hooks.md#available-events)을 발생시킵니다.

### 피어 에이전트 스모크 확인

엄격한 턴별 멘션을 사용하는 멀티 봇 Slack 배포에서는 다음 프로필을 유지하세요.

```yaml
slack:
  require_mention: true
  strict_mention: true
  allow_bots: mentions
  allowed_channels: ""
```

게이트웨이 설정 변경, 배포 또는 재시작 후 다음 합성 스모크 대상을 실행합니다.

```bash
uv run --frozen pytest -q tests/gateway/test_slack_peer_agent_smoke.py -o addopts=''
```

이 대상은 프로세스 내부의 합성 Slack 이벤트만 사용합니다. 실제 Slack 메시지를 보내지 않으며 기본적으로 실제 봇 토큰도 필요하지 않습니다.

실패 범주:

- `config:` `test_peer_agent_smoke_preflight_contract`가 프로필 불일치(`require_mention`, `strict_mention`, `allow_bots` 또는 `allowed_channels`)를 감지했습니다.
- `platform_connectivity:` 어댑터/클라이언트가 초기화되지 않아 라우팅 스모크가 아직 신뢰할 수 없습니다.
- `bot_identity:` 어댑터가 봇 사용자 ID를 확인하지 못해 현재 메시지의 멘션 검사가 작동할 수 없습니다.
- `routing_logic:` Slack 어댑터가 피어 에이전트 불변식(사람의 멘션 라우팅, 피어 봇 무시, 명시적 피어 멘션 허용, 수동 확인/상태/오류 억제) 중 하나에서 회귀했습니다.

이 대상은 통과했지만 실제 워크스페이스에서 메시지가 잘못 라우팅된다면 라우팅 로직 자체가 아니라 Slack 토큰/워크스페이스 연결과
런타임 배포 상태를 조사하세요.

### 채널 허용 목록 (`allowed_channels`)

봇이 많은 채널에 초대되었지만 일부에서만 응답해야 할 때 유용하도록 Slack 채널을 고정 목록으로 제한합니다. 설정하면 이 목록에
**없는** 채널의 메시지는 봇이 @멘션되었더라도 **조용히 무시됩니다**.

**1:1 DM은** 이 필터에서 제외되므로 인증된 사용자는 DM으로 항상 봇에 도달할 수 있습니다. **그룹 DM(MPIM)은 제외되지 않습니다**.
채널과 마찬가지로 MPIM은 허용 목록에 있어야 하며(ID는 `G`로 시작), 그렇지 않으면 메시지가 버려집니다.

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

또는 환경 변수(쉼표로 구분)로 설정합니다.

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

동작:

- 비어 있거나 설정되지 않음 → 제한 없음(완전한 이전 버전 호환).
- 비어 있지 않음 → 다른 게이팅(멘션 요구, `free_response_channels` 등)이 실행되기 전에 채널 ID가 목록에 있어야 하며, 아니면 메시지가 버려집니다.
- Slack 채널 ID는 `C`(공개), `G`(비공개) 또는 `D`(DM)로 시작합니다. Slack UI의 "Open channel details" → "About" 패널 또는 API로 확인할 수 있습니다.

참고: [관리자/사용자 슬래시 명령 분리](../../reference/slash-commands.md#permissions-and-adminuser-split).

### 인증되지 않은 사용자 처리

```yaml
slack:
  # What happens when an unauthorized user (not in SLACK_ALLOWED_USERS) DMs the bot
  # "pair"   — prompt them for a pairing code (default)
  # "ignore" — silently drop the message
  unauthorized_dm_behavior: "pair"
```

모든 플랫폼에 적용되는 전역 설정도 사용할 수 있습니다.

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` 아래의 플랫폼별 설정이 전역 설정에 우선합니다.

### 음성 변환

```yaml
# Global setting — enable/disable automatic transcription of incoming voice messages
stt_enabled: true
```

`true`(기본값)이면 수신 오디오 메시지가 에이전트에서 처리되기 전에 구성된 STT 제공자를 사용해 자동으로 변환됩니다.

### 전체 예시

```yaml
# Global gateway settings
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack-specific settings
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# Platform config
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---

## 홈 채널

Hermes가 예약 메시지, cron 작업 결과 및 기타 사전 알림을 전달할 채널 ID를 `SLACK_HOME_CHANNEL`로 설정합니다.
채널 ID를 찾으려면:

1. Slack에서 채널 이름을 오른쪽 클릭합니다.
2. **View channel details**를 클릭합니다.
3. 아래쪽으로 스크롤합니다 — Channel ID가 표시됩니다.

```bash
SLACK_HOME_CHANNEL=C01234567890
```

봇이 채널에 **초대되었는지** 확인합니다(`/invite @Hermes Agent`).

### Cron 전달 대상

Cron 작업([cron 가이드](../features/cron.md#delivery-options) 참조)은 세 가지 방식으로 Slack을 대상으로 지정할 수 있습니다.

| `deliver:` 값 | 도착 위치 |
|------------------|----------------|
| `slack` | 홈 채널(`SLACK_HOME_CHANNEL`) |
| `slack:C0123456789` | ID로 지정한 특정 채널 |
| `slack:U0123456789` | 해당 사용자의 **DM** — 사용자 ID만 지정하면 자동으로 DM 대화로 확인됨(`im:write` 스코프 필요) |

Cron 프로세스가 게이트웨이와 같은 위치에 있지 않아도 전달됩니다. Hermes는 `SLACK_BOT_TOKEN`을 사용하는 독립 Web API
전송기로 대체합니다. Cron 출력의 `MEDIA:` 첨부 파일은 같은 대상에 네이티브 Slack 파일 공유로 업로드됩니다.

### 메시지 및 미디어 보내기 (`send_message`)

에이전트의 `send_message` 도구는 동일한 대상 형식을 받습니다. 채널 ID(`C…`/`G…`), DM 대화(`D…`) 또는 일반 사용자 ID
(`U…`/`W…`)를 사용할 수 있으며, 모든 전송 경로(텍스트, 미디어, 대화형 프롬프트)에서 사용자의 DM으로 확인됩니다.
`MEDIA:<path>` 첨부 파일(이미지, PDF, 문서)은 네이티브 파일 공유로 업로드됩니다. 짧은 메시지가 단일 첨부 파일과 함께
전달되면 별도 메시지 대신 파일의 캡션으로 포함됩니다. 없는 파일은 전체 전송을 실패시키지 않고 파일별 경고로 보고됩니다.

---

## 멀티 워크스페이스 지원

Hermes는 단일 게이트웨이 인스턴스로 여러 Slack 워크스페이스에 동시에 연결할 수 있습니다. 각 워크스페이스는 자체 봇 사용자
ID로 독립적으로 인증됩니다.

### 설정

`SLACK_BOT_TOKEN`에 **쉼표로 구분한 목록**으로 여러 봇 토큰을 제공합니다.

```bash
# Multiple bot tokens — one per workspace
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# A single app-level token is still used for Socket Mode
SLACK_APP_TOKEN=xapp-your-app-token
```

또는 `~/.hermes/config.yaml`에 다음과 같이 설정합니다.

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuth 토큰 파일

환경 또는 설정의 토큰 외에도 Hermes는 다음 위치의 **OAuth 토큰 파일**에서 토큰을 읽습니다.

```
~/.hermes/slack_tokens.json
```

이 파일은 팀 ID를 토큰 항목에 매핑하는 JSON 객체입니다.

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

이 파일의 토큰은 `SLACK_BOT_TOKEN`으로 지정한 토큰과 병합됩니다. 중복 토큰은 자동으로 제거됩니다.

### 동작 방식

- 목록의 **첫 번째 토큰**이 기본 토큰이며 Socket Mode 연결(AsyncApp)에 사용됩니다.
- 시작 시 각 토큰을 `auth.test`로 인증합니다. 게이트웨이는 각 `team_id`를 자체 `WebClient` 및 `bot_user_id`에 매핑합니다.
- 메시지가 도착하면 Hermes는 올바른 워크스페이스별 클라이언트를 사용해 응답합니다.
- 첫 번째 토큰의 기본 `bot_user_id`는 단일 봇 ID를 기대하는 기능과의 이전 버전 호환성에 사용됩니다.

---

## 음성 메시지

Hermes는 Slack에서 음성을 지원합니다.

- **수신:** 구성된 STT 제공자(로컬 `faster-whisper`, Groq Whisper(`GROQ_API_KEY`) 또는 OpenAI Whisper(`VOICE_TOOLS_OPENAI_KEY`))를 사용해 음성/오디오 메시지를 자동으로 변환합니다.
- **발신:** TTS 응답을 오디오 파일 첨부로 보냅니다.

---

## 채널별 프롬프트

특정 Slack 채널에 에페메럴 시스템 프롬프트를 지정합니다. 프롬프트는 매 턴 런타임에 주입되고 대화 기록에는 저장되지
않으므로 변경사항이 즉시 적용됩니다.

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "C02ENGINEERING": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

키는 Slack 채널 ID입니다(채널 세부정보 → "About" → 아래로 스크롤하여 찾음). 일치하는 채널의 모든 메시지에 프롬프트가
에페메럴 시스템 지시로 주입됩니다.

## 채널별 스킬 바인딩

특정 채널 또는 DM에서 새 세션이 시작될 때마다 스킬을 자동으로 불러옵니다. 매 턴 주입되는 채널별 프롬프트와 달리 스킬
바인딩은 **세션 시작 시 사용자 메시지**로 스킬 내용을 주입하므로 대화 기록의 일부가 되며 이후 턴마다 다시 불러올 필요가 없습니다.

모델 자체의 스킬 선택기가 모든 짧은 답글에서 로드할지 결정하지 않도록, 전용 목적(플래시카드, 도메인별 Q&A 봇, 지원 분류 채널 등)이
있는 DM이나 채널에 적합합니다.

```yaml
slack:
  channel_skill_bindings:
    # DM channel — always runs in "german-flashcards" mode
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # Research channel — preload multiple skills in order
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # Short form: single skill as a string
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

참고:
- 바인딩은 채널 ID로 일치합니다. 바인딩된 채널의 스레드 메시지는 부모 채널의 바인딩을 상속합니다.
- 스킬은 세션 시작 시에만 로드됩니다(새 세션 또는 자동 재설정 후). 바인딩을 변경했다면 `/new`를 실행하거나 세션이 자동 재설정될 때까지 기다려 적용합니다.
- `channel_prompts`와 함께 사용하면 스킬 지시 위에 채널별 말투/제약 조건을 추가할 수 있습니다.

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| 봇이 DM에 응답하지 않음 | 이벤트 구독에 `message.im`이 있고 앱을 다시 설치했는지 확인 |
| 봇이 DM에서는 작동하지만 채널에서는 작동하지 않음 | **가장 흔한 문제입니다.** 이벤트 구독에 `message.channels`와 `message.groups`를 추가하고 앱을 다시 설치한 뒤 `/invite @Hermes Agent`로 채널에 봇을 초대 |
| 봇이 채널의 @멘션에 응답하지 않음 | 1) `message.channels` 이벤트 구독 확인. 2) 봇을 채널에 초대. 3) `channels:history` 스코프 확인. 4) 스코프/이벤트 변경 후 앱 재설치 |
| 봇이 비공개 채널의 메시지를 무시함 | `message.groups` 이벤트 구독과 `groups:history` 스코프를 모두 추가하고 앱을 재설치한 뒤 봇을 `/invite` |
| 봇이 그룹 DM(다인 DM)에 응답하지 않음 | `message.mpim` 이벤트 구독과 `mpim:history` 스코프(및 `mpim:read`)를 추가한 뒤 **앱을 다시 설치**. `message.mpim`이 없으면 1:1 DM이 작동해도 Slack은 그룹 DM 메시지를 봇에 전달하지 않음 |
| DM에서 "Sending messages to this app has been turned off" 표시 | App Home 설정에서 **Messages Tab** 활성화(5단계 참조) |
| "not_authed" 또는 "invalid_auth" 오류 | Bot Token과 App Token을 다시 생성하고 `.env` 업데이트 |
| 봇은 응답하지만 채널에 게시하지 못함 | `/invite @Hermes Agent`로 봇을 채널에 초대 |
| 봇은 대화할 수 있지만 업로드된 이미지/파일을 읽지 못함 | `files:read`를 추가하고 **앱을 다시 설치**. Slack이 스코프/인증/권한 실패를 반환하면 Hermes가 이제 채팅에 첨부 파일 접근 진단을 표시함 |
| `missing_scope` 오류 | OAuth & Permissions에서 필요한 스코프를 추가하고 **앱을 다시 설치** |
| Socket이 자주 끊김 | 네트워크 확인. Bolt가 자동 재연결하지만 불안정한 연결은 지연을 유발함 |
| 스코프/이벤트를 변경했지만 아무 변화가 없음 | 스코프나 이벤트 구독을 변경할 때마다 워크스페이스에 앱을 **반드시 다시 설치** |

### 빠른 확인 목록

봇이 채널에서 작동하지 않으면 다음을 **모두** 확인합니다.

1. ✅ `message.channels` 이벤트가 구독됨(공개 채널용)
2. ✅ `message.groups` 이벤트가 구독됨(비공개 채널용)
3. ✅ `app_mention` 이벤트가 구독됨
4. ✅ `channels:history` 스코프가 추가됨(공개 채널용)
5. ✅ `groups:history` 스코프가 추가됨(비공개 채널용)
6. ✅ 스코프/이벤트 추가 후 앱을 **재설치**함
7. ✅ 봇을 채널에 **초대**함(`/invite @Hermes Agent`)
8. ✅ 메시지에서 봇을 **@멘션**함

---

## 보안

:::warning
인증된 사용자의 Member ID로 **항상 `SLACK_ALLOWED_USERS`를 설정하세요**. 이 설정이 없으면 안전 조치로
게이트웨이가 기본적으로 모든 메시지를 거부합니다. 봇 토큰은 비밀번호처럼 취급하고 절대 공유하지 마세요.
:::

- 토큰은 `~/.hermes/.env`에 저장해야 합니다(파일 권한 `600`).
- Slack 앱 설정에서 주기적으로 토큰을 교체합니다.
- Hermes 설정 디렉터리에 접근할 수 있는 사용자를 감사하고 관리합니다.
- Socket Mode는 공개 엔드포인트를 노출하지 않으므로 공격 표면이 하나 줄어듭니다.
