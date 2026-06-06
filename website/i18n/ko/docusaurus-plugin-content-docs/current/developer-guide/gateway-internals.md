---
sidebar_position: 7
title: "게이트웨이 내부 구조 (Gateway Internals)"
description: "메시징 게이트웨이가 부팅, 사용자 권한 부여, 세션 라우팅 및 메시지 전달을 처리하는 방법"
---

# 게이트웨이 내부 구조 (Gateway Internals)

메시징 게이트웨이는 통합된 아키텍처를 통해 Hermes를 20개 이상의 외부 메시징 플랫폼과 연결하는 장기 실행 프로세스(long-running process)입니다.

## 주요 파일

| 파일 | 목적 |
|------|---------|
| `gateway/run.py` | `GatewayRunner` — 메인 루프, 슬래시 커맨드, 메시지 디스패치 (큰 파일; 현재 LOC는 git에서 확인) |
| `gateway/session.py` | `SessionStore` — 대화 지속성 및 세션 키 생성 |
| `gateway/delivery.py` | 대상 플랫폼/채널로 발신 메시지 전달 |
| `gateway/pairing.py` | 사용자 권한 부여를 위한 DM 페어링 흐름 |
| `gateway/channel_directory.py` | cron 전송을 위해 채팅 ID를 사람이 읽을 수 있는 이름으로 매핑 |
| `gateway/hooks.py` | 훅 검색, 로딩 및 생명주기 이벤트 디스패치 |
| `gateway/mirror.py` | `send_message`에 대한 교차 세션 메시지 미러링 |
| `gateway/status.py` | 프로필 범위 게이트웨이 인스턴스의 토큰 잠금 관리 |
| `gateway/builtin_hooks/` | 항상 등록되는 훅을 위한 확장 포인트 (기본 제공 없음) |
| `gateway/platforms/` | 플랫폼 어댑터 (메시징 플랫폼당 1개) |

## 아키텍처 개요

```text
┌─────────────────────────────────────────────────┐
│                  GatewayRunner                  │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Telegram │  │ Discord  │  │  Slack   │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │
│       └─────────────┼─────────────┘             │
│                     ▼                           │
│              _handle_message()                  │
│                     │                           │
│         ┌───────────┼───────────┐               │
│         ▼           ▼           ▼               │
│  Slash command   AIAgent    Queue/BG            │
│    dispatch      creation   sessions            │
│                     │                           │
│                     ▼                           │
│                 SessionStore                    │
│              (SQLite persistence)               │
└───────┴─────────────┴─────────────┴─────────────┘
```

## 메시지 흐름

어떤 플랫폼에서든 메시지가 도착하면 다음과 같은 과정이 진행됩니다:

1. **플랫폼 어댑터(Platform adapter)**가 원시 이벤트를 수신하고, 이를 정규화하여 `MessageEvent`로 만듭니다.
2. **기본 어댑터(Base adapter)**가 활성 세션 가드(active session guard)를 검사합니다:
   - 이 세션에서 에이전트가 실행 중인 경우 → 메시지를 큐에 넣고 인터럽트 이벤트를 설정합니다.
   - `/approve`, `/deny`, `/stop`인 경우 → 가드를 우회합니다 (인라인 디스패치).
3. **GatewayRunner._handle_message()**가 이벤트를 수신합니다:
   - `_session_key_for_source()`를 통해 세션 키를 확인합니다 (형식: `agent:main:{platform}:{chat_type}:{chat_id}`)
   - 권한을 검사합니다 (아래 권한 부여(Authorization) 참조).
   - 슬래시 커맨드인지 확인합니다 → 커맨드 핸들러로 디스패치합니다.
   - 에이전트가 이미 실행 중인지 확인합니다 → `/stop`, `/status` 같은 커맨드를 가로챕니다.
   - 그렇지 않은 경우 → `AIAgent` 인스턴스를 생성하고 대화를 실행합니다.
4. **응답(Response)**은 플랫폼 어댑터를 통해 다시 전송됩니다.

### 세션 키 형식

세션 키는 전체 라우팅 컨텍스트를 인코딩합니다:

```
agent:main:{platform}:{chat_type}:{chat_id}
```

예: `agent:main:telegram:private:123456789`

스레드를 지원하는 플랫폼(Telegram 포럼 주제, Discord 스레드, Slack 스레드)은 chat_id 부분에 스레드 ID를 포함할 수 있습니다. **세션 키를 수동으로 구성하지 마세요.** 항상 `gateway/session.py`의 `build_session_key()`를 사용하세요.

### 2단계 메시지 가드

에이전트가 활발하게 실행 중일 때, 들어오는 메시지는 두 개의 순차적인 가드를 통과합니다:

1. **레벨 1 — 기본 어댑터** (`gateway/platforms/base.py`): `_active_sessions`를 검사합니다. 세션이 활성화된 경우 메시지를 `_pending_messages` 큐에 넣고 인터럽트 이벤트를 설정합니다. 이 과정은 메시지가 게이트웨이 러너(runner)에 도달하기 *전에* 메시지를 잡습니다.

2. **레벨 2 — 게이트웨이 러너** (`gateway/run.py`): `_running_agents`를 검사합니다. 특정 커맨드(`/stop`, `/new`, `/queue`, `/status`, `/approve`, `/deny`)를 가로채고 적절하게 라우팅합니다. 그 외의 모든 것은 `running_agent.interrupt()`를 트리거합니다.

에이전트가 차단된 상태에서 러너에 도달해야 하는 커맨드(`/approve` 등)는 `await self._message_handler(event)`를 통해 **인라인(inline)**으로 디스패치됩니다. 레이스 컨디션을 피하기 위해 백그라운드 작업 시스템을 우회합니다.

## 권한 부여 (Authorization)

게이트웨이는 다중 계층 권한 검사를 사용하며 다음 순서대로 평가됩니다:

1. **플랫폼별 모두 허용 플래그** (예: `TELEGRAM_ALLOW_ALL_USERS`) — 설정된 경우, 해당 플랫폼의 모든 사용자에게 권한이 부여됩니다.
2. **플랫폼 허용 목록** (예: `TELEGRAM_ALLOWED_USERS`) — 쉼표로 구분된 사용자 ID.
3. **DM 페어링** — 인증된 사용자가 페어링 코드를 통해 새 사용자를 페어링할 수 있습니다.
4. **글로벌 모두 허용** (`GATEWAY_ALLOW_ALL_USERS`) — 설정된 경우, 모든 플랫폼의 모든 사용자에게 권한이 부여됩니다.
5. **기본값: 거부** — 승인되지 않은 사용자는 거부됩니다.

### DM 페어링 흐름

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

페어링 상태는 `gateway/pairing.py`에 유지되며 재시작 후에도 유지됩니다.

## 슬래시 커맨드 디스패치

게이트웨이 흐름의 모든 슬래시 커맨드는 동일한 해결 파이프라인(resolution pipeline)을 거칩니다:

1. `hermes_cli/commands.py`의 `resolve_command()`가 입력을 정규 이름(canonical name)에 매핑합니다 (별칭, 접두사 일치 처리).
2. 정규 이름은 `GATEWAY_KNOWN_COMMANDS`와 대조하여 확인됩니다.
3. `_handle_message()` 내의 핸들러는 정규 이름을 기반으로 디스패치합니다.
4. 일부 커맨드는 설정(Config)에 따라 게이트 처리됩니다 (`CommandDef`의 `gateway_config_gate`).

### 실행 중인 에이전트 가드 (Running-Agent Guard)

에이전트가 처리하는 동안 실행되어서는 안 되는 커맨드는 조기에 거부됩니다:

```python
if _quick_key in self._running_agents:
    if canonical == "model":
        return "⏳ Agent is running — wait for it to finish or /stop first."
```

우회(Bypass) 커맨드(`/stop`, `/new`, `/approve`, `/deny`, `/queue`, `/status`)는 특별 처리를 가집니다.

## 설정 소스 (Config Sources)

게이트웨이는 여러 소스에서 구성을 읽습니다:

| 소스 | 제공하는 정보 |
|--------|-----------------|
| `~/.hermes/.env` | API 키, 봇 토큰, 플랫폼 자격 증명 |
| `~/.hermes/config.yaml` | 모델 설정, 도구 구성, 표시 옵션 |
| 환경 변수 (Environment variables) | 위의 항목을 재정의 |

하드코딩된 기본값과 함께 `load_cli_config()`를 사용하는 CLI와 달리, 게이트웨이는 YAML 로더를 통해 직접 `config.yaml`을 읽습니다. 이는 CLI의 기본 딕셔너리(defaults dict)에는 있지만 사용자의 구성 파일에는 없는 구성 키가 CLI와 게이트웨이 간에 다르게 작동할 수 있음을 의미합니다.

## 플랫폼 어댑터

각 메시징 플랫폼에는 `gateway/platforms/`에 어댑터가 있습니다:

```text
gateway/platforms/
├── base.py              # BaseAdapter — 모든 플랫폼의 공유 로직
├── telegram.py          # Telegram Bot API (롱 폴링 또는 웹훅)
├── discord.py           # discord.py를 통한 Discord 봇
├── slack.py             # Slack Socket Mode
├── whatsapp.py          # WhatsApp Business Cloud API
├── signal.py            # signal-cli REST API를 통한 Signal
├── matrix.py            # mautrix를 통한 Matrix (선택적 E2EE)
├── mattermost.py        # Mattermost WebSocket API
├── email.py             # IMAP/SMTP를 통한 Email
├── sms.py               # Twilio를 통한 SMS
├── dingtalk.py          # DingTalk WebSocket
├── feishu.py            # Feishu/Lark WebSocket 또는 webhook
├── wecom.py             # WeCom (WeChat Work) callback
├── weixin.py            # iLink Bot API를 통한 Weixin (개인 WeChat)
├── bluebubbles.py       # BlueBubbles macOS 서버를 통한 Apple iMessage
├── qqbot/               # Official API v2를 통한 QQ Bot (Tencent QQ) (하위 패키지: adapter.py, crypto.py, keyboards.py, …)
├── yuanbao.py           # Yuanbao (Tencent) DM/group 어댑터
├── feishu_comment.py    # Feishu 문서/드라이브 댓글 답글 핸들러
├── msgraph_webhook.py   # Microsoft Graph 변경 알림 웹훅 (Teams, Outlook 등)
├── webhook.py           # 인바운드/아웃바운드 웹훅 어댑터
├── api_server.py        # REST API 서버 어댑터
└── homeassistant.py     # Home Assistant 대화 연동
```

어댑터는 공통 인터페이스를 구현합니다:
- `connect()` / `disconnect()` — 생명주기 관리
- `send_message()` — 아웃바운드 메시지 전달
- `on_message()` — 인바운드 메시지 정규화 → `MessageEvent`

### 토큰 잠금 (Token Locks)

고유한 자격 증명으로 연결하는 어댑터는 `connect()`에서 `acquire_scoped_lock()`을 호출하고 `disconnect()`에서 `release_scoped_lock()`을 호출합니다. 이렇게 하면 두 프로필이 동일한 봇 토큰을 동시에 사용할 수 없습니다.

## 전달 경로 (Delivery Path)

아웃바운드 전달(`gateway/delivery.py`)은 다음을 처리합니다:

- **직접 응답 (Direct reply)** — 응답을 메시지가 시작된 채팅으로 다시 전송
- **홈 채널 전달 (Home channel delivery)** — cron 작업 결과 및 백그라운드 결과를 구성된 홈 채널로 라우팅
- **명시적 대상 전달 (Explicit target delivery)** — `telegram:-1001234567890`을 지정하는 `send_message` 도구 또는 쉘 스크립트에 동일한 도구를 래핑하는 [`hermes send` CLI](/guides/pipe-script-output)
- **교차 플랫폼 전달 (Cross-platform delivery)** — 원본 메시지와 다른 플랫폼으로 전달

Cron 작업 전달은 게이트웨이 세션 기록에 미러링되지 않으며 자체 cron 세션에만 존재합니다. 이는 메시지 교대 규칙(message alternation violations) 위반을 피하기 위한 의도적인 설계 선택입니다.

## 훅 (Hooks)

게이트웨이 훅은 생명주기 이벤트에 반응하는 Python 모듈입니다:

### 게이트웨이 훅 이벤트

| 이벤트 | 실행 시점 |
|-------|-----------|
| `gateway:startup` | 게이트웨이 프로세스 시작 시 |
| `session:start` | 새 대화 세션 시작 시 |
| `session:end` | 세션 완료 또는 시간 초과 시 |
| `session:reset` | 사용자가 `/new`로 세션을 재설정할 때 |
| `agent:start` | 에이전트가 메시지 처리를 시작할 때 |
| `agent:step` | 에이전트가 도구 호출 반복 1회를 완료할 때 |
| `agent:end` | 에이전트가 완료되고 응답을 반환할 때 |
| `command:*` | 임의의 슬래시 커맨드가 실행될 때 |

훅은 `gateway/builtin_hooks/`(확장 포인트 — 배포판에서는 현재 비어 있음; `_register_builtin_hooks()`는 no-op 스텁임) 및 `~/.hermes/hooks/`(사용자 설치)에서 검색됩니다. 각 훅은 `HOOK.yaml` 매니페스트와 `handler.py`가 있는 디렉터리입니다.

## 메모리 프로바이더 통합

메모리 프로바이더 플러그인(예: Honcho)이 활성화된 경우:

1. 게이트웨이는 세션 ID와 함께 메시지당 `AIAgent`를 생성합니다.
2. `MemoryManager`는 세션 컨텍스트로 프로바이더를 초기화합니다.
3. 프로바이더 도구(예: `honcho_profile`, `viking_search`)는 다음 경로를 통해 라우팅됩니다:

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. 세션 종료/재설정 시, 정리 및 최종 데이터 플러시를 위해 `on_session_end()`가 발생합니다.

### 메모리 플러시 생명주기

세션이 재설정, 재개 또는 만료될 때:
1. 내장 메모리가 디스크에 플러시됩니다.
2. 메모리 프로바이더의 `on_session_end()` 훅이 발생합니다.
3. 임시 `AIAgent`가 메모리 전용 대화 턴(turn)을 실행합니다.
4. 그런 다음 컨텍스트가 삭제되거나 보관됩니다.

## 백그라운드 유지보수

게이트웨이는 메시지 처리와 함께 주기적인 유지 보수를 실행합니다:

- **Cron 틱(Cron ticking)** — 작업 일정을 확인하고 만기가 도래한 작업을 실행합니다.
- **세션 만료 (Session expiry)** — 시간 초과 후 방치된 세션을 정리합니다.
- **메모리 플러시 (Memory flush)** — 세션 만료 전에 사전에 메모리를 플러시합니다.
- **캐시 새로 고침 (Cache refresh)** — 모델 목록 및 프로바이더 상태를 새로 고칩니다.

## 프로세스 관리

게이트웨이는 다음을 통해 관리되는 장기 실행 프로세스(long-lived process)로 실행됩니다:

- `hermes gateway start` / `hermes gateway stop` — 수동 제어
- `systemctl` (Linux) 또는 `launchctl` (macOS) — 서비스 관리
- `~/.hermes/gateway.pid`의 PID 파일 — 프로필 범위의 프로세스 추적

**프로필 범위(Profile-scoped) vs 글로벌(Global)**: `start_gateway()`는 프로필 범위의 PID 파일을 사용합니다. `hermes gateway stop`은 현재 프로필의 게이트웨이만 중지합니다. `hermes gateway stop --all`은 글로벌 `ps aux` 스캔을 사용하여 모든 게이트웨이 프로세스를 종료합니다(업데이트 중 사용됨).

## 관련 문서

- [세션 저장소 (Session Storage)](./session-storage.md)
- [Cron 내부 구조 (Cron Internals)](./cron-internals.md)
- [ACP 내부 구조 (ACP Internals)](./acp-internals.md)
- [에이전트 루프 내부 구조 (Agent Loop Internals)](./agent-loop.md)
- [메시징 게이트웨이 사용자 가이드 (Messaging Gateway User Guide)](/user-guide/messaging)
