---
sidebar_position: 7
title: "게이트웨이 내부 구조"
description: "메시징 게이트웨이의 시작, 사용자 인증, 세션 라우팅 및 메시지 전달 방식"
---

# 게이트웨이 내부 구조

메시징 게이트웨이는 통합 아키텍처를 통해 Hermes를 20개 이상의 외부 메시징 플랫폼에 연결하는 장기 실행 프로세스입니다.

## 주요 파일

| 파일 | 용도 |
|------|---------|
| `gateway/run.py` | `GatewayRunner` — 메인 루프, 슬래시 명령, 메시지 디스패치 (대형 파일이므로 현재 LOC는 git에서 확인) |
| `gateway/session.py` | `SessionStore` — 대화 영속화 및 세션 키 구성 |
| `gateway/delivery.py` | 대상 플랫폼/채널로의 아웃바운드 메시지 전달 |
| `gateway/pairing.py` | 사용자 인증을 위한 DM 페어링 흐름 |
| `gateway/channel_directory.py` | cron 전달을 위해 채팅 ID를 사람이 읽을 수 있는 이름으로 매핑 |
| `gateway/hooks.py` | 훅 검색, 로드 및 수명 주기 이벤트 디스패치 |
| `gateway/mirror.py` | `send_message`를 위한 세션 간 메시지 미러링 |
| `gateway/status.py` | 프로필 범위 게이트웨이 인스턴스를 위한 토큰 잠금 관리 |
| `gateway/builtin_hooks/` | 항상 등록되는 훅을 위한 확장 지점 (기본 제공 훅 없음) |
| `gateway/platform_registry.py` | 번들 플랫폼 플러그인을 위한 어댑터 레지스트리, 팩토리 및 지연 로더 |
| `plugins/platforms/<name>/` | 번들 메시징 어댑터 (대부분의 플랫폼: `adapter.py` + `plugin.yaml`) |
| `gateway/platforms/` | 공유 `base.py`와 레거시/직접 어댑터 (Signal, API 서버, 웹훅 등) |

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

어떤 플랫폼에서든 메시지가 도착하면 다음과 같이 처리됩니다.

1. **플랫폼 어댑터**가 원시 이벤트를 수신하고 `MessageEvent`로 정규화합니다.
2. **기본 어댑터**가 활성 세션 가드를 확인합니다.
   - 이 세션에서 에이전트가 실행 중이면 메시지를 큐에 넣고 인터럽트 이벤트를 설정합니다.
   - `/approve`, `/deny`, `/stop`이면 가드를 우회하여 인라인으로 디스패치합니다.
3. **`GatewayRunner._handle_message()`**가 이벤트를 수신합니다.
   - `_session_key_for_source()`를 통해 세션 키를 확인합니다 (형식: `agent:main:{platform}:{chat_type}:{chat_id}`).
   - 인증을 확인합니다 (아래 인증 참고).
   - 슬래시 명령인지 확인하고 명령 핸들러로 디스패치합니다.
   - 에이전트가 이미 실행 중인지 확인하고 `/stop`, `/status` 같은 명령을 가로챕니다.
   - 그 외의 경우 `AIAgent` 인스턴스를 생성하고 대화를 실행합니다.
4. **응답**이 플랫폼 어댑터를 통해 다시 전송됩니다.

### 세션 키 형식

세션 키에는 전체 라우팅 컨텍스트가 인코딩됩니다.

```
agent:main:{platform}:{chat_type}:{chat_id}
```

예: `agent:main:telegram:private:123456789`

스레드를 인식하는 플랫폼(Telegram 포럼 토픽, Discord 스레드, Slack 스레드)은 `chat_id` 부분에 스레드 ID를 포함할 수 있습니다. **세션 키를 직접 구성하지 마세요.** 항상 `gateway/session.py`의 `build_session_key()`를 사용하세요.

### 2단계 메시지 가드

에이전트가 실행 중일 때 수신 메시지는 두 개의 순차적인 가드를 통과합니다.

1. **1단계 — 기본 어댑터** (`gateway/platforms/base.py`): `_active_sessions`를 확인합니다. 세션이 활성 상태이면 메시지를 `_pending_messages`에 큐에 넣고 인터럽트 이벤트를 설정합니다. 이 단계는 메시지가 게이트웨이 러너에 도달하기 전에 메시지를 포착합니다.

2. **2단계 — 게이트웨이 러너** (`gateway/run.py`): `_running_agents`를 확인합니다. 특정 명령(`/stop`, `/new`, `/queue`, `/status`, `/approve`, `/deny`)을 가로채 적절히 라우팅합니다. 그 외의 모든 입력은 `running_agent.interrupt()`를 호출합니다.

에이전트가 차단된 동안 러너에 도달해야 하는 명령(`/approve` 등)은 **`await self._message_handler(event)`**를 통해 인라인으로 디스패치됩니다. 경쟁 조건을 피하기 위해 백그라운드 작업 시스템을 우회합니다.

## 인증

게이트웨이는 다음 순서로 평가되는 다층 인증 확인을 사용합니다.

1. **플랫폼별 전체 허용 플래그** (예: `TELEGRAM_ALLOW_ALL_USERS`) — 설정되면 해당 플랫폼의 모든 사용자를 인증합니다.
2. **플랫폼 허용 목록** (예: `TELEGRAM_ALLOWED_USERS`) — 쉼표로 구분된 사용자 ID입니다.
3. **DM 페어링** — 인증된 사용자가 페어링 코드를 통해 새 사용자를 페어링할 수 있습니다.
4. **전역 전체 허용** (`GATEWAY_ALLOW_ALL_USERS`) — 설정되면 모든 플랫폼의 모든 사용자를 인증합니다.
5. **기본값: 거부** — 인증되지 않은 사용자는 거부됩니다.

### DM 페어링 흐름

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

페어링 상태는 `gateway/pairing.py`에 영속화되며 재시작 후에도 유지됩니다.

## 슬래시 명령 디스패치

게이트웨이의 모든 슬래시 명령은 동일한 확인 파이프라인을 거칩니다.

1. `hermes_cli/commands.py`의 `resolve_command()`가 입력을 표준 이름으로 매핑합니다 (별칭 및 접두사 매칭 처리).
2. 표준 이름을 `GATEWAY_KNOWN_COMMANDS`와 대조합니다.
3. `_handle_message()`의 핸들러가 표준 이름에 따라 디스패치합니다.
4. 일부 명령은 설정(`CommandDef`의 `gateway_config_gate`)에 따라 게이트됩니다.

### 실행 중인 에이전트 가드

에이전트가 처리 중일 때 실행해서는 안 되는 명령은 이른 단계에서 거부됩니다.

```python
if _quick_key in self._running_agents:
    if canonical == "model":
        return "⏳ Agent is running — wait for it to finish or /stop first."
```

우회 명령(`/stop`, `/new`, `/approve`, `/deny`, `/queue`, `/status`)은 특별히 처리됩니다.

## 설정 소스

게이트웨이는 여러 소스에서 설정을 읽습니다.

| 소스 | 제공 항목 |
|------|---------|
| `~/.hermes/.env` | API 키, 봇 토큰, 플랫폼 자격 증명 |
| `~/.hermes/config.yaml` | 모델 설정, 도구 구성, 표시 옵션 |
| 환경 변수 | 위 설정을 모두 재정의 |

`load_cli_config()`에서 하드코딩된 기본값과 사용자 설정을 병합하는 CLI와 달리, 게이트웨이는 YAML 로더를 통해 `config.yaml`을 직접 읽습니다. 따라서 CLI의 기본값 딕셔너리에는 존재하지만 사용자의 설정 파일에는 없는 설정 키가 CLI와 게이트웨이에서 다르게 동작할 수 있습니다.

## 플랫폼 어댑터

대부분의 메시징 플랫폼은 `plugins/platforms/<name>/adapter.py` 아래의 플러그인 어댑터로 제공되며, 일부 레거시 어댑터는 여전히 `gateway/platforms/`에 직접 있습니다. 모든 어댑터는 `gateway/platforms/base.py`의 `BasePlatformAdapter`를 확장합니다.

```text
plugins/platforms/                  # plugin-packaged adapters (one dir each)
├── telegram/adapter.py     # Telegram Bot API (long polling or webhook)
├── discord/adapter.py      # Discord bot via discord.py
├── slack/adapter.py        # Slack Socket Mode
├── whatsapp/adapter.py     # WhatsApp Business Cloud API
├── matrix/adapter.py       # Matrix via mautrix (optional E2EE)
├── mattermost/adapter.py   # Mattermost WebSocket API
├── email/adapter.py        # Email via IMAP/SMTP
├── sms/adapter.py          # SMS via Twilio
├── dingtalk/adapter.py     # DingTalk WebSocket
├── feishu/adapter.py       # Feishu/Lark WebSocket or webhook
├── wecom/adapter.py        # WeCom (WeChat Work) callback
├── line/adapter.py         # LINE Messaging API
├── teams/adapter.py        # Microsoft Teams
├── irc/adapter.py          # IRC (canonical scoped-lock example)
├── homeassistant/adapter.py # Home Assistant conversation integration
└── …                       # google_chat, ntfy, photon, raft, simplex, …

gateway/platforms/                  # core base + legacy direct adapters
├── base.py              # BasePlatformAdapter — shared logic for all platforms
├── signal.py            # Signal via signal-cli REST API
├── weixin.py            # Weixin (personal WeChat) via iLink Bot API
├── bluebubbles.py       # Apple iMessage via BlueBubbles macOS server
├── qqbot/               # QQ Bot (Tencent QQ) via Official API v2 (sub-package)
├── yuanbao.py           # Yuanbao (Tencent) DM/group adapter
├── msgraph_webhook.py   # Microsoft Graph change-notification webhook (Teams, Outlook, etc.)
├── webhook.py           # Inbound/outbound webhook adapter
└── api_server.py        # REST API server adapter
```

**지연 로딩:** `kind: platform` 번들 플러그인은 `hermes_cli/plugins.py`를 통해 `gateway/platform_registry.py`에 비용이 낮은 `register_deferred` 로더를 등록합니다. 따라서 플랫폼 SDK는 일반 `hermes chat`이 아니라 게이트웨이가 시작되거나, 메시지를 전달하거나, 설정/상태를 실행할 때만 import됩니다. 확인 시에는 하나의 어댑터를 로드하고, 전체 열거가 필요한 경로에서만 대기 중인 로더를 모두 실행합니다.

실험적인 커넥터 기반 플랫폼은 직접 플랫폼 모듈 대신 `gateway/relay/`의 범용 릴레이 어댑터를 사용합니다. `GATEWAY_RELAY_URL` 또는 `gateway.relay_url`이 설정되면 게이트웨이는 `relay` 플랫폼을 등록하고, 아웃바운드 WebSocket을 통해 커넥터에 연결하며, 동일한 소켓에서 `descriptor`, `inbound`, `interrupt_inbound` 프레임을 수신합니다. 커넥터는 `CapabilityDescriptor`를 알립니다. Hermes는 릴레이를 통해 일반 아웃바운드 답변, 토큰이 필요 없는 `follow_up` 작업 및 인터럽트 프레임을 다시 보낼 수 있습니다. 소스에 근거한 와이어 계약은 [`docs/relay-connector-contract.md`](https://github.com/NousResearch/hermes-agent/blob/main/docs/relay-connector-contract.md)에 있습니다.

어댑터는 공통 인터페이스를 구현합니다.

- `connect()` / `disconnect()` — 수명 주기 관리
- `send()` — 아웃바운드 메시지 전달
- 인바운드 이벤트는 `MessageEvent`로 정규화되어 `handle_message()`를 통해 전달됩니다.

### 토큰 잠금

고유한 자격 증명으로 연결하는 어댑터는 `connect()`에서 `acquire_scoped_lock()`을 호출하고 `disconnect()`에서 `release_scoped_lock()`을 호출합니다. 이를 통해 두 프로필이 동일한 봇 토큰을 동시에 사용하는 것을 방지합니다.

## 전달 경로

아웃바운드 전달(`gateway/delivery.py`)은 다음을 처리합니다.

- **직접 답변** — 응답을 원래 채팅으로 전송
- **홈 채널 전달** — cron 작업 출력과 백그라운드 결과를 설정된 홈 채널로 라우팅
- **명시적 대상 전달** — `telegram:-1001234567890`을 지정하는 전송 엔진. 셸 스크립트에서는 [`hermes send` CLI](/guides/pipe-script-output)를 통해, cron에서는 `deliver:` 대상으로 노출됩니다.
- **플랫폼 간 전달** — 원래 메시지가 온 플랫폼과 다른 플랫폼으로 전달

Cron 작업 전달은 게이트웨이 세션 기록에 미러링되지 않습니다. 대신 자체 cron 세션에 저장됩니다. 이는 메시지 교대 규칙 위반을 피하기 위한 의도적인 설계입니다.

## 훅

게이트웨이 훅은 수명 주기 이벤트에 응답하는 Python 모듈입니다.

### 게이트웨이 훅 이벤트

| 이벤트 | 발생 시점 |
|--------|-----------|
| `gateway:startup` | 게이트웨이 프로세스가 시작될 때 |
| `session:start` | 새 대화 세션이 시작될 때 |
| `session:end` | 세션이 완료되거나 시간이 초과될 때 |
| `session:reset` | 사용자가 `/new`로 세션을 초기화할 때 |
| `agent:start` | 에이전트가 메시지 처리를 시작할 때 |
| `agent:step` | 에이전트가 도구 호출 한 번을 완료할 때 |
| `agent:end` | 에이전트가 완료되고 응답을 반환할 때 |
| `command:*` | 슬래시 명령이 실행될 때 |

훅은 `gateway/builtin_hooks/`(확장 지점 — 제공되는 배포판에서는 현재 비어 있으며 `_register_builtin_hooks()`는 아무 작업도 하지 않는 스텁임)와 `~/.hermes/hooks/`(사용자 설치 훅)에서 검색됩니다. 각 훅은 `HOOK.yaml` 매니페스트와 `handler.py`를 포함하는 디렉터리입니다.

## 메모리 제공자 통합

메모리 제공자 플러그인(예: Honcho)이 활성화되면 다음과 같이 동작합니다.

1. 게이트웨이는 각 메시지마다 세션 ID와 함께 `AIAgent`를 생성합니다.
2. `MemoryManager`가 세션 컨텍스트로 제공자를 초기화합니다.
3. 제공자 도구(예: `honcho_profile`, `viking_search`)는 다음 경로로 라우팅됩니다.

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. 세션이 종료되거나 초기화될 때 정리를 위해 `on_session_end()`가 호출되고 최종 데이터가 플러시됩니다.

### 메모리 플러시 수명 주기

세션이 초기화되거나, 재개되거나, 만료되면 다음이 수행됩니다.

1. 기본 제공 메모리가 디스크로 플러시됩니다.
2. 메모리 제공자의 `on_session_end()` 훅이 실행됩니다.
3. 임시 `AIAgent`가 메모리 전용 대화 턴을 실행합니다.
4. 이후 컨텍스트가 폐기되거나 보관됩니다.

## 백그라운드 유지 관리

게이트웨이는 메시지 처리와 함께 주기적인 유지 관리도 실행합니다.

- **Cron 틱** — 작업 일정을 확인하고 기한이 된 작업을 실행
- **세션 만료** — 타임아웃 후 방치된 세션 정리
- **메모리 플러시** — 세션 만료 전에 메모리를 선제적으로 플러시
- **캐시 새로 고침** — 모델 목록과 제공자 상태 새로 고침

## 프로세스 관리

게이트웨이는 다음을 통해 관리되는 장기 실행 프로세스로 실행됩니다.

- `hermes gateway start` / `hermes gateway stop` — 수동 제어
- `systemctl` (Linux) 또는 `launchctl` (macOS) — 서비스 관리
- `~/.hermes/gateway.pid`의 PID 파일 — 프로필 범위 프로세스 추적

**프로필 범위와 전역 범위의 차이:** `start_gateway()`는 프로필 범위 PID 파일을 사용합니다. `hermes gateway stop`은 현재 프로필의 게이트웨이만 중지합니다. `hermes gateway stop --all`은 전역 `ps aux` 스캔을 사용해 모든 게이트웨이 프로세스를 종료합니다 (업데이트 중 사용).

## 관련 문서

- [세션 저장소](./session-storage.md)
- [Cron 내부 구조](./cron-internals.md)
- [ACP 내부 구조](./acp-internals.md)
- [에이전트 루프 내부 구조](./agent-loop.md)
- [메시징 게이트웨이 (사용자 가이드)](/user-guide/messaging)
