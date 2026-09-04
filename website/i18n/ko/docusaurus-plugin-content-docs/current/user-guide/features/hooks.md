---
sidebar_position: 6
title: "이벤트 훅"
description: "주요 수명 주기 지점에서 사용자 지정 코드를 실행하여 활동을 기록하고, 알림을 보내고, 웹훅을 게시합니다"
---

# 이벤트 훅

Hermes에는 주요 수명 주기 지점에서 사용자 지정 코드를 실행하는 네 가지 훅 시스템이 있습니다.

| 시스템 | 등록 방법 | 실행 위치 | 사용 사례 |
|--------|---------------|---------|----------|
| **[게이트웨이 훅](#gateway-event-hooks)** | `~/.hermes/hooks/`의 `HOOK.yaml` + `handler.py` | 게이트웨이만 | 로깅, 알림, 웹훅 |
| **[플러그인 훅](#plugin-hooks)** | [플러그인](/user-guide/features/plugins)의 `ctx.register_hook()` | CLI + 게이트웨이 | 도구 가로채기, 메트릭, 가드레일 |
| **[셸 훅](#shell-hooks)** | 셸 스크립트를 가리키는 `~/.hermes/config.yaml`의 `hooks:` 블록 | CLI + 게이트웨이 | 차단, 자동 포맷팅, 컨텍스트 주입을 위한 즉시 사용 가능한 스크립트 |
| **[아웃바운드 웹훅](#outbound-webhooks)** | `~/.hermes/config.yaml`의 `hooks.outbound:` 목록 | CLI + 게이트웨이 | 외부 HTTP 엔드포인트(CI, 대시보드, 다른 에이전트)에 서명된 수명 주기 이벤트 푸시 |

훅 콜백 오류는 에이전트를 중단시키는 대신 격리되어 로그에 기록됩니다. 훅이 모두 수동적인 것은 아닙니다. 지시/제어 훅은 흐름을 변경할 수 있고, 변환은 콘텐츠를 대체할 수 있으며, 셸 `pre_tool_call` 훅은 차단하거나 안전하게 실패할 수 있습니다.

## 게이트웨이 이벤트 훅

게이트웨이 훅은 주 에이전트 파이프라인을 차단하지 않고 게이트웨이 작동 중(Telegram, Discord, Slack, WhatsApp, Teams)에 자동으로 실행됩니다.

### 훅 만들기

각 훅은 `~/.hermes/hooks/` 아래에 있는 디렉터리이며, 다음 두 파일을 포함합니다.

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # Declares which events to listen for
    └── handler.py     # Python handler function
```

#### HOOK.yaml

```yaml
name: my-hook
description: Log all agent activity to a file
events:
  - agent:start
  - agent:end
  - agent:step
```

`events` 목록은 어떤 이벤트가 핸들러를 트리거할지 결정합니다. `command:*` 같은 와일드카드를 포함하여 원하는 조합의 이벤트를 구독할 수 있습니다.

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """Called for each subscribed event. Must be named 'handle'."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**핸들러 규칙:**
- 이름은 `handle`이어야 합니다.
- `event_type`(문자열)과 `context`(dict)를 받습니다.
- `async def` 또는 일반 `def`로 작성할 수 있으며 둘 다 작동합니다.
- 오류는 처리되어 로그에 기록되며 에이전트를 중단시키지 않습니다.

### 사용 가능한 이벤트

| 이벤트 | 실행 시점 | 컨텍스트 키 |
|-------|---------------|--------------|
| `gateway:startup` | 게이트웨이 프로세스가 시작될 때 | `platforms`(활성 플랫폼 이름 목록) |
| `session:start` | 새 메시징 세션이 생성될 때 | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | 세션이 종료될 때(초기화 전) | `platform`, `user_id`, `session_key` |
| `session:reset` | 사용자가 `/new` 또는 `/reset`을 실행할 때 | `platform`, `user_id`, `session_key` |
| `session:compress` | 세션의 컨텍스트 압축이 완료될 때 | `platform`, `session_id`, `old_session_id`(현재 위치에서 압축된 경우 비어 있음), `in_place`(bool — `true` = 같은 ID에서 트랜스크립트 압축, `false` = `old_session_id`에서 교체), `compression_count` |
| `agent:start` | 에이전트가 메시지 처리를 시작할 때 | `platform`, `user_id`, `chat_id`, `thread_id`(포럼 토픽 / 스레드 루트 ID; 스레드가 아니면 비어 있음), `chat_type`(`"dm"` | `"group"` | `"forum"`; 알 수 없으면 비어 있음), `session_id`, `message`(최대 500자로 잘림) |
| `agent:step` | 도구 호출 루프의 각 반복마다 | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | 에이전트가 처리를 완료할 때 | `agent:start`와 동일한 키 및 `response`(최대 500자로 잘림) |
| `reaction:added` | 봇이 볼 수 있는 메시지에 이모지 반응이 추가될 때(현재 Slack 어댑터) | `reactions:read` 범위 + `reaction_added` 봇 이벤트 구독이 필요하며, 봇은 채널의 멤버여야 합니다. `platform`, `reaction`, `user_id`, `item_user_id`, `item_type`, `channel_id`, `message_ts`, `team_id`, `event_ts`, `raw_event` |
| `reaction:removed` | 봇이 볼 수 있는 메시지에서 이모지 반응이 제거될 때 | `reaction:added`와 동일한 형태 |
| `command:*` | 슬래시 명령이 실행될 때 | `platform`, `user_id`, `command`, `args` |

#### 와일드카드 매칭

`command:*`에 등록된 핸들러는 모든 `command:` 이벤트(`command:model`, `command:reset` 등)에 대해 실행됩니다. 하나의 구독으로 모든 슬래시 명령을 모니터링할 수 있습니다.

:::tip 스레드 답글
같은 Telegram 포럼 토픽에 후속 메시지를 게시하는 핸들러는 `chat_type == "forum"`이고 `thread_id`가 비어 있지 않을 때 `message_thread_id=int(thread_id)`를 포함해야 합니다.
:::

### 예시

#### 긴 작업에 대한 Telegram 알림

에이전트가 10단계 넘게 실행되면 자신에게 메시지를 보냅니다.

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Alert when agent is taking many steps
events:
  - agent:step
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
import os
import httpx

THRESHOLD = 10
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_HOME_CHANNEL")

async def handle(event_type: str, context: dict):
    iteration = context.get("iteration", 0)
    if iteration == THRESHOLD and BOT_TOKEN and CHAT_ID:
        tools = ", ".join(context.get("tool_names", []))
        text = f"⚠️ Agent has been running for {iteration} steps. Last tools: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### 명령 사용량 로거

어떤 슬래시 명령이 사용되는지 추적합니다.

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: Log slash command usage
events:
  - command:*
```

```python
# ~/.hermes/hooks/command-logger/handler.py
import json
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".hermes" / "logs" / "command_usage.jsonl"

def handle(event_type: str, context: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "command": context.get("command"),
        "args": context.get("args"),
        "platform": context.get("platform"),
        "user": context.get("user_id"),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

#### 세션 시작 웹훅

새 세션이 시작될 때 외부 서비스에 POST합니다.

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: Notify external service on new sessions
events:
  - session:start
  - session:reset
```

```python
# ~/.hermes/hooks/session-webhook/handler.py
import httpx

WEBHOOK_URL = "https://your-service.example.com/hermes-events"

async def handle(event_type: str, context: dict):
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={
            "event": event_type,
            **context,
        }, timeout=5)
```

### 튜토리얼: BOOT.md — 게이트웨이가 부팅될 때마다 시작 체크리스트 실행하기

커뮤니티에서 널리 쓰이는 패턴은 `~/.hermes/BOOT.md`에 Markdown 체크리스트를 넣고 게이트웨이가 시작될 때마다 에이전트가 한 번 실행하도록 하는 것입니다. “부팅할 때마다 야간 cron 실패를 확인하고 실패한 항목이 있으면 Discord로 알려 줘” 또는 “최근 24시간의 deploy.log를 요약해 Slack #ops에 게시해 줘”와 같은 작업에 유용합니다.

이 튜토리얼에서는 사용자 정의 훅으로 직접 만드는 방법을 설명합니다. Hermes는 기본 제공 `BOOT.md` 훅을 제공하지 않으므로 원하는 동작을 정확히 직접 연결합니다.

#### 만들 내용

1. 자연어로 된 시작 지침을 담은 `~/.hermes/BOOT.md` 파일
2. `gateway:startup`에서 실행되고, 게이트웨이에서 확인된 모델/자격 증명으로 일회성 에이전트를 생성한 뒤 BOOT.md 지침을 실행하는 게이트웨이 훅
3. 보고할 내용이 없을 때 에이전트가 메시지 전송을 생략할 수 있도록 하는 `[SILENT]` 규칙

#### 1단계: 체크리스트 작성

`~/.hermes/BOOT.md`를 만듭니다. 사람 비서에게 지시한다고 생각하고 작성하세요.

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, summarize them for Discord #ops (the hook delivers your final response to its configured target).
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same report.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

에이전트는 이를 프롬프트의 일부로 보기 때문에, 자연어로 설명할 수 있는 것이라면 무엇이든 작동합니다. 도구 호출, 셸 명령, 메시지 전송, 파일 요약 등이 이에 해당합니다.

#### 2단계: 훅 만들기

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: Run ~/.hermes/BOOT.md on gateway startup
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""Run ~/.hermes/BOOT.md on every gateway startup."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "You are running a startup boot checklist. Follow the instructions "
        "below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. Put any user-facing summary in your "
        "final response — the hook delivers it to the configured channel "
        "(e.g. Discord or Slack); you do not send messages yourself.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent and execute the checklist.

    Uses the gateway's resolved model and runtime credentials so this works
    against custom endpoints, aggregators, and OAuth-based providers alike.
    """
    try:
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from run_agent import AIAgent

        agent = AIAgent(
            model=_resolve_gateway_model(),
            **_resolve_runtime_agent_kwargs(),
            platform="gateway",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(_build_prompt(content))
        response = (result.get("final_response", "") or "").strip()
        if response.upper() not in {"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"}:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    if not BOOT_FILE.exists():
        return
    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Background thread so gateway startup isn't blocked on a full agent turn.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

핵심은 다음 두 줄입니다.

- `_resolve_gateway_model()`은 게이트웨이에 현재 구성된 모델을 읽습니다.
- `_resolve_runtime_agent_kwargs()`는 일반 게이트웨이 턴과 동일한 방식으로 공급자 자격 증명을 확인합니다. 여기에는 API 키, 기본 URL, OAuth 토큰, 자격 증명 풀이 모두 포함됩니다.

이 값들이 없으면 기본 `AIAgent()`는 내장 기본값으로 대체되며, 기본값이 아닌 엔드포인트에서는 401 오류가 발생합니다.
#### 3단계: 테스트

게이트웨이를 재시작합니다:

```bash
hermes gateway restart
```

로그를 확인합니다:

```bash
hermes logs --follow --level INFO | grep boot-md
```

`Running BOOT.md (N chars)`가 출력된 다음, 에이전트가 수행한 작업의 요약인 `boot-md completed: ...` 또는 에이전트가 `[SILENT]` 같은 정확한 무음 토큰으로 응답했을 때의 `boot-md completed (nothing to report)`가 출력됩니다.

체크리스트를 비활성화하려면 `~/.hermes/BOOT.md`를 삭제합니다. 훅은 계속 로드되지만 파일이 없으면 조용히 건너뜁니다.

#### 패턴 확장

- **일정 인식 체크리스트:** BOOT.md의 지침에서 `datetime.now().weekday()`를 기준으로 삼습니다("월요일이면 주간 배포 로그도 확인"). 지침은 자유 형식의 텍스트이므로 에이전트가 추론할 수 있는 내용이라면 무엇이든 사용할 수 있습니다.
- **여러 체크리스트:** 훅이 다른 파일(`STARTUP.md`, `MORNING.md` 등)을 가리키도록 하고, 각 체크리스트에 별도의 훅 디렉터리를 등록합니다.
- **에이전트가 아닌 변형:** 완전한 에이전트 루프가 필요하지 않다면 `AIAgent`를 아예 건너뛰고, 핸들러가 `httpx`를 통해 고정 알림을 직접 게시하도록 합니다. 더 저렴하고 빠르며 프로바이더 의존성도 없습니다.

#### 왜 기본 제공 기능이 아닌가

Hermes의 이전 버전은 이 기능을 기본 제공 훅으로 포함했고, 게이트웨이가 부팅될 때마다 기본 설정으로 에이전트를 조용히 생성했습니다. 이로 인해 사용자 지정 엔드포인트를 사용하는 사용자가 예상하지 못한 동작을 겪었고, 실행 중인 사실을 모르는 사용자에게는 기능이 보이지 않았습니다. 이제 문서화된 패턴으로 유지하고, 사용자가 자신의 훅 디렉터리에서 직접 구축하게 하면 파일을 작성하는 순간 정확히 무엇을 하는지 확인하고 명시적으로 선택할 수 있습니다.

### 작동 방식

1. 게이트웨이가 시작될 때 `HookRegistry.discover_and_load()`가 `~/.hermes/hooks/`를 검색합니다.
2. `HOOK.yaml`과 `handler.py`가 있는 각 하위 디렉터리를 동적으로 로드합니다.
3. 선언된 이벤트에 핸들러를 등록합니다.
4. 각 수명 주기 지점에서 `hooks.emit()`이 일치하는 모든 핸들러를 실행합니다.
5. 모든 핸들러의 오류를 잡아 기록하므로, 고장 난 훅 하나가 에이전트를 중단시키지 않습니다.

:::info
게이트웨이 훅은 **게이트웨이**(Telegram, Discord, Slack, WhatsApp, Teams)에서만 실행됩니다. CLI는 게이트웨이 훅을 로드하지 않습니다. 어디서나 작동하는 훅이 필요하면 [플러그인 훅](#plugin-hooks)을 사용하세요.
:::

## 플러그인 훅

[플러그인](/user-guide/features/plugins)은 **CLI와 게이트웨이** 세션 모두에서 실행되는 훅을 등록할 수 있습니다. 플러그인의 `register()` 함수에서 `ctx.register_hook()`을 프로그래밍 방식으로 호출해 등록합니다.

플러그인 패키징 및 등록에 관한 자세한 내용은
[플러그인 가이드](/docs/user-guide/features/plugins)를 참조하세요.

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
    # Kanban board lifecycle (dependency-wait blocking may fire inside its transaction):
    ctx.register_hook("kanban_task_claimed", my_claim_callback)     # dispatcher process
    ctx.register_hook("kanban_task_completed", my_done_callback)    # worker process
    ctx.register_hook("kanban_task_blocked", my_blocked_callback)   # worker process
```

**모든 훅에 적용되는 일반 규칙:**

- 콜백은 **키워드 인수**를 받습니다. 향후 호환성을 위해 항상 `**kwargs`를 허용하세요.
- 콜백 예외는 기록한 뒤 건너뛰며, 이후 콜백은 계속 실행됩니다.
- 아래 카탈로그는 설명을 위한 것입니다. **관찰자**는 반환값을 무시하고, **변환**은 처음으로 유효한 문자열 대체값을 받으며, **지시/제어** 훅은 문서화된 반환 형태를 사용합니다. 플러그인 미들웨어는 별도의 레지스트리와 표면이며, 또 하나의 훅 범주가 아닙니다.
- `turn_id`, `api_request_id`, `task_id`, `session_id`, `api_call_count` 같은 상관관계 필드는 훅별로 다르며 없을 수도 있습니다. ID는 불투명한 값으로 취급하세요.
- 런타임 이벤트 이름의 유효성은 `hermes_cli.plugins.VALID_HOOKS`에서 결정됩니다. `hermes hooks list`는 사용 가능한 모든 이벤트가 아니라 구성된 셸/아웃바운드 훅을 나열합니다. `hermes hooks test <event>`는 유효하지 않은 이벤트가 제공된 경우에만 유효한 집합을 보고합니다.

### 캐시 안전 시스템 프롬프트 섹션

항상 적용되는 지속적인 지침이 필요한 플러그인은 매 턴 같은 텍스트를 `pre_llm_call`을 통해 주입하는 대신, 제한된 시스템 프롬프트 섹션을 등록할 수 있습니다:

```python
def board_rules(session_info):
    return f"Apply the worker rules for profile {session_info['profile_name']}."

def register(ctx):
    ctx.register_system_prompt_section(
        "kanban-advanced.worker-rules",
        board_rules,                       # a string is also accepted
        position="after_memory",
        max_chars=4000,
    )
```

계약은 의도적으로 좁게 정의되어 있습니다.

- ID는 전역적으로 안정적인 1~128자 소문자 식별자이며, 문자·숫자·`.`, `_`, `-`만 사용할 수 있습니다. 중복 ID는 거부됩니다.
- 배치 기준점은 `after_memory` 하나뿐입니다. 섹션은 ID순으로 정렬되어 메모리/프로필 컨텍스트 뒤, 세션 메타데이터 앞에 렌더링됩니다. 플러그인은 핵심 프롬프트의 순서를 바꾸거나 대체할 수 없습니다.
- 호출 가능한 값에는 `session_id`, `model`, `provider`, `platform`, `profile_name`, `cwd`가 담긴 읽기 전용 매핑이 전달됩니다. 새 세션에 대해 **한 번만** 실행됩니다. 렌더링된 바이트는 압축 시 고정되며, 프로세스 재시작/재개 후에는 이미 저장된 전체 시스템 프롬프트에서 복구됩니다. 기존 세션에서는 플러그인 상태를 다시 읽지 않습니다.
- `max_chars`는 4,000자로 제한됩니다. 감사 제목을 포함한 모든 플러그인 섹션의 총합은 8,000자 및 32개 섹션으로 제한됩니다. 비어 있거나 문자열이 아니거나, 크기를 초과하거나, 전체 예산을 초과하거나, 예외를 발생시키는 섹션은 경고와 함께 건너뛰며 프롬프트 구성은 계속됩니다.
- 허용된 모든 섹션은 프롬프트에 이름이 표시되고, 플러그인·위치·문자 수와 함께 세션 시작 시 기록됩니다.

진정으로 동적인 턴별 컨텍스트에는 `pre_llm_call`을 사용하세요. 이 계약에는 의도적으로 플러그인 환경 힌트 훅이 없습니다. cwd, 브랜치 또는 기타 환경 데이터를 변경해 세션의 캐시된 프롬프트를 조용히 바꾸어서는 안 되기 때문입니다. 이러한 훅을 추가하려면 구체적인 소비자와 동일한 고정·재개 안전 의미론이 필요합니다.

### 기본 제공 플러그인 훅 카탈로그

아래 페이로드 필드는 각 호출 지점에서 제공되는 정확한 이벤트별 필드입니다. 이전 버전과의 호환성을 위해 `PluginManager`는 모든 플러그인 훅 콜백에 `telemetry_schema_version="hermes.observer.v1"`도 추가합니다. 이 레거시 envelope 표식이 모든 훅 페이로드가 동일한 의미 체계를 공유한다는 뜻은 아닙니다. 새로운 버전 관리 계약은 구체적인 이벤트 또는 기능군에 속합니다.

| 훅 | 범주 | 정확한 시점 및 반환 동작 | 명시적 페이로드 필드 | 개인정보 보호 / 민감도 |
|---|---|---|---|---|
| `pre_tool_call` | 지시/제어 | 실행 전에 한 번 실행되며, 처음 유효한 `block` 또는 `approve` 지시가 적용됩니다. | `tool_name`, `args`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace` | 원시 인수에 사용자 콘텐츠, 경로, 명령 또는 비밀이 포함될 수 있습니다. |
| `post_tool_call` | 관찰자 | 차단, 오류 또는 성공 결과 후 실행되며 반환값은 무시됩니다. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message`, `middleware_trace` | 결과/오류 텍스트에 임의의 도구 또는 사용자 콘텐츠와 비밀이 포함될 수 있습니다. |
| `transform_tool_result` | 변환 | `post_tool_call` 후, 대화에 추가하기 전에 실행되며 처음 나온 문자열이 결과를 대체합니다. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message` | 모델에 전달되는 전체 결과와 인수를 노출합니다. |
| `transform_terminal_output` | 변환 | 제한된 포그라운드 프로세스 캡처 후, 최종 출력 제한 전에 실행되며 처음 나온 문자열이 출력을 대체합니다. | `command`, `output`, `returncode`, `task_id`, `env_type` | 명령/출력에 자격 증명이 포함될 수 있습니다. |
| `pre_transcription` | 변환 | 프로바이더 확인 후, 백엔드(기본 제공, 명령 유형 또는 플러그인 등록)가 호출되기 전에 STT 디스패처가 실행합니다. dict 결과는 등록 순서대로 적용되며 필드별로 마지막 기록이 우선합니다(`prompt`, `language`, `model`; `file_path`는 읽기 전용). | `file_path`, `provider`, `model`, `language`, `prompt`, `source` | 최종 프롬프트는 오디오와 함께 구성된 STT 프로바이더로 업로드되므로, 훅 반환값에 비밀을 넣지 마세요. |
| `pre_llm_call` | 지시/제어 | 루프 전에 턴마다 한 번 실행되며, 유효한 모든 문자열/`{"context": ...}` 반환값을 합쳐 사용자 메시지에 주입합니다. | `session_id`, `task_id`, `turn_id`, `user_message`, `conversation_history`, `is_first_turn`, `model`, `platform`, `parent_session_id`, `sender_id` | 전체 사용자 메시지와 대화 기록입니다. |
| `post_llm_call` | 관찰자 | 성공적으로 중단되지 않은 턴을 마무리할 때 실행되며 반환값은 무시됩니다. | `session_id`, `task_id`, `turn_id`, `user_message`, `assistant_response`, `conversation_history`, `model`, `platform` | 전체 프롬프트, 응답 및 기록입니다. |
| `transform_llm_output` | 변환 | `post_llm_call` 및 최종 전달 전에 실행되며, 처음 나온 비어 있지 않은 문자열이 응답을 대체합니다. | `response_text`, `session_id`, `model`, `platform` | 최종 어시스턴트 텍스트 전체입니다. |
| `pre_verify` | 지시/제어 | 제한된 코드 수정 검증 게이트에서 실행되며, 처음 유효한 계속/차단-중지 지시가 턴을 계속 진행하게 합니다. | `session_id`, `platform`, `model`, `coding`, `attempt`, `final_response`, `changed_paths` | 초안 응답과 변경된 경로입니다. |
| `pre_api_request` | 관찰자 | 프로바이더 시도마다 요청 직전에 실행되며 반환값은 무시됩니다. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `user_message`, `conversation_history`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `retry_count`, `request_messages`, `message_count`, `tool_count`, `approx_input_tokens`, `request_char_count`, `max_tokens`, `started_at`, `middleware_trace`, `request` | 높은 민감도: 레거시 `user_message`, `conversation_history`, `request_messages`는 의도적으로 원시 값입니다. 정제된 `request`를 우선 사용하세요. |
| `post_api_request` | 관찰자 | 정규화된 프로바이더 성공 후 실행되며 반환값은 무시됩니다. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `finish_reason`, `message_count`, `response_model`, `response`, `usage`, `assistant_message`, `assistant_content_chars`, `assistant_tool_call_count` | 정제된 `response`를 사용할 수 있지만, 정규화된 원시 `assistant_message`에는 모델/사용자 콘텐츠가 포함될 수 있습니다. `usage`는 사용량 회계 데이터입니다. |
| `api_request_error` | 관찰자 | 실패한 각 프로바이더 시도에서 실행되며 반환값은 무시됩니다. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `status_code`, `retry_count`, `max_retries`, `retryable`, `reason`, `error`, `request` | 오류 텍스트에 프로바이더/사용자 데이터가 포함될 수 있으며, `request`는 정제된 값을 전달하기 위한 것입니다. |
| `on_stream_start` | 관찰자 | 스트리밍 LLM 응답이 시작될 때 디스패치됩니다. 토큰 경로 외부에서 호스트가 소유한 제한 큐를 통해 콜백당 하나의 워커로 전달되며 반환값은 무시됩니다. | `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | 식별자와 라우팅 메타데이터만 포함합니다. |
| `on_stream_delta` | 관찰자 | 제한된 관찰자 큐를 통해 정규화된 스트리밍 텍스트 델타마다 디스패치됩니다. 멈춘 콜백은 자신에게 가장 오래된 이벤트만 버리며 반환값은 무시됩니다. | `delta`, `kind` (`text` 또는 `reasoning`), `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | 델타 텍스트는 원시 모델 출력이며, 추론 델타에는 `plugins.stream_reasoning_deltas` 옵트인이 필요합니다. |
| `on_stream_end` | 관찰자 | 스트림이 닫힌 후 스트리밍 응답이 완료되거나 오류가 발생할 때 디스패치되며 반환값은 무시됩니다. | `final_text`, `finished`, `error`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | 조합된 응답 텍스트 전체이며 오류 텍스트에 프로바이더 데이터가 포함될 수 있습니다. |
| `on_interim_message` | 관찰자 | 최종 답변 전에 루프 중간의 어시스턴트 메시지가 표시될 때(스트리밍 또는 비스트리밍) 디스패치되며 반환값은 무시됩니다. | `text`, `already_streamed`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | 중간 어시스턴트 텍스트 전체입니다. |
| `transform_api_error_classification` | 변환 | 실패한 각 프로바이더 시도에서 기본 제공 분류기의 시작 부분에 실행됩니다. 모든 콜백을 실행한 후 `reason`이 유효한 첫 dict가 우선하며(모두 실행한 다음 첫 값을 선택), 건너뛴 유효한 결과는 런타임 경고를 기록합니다. Python 플러그인에만 적용됩니다. | `provider`, `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages` | `error_message`와 `error_body`에 원시 프로바이더/사용자 데이터가 포함될 수 있습니다. |
| `on_session_start` | 관찰자 | 새 세션의 첫 번째 턴에서 실행되며 반환값은 무시됩니다. | `session_id`, `model`, `platform` | 식별자와 라우팅 메타데이터만 포함합니다. |
| `on_session_end` | 관찰자 | 각 턴을 마무리할 때 표준 방식으로 실행되며, CLI/TUI 종료에는 축소된 레거시 형태가 추가로 있습니다. 반환값은 무시됩니다. | 표준: `session_id`, `task_id`, `turn_id`, `completed`, `failed`, `interrupted`, `turn_exit_reason`, `model`, `platform`; 종료 경로에서는 `reason`/`api_request_id`가 추가되고 일부 필드가 생략될 수 있습니다. | ID, 모델/플랫폼 및 결과이며, 표준 페이로드에는 메시지 본문이 없습니다. |
| `on_session_finalize` | 관찰자 | `finalize_session`을 통한 CLI/TUI/게이트웨이 종료에서 실행됩니다. 게이트웨이 종료 또는 만료 시에는 초기화 없이 마무리될 수 있습니다. 반환값은 무시됩니다. | 표면에 따라 `session_id`, `platform`, 선택적으로 `reason`, `old_session_id`, `new_session_id` | 세션 및 라우팅 식별자입니다. |
| `on_session_reset` | 관찰자 | CLI/TUI 세션 경계와 교체 세션이 생성된 후의 게이트웨이에서 실행되며 반환값은 무시됩니다. | CLI: `session_id`, `platform`, `reason`; TUI: `session_id`, `platform`; 게이트웨이: 여기에 `reason`, `old_session_id`, `new_session_id` 추가 | 세션 및 라우팅 식별자입니다. |
| `on_skill_lifecycle` | 관찰자 | 권위 있는 스킬 사용 상태가 변경된 후 실행되며 반환값은 무시됩니다. | `action`, `skill_name`, `provenance`, `task_id`, `session_id`, `use_count`, `reused`, `reuse_after_patch` | 로컬 스킬 이름과 출처를 노출합니다. |
| `subagent_start` | 관찰자 | 자식 에이전트가 생성되고 실행 직전에 실행되며 반환값은 무시됩니다. | `parent_session_id`, `parent_turn_id`, `parent_subagent_id`, `child_session_id`, `child_subagent_id`, `child_role`, `child_goal` | 자식 목표에 사용자/프로젝트 콘텐츠가 포함될 수 있습니다. |
| `subagent_stop` | 관찰자 | 자식 에이전트가 종료될 때 실행되며 반환값은 무시됩니다. | `parent_session_id`, `parent_turn_id`, `child_session_id`, `child_role`, `child_summary`, `child_status`, `tool_call_history`, `duration_ms` | 요약과 삭제 처리된 도구 기록 메타데이터가 프로젝트 구조를 드러낼 수 있습니다. |
| `pre_gateway_dispatch` | 지시/제어 | 인증/페어링/디스패치 전에 들어온 비내부 메시지에서 실행되며, 처음 유효한 `skip`, `rewrite` 또는 `allow`가 흐름을 제어합니다. | `event`, `gateway`, `session_store` | 매우 높은 권한의 프로세스 내부 객체로, 수신 사용자/라우팅 데이터와 호스트 핸들에 접근할 수 있습니다. |
| `gateway_platform_event` | 관찰자 | 게이트웨이의 프로필 범위 권한 부여가 성공한 후, 지원되는 플랫폼 네이티브 이벤트가 게이트웨이 경계에서 정규화될 때 실행됩니다(Telegram: 리액션, 메시지 편집; Discord: 메시지 편집/삭제, 스레드 생성/이름 변경). 반환값은 무시됩니다. | `platform`, `event_type`, `payload` (이벤트 유형별 dict — 아래 개별 이벤트 계약 참조) | 정규화된 일반 dict 봉투만 전달되며, 원시 SDK 객체, 어댑터 핸들, 봇 클라이언트는 노출되지 않습니다. |
| `pre_command` | 관찰자 | 인식된 슬래시 명령이 CLI 및 게이트웨이의 콜드 패스 디스패치에서 핸들러 실행 직전에 디스패치되며 반환값은 무시됩니다. v1에서는 지시 형태의 dict를 디버그 수준으로 기록합니다. 게이트웨이에서 실행 중인 에이전트가 가로채는 명령(`/stop`, 실행 중 `/approve`)은 의도적으로 제외됩니다. 제어면의 비상 탈출구는 플러그인의 접근 범위 밖에 있어야 합니다. | `surface` (`"cli"` | `"gateway"`), `command` (표준 이름), `alias_used`, `args_raw`, `session_key`, `platform` | `args_raw`에 명령 뒤에 입력한 사용자 콘텐츠나 비밀이 포함될 수 있습니다. |
| `pre_approval_request` | 관찰자 | 프롬프트 또는 스마트 승인 전에 실행되며 반환값은 무시됩니다. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id` | 명령에 비밀이 포함될 수 있습니다. 스마트 관찰자 준비 단계에서는 강제 삭제 처리를 하지만, 표면마다 삭제 처리가 동일하지는 않습니다. |
| `post_approval_response` | 관찰자 | 결정, 시간 초과 또는 게이트웨이 알림 실패 후 실행되며 반환값은 무시됩니다. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id`, `choice`; 스마트 경로에서는 `decided_by`가 추가될 수 있습니다. | 동일한 명령 민감도와 결정 메타데이터입니다. |
| `kanban_task_claimed` | 관찰자 | 클레임 커밋 후, 워커 생성 전에 디스패처 프로세스에서 실행되며 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id` | 보드/작업/프로필/담당자 식별자입니다. |
| `kanban_task_completed` | 관찰자 | 완료 및 정리 후, 대개 워커 프로세스에서 실행되며 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `summary` | 요약에 프로젝트/사용자 콘텐츠가 포함될 수 있습니다. |
| `kanban_task_blocked` | 관찰자 | 차단 상태로 전환된 후 실행됩니다. 의존성 대기 경로는 트랜잭션이 종료되기 전에 실행됩니다. 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `reason` | 사유에 프로젝트/사용자 콘텐츠가 포함될 수 있습니다. |
| `on_kanban_worker_spawned` | 관찰자 | `spawn_fn`이 반환되고 워커 PID가 저장된 후 실행됩니다. 디스패치 잠금 안에서 실행되므로 콜백을 빠르게 유지하세요. 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `workspace_path` | `workspace_path`는 파일 시스템 경로이며 프로젝트 구조나 사용자 이름을 드러낼 수 있습니다. |
| `on_kanban_worker_exited` | 관찰자 | 틱에서 파생됩니다. `detect_crashed_workers`가 사망한 PID의 작업을 회수하고 회수 작업을 커밋한 후 실행됩니다. 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status` | 식별자와 종료 메타데이터만 포함합니다. |
| `on_kanban_worker_stale_claim` | 관찰자 | TTL이 만료된 클레임을 회수한 후 실행되며, 살아 있는 PID의 연장은 실행되지 않습니다. 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `heartbeat_stale`, `retry_status` | 식별자와 클레임 메타데이터입니다. |
| `on_kanban_task_updated` | 관찰자 | 클레임/완료/차단 수명 주기 외부에서 작업 필드를 커밋해 기록한 후 실행됩니다(할당, 재정의, 대시보드 편집기). 반환값은 무시됩니다. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `changed_fields` | `changed_fields`에는 필드 이름만 담기고 값은 담기지 않습니다. 보드 DB에서 해당 제목/본문 값에는 사용자/프로젝트 콘텐츠가 포함될 수 있습니다. |
| `on_kanban_dispatch_tick` | 관찰자 | 디스패처 틱마다 한 번, 디스패치 잠금이 해제된 직후에 실행됩니다. 유휴 틱과 경합 틱에서도 실행됩니다. 반환값은 무시됩니다. | `board`, `profile_name`, `dry_run`, `outcome`, `result` | `result`는 작업 ID, 담당자 및 워크스페이스 경로를 담은 해당 틱의 `DispatchResult`입니다. |

---
### 스트리밍 출력 훅

이러한 옵저버 전용 훅을 사용하면 플러그인이 스트리밍 LLM 출력을 소비하여 원격 측정, 실시간 대시보드 또는 TTS 파이프라인에 활용할 수 있습니다. 출력은 호스트가 소유한 제한된 큐를 통해 전달되며, 등록된 콜백마다 백그라운드 워커가 하나씩 실행되므로 플러그인 콜백은 토큰 경로에서 인라인으로 실행되지 않습니다. 콜백 하나가 멈추더라도 해당 콜백의 큐만 가득 차서 대기 중인 가장 오래된 옵저버 이벤트를 버립니다. 다른 옵저버는 서로 독립적으로 계속 이벤트를 수신합니다.

다른 플러그인 훅과 마찬가지로 등록합니다.

```python
def on_delta(delta, kind, model, provider, **kwargs):
    if kind == "text":
        print(delta, end="", flush=True)

def register(ctx):
    ctx.register_hook("on_stream_delta", on_delta)
```

네 가지 훅에 공통으로 제공되는 필드:

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `turn_id` | `str` | 사용 가능한 경우의 불투명한 턴 식별자 |
| `iteration` | `int` | 현재 API 호출/도구 루프 반복 횟수 |
| `session_id` | `str` | 현재 Hermes 세션 ID |
| `model` | `str` | 활성 모델 식별자 |
| `provider` | `str` | 활성 프로바이더 이름 |
| `surface` | `str` | 호출 표면. 예: `cli`, `discord`, `telegram` |

추가 필드:

| 훅 | 추가 필드 |
|------|--------------|
| `on_stream_start` | 없음 |
| `on_stream_delta` | `delta: str`, `kind: "text" | "reasoning"` |
| `on_stream_end` | `final_text: str`, `finished: bool`, `error: str | None` |
| `on_interim_message` | `text: str`, `already_streamed: bool` |

`on_interim_message`는 스트리밍하지 않은 응답 이후에도 실행될 수 있으므로, 이 훅만 등록해도 프로바이더 호출이 스트리밍 전송을 사용하도록 강제되지는 않습니다.

기본적으로 플러그인에는 추론 델타가 노출되지 않습니다. 명시적으로 활성화하려면 다음을 설정하세요.

```yaml
plugins:
  stream_reasoning_deltas: true
```

반환 값은 무시됩니다. 스트림을 빠르게 유지하려면 콜백이 자체 작업을 큐에 넣고 신속히 반환해야 합니다. 예외는 로그에 기록되며 스트림을 중단하지 않습니다.

---

### `pre_tool_call`

모든 도구 실행(기본 제공 도구와 플러그인 도구 모두) **직전에** 실행됩니다.

**콜백 시그니처:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `tool_name` | `str` | 곧 실행될 도구의 이름(예: `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | 모델이 도구에 전달한 인수 |
| `task_id` | `str` | 세션/작업 식별자. 설정되지 않았으면 빈 문자열입니다. |

**실행 위치:** 도구 핸들러가 실행되기 전 `model_tools.py`의 `handle_function_call()` 내부에서 실행됩니다. 도구 호출마다 한 번 실행됩니다. 모델이 도구 3개를 병렬로 호출하면 3번 실행됩니다.

**반환 값 — 차단 또는 승인 요구:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
# or
return {"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}
```

유효한 지시문 중 첫 번째 지시가 적용됩니다. `block`에는 비어 있지 않은 `message`가 필요하며, 도구 호출을 즉시 종료하고 해당 텍스트를 모델에 반환되는 오류로 사용합니다. `approve`는 호출을 기존 사람 승인 게이트로 전달합니다. `message`와 `rule_key`는 선택 사항이며, 거부·시간 초과·게이트 오류가 발생하면 안전하게 실패합니다. 그 밖의 반환 값은 무시됩니다.

**사용 사례:** 로깅, 감사 추적, 도구 호출 횟수 집계, 위험한 작업 차단, 속도 제한, 사용자별 정책 적용.

**예시 — 도구 호출 감사 로그:**

```python
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def audit_tool_call(tool_name, args, task_id, **kwargs):
    logger.info("TOOL_CALL session=%s tool=%s args=%s",
                task_id, tool_name, json.dumps(args)[:200])

def register(ctx):
    ctx.register_hook("pre_tool_call", audit_tool_call)
```

**예시 — 위험한 도구 경고:**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ Executing potentially dangerous tool: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

모든 도구 실행이 반환된 **직후** 실행됩니다.

**콜백 시그니처:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `tool_name` | `str` | 방금 실행된 도구의 이름 |
| `args` | `dict` | 모델이 도구에 전달한 인수 |
| `result` | `str` | 도구의 반환 값(항상 JSON 문자열) |
| `task_id` | `str` | 세션/작업 식별자. 설정되지 않았으면 빈 문자열입니다. |
| `duration_ms` | `int` | 도구 디스패치에 걸린 시간(밀리초). `registry.dispatch()` 전후를 `time.monotonic()`으로 측정합니다. |

**실행 위치:** 도구 핸들러가 반환된 후 `model_tools.py`의 `handle_function_call()` 내부에서 실행됩니다. 도구 호출마다 한 번 실행됩니다. 처리되지 않은 예외를 도구가 발생시킨 경우에는 실행되지 않습니다(오류가 포착되어 오류 JSON 문자열로 반환되는 경우에는 예외가 발생하더라도 `post_tool_call`이 해당 오류 문자열을 `result`로 받아 실행됩니다).

**반환 값:** 무시됩니다.

**사용 사례:** 도구 결과 로깅, 메트릭 수집, 도구별 성공/실패율 추적, 지연 시간 대시보드, 도구별 예산 경고, 특정 도구 완료 시 알림 전송.

**예시 — 도구 사용 메트릭 추적:**

```python
from collections import Counter, defaultdict
import json

_tool_counts = Counter()
_error_counts = Counter()
_latency_ms = defaultdict(list)

def track_metrics(tool_name, result, duration_ms=0, **kwargs):
    _tool_counts[tool_name] += 1
    _latency_ms[tool_name].append(duration_ms)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            _error_counts[tool_name] += 1
    except (json.JSONDecodeError, TypeError):
        pass

def register(ctx):
    ctx.register_hook("post_tool_call", track_metrics)
```

---

### `pre_llm_call`

도구 호출 루프가 시작되기 전에 턴마다 **한 번** 실행됩니다. 유효한 콜백 반환 값은 모두 플러그인 순서대로 합쳐져 현재 턴의 사용자 메시지에 삽입됩니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 현재 세션의 고유 식별자 |
| `user_message` | `str` | 이 턴의 원래 사용자 메시지(스킬 삽입 전) |
| `conversation_history` | `list` | 전체 메시지 목록의 복사본(OpenAI 형식: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | 새 세션의 첫 턴이면 `True`, 이후 턴이면 `False` |
| `model` | `str` | 모델 식별자(예: `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | 세션이 실행되는 위치: `"cli"`, `"telegram"`, `"discord"` 등 |

**실행 위치:** `run_agent.py`의 `run_conversation()` 내부에서 컨텍스트 압축 이후, 메인 `while` 루프 전에 실행됩니다. API 호출마다가 아니라 `run_conversation()` 호출마다(즉, 사용자 턴마다) 한 번 실행됩니다.

**반환 값:** 콜백이 `"context"` 키가 있는 딕셔너리 또는 일반적인 비어 있지 않은 문자열을 반환하면 해당 텍스트가 현재 턴의 사용자 메시지에 추가됩니다. 삽입하지 않으려면 `None`을 반환하세요.

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**컨텍스트 삽입 위치:** 항상 **사용자 메시지**이며, 시스템 프롬프트가 아닙니다. 시스템 프롬프트가 턴마다 동일하게 유지되어 캐시된 토큰을 재사용할 수 있으므로 프롬프트 캐시가 보존됩니다. 시스템 프롬프트는 Hermes의 영역입니다(모델 지침, 도구 적용, 개성, 스킬). 플러그인은 사용자 입력과 함께 컨텍스트를 제공합니다.

깨끗한 사용자 메시지의 `content`는 변경되지 않습니다. 재생 및 프롬프트 캐시 안정성을 위해 Hermes는 플러그인이 삽입한 컨텍스트를 포함한 API 전송용 정확한 메시지를 행의 `api_content` 사이드카에 저장할 수 있습니다.

**여러 플러그인이** 컨텍스트를 반환하면 출력은 플러그인 검색 순서(디렉터리 이름의 알파벳순)에 따라 빈 줄 두 개로 연결됩니다.

**사용 사례:** 메모리 검색, RAG 컨텍스트 삽입, 가드레일, 턴별 분석.

**예시 — 메모리 검색:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall(session_id, user_message, is_first_turn, **kwargs):
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None
        text = "Recalled context:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**예시 — 가드레일:**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

도구 호출 루프가 완료되고 에이전트가 최종 응답을 생성한 후 턴마다 **한 번** 실행됩니다. 성공한 턴에서만 실행되며, 턴이 중단된 경우에는 실행되지 않습니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 현재 세션의 고유 식별자 |
| `user_message` | `str` | 이 턴의 원래 사용자 메시지 |
| `assistant_response` | `str` | 이 턴에 대한 에이전트의 최종 텍스트 응답 |
| `conversation_history` | `list` | 턴 완료 후 전체 메시지 목록의 복사본 |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행되는 위치 |

**실행 위치:** `run_agent.py`의 `run_conversation()` 내부에서 최종 응답과 함께 도구 루프가 종료된 후 실행됩니다. `if final_response and not interrupted`로 보호되므로, 사용자가 턴 중간에 인터럽트를 발생시키거나 에이전트가 응답을 생성하지 못한 채 반복 횟수 제한에 도달하면 실행되지 않습니다.

**반환 값:** 무시됩니다.

**사용 사례:** 대화 데이터를 외부 메모리 시스템과 동기화, 응답 품질 메트릭 계산, 턴 요약 로깅, 후속 작업 트리거.

**예시 — 외부 메모리와 동기화:**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def sync_memory(session_id, user_message, assistant_response, **kwargs):
    try:
        httpx.post(f"{MEMORY_API}/store", json={
            "session_id": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }, timeout=5)
    except Exception:
        pass  # best-effort

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**예시 — 응답 길이 추적:**

```python
import logging
logger = logging.getLogger(__name__)

def log_response_length(session_id, assistant_response, model, **kwargs):
    logger.info("RESPONSE session=%s model=%s chars=%d",
                session_id, model, len(assistant_response or ""))

def register(ctx):
    ctx.register_hook("post_llm_call", log_response_length)
```

---

### `pre_verify`

에이전트가 코드를 편집한 경우 턴마다 **한 번**, 종료 직전에 실행됩니다(기본 제공 verify-on-stop 가드 이후). 이 훅은 사용자/플러그인 정책 게이트입니다. 콜백이 에이전트를 계속 실행하도록 할 수 있습니다. 즉, 에이전트가 종료되는 대신 검사를 실행하거나, 검사를 연기하거나, diff를 정리할 수 있습니다.

Hermes에 포함된 검증 지침은 기본 `pre_verify` 훅이 아닙니다. 편집된 코드에 최신 검증 증거가 없을 때 증거 기반 verify-on-stop 알림에 추가되므로, 두 번째 기본 계속 실행 경로를 만들지 않습니다. 기본 제공 증거 알림을 간결하게 유지하려면 `agent.verify_guidance: false`로 설정하세요.

**콜백 시그니처:**

```python
def my_callback(session_id: str, platform: str, model: str, coding: bool,
                attempt: int, final_response: str, changed_paths: list, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 현재 세션의 고유 식별자 |
| `platform` | `str` | 세션이 실행되는 위치(`"cli"`, `"telegram"`, …) |
| `model` | `str` | 모델 식별자 |
| `coding` | `bool` | 턴이 코딩 자세인지 여부(코드 작업 공간에서 실행되는지) — 이 값을 기준으로 훅 범위를 지정하세요. |
| `attempt` | `int` | 이 턴에 이미 알림을 보낸 횟수(첫 번째는 0) — 이 값을 기준으로 자체 제한하세요. |
| `final_response` | `str` | 에이전트가 곧 전달할 답변 |
| `changed_paths` | `list` | 이 턴에 에이전트가 편집한 파일(정렬되어 있으며 여기서는 항상 비어 있지 않음) |

`coding`을 확인하여 코딩 컨텍스트에 훅을 적용하고, 셸 훅이 `.extra`에서 두 값을 모두 읽는 것과 같은 방식으로 `attempt`를 사용해 한 번만 실행되도록 하세요. 이는 `pre_tool_call` 훅이 `tool_name`을 기준으로 범위를 지정하는 방식과 같습니다. 이렇게 하면 여러 `pre_verify` 훅을 등록하더라도 각각 필요한 위치에서만 실행됩니다.

**실행 위치:** 에이전트가 최종 답변을 수락하려는 시점, 즉 verify-on-stop 검사 직후 `agent/conversation_loop.py`에서 실행됩니다. 단, 해당 턴에 에이전트가 코드를 편집했고 하나 이상의 `pre_verify` 훅이 등록된 경우에만 실행됩니다.

**반환 값 — 에이전트를 계속 실행:**

```python
return {"action": "continue", "message": "Run the formatter on your changes, then finish."}
```

`message`가 합성 사용자 턴으로 추가되고 루프가 다시 실행됩니다. 종료를 차단한다는 의미의 Claude-Code Stop 형식(`{"decision": "block", "reason": "..."}`)도 허용됩니다. 메시지가 없는 지시문이나 그 밖의 반환 값은 턴을 종료합니다.

**제한:** 한 턴에서 연속된 계속 실행 지시 횟수는 `agent.max_verify_nudges`(기본값 3)로 제한되므로, 항상 continue를 반환하는 훅도 루프를 가둘 수 없습니다. 에이전트가 알림을 받는 동안 시도한 답변은 기록에 유지되지만 사용자에게 표시되지는 않습니다.

**멱등성 유지:** 알림이 발생할 때마다 훅이 다시 실행되므로 `attempt`를 기준으로 제한하세요(`if attempt: return None`). 그렇지 않으면 제한에 도달할 때까지 계속 알림을 보냅니다.

**사용 사례:** 창의적인 반복 작업 중 테스트/린트 연기, 특정 경로에 대한 통과 검사 요구, 변경 로그 항목이 생길 때까지 "완료" 차단, 프로젝트별 검증 체크리스트 실행.

**예시 — 범위를 지정해 창의적인 UI 작업의 검사를 연기하고 한 번만 실행:**

```python
UI = (".tsx", ".jsx", ".css", ".scss")

def defer_ui_checks(coding, attempt, changed_paths, **kwargs):
    if attempt or not coding:
        return None  # one-shot, coding only
    if not all(p.endswith(UI) for p in changed_paths):
        return None  # only pure-UI edits
    return {
        "action": "continue",
        "message": "This is UI work — don't run tests/lints yet; ask the user to "
                   "eyeball it first, and clean the diff before any commit.",
    }

def register(ctx):
    ctx.register_hook("pre_verify", defer_ui_checks)
```

기본 제공 누락 증거 알림에 반영되어야 하는 상시 지침에는 `agent.verify_guidance`를 사용하세요. 검증을 *게이트*할 필요가 없는 더 광범위한 코딩 자세 규칙에는 `config.yaml`의 `agent.coding_instructions`를 사용하는 편이 좋습니다. 이 설정은 코딩 브리프에 포함되며 추가 턴을 발생시키지 않습니다.

---
### `transform_api_error_classification`

실패한 API 호출마다 `agent/error_classifier.classify_api_error()`의 최상단에서, 내장 파이프라인보다 먼저 한 번 실행됩니다. 공급자 플러그인은 코어를 수정하지 않고 자신이 담당하는 공급자의 오류 특성을 처리할 수 있습니다. 이는 동작을 변경하는(transform family) 훅입니다. 반환된 분류가 재시도, 압축, 자격 증명 교체 및 대체 경로 라우팅을 결정합니다.

콜백은 파싱된 오류 컨텍스트를 kwargs로 받습니다 — `provider` (이 콜백이 속한 self 범위), `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages`. 처리를 거부하려면 `None`을 반환하고, 오류를 처리하려면 분류를 나타내는 dict를 반환합니다.

```python
return {"reason": "model_not_found",   # required: a FailoverReason name
        "retryable": False, "should_fallback": True}  # optional recovery-hint overrides
```

디스패치는 모두 실행한 뒤 첫 번째를 선택하는 방식입니다. 모든 콜백이 실행되며, 각 콜백의 실패는 격리되고, 등록 순서상 유효한 첫 번째 결과가 승자가 됩니다(유효하지만 선택되지 않은 결과는 런타임 경고를 기록합니다). 유효하지 않은 dict와 알 수 없는 reason은 건너뛰므로, 고장 난 플러그인 하나 때문에 분류가 중단되는 일은 없습니다.

**개인정보 보호:** `error_message`와 `error_body`에는 마스킹되지 않은 공급자 데이터가 포함될 수 있습니다. **Python 플러그인만 허용됩니다** — 셸 등록은 설정을 파싱할 때 경고와 함께 거부됩니다.

---

### `on_session_start`

새 세션이 생성될 때 **한 번** 실행됩니다. 세션을 계속 이어갈 때(사용자가 기존 세션에서 두 번째 메시지를 보낼 때)에는 실행되지 않습니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 새 세션의 고유 식별자 |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행되는 위치 |

**실행 위치:** `run_agent.py`의 `run_conversation()`에서 새 세션의 첫 번째 턴 중 — 시스템 프롬프트를 만든 후, 도구 루프를 시작하기 전에 실행됩니다. 검사는 `if not conversation_history`입니다(이전 메시지가 없으면 새 세션).

**반환값:** 무시됩니다.

**사용 사례:** 세션 범위 상태 초기화, 캐시 예열, 외부 서비스에 세션 등록, 세션 시작 로깅.

**예시 — 세션 캐시 초기화:**

```python
_session_caches = {}

def init_session(session_id, model, platform, **kwargs):
    _session_caches[session_id] = {
        "model": model,
        "platform": platform,
        "tool_calls": 0,
        "started": __import__("datetime").datetime.now().isoformat(),
    }

def register(ctx):
    ctx.register_hook("on_session_start", init_session)
```

---

### `on_session_end`

결과와 관계없이 모든 `run_conversation()` 호출이 **완전히 끝나는 시점**에 실행됩니다. 에이전트가 사용자가 종료했을 때 턴을 진행 중이었다면 CLI의 종료 핸들러에서도 실행됩니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 세션의 고유 식별자 |
| `completed` | `bool` | 에이전트가 최종 응답을 생성했으면 `True`, 그렇지 않으면 `False` |
| `interrupted` | `bool` | 턴이 중단되었으면 `True` (사용자가 새 메시지를 보냈거나 `/stop`을 실행했거나 종료한 경우) |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행되는 위치 |

**실행 위치:** 두 곳에서 실행됩니다.
1. **`run_agent.py`** — 모든 정리가 끝난 뒤 모든 `run_conversation()` 호출의 마지막에 실행됩니다. 턴에서 오류가 발생한 경우에도 항상 실행됩니다.
2. **`cli.py`** — CLI의 atexit 핸들러에서 실행되지만, 종료 시 에이전트가 턴을 진행 중(`_agent_running=True`)인 경우에만 실행됩니다. 처리 중 Ctrl+C와 `/exit`를 포착합니다. 이 경우 `completed=False`, `interrupted=True`입니다.

**반환값:** 무시됩니다.

**사용 사례:** 버퍼 플러시, 연결 종료, 세션 상태 저장, 세션 기간 로깅, `on_session_start`에서 초기화한 리소스 정리.

**예시 — 플러시 및 정리:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # Flush accumulated data to disk or external service
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**예시 — 세션 기간 추적:**

```python
import time, logging
logger = logging.getLogger(__name__)

_start_times = {}

def on_start(session_id, **kwargs):
    _start_times[session_id] = time.time()

def on_end(session_id, completed, interrupted, **kwargs):
    start = _start_times.pop(session_id, None)
    if start:
        duration = time.time() - start
        logger.info("SESSION_DURATION session=%s seconds=%.1f completed=%s interrupted=%s",
                     session_id, duration, completed, interrupted)

def register(ctx):
    ctx.register_hook("on_session_start", on_start)
    ctx.register_hook("on_session_end", on_end)
```

---

### `on_session_finalize`

CLI 또는 gateway가 활성 세션을 **해체할** 때 실행됩니다 — 예를 들어 사용자가 `/new`를 실행했거나, gateway가 유휴 세션을 정리했거나, 활성 에이전트가 있는 상태에서 CLI가 종료된 경우입니다. 나가는 세션 ID에 연결된 상태를 플러시할 때 사용합니다. gateway 재설정 시에는 이 콜백이 실행되기 전에 교체 세션이 이미 존재합니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` 또는 `None` | 나가는 세션 ID. 활성 세션이 없었다면 `None`일 수 있습니다. |
| `platform` | `str` | `"cli"` 또는 메시징 플랫폼 이름(`"telegram"`, `"discord"` 등). |

**실행 위치:** CLI/TUI 해체 과정과 gateway 재설정, 종료 또는 유휴 만료 경로에서 실행됩니다. gateway 종료 및 만료는 일치하는 `on_session_reset` 없이도 finalize할 수 있습니다.

**반환값:** 무시됩니다.

**사용 사례:** 세션 ID가 폐기되기 전에 최종 세션 지표 저장, 세션별 리소스 종료, 최종 텔레메트리 이벤트 발행, 대기 중인 쓰기 작업 처리.

---

### `on_session_reset`

CLI 또는 TUI 세션 경계에서, 또는 gateway가 활성 채팅에 새 세션 키를 **교체해 넣을** 때 실행됩니다. 다음 `on_session_start`를 기다리지 않고, 대화 상태가 초기화된 것에 플러그인이 반응할 수 있습니다.

**콜백 시그니처:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 새 세션의 ID(이미 새 값으로 교체됨). |
| `platform` | `str` | `"cli"`, `"tui"` 또는 메시징 플랫폼 이름. |
| `reason` | `str`, 선택 사항 | CLI 및 gateway 재설정 경로에서 제공됩니다. |
| `old_session_id` | `str`, 선택 사항 | gateway 전용으로, 나가는 세션 ID입니다. |
| `new_session_id` | `str`, 선택 사항 | gateway 전용으로, 교체된 세션 ID입니다. |

**실행 위치:** CLI는 `session_id`, `platform`, `reason`을 제공합니다. TUI는 `session_id`와 `platform`을 제공합니다. gateway는 교체 키를 할당한 후 `reason`, `old_session_id`, `new_session_id`를 추가합니다. gateway 재설정 순서는 다음과 같습니다: 교체 세션 생성 및 저장 → `on_session_finalize(old_id)` → `on_session_reset(new_id)` → 첫 번째 인바운드 턴에서 `on_session_start(new_id)` 실행.

**반환값:** 무시됩니다.

**사용 사례:** `session_id`를 키로 사용하는 세션별 캐시 초기화, "세션 교체" 분석 이벤트 발행, 새 상태 버킷 예열.

---

**[플러그인 빌드 가이드](/developer-guide/plugins)**에서 도구 스키마, 핸들러 및 고급 훅 패턴을 포함한 전체 안내를 확인할 수 있습니다.

---

### `subagent_start`

`delegate_task`가 자식 `AIAgent`를 구성한 후, 해당 자식이 실행되기 전에 **자식 에이전트마다 한 번씩** 실행됩니다. 단일 작업을 위임하든 세 작업의 배치를 위임하든, 각 자식마다 한 번씩 이 훅이 실행됩니다.

이 훅은 위임/서브에이전트 수명 주기에만 해당합니다. gateway, CLI, cron, batch, MoA 또는 기타 실행기가 시작한 에이전트 실행을 포괄하는 "모든 에이전트 호출 전" 게이트가 아닙니다.

**콜백 시그니처:**

```python
def my_callback(parent_session_id: str | None,
                parent_turn_id: str,
                parent_subagent_id: str | None,
                child_session_id: str | None,
                child_subagent_id: str,
                child_role: str,
                child_goal: str,
                **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `parent_session_id` | `str \| None` | 위임하는 부모 에이전트의 세션 ID. |
| `parent_turn_id` | `str` | 위임을 요청한 부모 에이전트 턴의 턴 ID(사용 가능한 경우). |
| `parent_subagent_id` | `str \| None` | 이 자식이 다른 서브에이전트에 의해 생성된 경우 부모 서브에이전트 ID이며, 최상위 부모 에이전트이면 `None`. |
| `child_session_id` | `str \| None` | 자식 에이전트에 할당된 세션 ID. |
| `child_subagent_id` | `str` | 위임 관찰 및 제어에 사용되는 안정적인 서브에이전트 ID. |
| `child_role` | `str` | 위임 정책 적용 후의 유효 자식 역할. 예를 들어 `"leaf"` 또는 `"orchestrator"`. |
| `child_goal` | `str` | 자식 에이전트가 실행할 위임된 목표/프롬프트. |

**실행 위치:** `tools/delegate_tool.py`의 `_build_child_agent()` 내부에서, 자식 `AIAgent`가 구성되고 서브에이전트 식별 메타데이터가 추가된 후, `_run_single_child()`가 자식을 실행하기 전에 실행됩니다.

**반환값:** 무시됩니다. 이 훅은 관찰 전용이며, 값을 반환해도 자식 에이전트 실행을 차단하거나 변경하지 않습니다.

**사용 사례:** 서브에이전트 생성 로깅, 부모/자식 세션 관계 매핑, 중첩된 위임 트리 추적, 실행 전 감사 기록 발행, 자식별 관찰 리소스 사전 할당.

**예시 — 서브에이전트 생성 로깅:**

```python
import logging

logger = logging.getLogger(__name__)

def log_subagent_start(
    parent_session_id,
    parent_turn_id,
    child_session_id,
    child_subagent_id,
    child_role,
    child_goal,
    **kwargs,
):
    logger.info(
        "SUBAGENT_START parent=%s turn=%s child_session=%s child=%s role=%s goal=%r",
        parent_session_id,
        parent_turn_id,
        child_session_id,
        child_subagent_id,
        child_role,
        child_goal[:200],
    )

def register(ctx):
    ctx.register_hook("subagent_start", log_subagent_start)
```

:::info
`subagent_start`는 위임 관찰에 유용하지만 차단 정책 훅은 아닙니다. 자식을 구성하기 전에 위임을 차단하려면 [`pre_tool_call`](#pre_tool_call)을 사용해 `delegate_task` 도구 호출을 차단하세요.
:::

---

### `subagent_stop`

`delegate_task`가 완료된 후 **자식 에이전트마다 한 번씩** 실행됩니다. 단일 작업을 위임하든 세 작업의 배치를 위임하든, 부모 스레드에서 직렬화되어 각 자식마다 한 번씩 실행됩니다.

**콜백 시그니처:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                tool_call_history: list[dict], duration_ms: int, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `parent_session_id` | `str` | 위임하는 부모 에이전트의 세션 ID |
| `child_role` | `str \| None` | 자식에 설정된 오케스트레이터 역할 태그(기능이 활성화되지 않았으면 `None`) |
| `child_summary` | `str \| None` | 자식이 부모에게 반환한 최종 응답 |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"` 또는 `"error"` |
| `tool_call_history` | `list[dict]` | 순서가 있는 메타데이터 전용 도구 호출: `tool_name`, 제한된 `tool_input`, `input_bytes`, `output_bytes`, `status`; 원본 입력과 출력은 제외됩니다 |
| `duration_ms` | `int` | 자식 실행에 걸린 벽시계 시간(밀리초) |

**실행 위치:** `tools/delegate_tool.py`에서 `ThreadPoolExecutor.as_completed()`가 모든 자식 future를 처리한 후 실행됩니다. 실행은 부모 스레드로 전달되므로 훅 작성자가 콜백의 동시 실행을 고려할 필요가 없습니다.

**반환값:** 무시됩니다.

**사용 사례:** 오케스트레이션 활동 로깅, 과금을 위한 자식 실행 시간 누적, 위임 후 감사 기록 작성.

**예시 — 오케스트레이터 활동 로깅:**

```python
import logging
logger = logging.getLogger(__name__)

def log_subagent(parent_session_id, child_role, child_status, duration_ms, **kwargs):
    logger.info(
        "SUBAGENT parent=%s role=%s status=%s duration_ms=%d",
        parent_session_id, child_role, child_status, duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_stop", log_subagent)
```

:::info
대규모 위임(예: 오케스트레이터 역할  × 리프 5개 × 중첩 깊이)에서는 턴마다 `subagent_stop`이 여러 번 실행됩니다. 콜백을 빠르게 유지하고, 비용이 큰 작업은 백그라운드 큐로 넘기세요.
:::

---
### `pre_gateway_dispatch`

게이트웨이에서 **각 `MessageEvent`가 들어올 때마다 한 번** 실행되며, 내부 이벤트 가드 이후, 인증/페어링 및 에이전트 디스패치 전에 실행됩니다. 단일 플랫폼 어댑터에 깔끔하게 들어맞지 않는 게이트웨이 수준의 메시지 흐름 정책(수신 전용 시간대, 사람에게 인계, 채팅별 라우팅 등)을 적용할 수 있는 지점입니다.

**콜백 시그니처:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `event` | `MessageEvent` | 정규화된 인바운드 메시지입니다(`.text`, `.source`, `.message_id`, `.internal` 등을 가짐). |
| `gateway` | `GatewayRunner` | 활성 게이트웨이 실행기입니다. 플러그인은 `gateway.adapters[platform].send(...)`를 호출해 사이드 채널로 답장(소유자 알림 등)을 보낼 수 있습니다. |
| `session_store` | `SessionStore` | `session_store.append_to_transcript(...)`를 통한 무응답 트랜스크립트 수집에 사용합니다. |

**실행 위치:** `gateway/run.py`의 `GatewayRunner._handle_message()`에서 `is_internal`을 계산한 직후입니다. **내부 이벤트는 훅 전체를 건너뜁니다**(백그라운드 프로세스 완료 등 시스템이 생성한 이벤트이며, 사용자 대상 정책의 게이트를 적용하면 안 됩니다).

**반환값:** `None` 또는 딕셔너리입니다. 처음 인식된 액션 딕셔너리가 우선하며, 나머지 플러그인 결과는 무시됩니다. 플러그인 콜백의 예외는 잡아서 로그에 기록하며, 오류가 발생해도 게이트웨이는 항상 일반 디스패치로 계속 진행합니다.

| 반환값 | 효과 |
|--------|------|
| `{"action": "skip", "reason": "..."}` | 메시지를 버립니다. 에이전트 답장, 페어링 흐름, 인증이 발생하지 않습니다. 플러그인이 처리한 것으로 간주합니다(예: 트랜스크립트에 조용히 수집). |
| `{"action": "rewrite", "text": "new text"}` | `event.text`를 바꾼 뒤 수정된 이벤트로 일반 디스패치를 계속합니다. 버퍼링된 주변 메시지를 하나의 프롬프트로 합칠 때 유용합니다. |
| `{"action": "allow"}` / `None` | 일반 디스패치로 진행합니다. 전체 인증/페어링/에이전트 루프 체인을 실행합니다. |

**사용 사례:** 수신 전용 그룹 채팅(태그되었을 때만 응답하고 주변 메시지는 컨텍스트에 버퍼링); 사람에게 인계(소유자가 수동으로 채팅을 처리하는 동안 고객 메시지를 조용히 수집); 프로필별 속도 제한; 정책 기반 라우팅.

**예시 — 페어링 코드를 트리거하지 않고 승인되지 않은 DM을 조용히 삭제하기:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**예시 — 멘션 시 주변 메시지 버퍼를 하나의 프롬프트로 다시 작성하기:**

```python
_buffers = {}

def buffer_or_rewrite(event, **kwargs):
    key = (event.source.platform, event.source.chat_id)
    buf = _buffers.setdefault(key, [])
    if _bot_mentioned(event.text):
        combined = "\n".join(buf + [event.text])
        buf.clear()
        return {"action": "rewrite", "text": combined}
    buf.append(event.text)
    return {"action": "skip", "reason": "ambient-buffered"}

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", buffer_or_rewrite)
```

---

### `gateway_platform_event`

게이트웨이의 일반적인 프로필 범위 인증 검사가 성공한 **후에만** 지원되는 플랫폼 네이티브 이벤트에 대해 실행됩니다. 콜백에는 일반 딕셔너리가 전달되며, 원시 SDK 객체, 어댑터 핸들, 봇 클라이언트, 콜백 컨텍스트는 안정적인 계약의 일부가 아닙니다.

Telegram 메시지 리액션이 처음 지원된 이벤트였고, 이후 메시지 수정, 삭제, 스레드 생명주기 이벤트가 추가되었습니다.

```python
def on_platform_event(platform, event_type, payload, **kwargs):
    if platform == "telegram" and event_type == "reaction":
        print(payload["chat_id"], payload["message_id"], payload["emojis"])
    elif event_type == "message_edited":
        print(platform, payload["chat_id"], payload["message_id"], payload["text"])

def register(ctx):
    ctx.register_hook("gateway_platform_event", on_platform_event)
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `platform` | `str` | 안정적인 플랫폼 id입니다(`"telegram"`, `"discord"`). |
| `event_type` | `str` | 이벤트별 계약 id입니다(아래 표 참조). |
| `payload` | `dict` | 이벤트 타입별 필드입니다. 아래에서 이벤트 타입별로 설명합니다. |

모든 페이로드는 추가적이고 이벤트별로 다르며, 하나의 통합 게이트웨이 페이로드 버전은 없습니다. 모든 id는 문자열이고, 누락되었거나 사용할 수 없는 필드는 추측하지 않고 항상 `None`으로 표시합니다. 형식이 잘못된 이벤트와 출처를 인증할 수 없는 이벤트는 삭제됩니다(페일 클로즈). 일시적인 Telegram Application 재구축이 발생하면 코어 핸들러와 함께 옵저버를 다시 등록합니다.

**이벤트별 페이로드 계약(v1, 추가적):**

| `event_type` | 플랫폼 | 페이로드 필드 |
|--------------|-----------|----------------|
| `reaction` | telegram | `emojis: list[str]`, `custom_emoji_ids: list[str]`, `chat_id: str`, `message_id: str`, `thread_id: str \| None` (Telegram 리액션 업데이트에는 토픽 id가 없으므로 현재는 항상 `None`입니다). |
| `message_edited` | telegram, discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `text: str \| None` (수정된 텍스트 또는 캡션이며 길이가 제한됩니다. 미디어만 수정되었거나 캐시되지 않은 경우 `None`), `edited_at: str \| None` (ISO 8601). |
| `message_deleted` | discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `author_id: str \| None`. Discord의 삭제 이벤트는 삭제한 사람을 식별하지 않습니다. 인증된 출처는 삭제된 메시지의 작성자이며, 캐시되지 않은 삭제에는 이벤트가 발생하지 않습니다. |
| `thread_created` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `name: str \| None`, `owner_id: str \| None`. |
| `thread_renamed` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `old_name: str \| None`, `new_name: str`. 이름이 실제로 변경된 경우에만 발생하며, 다른 스레드 업데이트(보관, slowmode, 태그)는 삭제됩니다. Discord의 스레드 업데이트 이벤트에는 행위자가 없으므로 스레드 소유자가 인증된 출처입니다. |

봇 자체의 점진적 메시지 수정(스트리밍)은 Discord에서 `message_edited`를 발생시키지 않습니다. 봇이 작성한 이벤트는 발생 지점에서 삭제됩니다.

이 훅은 옵저버 전용입니다. **원시 이벤트 접근이나 어댑터 접근을 추가하지 않습니다.** **원시 SDK 페이로드 접근은 의도적으로 제공하지 않습니다.** 어댑터 SDK 객체는 예고 없이 형태가 바뀌며, 이를 노출하면 발전시킬 수 없는 API 표면이 되기 때문입니다. 정말 필요한 경우에는 자체적인 명시적 기능(`gateway.raw_events`)과 "안정성 보장 없음" 라벨 및 별도의 설계가 필요합니다(#64228에서 추적 중). 플랫폼에서 동작을 수행하려면(리액션 추가, 스레드 이름 변경 등) [플러그인 가이드](plugins.md#platform-actions)에 설명된 기능 게이트 방식의 `ctx.platform_actions` 파사드를 사용합니다. 기본적으로 `gateway.platform_actions` 기능 뒤에서 꺼져 있습니다. `PluginContext.dispatch_tool()`은 도구 레지스트리에 등록된 도구만 호출할 수 있으며, `send_message`는 의도적으로 해당 레지스트리에 등록하지 않습니다(전송은 명시적인 CLI, cron, kanban, MCP 전달 경로에만 예약되어 있습니다). 향후 아웃바운드 전달 계약을 만들려면 먼저 모든 어댑터에서 안정적으로 전달된 콘텐츠/핸들을 제공해야 합니다. 이 조각에서는 동작하지 않는 `gateway_message_delivered` 훅을 미리 등록하지 않습니다.

---

### `pre_approval_request`

승인 결정을 요청하기 전에 실행됩니다. 대화형 CLI, Ink TUI, 게이트웨이 플랫폼, ACP 클라이언트 등 프롬프트가 표시되는 표면과, 사람에게 프롬프트를 표시하지 않고 `approvals.mode=smart`로 결정하는 경우(`surface="smart"`)를 모두 다룹니다. 스마트 모드에서는 보조 LLM을 호출하기 전에 훅이 실행됩니다.

사용자 지정 알림기를 연결하기에 적합한 위치입니다. 예를 들어 허용/거부 알림을 띄우는 macOS 메뉴 막대 앱이나, 컨텍스트와 함께 모든 승인 요청을 기록하는 감사 로그를 연결할 수 있습니다.

**콜백 시그니처:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    **kwargs,
):
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `command` | `str` | 평가 중인 터미널 명령 또는 `execute_code` 스크립트입니다. 스마트 및 게이트웨이 페이로드는 옵저버 디스패치 전에 비식별화됩니다. `security.redact_secrets`가 비활성화되어 있어도 스마트 옵저버의 비식별화는 필수이며, 비식별화에 실패하면 스마트 훅을 건너뜁니다. |
| `description` | `str` | 명령이 표시된 사람이 읽을 수 있는 이유입니다. 여러 패턴이 일치하면 이유가 결합됩니다. |
| `pattern_key` | `str` | 승인을 트리거한 기본 패턴 키입니다(예: `"rm_rf"`, `"sudo"`). |
| `pattern_keys` | `list[str]` | 일치한 모든 패턴 키입니다. |
| `session_key` | `str` | 세션 식별자이며, 채팅별 알림 범위를 지정할 때 유용합니다. |
| `surface` | `str` | 대화형 CLI/TUI 프롬프트의 경우 `"cli"`, 비동기 플랫폼 승인 요청의 경우 `"gateway"`, 보조 LLM의 자동 승인/거부 결정의 경우 `"smart"`입니다. |

**반환값:** 무시됩니다. 이 훅은 옵저버 전용이므로 승인을 거부하거나 미리 응답할 수 없습니다. 승인 시스템에 도달하기 전에 도구를 차단하려면 [`pre_tool_call`](#pre_tool_call)을 사용합니다.

**사용 사례:** 데스크톱 알림, 푸시 알림, 감사 로깅, Slack 웹훅, 에스컬레이션 라우팅, 메트릭.

**예시 — macOS에서 데스크톱 알림 보내기:**

```python
import subprocess

def notify_approval(command, description, session_key, **kwargs):
    title = "Hermes needs approval"
    body = f"{description}: {command[:80]}"
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "{body}" with title "{title}"',
    ])

def register(ctx):
    ctx.register_hook("pre_approval_request", notify_approval)
```

---

### `post_approval_response`

프롬프트가 표시된 승인 또는 스마트 승인 결정 이후, 프롬프트 시간이 초과된 후, 또는 게이트웨이가 승인 알림을 전달하지 못했을 때 실행됩니다. 알림 전달 실패 시 승인 결정이 존재하기 전에 `choice="notify_failed"`가 발생합니다.

**콜백 시그니처:**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    choice: str,
    **kwargs,
):
```

`pre_approval_request`와 동일한 kwargs에 다음 항목이 추가됩니다.

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `choice` | `str` | 프롬프트 표면에서는 `"once"`, `"session"`, `"always"`, `"deny"`, `"timeout"`, `"notify_failed"` 중 하나이고, 스마트 결정에서는 `"smart_approve"` 또는 `"smart_deny"`입니다. |
| `decided_by` | `str` | 스마트 결정에서는 `"aux_llm"`이며, 프롬프트 표면에서는 존재하지 않습니다. |

**반환값:** 무시됩니다.

**사용 사례:** 연결된 데스크톱 알림 닫기, 감사 로그에 최종 결정 기록, 메트릭 업데이트, 속도 제한기 진행.

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `pre_transcription`

STT 디스패처(`tools.transcription_tools.transcribe_audio`) 내부에서, 프로바이더가 확인된 후 백엔드가 호출되기 전에 실행됩니다. 이 백엔드는 내장 백엔드, `type: command` 프로바이더, 플러그인으로 등록된 프로바이더 중 무엇이든 해당합니다. 플러그인이 나중에 트랜스크립트를 관찰하는 데 그치지 않고 트랜스크립션 요청 자체를 조정할 수 있습니다.

**콜백 시그니처:**

```python
def my_callback(
    file_path: str,
    provider: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    source: str | None,
    **kwargs,
) -> dict | None:
```

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `file_path` | `str` | 곧 트랜스크립션할 오디오 파일의 절대 경로입니다. 읽기 전용입니다. |
| `provider` | `str` | 확인된 STT 프로바이더입니다(`local`, `groq`, `openai`, `mistral`, `xai`, `elevenlabs`, `deepinfra`, `local_command`, 명령 프로바이더 이름 또는 플러그인 프로바이더 이름). |
| `model` | `str \| None` | 지금까지 확인된 모델이며, 백엔드 기본값이 적용되는 경우 `None`입니다. |
| `language` | `str \| None` | 프로바이더의 config 섹션에서 가져온 언어이며, 없으면 `None`입니다. |
| `prompt` | `str \| None` | 정적 [`stt.prompt`](/user-guide/configuration#transcription-prompt-vocabulary-hints) 값이며, 없으면 `None`입니다. |
| `source` | `str \| None` | 호출 표면 라벨(`gateway`, `voice_mode`, …)입니다. 관찰용일 뿐 디스패치에는 사용되지 않습니다. |

**반환값:** 문자열에 매핑된 `"prompt"`, `"language"`, `"model"` 중 어느 것이든 포함하는 딕셔너리이거나, 요청을 변경하지 않으려면 `None`입니다. 문자열이 아닌 값, 알 수 없는 키, `file_path`는 무시됩니다(`file_path`를 변경하려는 시도는 경고로 기록됩니다). 결과는 `stt.prompt` config 값 위에 등록 순서대로 적용되며, 필드별로 마지막 작성자가 승리합니다. `prompt`에 `""`을 반환하면 해당 요청에 설정된 프롬프트를 지웁니다.

**사용 사례:** 오디오를 업로드하기 전에 사용자별 또는 채팅별 어휘 목록 주입, 호출자 로케일에 따른 `language` 강제, 긴 녹음에 더 낮은 `model` 적용, 잡음이 많은 소스를 다른 모델로 라우팅.

```python
VOCAB = "Hermes, Teknium, Nous Research, kanban"

def add_vocab(provider, prompt, source, **kwargs):
    if source != "gateway":
        return None
    return {"prompt": f"{prompt}. {VOCAB}" if prompt else VOCAB}

def register(ctx):
    ctx.register_hook("pre_transcription", add_vocab)
```

모든 백엔드가 프롬프트를 지원하는 것은 아닙니다. `local`은 이를 faster-whisper의 `initial_prompt`로 매핑하고, `openai`, `groq`, `mistral`, `deepinfra`는 `prompt`로 전송합니다. `xai`, `elevenlabs`, `local_command`, `type: command` 프로바이더는 DEBUG 수준으로 기록하고 프롬프트 없이 트랜스크립션합니다. 전체 매트릭스와 개인정보 경계는 [프로바이더 지원 표](/user-guide/configuration#transcription-prompt-vocabulary-hints)를 참조하세요. 훅 연결 오류는 페일 오픈으로 처리되므로 디스패치는 수정되지 않은 요청으로 계속 진행합니다.
### `transform_tool_result`

도구가 결과를 반환한 **후**, 그 결과가 대화에 추가되기 **전에** 실행됩니다. 터미널 출력뿐 아니라 모든 도구의 결과 문자열을 모델이 보기 전에 플러그인이 다시 작성할 수 있습니다.

**콜백 시그니처:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs) -> str | None:
```

전체 페이로드에는 `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message`도 포함됩니다. `result`는 도구 디스패치가 반환한 최종 결과이며, `result`와 `args`에는 임의의 사용자/도구 콘텐츠와 비밀 정보가 포함될 수 있습니다.

**반환값:** 첫 번째 `str`이 결과를 대체합니다(빈 문자열 포함). `None`이면 변경되지 않습니다.

**사용 사례:** `web_extract` 출력에서 조직별 PII를 삭제하거나, 긴 JSON 도구 응답을 요약 헤더로 감싸거나, `read_file` 결과에 검색 증강 컨텍스트를 주입하거나, `delegate_task` 서브에이전트 보고서를 프로젝트별 스키마로 다시 작성할 수 있습니다.

```python
import re
SECRET = re.compile(r"sk-[A-Za-z0-9]{32,}")

def redact_secrets(tool_name, result, **kwargs):
    if SECRET.search(result):
        return SECRET.sub("[REDACTED]", result)
    return None

def register(ctx):
    ctx.register_hook("transform_tool_result", redact_secrets)
```

모든 도구에 적용됩니다. 터미널에만 적용되는 재작성은 아래의 `transform_terminal_output`을 참조하세요. 이 훅은 더 좁은 범위에서 동작하며 `transform_tool_result`보다 먼저 실행되고, 대체된 결과에도 터미널 도구의 최종 출력 제한이 적용됩니다.

---

### `transform_terminal_output`

포그라운드 프로세스 캡처가 환경에 의해 이미 제한된 후, 최종 출력 제한이 적용되기 전에 `terminal` 도구 내부에서 실행됩니다. 플러그인이 캡처된 stdout/stderr를 대체할 수 있으며, 대체된 결과에도 최종 출력 제한이 적용됩니다.

**콜백 시그니처:**

```python
def my_callback(
    command: str,
    output: str,
    returncode: int,
    task_id: str,
    env_type: str,
    **kwargs,
) -> str | None:
```

| 매개변수 | 유형 | 설명 |
|-----------|------|-------------|
| `command` | `str` | 출력을 생성한 셸 명령입니다. |
| `output` | `str` | 제한된 프로세스 캡처 후의 stdout/stderr 결합 결과입니다. |
| `returncode` | `int` | 프로세스 반환 코드입니다. |
| `task_id` | `str` | 유효한 작업 식별자이거나 빈 문자열입니다. |
| `env_type` | `str` | 실행 환경 유형입니다. |

**반환값:** 첫 번째 `str`이 출력을 대체합니다. `None`이면 변경되지 않습니다. 명령과 출력에는 자격 증명이나 기타 민감한 데이터가 포함될 수 있습니다.

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[summary: {lines} paths total, showing first 40]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

모든 도구에서 이후에 실행되는 `transform_tool_result`와 함께 동작하며, `terminal`도 포함됩니다.

---

### `transform_llm_output`

도구 호출 루프가 완료되고 모델이 최종 응답을 생성한 **후**, 해당 응답이 사용자에게 전달되기 **전에** 턴마다 **한 번** 실행됩니다(CLI, 게이트웨이 또는 프로그래밍 방식의 호출자). 플러그인이 고전적인 프로그래밍 방식으로 어시스턴트의 최종 텍스트를 다시 작성할 수 있습니다. SOUL의 말투 텍스트나 스킬 기반 변환에 추가 추론 토큰을 소모하지 않습니다.

**콜백 시그니처:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| 매개변수 | 유형 | 설명 |
|-----------|------|-------------|
| `response_text` | `str` | 이번 턴에 대한 어시스턴트의 최종 응답 텍스트입니다. |
| `session_id` | `str` | 이 대화의 세션 ID입니다(일회성 실행에서는 비어 있을 수 있음). |
| `model` | `str` | 응답을 생성한 모델 이름입니다(예: `anthropic/claude-sonnet-4.6`). |
| `platform` | `str` | 전달 플랫폼입니다(`cli`, `telegram`, `discord`, …; 설정되지 않으면 비어 있음). |

**반환값:** 비어 있지 않은 `str`은 응답 텍스트를 대체하고, `None` 또는 빈 문자열이면 변경되지 않습니다. 여러 플러그인이 등록된 경우 **비어 있지 않은 첫 번째 문자열이 적용됩니다**. 도구 및 터미널 변환과 달리 빈 문자열은 대체값으로 허용되지 않습니다.

**사용 사례:** 성격/어휘 변환(해적 말투, 스폰지밥 말투)을 적용하거나, 사용자별 식별자를 최종 텍스트에서 삭제하거나, 프로젝트별 서명 푸터를 추가하거나, SOUL 지침에 토큰을 소모하지 않고 사내 스타일 가이드를 적용할 수 있습니다.

CLI 스트리밍이 활성화된 경우 추가 전용 변환은 스트리밍된 본문 뒤에 출력됩니다. 응답을 대체하는 변환은 스트리밍된 본문 뒤에 전체가 출력되고 스트림 후 변환으로 표시되므로, 대체 콘텐츠가 조용히 사라지지 않습니다.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

이 훅은 비어 있지 않고 중단되지 않은 응답에 대해서만 활성화되므로, 중지 버튼으로 중단된 턴이나 빈 턴에서는 실행되지 않습니다. 예외는 경고로 기록되며 에이전트 실행을 중단하지 않습니다.

### API 요청 옵저버 훅

#### `pre_api_request`

각 프로바이더 시도에서 전송 직전에 실행됩니다. 옵저버 전용입니다. 레거시 `user_message`, `conversation_history`, `request_messages` 필드는 호환성을 위해 원본 그대로이며 의도적으로 정제되지 않습니다. 새 소비자는 정제된 `request` 엔벌로프를 우선 사용해야 합니다.

#### `post_api_request`

프로바이더 응답이 성공적으로 정규화된 후 실행됩니다. 옵저버 전용입니다. 정제된 `response`를 우선 사용하세요. `assistant_message`는 정규화된 원본 메시지이고 `usage`에는 사용량 집계 데이터가 포함됩니다.

#### `api_request_error`

상태/재시도 타이밍, `error` 객체, 정제된 `request`와 함께 프로바이더 시도가 실패했을 때 실행됩니다. 옵저버 전용입니다. 오류 메시지에는 여전히 프로바이더 또는 사용자 데이터가 포함될 수 있습니다.

### `on_skill_lifecycle`

권위 있는 스킬 사용 상태가 변경된 후 실행됩니다. 옵저버 전용이며 로컬 `skill_name`, 출처, 상관관계 ID, 사용 횟수, 재사용 플래그를 노출합니다.

### Kanban 수명 주기 옵저버

#### `kanban_task_claimed`

디스패처 프로세스에서 클레임 커밋이 완료된 후, 워커가 생성되기 직전에 실행됩니다.

#### `kanban_task_completed`

일반적으로 워커 프로세스에서 완료 및 정리가 끝난 후 실행됩니다. `summary`에는 프로젝트 또는 사용자 콘텐츠가 포함될 수 있습니다.

#### `kanban_task_blocked`

정상적인 차단 전환 후 실행됩니다. 의존성 대기 경로에서는 쓰기 트랜잭션이 종료되기 전에 호출됩니다. `reason`에는 프로젝트 또는 사용자 콘텐츠가 포함될 수 있습니다.

세 Kanban 훅은 모두 옵저버 전용이며 `task_id`, `profile_name`, `board`, `assignee`, `run_id`를 전달합니다. completed에는 `summary`가 추가되고 blocked에는 `reason`이 추가됩니다.

### Kanban 워커 수명 주기, 작업 변경 및 디스패치 옵저버

다섯 가지 추가 옵저버(RFC #58548)가 Kanban 계열을 확장합니다. 모두 옵저버 전용이며 관련 트랜잭션이 커밋된 후 실행되고 `has_hook`에서 단락 평가됩니다. 구독자가 없으면 디스패치 동작은 변경되지 않습니다. 작업 범위 훅은 위 훅과 동일한 공통 필드를 전달합니다.

- **`on_kanban_worker_spawned`** — `spawn_fn`이 반환되고 워커 PID가 저장된 후 실행됩니다. `worker_pid`(없을 수 있음)와 `workspace_path`를 추가합니다. 디스패치 잠금 안에서 실행되므로 콜백은 빠르게 처리하세요.
- **`on_kanban_worker_exited`** — `detect_crashed_workers`가 죽은 PID의 작업을 회수할 때 틱에서 파생되어 실행됩니다. `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status`를 추가합니다.
- **`on_kanban_worker_stale_claim`** — TTL이 만료된 클레임을 회수할 때 실행됩니다. 살아 있는 PID에 대한 확장에서는 실행되지 않습니다. `worker_pid`, `heartbeat_stale`, `retry_status`를 추가합니다.
- **`on_kanban_task_updated`** — 클레임/완료/차단 수명 주기 외부에서 커밋된 작업 필드 쓰기(`assign_task`, 모델/추론 재정의, 대시보드 편집기)가 끝난 후 실행됩니다. `changed_fields`를 추가하며, 값은 절대 포함하지 않고 필드 이름만 전달합니다.
- **`on_kanban_dispatch_tick`** — 디스패처 틱마다 한 번, 디스패치 잠금이 해제된 직후에 실행됩니다. 유휴 틱과 잠금 경합 틱도 포함됩니다. 페이로드: `board`, `profile_name`, `dry_run`, `outcome`, `result`.

---

## 셸 훅

`~/.hermes/config.yaml`에서 셸 스크립트 훅을 선언하면 Hermes가 CLI와 게이트웨이 세션 모두에서 해당 플러그인 훅 이벤트가 발생할 때마다 서브프로세스로 실행합니다. Python 플러그인을 작성할 필요가 없습니다.

다음과 같은 용도로 바로 사용할 수 있는 단일 파일 스크립트(Bash, Python, shebang이 있는 모든 프로그램)를 원한다면 셸 훅을 사용하세요.

- **도구 호출 차단** — 위험한 `terminal` 명령을 거부하고, 디렉터리별 정책을 적용하며, 파괴적인 `write_file`/`patch` 작업에 승인을 요구합니다.
- **도구 호출 후 실행** — 에이전트가 방금 작성한 Python 또는 TypeScript 파일을 자동 포맷하거나, API 호출을 기록하거나, CI 워크플로를 트리거합니다.
- **다음 LLM 턴에 컨텍스트 주입** — `git status` 출력, 현재 요일 또는 검색된 문서를 사용자 메시지 앞에 추가합니다([`pre_llm_call`](#pre_llm_call) 참조).
- **수명 주기 이벤트 관찰** — 서브에이전트가 완료되거나(`subagent_stop`) 세션이 시작될 때(`on_session_start`) 로그 한 줄을 기록합니다.

셸 훅은 CLI 시작 시(`hermes_cli/main.py`)와 게이트웨이 시작 시(`gateway/run.py`) 모두 `agent.shell_hooks.register_from_config(cfg)`를 호출하여 등록합니다. Python 플러그인 훅과 자연스럽게 함께 구성되며, 둘 다 동일한 디스패처를 통해 흐릅니다.

### 한눈에 보는 비교

| 차원 | 셸 훅 | [플러그인 훅](#plugin-hooks) | [게이트웨이 훅](#gateway-event-hooks) |
|-----------|------|-------------|-------------|
| 선언 위치 | `~/.hermes/config.yaml`의 `hooks:` 블록 | `plugin.yaml` 플러그인의 `register()` | `HOOK.yaml` + `handler.py` 디렉터리 |
| 위치 | `~/.hermes/agent-hooks/` (관례상) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 언어 | 모든 언어(Bash, Python, Go 바이너리 등) | Python만 | Python만 |
| 실행 환경 | CLI + 게이트웨이 | CLI + 게이트웨이 | 게이트웨이만 |
| 이벤트 | `VALID_HOOKS` (`subagent_stop` 포함) | `VALID_HOOKS` | 게이트웨이 수명 주기(`gateway:startup`, `agent:*`, `command:*`) |
| 도구 호출 차단 가능 | 예(`pre_tool_call`) | 예(`pre_tool_call`) | 아니요 |
| LLM 컨텍스트 주입 가능 | 예(`pre_llm_call`) | 예(`pre_llm_call`) | 아니요 |
| 동의 | `(event, command)` 쌍별 최초 사용 프롬프트 | 암묵적(Python 플러그인 신뢰) | 암묵적(디렉터리 신뢰) |
| 프로세스 간 격리 | 예(서브프로세스) | 아니요(인프로세스) | 아니요(인프로세스) |

### 구성 스키마

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300
      fail_closed: <bool>        # Optional; default false. pre_tool_call only.
                                 # `failClosed` also accepted (Cursor/Claude Code compat)

hooks_auto_accept: false         # See "Consent model" below
```

이벤트 이름은 [플러그인 훅 이벤트](#plugin-hooks) 중 하나여야 합니다. 오타가 있으면 "Did you mean X?" 경고가 표시되고 건너뜁니다. 단일 항목 안의 알 수 없는 키는 무시되고, `command`가 없으면 건너뛴다는 경고가 표시됩니다. `timeout > 300`은 경고와 함께 300으로 제한됩니다. `pre_tool_call` 이외의 이벤트에서 `fail_closed: true`를 사용하면 경고 후 무시됩니다(차단할 수 있는 이벤트만 fail closed가 될 수 있음).

### JSON 와이어 프로토콜

이벤트가 발생할 때마다 Hermes는 일치하는 각 훅에 대해 서브프로세스를 생성하고(매처가 허용하는 경우), JSON 페이로드를 **stdin**으로 전달한 뒤 **stdout**에서 JSON을 읽습니다.

**stdin — 스크립트가 받는 페이로드:**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

도구가 아닌 이벤트(`pre_llm_call`, `subagent_stop`, 세션 수명 주기)에서는 `tool_name`과 `tool_input`이 `null`입니다. `extra` 딕셔너리에는 이벤트별 모든 kwargs(`user_message`, `conversation_history`, `child_role`, `duration_ms`, …)가 포함됩니다. 직렬화할 수 없는 값은 생략되지 않고 문자열로 변환됩니다.

**stdout — 선택적 응답:**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Keep the agent going at the verify gate (pre_verify); both shapes accepted:
{"action": "continue", "message": "Run the formatter, then finish."}
{"decision": "block",  "reason":  "Run the formatter, then finish."}

// Silent no-op — any empty / non-matching output is fine:
```

잘못된 JSON, 0이 아닌 종료 코드, 타임아웃은 경고로 기록되지만 에이전트 루프를 중단하지 않습니다.
### 종료 코드 2 = 차단(Claude Code / Cursor 호환)

`pre_tool_call` 훅이 종료 코드 **2**를 반환하면 stdout에 차단 JSON이 없어도 도구 호출이 차단됩니다. 차단 메시지는 다음 우선순위에 따라 결정됩니다.

1. stdout의 차단 JSON(`reason` / `message`, 해당하는 경우)
2. stderr의 첫 400자
3. 일반적인 `"Blocked by shell hook."` 기본값

가장 간단한 차단 훅은 다음과 같습니다.

```bash
#!/usr/bin/env bash
echo "policy violation: rm -rf is not permitted" >&2
exit 2
```

차단 지시가 적용되지 않는 이벤트(현재는 `pre_tool_call` 이외의 모든 이벤트)에서는 종료 코드 2가 다른 비제로 종료 코드와 동일하게 처리됩니다. stdout이 계속 출력되고 파싱됩니다.

### 실패 허용과 실패 차단

기본적으로 셸 훅은 **실패를 허용**합니다. 즉, 생성 오류, 시간 초과 또는 파싱할 수 없는 stdout이 기록되고 작업은 계속됩니다. 관찰용 훅에는 적절한 동작이지만 보안 게이트에는 적절하지 않습니다. 중단된 비밀 스캐너가 검사하려던 도구 호출을 조용히 허용해서는 안 됩니다.

이를 반대로 바꾸려면 `pre_tool_call` 항목에 `fail_closed: true`(또는 Cursor/Claude Code 표기인 `failClosed: true`)를 설정합니다.

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal|write_file|patch"
      command: "~/.hermes/agent-hooks/secret-scan.sh"
      timeout: 10
      fail_closed: true
```

이제 `fail_closed: true`에서는 다음 각각이 `hook <command> failed closed: <reason>`과 함께 도구 호출을 **차단**합니다.

| 실패 | 실패 허용(기본값) | `fail_closed: true` |
|---------|--------------------|--------------------|
| 명령을 찾을 수 없음 / 실행할 수 없음 | 경고, 계속 진행 | **차단** |
| 시간 초과 | 경고, 계속 진행 | **차단** |
| JSON이 아닌 stdout(예: 스택 트레이스) | 경고, 계속 진행 | **차단** |
| 정상 종료, 유효한 무동작 JSON(`{}`) | 계속 진행 | 계속 진행 |

`fail_closed`는 차단 가능한 이벤트(현재는 `pre_tool_call`)에만 적용됩니다. 다른 이벤트에 설정하면 구성 파싱 시 경고가 기록되고 무시됩니다. `hermes hooks test`는 이러한 의미를 반영합니다. `parsed` 줄에는 디스패처가 받게 될 차단 형식이 그대로 표시됩니다.

### 예제

#### 1. 모든 쓰기 작업 후 Python 파일 자동 포맷

```yaml
# ~/.hermes/config.yaml
hooks:
  post_tool_call:
    - matcher: "write_file|patch"
      command: "~/.hermes/agent-hooks/auto-format.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/auto-format.sh
payload="$(cat -)"
path=$(echo "$payload" | jq -r '.tool_input.path // empty')
[[ "$path" == *.py ]] && command -v black >/dev/null && black "$path" 2>/dev/null
printf '{}\n'
```

에이전트의 컨텍스트 내 파일 내용은 **자동으로 다시 읽히지 않습니다**. 포맷 변경은 디스크의 파일에만 적용됩니다. 이후의 `read_file` 호출에서는 포맷이 적용된 버전을 가져옵니다.

#### 2. 파괴적인 `terminal` 명령 차단

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal"
      command: "~/.hermes/agent-hooks/block-rm-rf.sh"
      timeout: 5
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/block-rm-rf.sh
payload="$(cat -)"
cmd=$(echo "$payload" | jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf?[[:space:]]+/'; then
  printf '{"decision": "block", "reason": "blocked: rm -rf / is not permitted"}\n'
else
  printf '{}\n'
fi
```

#### 3. 매 턴에 `git status` 주입(Claude-Code `UserPromptSubmit`와 동일한 기능)

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # discard stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Code의 `UserPromptSubmit` 이벤트는 의도적으로 별도의 Hermes 이벤트가 아닙니다. `pre_llm_call`이 같은 지점에서 실행되며 이미 컨텍스트 주입을 지원하므로 여기서 이를 사용합니다.

#### 4. 모든 서브에이전트 완료 기록

```yaml
hooks:
  subagent_stop:
    - command: "~/.hermes/agent-hooks/log-orchestration.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/log-orchestration.sh
log=~/.hermes/logs/orchestration.log
jq -c '{ts: now, parent: .session_id, extra: .extra}' < /dev/stdin >> "$log"
printf '{}\n'
```

### 동의 모델

각 고유한 `(event, command)` 쌍에 대해 Hermes가 처음 이를 확인할 때 사용자에게 승인을 요청한 다음, 결정을 `~/.hermes/shell-hooks-allowlist.json`에 저장합니다. 이후 CLI 또는 게이트웨이 실행에서는 프롬프트를 건너뜁니다.

다음 세 가지 우회 방법 중 하나만 사용해도 대화형 프롬프트를 건너뛸 수 있습니다.

1. CLI의 `--accept-hooks` 플래그(예: `hermes --accept-hooks chat`)
2. `HERMES_ACCEPT_HOOKS=1` 환경 변수
3. `~/.hermes/config.yaml`의 `hooks_auto_accept: true`

비 TTY 실행(게이트웨이, cron, CI)에서는 이 세 가지 중 하나가 필요합니다. 그렇지 않으면 새로 추가된 훅이 조용히 등록되지 않고 경고를 기록합니다.

**스크립트 수정은 암묵적으로 신뢰됩니다.** 허용 목록은 스크립트의 해시가 아니라 정확한 명령 문자열을 기준으로 하므로, 디스크에서 스크립트를 수정해도 동의가 무효화되지 않습니다. `hermes hooks doctor`는 수정 사항을 확인할 수 있도록 mtime 변경을 표시하므로 재승인 여부를 결정할 수 있습니다.

#### 수동 허용 목록 등록

수동 허용 목록 등록은 운영자가 최초 사용 승인 프롬프트에 대화형으로 응답할 수 없는 비 TTY 또는 서비스 계정 배포에 유용합니다. 허용 목록 파일은 `~/.hermes/shell-hooks-allowlist.json`이며, 예상 형식은 `approvals` 배열입니다. 각 승인 항목에는 훅의 `event`와 정확한 `command` 문자열이 기록됩니다.

```json
{
  "approvals": [
    {
      "event": "post_llm_call",
      "command": "/home/hermes/.hermes/hooks/my-hook.py"
    }
  ]
}
```

명령 문자열은 구성된 훅 명령과 정확히 일치해야 합니다. `sha256` 필드가 있는 경로 키 객체는 예상 형식이 아니므로 훅을 승인하지 않습니다. `hermes hooks list`로 수동 항목을 확인하세요.

### `hermes hooks` CLI

| 명령 | 기능 |
|---------|--------------|
| `hermes hooks list` | matcher, timeout, 동의 상태와 함께 구성된 훅을 출력합니다 |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 합성 페이로드에 일치하는 모든 훅을 실행하고 파싱된 응답을 출력합니다 |
| `hermes hooks revoke <command>` | `<command>`와 일치하는 모든 허용 목록 항목을 제거합니다(다음 재시작부터 적용) |
| `hermes hooks doctor` | 구성된 각 훅에 대해 실행 비트, 허용 목록 상태, mtime 변경, JSON 출력 유효성, 대략적인 실행 시간을 확인합니다 |

### 보안

셸 훅은 **사용자의 전체 자격 증명**으로 실행됩니다. 즉 cron 항목이나 셸 별칭과 동일한 신뢰 경계를 가집니다. `config.yaml`의 `hooks:` 블록은 권한 있는 구성으로 취급하세요.

- 직접 작성했거나 완전히 검토한 스크립트만 참조하세요.
- 감사하기 쉽도록 스크립트를 `~/.hermes/agent-hooks/` 안에 보관하세요.
- 공유 구성을 가져온 후 `hermes hooks doctor`를 다시 실행하여 등록되기 전에 새로 추가된 훅을 확인하세요.
- `config.yaml`을 팀에서 버전 관리한다면, `hooks:` 섹션을 변경하는 PR을 CI 구성 변경을 검토하는 것과 같은 방식으로 검토하세요.

### 순서와 우선순위

Python 플러그인 훅과 셸 훅은 모두 동일한 `invoke_hook()` 디스패처를 거칩니다. Python 플러그인이 먼저 등록되고(`discover_and_load()`), 셸 훅이 그 다음에 등록되므로(`register_from_config()`), 동률인 경우 Python `pre_tool_call` 차단 결정이 우선합니다. 첫 번째 유효한 차단이 승리합니다. 즉 콜백 중 하나가 `{"action": "block", "message": str}`을 생성하고 메시지가 비어 있지 않으면 집계기가 즉시 반환합니다.

## 아웃바운드 웹훅

아웃바운드 웹훅은 [인바운드 웹훅 플랫폼](/user-guide/messaging/webhooks)의 푸시 측 반대편입니다. 인바운드 웹훅은 세상이 변할 때 Hermes를 깨우고, 아웃바운드 웹훅은 Hermes가 무언가를 수행할 때 세상에 알립니다. HTTP 엔드포인트 목록과 각 엔드포인트가 관심을 가질 수명 주기 이벤트를 구성하면, 일치하는 이벤트가 발생할 때마다 Hermes가 서명된 JSON 페이로드를 각 엔드포인트로 POST합니다. 수신 측에서 폴링할 필요가 없습니다.

일반적인 용도:

- 에이전트 턴이 끝날 때(`on_session_end`) CI 시스템이나 대시보드에 알림
- 여러 인스턴스에 걸친 서브에이전트 완료 추적(`subagent_stop`)
- 외부 모니터링으로 도구 활동 전달(`matcher`가 있는 `post_tool_call`)
- *다른* Hermes 인스턴스 깨우기: URL을 해당 인스턴스의 인바운드 웹훅으로 지정

### 구성

`~/.hermes/config.yaml`에 `hooks.outbound:` 목록을 추가합니다.

```yaml
hooks:
  outbound:
    - name: ci-notify                       # optional label for logs
      url: https://ci.example.com/hermes-events
      events: [on_session_end, subagent_stop]
      secret_env: HERMES_OUTBOUND_WEBHOOK_SECRET   # env var holding the HMAC secret
      timeout: 10                           # per-attempt seconds (1–60)

    - name: tool-monitor
      url: https://metrics.example.com/hooks/hermes
      events: [post_tool_call]
      matcher: "terminal|delegate_task"     # regex, tool-scoped events only
```

플러그인 훅 집합의 모든 이벤트가 유효합니다(`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `subagent_start`, `subagent_stop`, ...). 잘못된 항목은 경고 후 건너뜁니다. 웹훅이 잘못되어도 에이전트가 중단되지는 않습니다. 변경 사항은 다음 CLI 세션 또는 게이트웨이 재시작부터 적용됩니다.

비밀 정보는 인라인 `secret:` 리터럴보다 `secret_env`(환경 변수의 이름이며, 일반적으로 `~/.hermes/.env`에 설정)를 사용하는 것이 좋습니다. 이렇게 하면 구성 파일에 자격 증명이 남지 않습니다. 비밀 정보가 없는 항목은 서명 없이 전송되며 `hermes hooks list`에서 `UNSIGNED`로 표시됩니다.

### 와이어 형식

각 이벤트 발생 시 셸 훅의 표준 입력과 동일한 최상위 구조에 전송 메타데이터를 더한 JSON 본문을 POST합니다.

```json
{
  "hook_event_name": "on_session_end",
  "tool_name": null,
  "tool_input": null,
  "session_id": "sess_abc123",
  "cwd": "/home/user/project",
  "extra": {"completed": true, "interrupted": false, "model": "...", "platform": "cli"},
  "delivery_id": "3f2c9a...",
  "timestamp": "2026-07-22T14:00:00Z"
}
```

헤더:

| 헤더 | 값 |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Hermes-Event` | 훅 이벤트 이름 |
| `X-Hermes-Delivery` | 전달마다 고유한 ID — 본문의 `delivery_id`와 동일한 값 |
| `X-Hermes-Signature-256` | `sha256=<hex>` — 원시 본문의 HMAC-SHA256(GitHub 방식), 비밀 정보가 구성된 경우에만 존재 |

GitHub 웹훅을 검증하는 것과 정확히 같은 방식으로 서명을 검증합니다.

```python
import hashlib, hmac

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

`delivery_id`와 `timestamp`가 **서명된 본문 안에** 있으므로, 검증된 수신자는 재생 공격 방지도 별도의 비용 없이 얻습니다.

- `delivery_id`(또는 이에 대응하는 `X-Hermes-Delivery` 헤더)를 기준으로 **중복 제거** — 최근 확인한 ID를 기억하고 중복을 건너뜁니다. Hermes는 실패한 전달을 한 번 재시도하므로 동일한 ID가 정상적으로 두 번 도착할 수 있습니다.
- 시계를 기준으로 `timestamp`를 허용 시간 범위(일반적인 기본값은 5분)와 비교하여 **오래된 이벤트 거부** — 캡처된 요청을 재생하는 공격자는 비밀 정보 없이 새로운 타임스탬프를 위조할 수 없습니다.

### 전달 의미

- **실행 후 잊기 방식이며, 주요 경로에서 분리됩니다.** 이벤트는 즉시 직렬화되어 큐에 들어가고, 하나의 백그라운드 스레드가 HTTP POST를 수행합니다. 느리거나 응답하지 않는 엔드포인트도 도구 호출이나 에이전트 턴을 멈출 수 없습니다.
- **알림 전용입니다.** 셸 훅과 달리 아웃바운드 웹훅은 도구 호출을 차단하거나 컨텍스트를 주입할 수 없으며 응답 본문은 무시됩니다. 관찰만 하며 흐름을 제어하지 않습니다.
- **재시도 횟수는 제한됩니다.** 연결 오류와 5xx 응답은 백오프를 적용해 한 번 재시도합니다. 4xx 응답은 수신자가 요청 자체가 잘못되었다고 알린 것이므로 재시도하지 않습니다. 실패는 기록 후 폐기되며 전달은 최선 노력 방식으로 처리되고 보장되지 않습니다.
- **리디렉션은 절대 따라가지 않습니다.** 3xx 응답은 잘못된 구성으로 간주되어 기록됩니다. 리디렉션된 POST를 따라가면 서명된 페이로드가 조용히 유실될 수 있기 때문입니다. `url`은 최종 엔드포인트를 가리키도록 지정하세요.
- **큐 크기는 제한됩니다.** 큐가 밀리면(응답하지 않는 엔드포인트, 이벤트 폭주) 무제한 메모리 사용을 막기 위해 새 이벤트를 경고와 함께 버립니다.
- **동의 프롬프트가 없습니다.** 아웃바운드 대상은 사용자의 컴퓨터에서 코드를 실행하지 않고, 사용자가 구성한 URL에서 데이터를 받기만 합니다. 플러그인 및 셸 훅과 마찬가지로 `HERMES_SAFE_MODE=1`에서는 등록도 건너뜁니다. 페이로드에는 도구 입력과 이벤트 메타데이터가 포함되므로 신뢰하는 엔드포인트만 대상으로 지정하고 `https://`를 우선 사용하세요.

`hermes hooks list`는 각 대상의 서명 여부를 포함하여 구성된 아웃바운드 대상과 셸 훅을 함께 표시합니다.
