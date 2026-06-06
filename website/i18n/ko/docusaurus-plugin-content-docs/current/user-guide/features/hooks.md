---
sidebar_position: 6
title: "이벤트 훅 (Event Hooks)"
description: "핵심 수명 주기 지점에서 사용자 지정 코드 실행 — 활동 로깅, 경고 전송, 웹훅 게시"
---

# 이벤트 훅 (Event Hooks)

Hermes에는 주요 수명 주기 지점에서 사용자 지정 코드를 실행하는 세 가지 훅 시스템이 있습니다.

| 시스템 | 등록 방법 | 실행 환경 | 사용 사례 |
|--------|---------------|---------|----------|
| **[게이트웨이 훅 (Gateway hooks)](#gateway-event-hooks)** | `~/.hermes/hooks/` 디렉토리의 `HOOK.yaml` + `handler.py` | 게이트웨이 전용 | 로깅, 경고, 웹훅 |
| **[플러그인 훅 (Plugin hooks)](#plugin-hooks)** | [플러그인](/user-guide/features/plugins)에서 `ctx.register_hook()` 사용 | CLI + 게이트웨이 | 도구 가로채기, 메트릭, 가드레일 |
| **[셸 훅 (Shell hooks)](#shell-hooks)** | `~/.hermes/config.yaml`의 `hooks:` 블록에서 셸 스크립트 지정 | CLI + 게이트웨이 | 차단, 자동 포맷팅, 컨텍스트 주입을 위한 드롭인 스크립트 |

세 시스템 모두 비차단(non-blocking) 방식입니다. 어떤 훅에서든 발생하는 오류는 포착되어 기록되며, 에이전트의 작동을 멈추게 하지 않습니다.

## 게이트웨이 이벤트 훅 (Gateway Event Hooks)

게이트웨이 훅은 게이트웨이(Telegram, Discord, Slack, WhatsApp, Teams)가 작동하는 동안 메인 에이전트 파이프라인을 차단하지 않고 자동으로 실행됩니다.

### 훅 생성하기

각 훅은 두 개의 파일을 포함하는 `~/.hermes/hooks/` 아래의 디렉토리입니다.

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # 수신 대기할 이벤트 선언
    └── handler.py     # Python 핸들러 함수
```

#### HOOK.yaml

```yaml
name: my-hook
description: 모든 에이전트 활동을 파일에 기록
events:
  - agent:start
  - agent:end
  - agent:step
```

`events` 목록은 핸들러를 트리거할 이벤트를 결정합니다. `command:*`와 같은 와일드카드를 포함하여 모든 이벤트 조합을 구독할 수 있습니다.

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """구독된 각 이벤트에 대해 호출됩니다. 이름은 반드시 'handle'이어야 합니다."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**핸들러 규칙:**
- 이름은 반드시 `handle`이어야 합니다.
- `event_type` (문자열)과 `context` (사전)를 받습니다.
- `async def` 또는 일반 `def` 모두 작동합니다.
- 오류는 포착되어 기록되며, 절대로 에이전트를 중단시키지 않습니다.

### 사용 가능한 이벤트

| 이벤트 | 실행 시기 | 컨텍스트 키 |
|-------|---------------|--------------|
| `gateway:startup` | 게이트웨이 프로세스 시작 시 | `platforms` (활성 플랫폼 이름 목록) |
| `session:start` | 새 메시징 세션이 생성됨 | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | 세션이 종료됨 (재설정 전) | `platform`, `user_id`, `session_key` |
| `session:reset` | 사용자가 `/new` 또는 `/reset`을 실행함 | `platform`, `user_id`, `session_key` |
| `agent:start` | 에이전트가 메시지 처리를 시작함 | `platform`, `user_id`, `session_id`, `message` |
| `agent:step` | 도구 호출 루프의 각 반복 | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | 에이전트 처리가 완료됨 | `platform`, `user_id`, `session_id`, `message`, `response` |
| `command:*` | 모든 슬래시 명령어가 실행됨 | `platform`, `user_id`, `command`, `args` |

#### 와일드카드 매칭

`command:*`에 등록된 핸들러는 모든 `command:` 이벤트(`command:model`, `command:reset` 등)에 대해 실행됩니다. 단일 구독으로 모든 슬래시 명령어를 모니터링할 수 있습니다.

### 예시

#### 작업이 길어질 때 Telegram 경고

에이전트가 10단계를 초과할 때 자신에게 메시지를 보냅니다.

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: 에이전트가 많은 단계를 수행 중일 때 경고
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
        text = f"⚠️ 에이전트가 {iteration}단계 동안 실행 중입니다. 마지막 도구: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### 명령어 사용 기록기 (Command Usage Logger)

어떤 슬래시 명령어가 사용되는지 추적합니다.

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: 슬래시 명령어 사용 기록
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

#### 세션 시작 웹훅 (Session Start Webhook)

새 세션 시 외부 서비스에 POST를 보냅니다.

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: 새 세션 시 외부 서비스에 알림
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

### 튜토리얼: BOOT.md — 게이트웨이 부팅 시마다 시작 체크리스트 실행하기

커뮤니티에서 인기 있는 패턴: `~/.hermes/BOOT.md`에 마크다운 체크리스트를 놓아두고 게이트웨이가 시작될 때마다 에이전트가 이를 한 번 실행하도록 합니다. "매 부팅 시 밤새 실패한 크론 작업을 확인하고 실패한 것이 있으면 Discord로 핑(ping)해줘" 또는 "지난 24시간 동안의 deploy.log를 요약해서 Slack #ops에 올려줘" 등에 유용합니다.

이 튜토리얼은 사용자 정의 훅으로 직접 구축하는 방법을 보여줍니다. Hermes는 내장형 BOOT.md 훅을 제공하지 않으므로 사용자가 원하는 동작을 정확하게 연결해야 합니다.

#### 구축할 내용

1. 자연어 시작 지침이 있는 `~/.hermes/BOOT.md` 파일.
2. `gateway:startup` 시 실행되고, 게이트웨이의 해결된 모델/자격 증명으로 일회성 에이전트를 생성하며, BOOT.md 지침을 실행하는 게이트웨이 훅.
3. 보고할 내용이 없을 때 메시지 전송을 선택 해제할 수 있도록 하는 `[SILENT]` 규칙.

#### 1단계: 체크리스트 작성

`~/.hermes/BOOT.md`를 만듭니다. 인간 비서에게 지시하는 것처럼 작성하세요.

```markdown
# 시작 체크리스트 (Startup Checklist)

1. `hermes cron list`를 실행하여 밤새 예약된 작업 중 실패한 것이 있는지 확인하세요.
2. 실패한 항목이 있으면 `send_message` 도구를 사용하여 요약 내용을 Discord #ops로 전송하세요.
3. `/opt/app/deploy.log`에 지난 24시간 동안 ERROR 줄이 있는지 확인하세요. 있는 경우 이를 요약하고 동일한 Discord 메시지에 포함하세요.
4. 잘못된 것이 없다면 전송될 메시지가 없도록 `[SILENT]`라고만 응답하세요.
```

에이전트는 이를 프롬프트의 일부로 인식하므로, 도구 호출, 셸 명령어, 메시지 전송, 파일 요약 등 일반 언어로 설명할 수 있는 모든 작업이 가능합니다.

#### 2단계: 훅 생성

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: 게이트웨이 시작 시 ~/.hermes/BOOT.md 실행
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""모든 게이트웨이 시작 시 ~/.hermes/BOOT.md를 실행합니다."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "시작 부팅 체크리스트를 실행하고 있습니다. 아래 지침을 "
        "정확히 따르세요.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "각 지침을 실행하세요. send_message 도구를 사용하여 "
        "Discord나 Slack과 같은 플랫폼에 메시지를 전달하세요.\n"
        "주의가 필요한 사항이 없고 보고할 내용이 없으면 "
        "ONLY: [SILENT] 로 응답하세요."
    )


def _run_boot_agent(content: str) -> None:
    """일회성 에이전트를 생성하고 체크리스트를 실행합니다.

    게이트웨이의 해결된 모델 및 런타임 자격 증명을 사용하여
    사용자 지정 엔드포인트, 통합자 및 OAuth 기반 공급업체에 대해서도 모두 작동합니다.
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
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
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

    # 게이트웨이 시작이 전체 에이전트 턴에서 차단되지 않도록 하는 백그라운드 스레드입니다.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

두 가지 핵심 줄:

- `_resolve_gateway_model()`은 게이트웨이에 현재 구성된 모델을 읽습니다.
- `_resolve_runtime_agent_kwargs()`는 정상적인 게이트웨이 턴과 동일한 방식으로 제공업체 자격 증명(API 키, 기본 URL, OAuth 토큰 및 자격 증명 풀 포함)을 확인합니다.

이러한 항목이 없으면 베어(bare) `AIAgent()`는 내장된 기본값으로 폴백되며 기본이 아닌 엔드포인트에 대해 401 오류가 발생합니다.

#### 3단계: 테스트하기

게이트웨이 다시 시작:

```bash
hermes gateway restart
```

로그 확인:

```bash
hermes logs --follow --level INFO | grep boot-md
```

`Running BOOT.md (N chars)` 뒤에 `boot-md completed: ...`(에이전트가 수행한 작업 요약) 또는 에이전트가 `[SILENT]`로 응답한 경우 `boot-md completed (nothing to report)`가 표시되어야 합니다.

체크리스트를 비활성화하려면 `~/.hermes/BOOT.md`를 삭제하세요. 훅은 계속 로드된 상태로 유지되지만 파일이 없으면 자동으로 건너뜁니다.

#### 패턴 확장

- **일정 인지 체크리스트:** BOOT.md 지시 내에서 `datetime.now().weekday()`를 확인합니다("월요일이면 주간 배포 로그도 확인해"). 지침은 자유 형식의 텍스트이므로 에이전트가 추론할 수 있는 것은 무엇이든 사용할 수 있습니다.
- **다중 체크리스트:** 훅이 다른 파일(`STARTUP.md`, `MORNING.md` 등)을 가리키도록 하고 각각에 대해 별도의 훅 디렉토리를 등록합니다.
- **비 에이전트 변형:** 전체 에이전트 루프가 필요하지 않은 경우 `AIAgent`를 완전히 건너뛰고 핸들러가 `httpx`를 통해 고정 알림을 직접 게시하도록 합니다. 이 방법은 더 저렴하고, 빠르며, 공급업체 의존성이 없습니다.

#### 내장 기능이 아닌 이유

Hermes의 이전 버전에서는 이를 기본 제공 훅으로 제공하고 게이트웨이가 부팅될 때마다 베어(bare) 기본값으로 에이전트를 조용히 생성했습니다. 이는 사용자 지정 엔드포인트가 있는 사용자를 놀라게 했고, 훅이 실행되고 있다는 사실을 모르는 사용자에게는 이 기능이 보이지 않았습니다. 이를 문서화된 패턴(자신의 훅 디렉토리에 직접 구축)으로 유지하면 수행하는 작업을 정확히 볼 수 있고 파일을 작성하여 명시적으로 선택(opt in)할 수 있습니다.

### 작동 방식

1. 게이트웨이 시작 시 `HookRegistry.discover_and_load()`가 `~/.hermes/hooks/`를 스캔합니다.
2. `HOOK.yaml` + `handler.py`가 있는 각 하위 디렉토리가 동적으로 로드됩니다.
3. 핸들러는 선언된 이벤트에 대해 등록됩니다.
4. 각 수명 주기 시점에 `hooks.emit()`은 일치하는 모든 핸들러를 실행합니다.
5. 핸들러에서 발생한 오류는 포착되어 기록됩니다. 손상된 훅은 에이전트 작동을 중단시키지 않습니다.

:::info
게이트웨이 훅은 **게이트웨이**(Telegram, Discord, Slack, WhatsApp, Teams)에서만 실행됩니다. CLI는 게이트웨이 훅을 로드하지 않습니다. 어디서나 작동하는 훅을 사용하려면 [플러그인 훅](#plugin-hooks)을 사용하세요.
:::

## 플러그인 훅 (Plugin Hooks)

[플러그인](/user-guide/features/plugins)은 **CLI 및 게이트웨이** 세션 모두에서 실행되는 훅을 등록할 수 있습니다. 플러그인의 `register()` 함수에서 `ctx.register_hook()`을 통해 프로그래밍 방식으로 등록됩니다.

플러그인 패키징 및 등록 세부 정보는 [플러그인 가이드](/docs/user-guide/features/plugins)를 참조하세요.

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**모든 훅의 일반 규칙:**

- 콜백은 **키워드 인수(keyword arguments)**를 받습니다. 하위 호환성을 위해 항상 `**kwargs`를 허용하세요. 이후 버전에서 플러그인 중단 없이 새로운 파라미터가 추가될 수 있습니다.
- 콜백에서 **충돌이 발생하면** 로그가 기록되고 건너뜁니다. 다른 훅과 에이전트는 정상적으로 계속 작동합니다. 오작동하는 플러그인은 에이전트 작동을 방해할 수 없습니다.
- 두 개의 훅 반환 값이 동작에 영향을 줍니다. [`pre_tool_call`](#pre_tool_call)은 호출을 **차단**할 수 있고, [`pre_llm_call`](#pre_llm_call)은 LLM 호출에 **컨텍스트를 주입**할 수 있습니다. 다른 모든 훅은 잊고(fire-and-forget) 넘어가는 관찰자(observer)입니다.
- 관찰자 콜백은 `telemetry_schema_version`을 자동으로 받습니다. 표시되는 경우 `turn_id`, `api_request_id`, `task_id`, `session_id` 및 `api_call_count`는 별도의 상관관계 필드(correlation fields)입니다. `api_request_id`는 불투명한 식별자로 취급하고 해당 문자열 형식을 구문 분석하지 마세요.

### 빠른 참조

| 훅 | 실행 시기 | 반환값 |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | 도구가 실행되기 전 | 호출을 거부(veto)하려면 `{"action": "block", "message": str}` 반환 |
| [`post_tool_call`](#post_tool_call) | 도구가 반환된 후 | 무시됨 |
| [`pre_llm_call`](#pre_llm_call) | 턴당 한 번, 도구 호출 루프 시작 전 | 사용자 메시지 앞에 컨텍스트를 추가하려면 `{"context": str}` 반환 |
| [`post_llm_call`](#post_llm_call) | 턴당 한 번, 도구 호출 루프 후 | 무시됨 |
| [`on_session_start`](#on_session_start) | 새 세션 생성됨 (첫 번째 턴만) | 무시됨 |
| [`on_session_end`](#on_session_end) | 세션 종료 | 무시됨 |
| [`on_session_finalize`](#on_session_finalize) | CLI/게이트웨이가 활성 세션을 해제함 (플러시, 저장, 통계) | 무시됨 |
| [`on_session_reset`](#on_session_reset) | 게이트웨이가 새로운 세션 키로 교체함 (예: `/new`, `/reset`) | 무시됨 |
| [`subagent_stop`](#subagent_stop) | `delegate_task` 하위(child) 에이전트 종료됨 | 무시됨 |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | 인증 및 디스패치 전, 게이트웨이가 사용자 메시지를 수신함 | 흐름에 영향을 주려면 `{"action": "skip" \| "rewrite" \| "allow", ...}` 반환 |
| [`pre_approval_request`](#pre_approval_request) | 프롬프트/알림이 전송되기 전, 위험한 명령어의 사용자 승인 요청 시 | 무시됨 |
| [`post_approval_response`](#post_approval_response) | 사용자가 승인 프롬프트에 응답함 (또는 시간 초과됨) | 무시됨 |
| [`transform_tool_result`](#transform_tool_result) | 모든 도구가 반환된 후, 결과를 모델에 전달하기 전 | 결과를 바꾸려면 `str`, 변경하지 않으려면 `None` 반환 |
| [`transform_terminal_output`](#transform_terminal_output) | 자르기/ANSI 제거/편집 전에 `terminal` 도구 내부에서 | 원시 출력을 바꾸려면 `str`, 변경하지 않으려면 `None` 반환 |
| [`transform_llm_output`](#transform_llm_output) | 도구 호출 루프가 완료된 후, 최종 응답이 전달되기 전 | 응답 텍스트를 바꾸려면 `str`, 변경하지 않으려면 `None`/빈 문자열 반환 |

---

### `pre_tool_call`

기본 제공 도구 및 플러그인 도구와 같이 모든 도구 실행 **직전**에 실행됩니다.

**콜백 서명:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `tool_name` | `str` | 실행할 도구의 이름 (예: `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | 모델이 도구에 전달한 인수 |
| `task_id` | `str` | 세션/작업 식별자. 설정되지 않은 경우 빈 문자열. |

**실행 시기:** `model_tools.py`의 `handle_function_call()` 내부에서 도구 핸들러가 실행되기 전. 도구 호출당 한 번 실행됩니다. 모델이 도구 3개를 병렬로 호출하면 3번 실행됩니다.

**반환값 — 호출 거부(veto):**

```python
return {"action": "block", "message": "도구 호출이 차단된 이유"}
```

에이전트는 모델에 반환된 오류로 `message`를 사용하여 도구를 조기에 종료(short-circuits)합니다. 처음 일치하는 블록 지시문이 적용됩니다(Python 플러그인이 먼저 등록된 다음 셸 훅). 다른 반환 값은 무시되므로 기존 관찰자 전용 콜백은 변경 없이 계속 작동합니다.

**사용 사례:** 로깅, 감사 추적, 도구 호출 카운터, 위험한 작업 차단, 속도 제한, 사용자별 정책 시행.

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

**예시 — 위험한 도구 사용 경고:**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ 잠재적으로 위험한 도구 실행 중: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

모든 도구 실행이 반환된 **직후**에 실행됩니다.

**콜백 서명:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `tool_name` | `str` | 방금 실행된 도구 이름 |
| `args` | `dict` | 모델이 도구에 전달한 인수 |
| `result` | `str` | 도구의 반환값 (항상 JSON 문자열) |
| `task_id` | `str` | 세션/작업 식별자. 설정되지 않은 경우 빈 문자열. |
| `duration_ms` | `int` | 도구 파견에 걸린 시간(밀리초). (`time.monotonic()`으로 측정) |

**실행 시기:** `model_tools.py`의 `handle_function_call()` 내부에서 도구 핸들러가 반환된 후. 도구 호출당 한 번씩 발생합니다. 도구에서 처리되지 않은 예외가 발생한 경우에는 실행되지 **않습니다** (오류를 잡아 오류 JSON 문자열로 반환하며 `post_tool_call`은 `result`에 해당 오류 문자열과 함께 실행됩니다).

**반환값:** 무시됨.

**사용 사례:** 도구 결과 로깅, 메트릭 수집, 도구 성공/실패율 추적, 대기 시간 대시보드, 도구당 예산 경고, 특정 도구가 완료될 때 알림 전송.

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

도구 호출 루프가 시작되기 전 **턴(turn)당 한 번** 실행됩니다. 이는 **반환값이 사용되는 유일한 훅**입니다. 이 턴의 사용자 메시지에 컨텍스트를 주입할 수 있습니다.

**콜백 서명:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 현재 세션의 고유 식별자 |
| `user_message` | `str` | 이 턴에 대한 사용자의 원본 메시지 (스킬 주입 전) |
| `conversation_history` | `list` | 전체 메시지 목록의 사본 (OpenAI 형식: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | 새 세션의 첫 번째 턴인 경우 `True`, 이후 턴의 경우 `False` |
| `model` | `str` | 모델 식별자 (예: `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | 세션이 실행 중인 플랫폼: `"cli"`, `"telegram"`, `"discord"` 등 |

**실행 시기:** `run_agent.py`의 `run_conversation()` 내부에서, 컨텍스트 압축 후 메인 `while` 루프 이전에. 도구 루프 내의 API 호출당 한 번이 아니라, `run_conversation()` 호출당 한 번(즉, 사용자 턴당 한 번) 실행됩니다.

**반환값:** 콜백이 `"context"` 키가 포함된 dict 또는 비어 있지 않은 일반 문자열을 반환하면 해당 텍스트가 현재 턴의 사용자 메시지에 추가됩니다. 주입하지 않으려면 `None`을 반환합니다.

```python
# 컨텍스트 주입
return {"context": "기억해낸 정보:\n- 사용자는 Python을 좋아함\n- hermes-agent 작업 중"}

# 일반 문자열 (동일)
return "기억해낸 정보:\n- 사용자는 Python을 좋아함"

# 주입 없음
return None
```

**컨텍스트가 주입되는 위치:** 항상 시스템 프롬프트가 아닌 **사용자 메시지**입니다. 이렇게 하면 프롬프트 캐시가 유지됩니다. 시스템 프롬프트는 턴 간에 동일하게 유지되므로 캐시된 토큰이 재사용됩니다. 시스템 프롬프트는 Hermes의 영역(모델 가이던스, 도구 적용, 성격, 스킬)입니다. 플러그인은 사용자 입력과 함께 컨텍스트에 기여합니다.

주입된 모든 컨텍스트는 **일시적(ephemeral)**입니다 (API 호출 시에만 추가됨). 대화 기록에 있는 사용자의 원본 메시지는 절대로 변경되지 않으며, 어떤 내용도 세션 데이터베이스에 영구 저장되지 않습니다.

**여러 플러그인**이 컨텍스트를 반환할 때, 그 출력값은 플러그인 발견 순서(디렉토리 이름의 알파벳 순)대로 이중 줄 바꿈(`\n\n`)으로 결합됩니다.

**사용 사례:** 메모리 리콜(Memory recall), RAG 컨텍스트 주입, 가드레일, 턴당(per-turn) 분석.

**예시 — 메모리 리콜:**

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
        text = "기억해낸 컨텍스트:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**예시 — 가드레일:**

```python
POLICY = "사용자의 명시적인 확인 없이 파일을 삭제하는 명령을 실행하지 마십시오."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

도구 호출 루프가 완료되고 에이전트가 최종 응답을 생성한 후 **턴(turn)당 한 번** 실행됩니다. **성공적인** 턴에만 실행되며, 턴이 중간에 중단되면 실행되지 않습니다.

**콜백 서명:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 현재 세션의 고유 식별자 |
| `user_message` | `str` | 이 턴에 대한 사용자의 원본 메시지 |
| `assistant_response` | `str` | 이 턴에 대한 에이전트의 최종 텍스트 응답 |
| `conversation_history` | `list` | 턴이 완료된 후 전체 메시지 목록의 사본 |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행 중인 플랫폼 |

**실행 시기:** `run_agent.py`의 `run_conversation()` 내부에서 도구 루프가 최종 응답을 남기고 종료된 후. `if final_response and not interrupted`로 보호되므로 사용자가 중간에 중단하거나 에이전트가 응답을 생성하지 못한 채 반복 제한(iteration limit)에 도달하면 실행되지 **않습니다**.

**반환값:** 무시됨.

**사용 사례:** 외부 메모리 시스템에 대화 데이터 동기화, 응답 품질 메트릭 계산, 턴 요약 기록, 후속 조치(follow-up actions) 트리거.

**예시 — 외부 메모리에 동기화:**

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
        pass  # best-effort (최선을 다하되 실패 시 무시)

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

### `on_session_start`

완전히 새로운 세션이 생성될 때 **한 번** 발생합니다. 세션 재개(session continuation, 사용자가 기존 세션에서 두 번째 메시지를 보낼 때) 시에는 발생하지 **않습니다**.

**콜백 서명:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 새 세션의 고유 식별자 |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행 중인 플랫폼 |

**실행 시기:** `run_agent.py`의 `run_conversation()` 내부, 새 세션의 첫 번째 턴 중에 — 구체적으로 시스템 프롬프트가 구성된 후 도구 루프가 시작되기 전. 조건은 `if not conversation_history`(이전 메시지가 없으면 새 세션)입니다.

**반환값:** 무시됨.

**사용 사례:** 세션 범위의 상태(state) 초기화, 캐시 예열, 외부 서비스에 세션 등록, 세션 시작 기록.

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

결과에 관계없이 모든 `run_conversation()` 호출의 **가장 마지막**에 발생합니다. 사용자가 종료했을 때 에이전트가 처리 중인 턴이었다면 CLI의 종료 핸들러(exit handler)에서도 발생합니다.

**콜백 서명:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 세션의 고유 식별자 |
| `completed` | `bool` | 에이전트가 최종 응답을 생성한 경우 `True`, 그렇지 않은 경우 `False` |
| `interrupted` | `bool` | 턴이 중단된 경우 `True` (사용자가 새 메시지를 보냈거나, `/stop` 실행, 종료한 경우) |
| `model` | `str` | 모델 식별자 |
| `platform` | `str` | 세션이 실행 중인 플랫폼 |

**실행 시기:** 두 곳에서 발생합니다.
1. **`run_agent.py`** — 모든 정리가 완료된 후 모든 `run_conversation()` 호출의 끝에. 턴에서 오류가 발생했더라도 항상 실행됩니다.
2. **`cli.py`** — CLI의 `atexit` 핸들러 내부, 단 종료가 발생했을 때 에이전트가 턴 중간(`_agent_running=True`)이었던 경우에**만**. 이는 처리 중 Ctrl+C 및 `/exit`를 포착합니다. 이 경우 `completed=False` 및 `interrupted=True`입니다.

**반환값:** 무시됨.

**사용 사례:** 버퍼 플러시(flush), 연결 닫기, 세션 상태 저장, 세션 지속 시간 기록, `on_session_start`에서 초기화된 리소스 정리.

**예시 — 플러시 및 정리:**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # 누적된 데이터를 디스크나 외부 서비스에 플러시(Flush)합니다.
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"세션 {session_id} 종료: {status}, 도구 {cache['tool_calls']}회 호출됨")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**예시 — 세션 지속 시간 추적:**

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

CLI 또는 게이트웨이가 활성 세션을 **해제(tears down)**할 때 발생합니다 (예: 사용자가 `/new`를 실행하거나, 게이트웨이가 유휴 세션을 GC(가비지 컬렉션)했거나, CLI가 활성 에이전트와 함께 종료될 때). 이전 세션의 식별자(ID)가 사라지기 전에 나가는 세션에 묶인 상태를 플러시(flush)할 수 있는 마지막 기회입니다.

**콜백 서명:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` 또는 `None` | 해제되는 세션의 ID입니다. 활성 세션이 없었던 경우 `None`일 수 있습니다. |
| `platform` | `str` | `"cli"` 또는 메시징 플랫폼 이름 (`"telegram"`, `"discord"` 등). |

**실행 시기:** `cli.py` (`/new` / CLI 종료 시) 및 `gateway/run.py` (세션이 재설정되거나 GC될 때). 게이트웨이 측의 `on_session_reset`과 항상 쌍을 이룹니다.

**반환값:** 무시됨.

**사용 사례:** 세션 ID가 삭제되기 전에 최종 세션 메트릭 유지, 세션별 리소스 닫기, 최종 원격 측정 이벤트(telemetry event) 내보내기, 대기열에 있는 쓰기 작업 비우기(drain queued writes).

---

### `on_session_reset`

게이트웨이가 활성 채팅에 대해 **새 세션 키로 교체**할 때 발생합니다 (사용자가 `/new`, `/reset`, `/clear`를 호출했거나, 어댑터가 유휴 창이 지난 후 새 세션을 선택했을 때). 이를 통해 플러그인은 다음 `on_session_start`를 기다리지 않고 대화 상태가 지워졌다는 사실에 반응할 수 있습니다.

**콜백 서명:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `session_id` | `str` | 새 세션의 ID입니다 (이미 새 값으로 회전됨). |
| `platform` | `str` | 메시징 플랫폼 이름. |

**실행 시기:** `gateway/run.py`에서 새 세션 키가 할당된 직후지만 다음 인바운드 메시지가 처리되기 전. 게이트웨이에서의 순서는 `on_session_finalize(old_id)` → 전환(swap) → `on_session_reset(new_id)` → (첫 번째 인바운드 턴에서) `on_session_start(new_id)`입니다.

**반환값:** 무시됨.

**사용 사례:** `session_id`를 키로 하는 세션별 캐시 재설정, "session rotated" (세션 교체) 분석(analytics) 전송, 새로운 상태 버킷(state bucket) 준비.

---

도구 스키마, 핸들러, 고급 훅 패턴을 포함한 전체 과정은 **[플러그인 빌드 가이드](/guides/build-a-hermes-plugin)**를 참조하세요.

---

### `subagent_stop`

`delegate_task`가 완료된 후 **하위 에이전트(child agent)당 한 번** 실행됩니다. 단일 작업을 위임했든 세 개의 작업을 일괄 위임했든 관계없이 각 하위에 대해 한 번씩 상위(parent) 스레드에서 순차적으로(serialised) 발생합니다.

**콜백 서명:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `parent_session_id` | `str` | 위임하는 상위 에이전트의 세션 ID |
| `child_role` | `str \| None` | 하위 에이전트에 설정된 오케스트레이터 역할 태그 (기능이 활성화되지 않은 경우 `None`) |
| `child_summary` | `str \| None` | 하위 에이전트가 상위에게 반환한 최종 응답 |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"`, 또는 `"error"` |
| `duration_ms` | `int` | 하위 에이전트를 실행하는 데 소요된 실제 시간(밀리초, Wall-clock time) |

**실행 시기:** `tools/delegate_tool.py`에서 `ThreadPoolExecutor.as_completed()`가 모든 하위 선물(child futures)을 비운(drain) 후. 훅 작성자가 동시(concurrent) 콜백 실행에 대해 고민하지 않도록 실행이 상위 스레드로 마샬링(marshalled)됩니다.

**반환값:** 무시됨.

**사용 사례:** 오케스트레이션 활동 로깅, 청구를 위한 하위 소요 시간 누적, 위임 후 감사 기록 작성.

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
대규모 위임(예: 오케스트레이터 역할 × 5개 리프(leaves) × 중첩 깊이)의 경우 `subagent_stop`은 턴당 여러 번 발생합니다. 콜백을 빠르게 유지하세요. 무거운 작업은 백그라운드 대기열로 밀어넣으세요.
:::

---

### `pre_gateway_dispatch`

게이트웨이에서 **수신되는 각 `MessageEvent`당 한 번씩**, 내부 이벤트 보호(internal-event guard) 후지만 인증/페어링(auth/pairing) 및 에이전트 디스패치 **이전**에 발생합니다. 이는 단일 플랫폼 어댑터에 깔끔하게 맞지 않는 게이트웨이 수준의 메시지 흐름 정책(수신 전용 창, 사람에게 인계(human handover), 채팅별 라우팅 등)에 대한 가로채기 지점(interception point)입니다.

**콜백 서명:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `event` | `MessageEvent` | 정규화된 인바운드 메시지 (`.text`, `.source`, `.message_id`, `.internal` 등 포함). |
| `gateway` | `GatewayRunner` | 플러그인이 사이드 채널 응답(소유자 알림 등)을 위해 `gateway.adapters[platform].send(...)`를 호출할 수 있도록 하는 활성 게이트웨이 러너. |
| `session_store` | `SessionStore` | `session_store.append_to_transcript(...)`를 통한 조용한 텍스트(transcript) 수집(ingestion)용. |

**실행 시기:** `gateway/run.py`에서 `GatewayRunner._handle_message()` 내부, `is_internal`이 계산된 직후. **내부 이벤트는 훅을 완전히 건너뜁니다** (백그라운드 프로세스 완료 등 시스템에서 생성되는 이벤트이므로 사용자 대면 정책에 의해 제어되어서는 안 됩니다).

**반환값:** `None` 또는 dict. 첫 번째로 인식된 동작 dict가 적용됩니다(wins). 나머지 플러그인 결과는 무시됩니다. 플러그인 콜백의 예외는 포착되고 기록됩니다. 게이트웨이는 오류 발생 시 항상 정상적인 디스패치로 폴백(fall through)합니다.

| 반환값 | 효과 |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | 메시지를 삭제합니다 — 에이전트 응답 없음, 페어링 절차 없음, 인증 없음. 플러그인이 이 메시지를 처리한 것으로 간주됩니다(예: 대화 기록에 조용히 수집됨). |
| `{"action": "rewrite", "text": "새 텍스트"}` | `event.text`를 바꾼 후, 수정된 이벤트로 정상적인 디스패치를 계속합니다. 버퍼링된 주변 메시지(ambient messages)를 하나의 프롬프트로 병합(collapse)할 때 유용합니다. |
| `{"action": "allow"}` / `None` | 정상적인 디스패치 — 전체 인증 / 페어링 / 에이전트 루프 체인을 실행합니다. |

**사용 사례:** 수신 전용 그룹 채팅 (태그된 경우에만 응답; 주변 메시지를 컨텍스트로 버퍼링); 사람에게 인계 (소유자가 채팅을 수동으로 처리하는 동안 고객 메시지를 조용히 수집); 프로필별 속도 제한; 정책 기반 라우팅.

**예시 — 페어링 코드를 트리거하지 않고 조용히 권한 없는 DM 차단(drop)하기:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**예시 — 멘션이 있을 때 주변(ambient) 메시지 버퍼를 단일 프롬프트로 재작성하기:**

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

### `pre_approval_request`

승인 요청이 사용자에게 표시되기 **직전**에 발생합니다. 대화형 CLI, Ink TUI, 게이트웨이 플랫폼(Telegram, Discord, Slack, WhatsApp, Matrix 등) 및 ACP 클라이언트(VS Code, Zed, JetBrains)와 같은 모든 접점(surface)을 포괄합니다.

이는 허용/거부(allow/deny) 알림을 표시하는 macOS 메뉴 막대 앱이나 모든 승인 요청을 컨텍스트와 함께 기록하는 감사 로그와 같이 사용자 지정 알리미(notifier)를 연결하기에 적합한 위치입니다.

**콜백 서명:**

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

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `command` | `str` | 승인을 기다리는 셸 명령어 |
| `description` | `str` | 명령어가 플래그된 이유를 사람이 읽을 수 있는 형태로 표시 (여러 패턴이 일치할 때 결합됨) |
| `pattern_key` | `str` | 승인을 트리거한 기본 패턴 키 (예: `"rm_rf"`, `"sudo"`) |
| `pattern_keys` | `list[str]` | 일치하는 모든 패턴 키 |
| `session_key` | `str` | 채팅별로 알림의 범위를 지정할 때 유용한 세션 식별자 |
| `surface` | `str` | 대화형 CLI/TUI 프롬프트의 경우 `"cli"`, 비동기 플랫폼 승인의 경우 `"gateway"` |

**반환값:** 무시됨. 이 훅은 관찰자(observer) 역할만 수행합니다. 승인을 거부하거나 미리 답변할 수 없습니다. 승인 시스템에 도달하기 전에 도구를 차단하려면 [`pre_tool_call`](#pre_tool_call)을 사용하세요.

**사용 사례:** 데스크탑 알림, 푸시 알림, 감사 로깅, Slack 웹훅, 에스컬레이션 라우팅, 메트릭.

**예시 — macOS의 데스크탑 알림:**

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

사용자가 승인 프롬프트에 응답한(또는 프롬프트가 시간 초과된) **후**에 발생합니다.

**콜백 서명:**

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

`pre_approval_request`와 동일한 kwargs에 다음이 추가됩니다.

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `choice` | `str` | `"once"`, `"session"`, `"always"`, `"deny"`, 또는 `"timeout"` 중 하나 |

**반환값:** 무시됨.

**사용 사례:** 일치하는 데스크톱 알림 닫기, 감사 로그에 최종 결정 기록, 메트릭 업데이트, 속도 제한기 롤 포워드(roll forward).

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

도구가 반환된 **후** 그리고 결과가 대화에 추가되기 **전**에 실행됩니다. 플러그인이 모델이 결과를 보기 전에 (터미널 출력뿐만 아니라) 모든 도구의 결과 문자열을 바꿀 수 있습니다(rewrite).

**콜백 서명:**

```python
def my_callback(
    tool_name: str,
    arguments: dict,
    result: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `tool_name` | `str` | 결과를 생성한 도구 (`read_file`, `web_extract`, `delegate_task` 등). |
| `arguments` | `dict` | 모델이 도구를 호출할 때 사용한 인수. |
| `result` | `str` | 자르기 및 ANSI가 제거된(post-truncation, post-ANSI-strip) 이후 도구의 원시 결과 문자열. |
| `task_id` | `str \| None` | RL/벤치마크 환경 내에서 실행할 때의 작업/세션 ID. |

**반환값:** 결과를 대체하려면 `str`(반환된 문자열이 모델에 표시되는 내용), 변경하지 않으려면 `None`.

**사용 사례:** `web_extract` 출력에서 조직별 PII 수정, 긴 JSON 도구 응답을 요약 헤더로 감싸기, 검색 증강 힌트(retrieval-augmented hints)를 `read_file` 결과에 주입, `delegate_task` 하위 에이전트 보고서를 프로젝트별 스키마로 재작성(rewrite).

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

모든 도구에 적용됩니다. 터미널 전용 재작성의 경우 아래의 `transform_terminal_output`을 참조하세요. 이 훅이 더 제한적이며 파이프라인의 초기(자르기 전, 수정 전) 단계에서 실행됩니다.

---

### `transform_terminal_output`

기본 50KB 자르기(truncation), ANSI 제거(strip) 및 시크릿(secret) 수정(redaction) 전에 `terminal` 도구의 포그라운드 출력 파이프라인 내부에서 실행됩니다. 다운스트림 처리가 닿기 전에 플러그인이 셸 명령의 원시 stdout/stderr을 다시 작성(rewrite)할 수 있도록 합니다.

**콜백 서명:**

```python
def my_callback(
    command: str,
    output: str,
    exit_code: int,
    cwd: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `command` | `str` | 출력을 생성한 셸 명령입니다. |
| `output` | `str` | 원시 결합 stdout/stderr (매우 클 수 있음 — 이 훅 이후에 잘림(truncation)이 발생함). |
| `exit_code` | `int` | 프로세스 종료 코드. |
| `cwd` | `str` | 명령이 실행된 작업 디렉토리입니다. |

**반환값:** 출력을 교체할 `str`, 변경하지 않으려면 `None`.

**사용 사례:** 방대한 출력을 생성하는 명령(`du -ah`, `find`, `tree`)에 요약 주입, 프로젝트별 마커로 출력을 태그하여 다운스트림 훅이 처리 방법을 알 수 있도록 함, 실행 간 펄럭거리며(flaps) 프롬프트 캐시를 방해하는 타이밍 노이즈 제거.

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[요약: 총 {lines}개 경로, 처음 40개 표시]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

(다른 모든 도구를 포괄하는) `transform_tool_result`와 잘 어울립니다.

---

### `transform_llm_output`

도구 호출 루프가 완료되고 모델이 최종 응답을 생성한 후 해당 응답이 사용자(CLI, 게이트웨이 또는 프로그래밍 호출자)에게 전달되기 **전**에 **턴당 한 번** 실행됩니다. SOUL 관련 플레이버(flavor) 텍스트 또는 기술 기반(skill-driven) 변환에 대한 추가 추론 토큰을 소모하지 않고 클래식 프로그래밍 방식을 사용하여 플러그인이 어시스턴트의 최종 텍스트를 재작성할 수 있도록 합니다.

**콜백 서명:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| 파라미터 | 유형 | 설명 |
|-----------|------|-------------|
| `response_text` | `str` | 이번 턴에 대한 어시스턴트의 최종 응답 텍스트입니다. |
| `session_id` | `str` | 이 대화에 대한 세션 ID (원샷 실행의 경우 비어 있을 수 있음). |
| `model` | `str` | 응답을 생성한 모델 이름 (예: `anthropic/claude-sonnet-4.6`). |
| `platform` | `str` | 전달 플랫폼 (`cli`, `telegram`, `discord`, …; 설정되지 않은 경우 비어 있음). |

**반환값:** 응답 텍스트를 바꿀 비어 있지 않은 `str`, 변경하지 않으려면 `None` 또는 빈 문자열. `transform_tool_result`와 마찬가지로, 여러 플러그인이 등록된 경우 **첫 번째로 비어 있지 않은 문자열이 우선(wins)합니다**.

**사용 사례:** 성격/어휘 변환 적용 (해적어, 스폰지밥 등), 최종 텍스트에서 사용자 관련 식별자 숨기기(redact), 프로젝트 관련 서명 바닥글 추가, SOUL 지침에 토큰을 사용하지 않고 자체 스타일 가이드 강제 적용.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # 변경 없이 통과
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

이 훅은 응답이 비어있지 않고 중단되지 않은 경우에만 실행되도록 보호됩니다 — 정지 버튼 인터럽트 또는 빈 턴에서는 실행되지 않습니다. 예외는 경고로 기록되며 에이전트 실행을 중단시키지 않습니다.

---

## 셸 훅 (Shell Hooks)

`cli-config.yaml`에 셸 스크립트 훅을 선언하면, 해당 플러그인 훅 이벤트가 발생할 때마다 CLI와 게이트웨이 세션 모두에서 Hermes가 해당 스크립트를 하위 프로세스로 실행합니다. Python 플러그인을 작성할 필요가 없습니다.

셸 훅은 다음과 같이 즉시 사용 가능한(drop-in) 단일 파일 스크립트(Bash, Python, 셰뱅(shebang)이 있는 모든 스크립트)를 원할 때 사용합니다:

- **도구 호출 차단** — 위험한 `terminal` 명령어 거부, 디렉토리별 정책 시행, 파괴적인 `write_file` / `patch` 작업에 대한 승인 요구.
- **도구 호출 후 실행** — 에이전트가 방금 작성한 Python 또는 TypeScript 파일 자동 서식 지정(auto-format), API 호출 로깅, CI 워크플로 트리거.
- **다음 LLM 턴에 컨텍스트 주입** — `git status` 출력, 현재 요일 또는 검색된 문서를 사용자 메시지 앞에 추가 ([`pre_llm_call`](#pre_llm_call) 참조).
- **수명 주기 이벤트 모니터링** — 하위 에이전트가 완료될 때(`subagent_stop`) 또는 세션이 시작될 때(`on_session_start`) 로그 줄 작성.

셸 훅은 CLI 시작 시(`hermes_cli/main.py`)와 게이트웨이 시작 시(`gateway/run.py`) 모두에서 `agent.shell_hooks.register_from_config(cfg)`를 호출하여 등록됩니다. 이 훅들은 Python 플러그인 훅과 자연스럽게 결합되어 동일한 디스패처를 통해 실행됩니다.

### 한 눈에 비교하기

| 구분 | 셸 훅 (Shell hooks) | [플러그인 훅 (Plugin hooks)](#plugin-hooks) | [게이트웨이 훅 (Gateway hooks)](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| 선언 위치 | `~/.hermes/config.yaml`의 `hooks:` 블록 | `plugin.yaml` 플러그인의 `register()` | `HOOK.yaml` + `handler.py` 디렉토리 |
| 저장 위치 | `~/.hermes/agent-hooks/` (관례) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 언어 | 모두 가능 (Bash, Python, Go 바이너리 등) | Python 전용 | Python 전용 |
| 실행 환경 | CLI + 게이트웨이 | CLI + 게이트웨이 | 게이트웨이 전용 |
| 이벤트 | `VALID_HOOKS` (incl. `subagent_stop`) | `VALID_HOOKS` | 게이트웨이 수명 주기 (`gateway:startup`, `agent:*`, `command:*`) |
| 도구 호출 차단 여부 | 가능 (`pre_tool_call`) | 가능 (`pre_tool_call`) | 불가능 |
| LLM 컨텍스트 주입 | 가능 (`pre_llm_call`) | 가능 (`pre_llm_call`) | 불가능 |
| 동의 (Consent) | `(이벤트, 명령어)` 쌍마다 처음 사용할 때 프롬프트 | 암시적 (Python 플러그인 신뢰) | 암시적 (디렉토리 신뢰) |
| 프로세스 간 격리 | 예 (하위 프로세스) | 아니요 (in-process) | 아니요 (in-process) |

### 구성 스키마 (Configuration schema)

```yaml
hooks:
  <event_name>:                  # VALID_HOOKS에 포함되어야 함
    - matcher: "<regex>"         # 선택사항; pre/post_tool_call 전용
      command: "<shell command>" # 필수; shlex.split, shell=False를 통해 실행됨
      timeout: <seconds>         # 선택사항; 기본값 60, 최대 300
      
hooks_auto_accept: false         # 아래의 "동의 모델(Consent model)" 참조
```

이벤트 이름은 [플러그인 훅 이벤트](#plugin-hooks) 중 하나여야 합니다; 오타가 나면 "X를 의미했습니까?"라는 경고가 발생하고 건너뜁니다. 단일 항목 내의 알 수 없는 키는 무시되며, `command`가 누락된 경우 경고와 함께 건너뜁니다. `timeout > 300`인 경우 경고와 함께 조정(clamp)됩니다.

### JSON 와이어 프로토콜 (JSON wire protocol)

이벤트가 발생할 때마다, Hermes는 (매처(matcher)가 허용하는) 모든 일치하는 훅에 대한 하위 프로세스를 생성하고, JSON 페이로드를 **stdin**으로 파이프한 다음, **stdout**을 다시 JSON으로 읽어옵니다.

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

도구 이벤트가 아닌 경우(`pre_llm_call`, `subagent_stop`, 세션 수명 주기), `tool_name` 및 `tool_input`은 `null`이 됩니다. `extra` 딕셔너리는 모든 이벤트별 kwargs(`user_message`, `conversation_history`, `child_role`, `duration_ms` 등)를 전달합니다. 직렬화할 수 없는 값은 생략되지 않고 문자열화됩니다.

**stdout — 선택적 응답:**

```jsonc
// pre_tool_call 차단 (두 형태 모두 허용되며 내부적으로 정규화됨):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code 스타일
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical 스타일

// pre_llm_call용 컨텍스트 주입:
{"context": "오늘은 2026-04-17 금요일입니다"}

// 아무 반응 없음(Silent no-op) — 빈 값이나 일치하지 않는 어떤 출력이든 상관없음:
```

잘못된 형식의 JSON, 0이 아닌 종료 코드 및 시간 초과는 경고를 기록하지만 절대 에이전트 루프를 중단시키지 않습니다.

### 작동 예시

#### 1. 변경(write)이 있을 때마다 Python 파일 자동 포맷팅

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

파일에 대한 에이전트의 컨텍스트 내 뷰는 자동으로 다시 읽혀지지 **않으며**, 리포맷팅은 디스크에 있는 파일에만 영향을 미칩니다. 이후 `read_file` 호출은 포맷된 버전을 가져옵니다.

#### 2. 파괴적인 `terminal` 명령어 차단

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
  printf '{"decision": "block", "reason": "차단됨: rm -rf / 은(는) 허용되지 않습니다"}\n'
else
  printf '{}\n'
fi
```

#### 3. 모든 턴에 `git status` 주입 (Claude-Code `UserPromptSubmit`의 대응 항목)

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # stdin 페이로드 삭제
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("cwd에 커밋되지 않은 변경 사항:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Code의 `UserPromptSubmit` 이벤트는 의도적으로 별도의 Hermes 이벤트로 분리되지 않습니다 — `pre_llm_call`이 같은 위치에서 실행되며 이미 컨텍스트 주입을 지원합니다. 이 훅을 사용하세요.

#### 4. 모든 하위 에이전트 종료 시 기록

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

### 동의 모델 (Consent model)

각각의 고유한 `(event, command)` 쌍은 Hermes가 이를 처음 볼 때 사용자에게 승인을 요청한 다음, 이 결정을 `~/.hermes/shell-hooks-allowlist.json`에 영구적으로 저장합니다. 후속 실행(CLI 또는 게이트웨이) 시에는 프롬프트를 건너뜁니다.

대화형 프롬프트를 우회할 수 있는 세 가지 탈출구(escape hatch)가 있으며, 이 중 하나면 충분합니다.

1. CLI에서의 `--accept-hooks` 플래그 (예: `hermes --accept-hooks chat`)
2. `HERMES_ACCEPT_HOOKS=1` 환경 변수
3. `cli-config.yaml`의 `hooks_auto_accept: true`

TTY가 아닌 실행 (게이트웨이, 크론(cron), CI) 환경에서는 이 세 가지 중 하나가 필요합니다. 그렇지 않으면 새로 추가된 훅이 등록되지 않고 조용히 남아 경고를 기록하게 됩니다.

**스크립트 편집은 조용히 신뢰됩니다.** 허용 목록 키는 스크립트의 해시가 아니라 정확한 명령어 문자열을 기반으로 하므로, 디스크에 있는 스크립트를 편집하더라도 동의가 무효화되지 않습니다. `hermes hooks doctor`는 mtime 드리프트를 플래그하여 편집을 확인하고 재승인 여부를 결정할 수 있게 해줍니다.

### `hermes hooks` CLI

| 명령어 | 수행하는 작업 |
|---------|--------------|
| `hermes hooks list` | 매처(matcher), 시간 초과 및 동의 상태와 함께 설정된 훅 덤프 (Dump) |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 가상 페이로드에 대해 일치하는 모든 훅을 실행하고 파싱된(parsed) 응답 출력 |
| `hermes hooks revoke <command>` | `<command>`와 일치하는 모든 허용 목록 항목을 제거 (다음 재시작 시 적용됨) |
| `hermes hooks doctor` | 모든 구성된 훅에 대해 다음 사항을 확인: 실행 비트(exec bit), 허용 목록 상태, mtime 드리프트, JSON 출력 유효성 및 대략적인 실행 시간 |

### 보안 (Security)

셸 훅은 크론(cron) 항목이나 셸 별칭(shell alias)과 동일한 신뢰 경계로 **귀하의 모든 사용자 자격 증명**으로 실행됩니다. `config.yaml`의 `hooks:` 블록을 중요한 설정으로 취급하세요.

- 자신이 작성했거나 완전히 검토한 스크립트만 참조하세요.
- 경로 감사가 용이하도록 스크립트를 `~/.hermes/agent-hooks/` 내부에 보관하세요.
- 공유 설정을 가져온(pull) 후 새로 추가된 훅이 등록되기 전에 `hermes hooks doctor`를 다시 실행하여 발견하세요.
- `config.yaml`이 팀 전체에서 버전 제어되는 경우 CI 설정을 검토하는 것과 동일한 방식으로 `hooks:` 섹션을 변경하는 PR을 검토하세요.

### 순서 및 우선순위

Python 플러그인 훅과 셸 훅 모두 동일한 `invoke_hook()` 디스패처를 통과합니다. Python 플러그인이 먼저 등록되고(`discover_and_load()`), 셸 훅이 나중에 등록되므로(`register_from_config()`), 넥타이 케이스에서는 Python `pre_tool_call`의 차단 결정이 우선합니다. 첫 번째 유효한 블록이 우선합니다 — 콜백이 비어 있지 않은 메시지와 함께 `{"action": "block", "message": str}`을 생성하는 즉시 집계자(aggregator)는 반환합니다.
