---
sidebar_position: 6
title: "Event Hooks"
description: "Run custom code at key lifecycle points — log activity, send alerts, post to webhooks"
---

# Event Hooks

Hermes has two hook systems that run custom code at key lifecycle points:

| System | Registered via | Runs in | Use case |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/` | Gateway only | Logging, alerts, webhooks |
| **[Plugin hooks](#plugin-hooks)** | `ctx.register_hook()` in a [plugin](/docs/user-guide/features/plugins) | CLI + Gateway | Tool interception, metrics, guardrails |

Both systems are non-blocking — errors in any hook are caught and logged, never crashing the agent.

## Gateway Event Hooks

Gateway hooks fire automatically during gateway operation (Telegram, Discord, Slack, WhatsApp) without blocking the main agent pipeline.

### Creating a Hook

Each hook is a directory under `~/.hermes/hooks/` containing two files:

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

The `events` list determines which events trigger your handler. You can subscribe to any combination of events, including wildcards like `command:*`.

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

**Handler rules:**
- Must be named `handle`
- Receives `event_type` (string) and `context` (dict)
- Can be `async def` or regular `def` — both work
- Errors are caught and logged, never crashing the agent

### Available Events

| Event | When it fires | Context keys |
|-------|---------------|--------------|
| `gateway:startup` | Gateway process starts | `platforms` (list of active platform names) |
| `session:start` | New messaging session created | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | Session ended (before reset) | `platform`, `user_id`, `session_key` |
| `session:reset` | User ran `/new` or `/reset` | `platform`, `user_id`, `session_key` |
| `agent:start` | Agent begins processing a message | `platform`, `user_id`, `session_id`, `message` |
| `agent:step` | Each iteration of the tool-calling loop | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Agent finishes processing | `platform`, `user_id`, `session_id`, `message`, `response` |
| `command:*` | Any slash command executed | `platform`, `user_id`, `command`, `args` |

#### Wildcard Matching

Handlers registered for `command:*` fire for any `command:` event (`command:model`, `command:reset`, etc.). Monitor all slash commands with a single subscription.

### Examples

#### Boot Checklist (BOOT.md) — Built-in

The gateway ships with a built-in `boot-md` hook that looks for `~/.hermes/BOOT.md` on every startup. If the file exists, the agent runs its instructions in a background session. No installation needed — just create the file.

**Create `~/.hermes/BOOT.md`:**

```markdown
# Startup Checklist

1. Check if any cron jobs failed overnight — run `hermes cron list`
2. Send a message to Discord #general saying "Gateway restarted, all systems go"
3. Check if /opt/app/deploy.log has any errors from the last 24 hours
```

The agent runs these instructions in a background thread so it doesn't block gateway startup. If nothing needs attention, the agent replies with `[SILENT]` and no message is delivered.

:::tip
No BOOT.md? The hook silently skips — zero overhead. Create the file whenever you need startup automation, delete it when you don't.
:::

#### Telegram Alert on Long Tasks

Send yourself a message when the agent takes more than 10 steps:

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

#### Command Usage Logger

Track which slash commands are used:

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

#### Session Start Webhook

POST to an external service on new sessions:

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

### How It Works

1. On gateway startup, `HookRegistry.discover_and_load()` scans `~/.hermes/hooks/`
2. Each subdirectory with `HOOK.yaml` + `handler.py` is loaded dynamically
3. Handlers are registered for their declared events
4. At each lifecycle point, `hooks.emit()` fires all matching handlers
5. Errors in any handler are caught and logged — a broken hook never crashes the agent

:::info
Gateway hooks only fire in the **gateway** (Telegram, Discord, Slack, WhatsApp). The CLI does not load gateway hooks. For hooks that work everywhere, use [plugin hooks](#plugin-hooks).
:::

## Plugin Hooks

[Plugins](/docs/user-guide/features/plugins) can register hooks that fire in **both CLI and gateway** sessions. These are registered programmatically via `ctx.register_hook()` in your plugin's `register()` function.

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**General rules for all hooks:**

- Callbacks receive **keyword arguments**. Always accept `**kwargs` for forward compatibility — new parameters may be added in future versions without breaking your plugin.
- If a callback **crashes**, it's logged and skipped. Other hooks and the agent continue normally. A misbehaving plugin can never break the agent.
- All hooks are **fire-and-forget observers** whose return values are ignored — except `pre_llm_call`, which can [inject context](#pre_llm_call).

### Quick reference

| Hook | Fires when | Returns |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | Before any tool executes | ignored |
| [`post_tool_call`](#post_tool_call) | After any tool returns | ignored |
| [`pre_llm_call`](#pre_llm_call) | Once per turn, before the tool-calling loop | context injection |
| [`post_llm_call`](#post_llm_call) | Once per turn, after the tool-calling loop | ignored |
| [`on_session_start`](#on_session_start) | New session created (first turn only) | ignored |
| [`on_session_end`](#on_session_end) | Session ends | ignored |

---

### `pre_tool_call`

Fires **immediately before** every tool execution — built-in tools and plugin tools alike.

**Callback signature:**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool about to execute (e.g. `"terminal"`, `"web_search"`, `"read_file"`) |
| `args` | `dict` | The arguments the model passed to the tool |
| `task_id` | `str` | Session/task identifier. Empty string if not set. |

**Fires:** In `model_tools.py`, inside `handle_function_call()`, before the tool's handler runs. Fires once per tool call — if the model calls 3 tools in parallel, this fires 3 times.

**Return value:** Ignored.

**Use cases:** Logging, audit trails, tool call counters, blocking dangerous operations (print a warning), rate limiting.

**Example — tool call audit log:**

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

**Example — warn on dangerous tools:**

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

Fires **immediately after** every tool execution returns.

**Callback signature:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool that just executed |
| `args` | `dict` | The arguments the model passed to the tool |
| `result` | `str` | The tool's return value (always a JSON string) |
| `task_id` | `str` | Session/task identifier. Empty string if not set. |

**Fires:** In `model_tools.py`, inside `handle_function_call()`, after the tool's handler returns. Fires once per tool call. Does **not** fire if the tool raised an unhandled exception (the error is caught and returned as an error JSON string instead, and `post_tool_call` fires with that error string as `result`).

**Return value:** Ignored.

**Use cases:** Logging tool results, metrics collection, tracking tool success/failure rates, sending notifications when specific tools complete.

**Example — track tool usage metrics:**

```python
from collections import Counter
import json

_tool_counts = Counter()
_error_counts = Counter()

def track_metrics(tool_name, result, **kwargs):
    _tool_counts[tool_name] += 1
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

Fires **once per turn**, before the tool-calling loop begins. This is the **only hook whose return value is used** — it can inject context into the current turn's user message.

**Callback signature:**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `user_message` | `str` | The user's original message for this turn (before any skill injection) |
| `conversation_history` | `list` | Copy of the full message list (OpenAI format: `[{"role": "user", "content": "..."}]`) |
| `is_first_turn` | `bool` | `True` if this is the first turn of a new session, `False` on subsequent turns |
| `model` | `str` | The model identifier (e.g. `"anthropic/claude-sonnet-4.6"`) |
| `platform` | `str` | Where the session is running: `"cli"`, `"telegram"`, `"discord"`, etc. |

**Fires:** In `run_agent.py`, inside `run_conversation()`, after context compression but before the main `while` loop. Fires once per `run_conversation()` call (i.e. once per user turn), not once per API call within the tool loop.

**Return value:** If the callback returns a dict with a `"context"` key, or a plain non-empty string, the text is appended to the current turn's user message. Return `None` for no injection.

```python
# Inject context
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# Plain string (equivalent)
return "Recalled memories:\n- User likes Python"

# No injection
return None
```

**Where context is injected:** Always the **user message**, never the system prompt. This preserves the prompt cache — the system prompt stays identical across turns, so cached tokens are reused. The system prompt is Hermes's territory (model guidance, tool enforcement, personality, skills). Plugins contribute context alongside the user's input.

All injected context is **ephemeral** — added at API call time only. The original user message in the conversation history is never mutated, and nothing is persisted to the session database.

When **multiple plugins** return context, their outputs are joined with double newlines in plugin discovery order (alphabetical by directory name).

**Use cases:** Memory recall, RAG context injection, guardrails, per-turn analytics.

**Example — memory recall:**

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

**Example — guardrails:**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

Fires **once per turn**, after the tool-calling loop completes and the agent has produced a final response. Only fires on **successful** turns — does not fire if the turn was interrupted.

**Callback signature:**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `user_message` | `str` | The user's original message for this turn |
| `assistant_response` | `str` | The agent's final text response for this turn |
| `conversation_history` | `list` | Copy of the full message list after the turn completed |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Fires:** In `run_agent.py`, inside `run_conversation()`, after the tool loop exits with a final response. Guarded by `if final_response and not interrupted` — so it does **not** fire when the user interrupts mid-turn or the agent hits the iteration limit without producing a response.

**Return value:** Ignored.

**Use cases:** Syncing conversation data to an external memory system, computing response quality metrics, logging turn summaries, triggering follow-up actions.

**Example — sync to external memory:**

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

**Example — track response lengths:**

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

Fires **once** when a brand-new session is created. Does **not** fire on session continuation (when the user sends a second message in an existing session).

**Callback signature:**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the new session |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Fires:** In `run_agent.py`, inside `run_conversation()`, during the first turn of a new session — specifically after the system prompt is built but before the tool loop starts. The check is `if not conversation_history` (no prior messages = new session).

**Return value:** Ignored.

**Use cases:** Initializing session-scoped state, warming caches, registering the session with an external service, logging session starts.

**Example — initialize a session cache:**

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

Fires at the **very end** of every `run_conversation()` call, regardless of outcome. Also fires from the CLI's exit handler if the agent was mid-turn when the user quit.

**Callback signature:**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the session |
| `completed` | `bool` | `True` if the agent produced a final response, `False` otherwise |
| `interrupted` | `bool` | `True` if the turn was interrupted (user sent new message, `/stop`, or quit) |
| `model` | `str` | The model identifier |
| `platform` | `str` | Where the session is running |

**Fires:** In two places:
1. **`run_agent.py`** — at the end of every `run_conversation()` call, after all cleanup. Always fires, even if the turn errored.
2. **`cli.py`** — in the CLI's atexit handler, but **only** if the agent was mid-turn (`_agent_running=True`) when the exit occurred. This catches Ctrl+C and `/exit` during processing. In this case, `completed=False` and `interrupted=True`.

**Return value:** Ignored.

**Use cases:** Flushing buffers, closing connections, persisting session state, logging session duration, cleanup of resources initialized in `on_session_start`.

**Example — flush and cleanup:**

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

**Example — session duration tracking:**

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

See the **[Build a Plugin guide](/docs/guides/build-a-hermes-plugin)** for the full walkthrough including tool schemas, handlers, and advanced hook patterns.
