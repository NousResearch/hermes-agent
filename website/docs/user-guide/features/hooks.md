---
sidebar_position: 6
title: "Event Hooks"
description: "Run custom code at key lifecycle points — log activity, send alerts, post to webhooks"
---

# Event Hooks

Hermes has three hook systems that run custom code at key lifecycle points:

| System | Registered via | Runs in | Use case |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/` | Gateway only | Logging, alerts, webhooks |
| **[Plugin hooks](#plugin-hooks)** | `ctx.register_hook()` in a [plugin](/docs/user-guide/features/plugins) | CLI + Gateway | Tool interception, metrics, guardrails |
| **[Shell hooks](#shell-hooks)** | `hooks:` block in `~/.hermes/config.yaml` pointing at shell scripts | CLI + Gateway | Drop-in scripts for blocking, auto-formatting, context injection |

All three systems are non-blocking — errors in any hook are caught and logged, never crashing the agent.

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
- Two hooks' return values affect behavior: [`pre_tool_call`](#pre_tool_call) can **block** the tool, and [`pre_llm_call`](#pre_llm_call) can **inject context** into the LLM call. All other hooks are fire-and-forget observers.

### Quick reference

| Hook | Fires when | Returns |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | Before any tool executes | `{"action": "block", "message": str}` to veto the call |
| [`post_tool_call`](#post_tool_call) | After any tool returns | ignored |
| [`pre_llm_call`](#pre_llm_call) | Once per turn, before the tool-calling loop | `{"context": str}` to prepend context to the user message |
| [`post_llm_call`](#post_llm_call) | Once per turn, after the tool-calling loop | ignored |
| [`on_session_start`](#on_session_start) | New session created (first turn only) | ignored |
| [`on_session_end`](#on_session_end) | Session ends | ignored |
| [`on_session_finalize`](#on_session_finalize) | CLI/gateway tears down an active session (flush, save, stats) | ignored |
| [`on_session_reset`](#on_session_reset) | Gateway swaps in a fresh session key (e.g. `/new`, `/reset`) | ignored |
| [`subagent_stop`](#subagent_stop) | A `delegate_task` child has exited | ignored |
| [`subagent_batch_complete`](#subagent_batch_complete) | Every `delegate_task` call — fires once after all per-child `subagent_stop` firings, with an aggregate view of the batch | ignored |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | Gateway received a user message, before auth + dispatch | `{"action": "skip" \| "rewrite" \| "allow", ...}` to influence flow |

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

**Return value — veto the call:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
```

The agent short-circuits the tool with `message` as the error returned to the model. The first matching block directive wins (Python plugins registered first, then shell hooks). Any other return value is ignored, so existing observer-only callbacks keep working unchanged.

**Use cases:** Logging, audit trails, tool call counters, blocking dangerous operations, rate limiting, per-user policy enforcement.

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
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name of the tool that just executed |
| `args` | `dict` | The arguments the model passed to the tool |
| `result` | `str` | The tool's return value (always a JSON string) |
| `task_id` | `str` | Session/task identifier. Empty string if not set. |
| `duration_ms` | `int` | How long the tool's dispatch took, in milliseconds (measured with `time.monotonic()` around `registry.dispatch()`). |

**Fires:** In `model_tools.py`, inside `handle_function_call()`, after the tool's handler returns. Fires once per tool call. Does **not** fire if the tool raised an unhandled exception (the error is caught and returned as an error JSON string instead, and `post_tool_call` fires with that error string as `result`).

**Return value:** Ignored.

**Use cases:** Logging tool results, metrics collection, tracking tool success/failure rates, latency dashboards, per-tool budget alerts, sending notifications when specific tools complete.

**Example — track tool usage metrics:**

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

### `on_session_finalize`

Fires when the CLI or gateway **tears down** an active session — for example, when the user runs `/new`, the gateway GC'd an idle session, or the CLI quit with an active agent. This is the last chance to flush state tied to the outgoing session before its identity is gone.

**Callback signature:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` or `None` | The outgoing session ID. May be `None` if no active session existed. |
| `platform` | `str` | `"cli"` or the messaging platform name (`"telegram"`, `"discord"`, etc.). |

**Fires:** In `cli.py` (on `/new` / CLI exit) and `gateway/run.py` (when a session is reset or GC'd). Always paired with `on_session_reset` on the gateway side.

**Return value:** Ignored.

**Use cases:** Persist final session metrics before the session ID is discarded, close per-session resources, emit a final telemetry event, drain queued writes.

---

### `on_session_reset`

Fires when the gateway **swaps in a new session key** for an active chat — the user invoked `/new`, `/reset`, `/clear`, or the adapter picked a fresh session after an idle window. This lets plugins react to the fact that conversation state has been wiped without waiting for the next `on_session_start`.

**Callback signature:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The new session's ID (already rotated to the fresh value). |
| `platform` | `str` | The messaging platform name. |

**Fires:** In `gateway/run.py`, immediately after the new session key is allocated but before the next inbound message is processed. On the gateway, the order is: `on_session_finalize(old_id)` → swap → `on_session_reset(new_id)` → `on_session_start(new_id)` on the first inbound turn.

**Return value:** Ignored.

**Use cases:** Reset per-session caches keyed by `session_id`, emit "session rotated" analytics, prime a fresh state bucket.

---

See the **[Build a Plugin guide](/docs/guides/build-a-hermes-plugin)** for the full walkthrough including tool schemas, handlers, and advanced hook patterns.

---

### `subagent_stop`

Fires **once per child agent** after `delegate_task` finishes. Whether you delegated a single task or a batch of three, this hook fires once for each child, serialised on the parent thread.

**Callback signature:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_session_id` | `str` | Session ID of the delegating parent agent |
| `child_role` | `str \| None` | Orchestrator role tag set on the child (`None` if the feature isn't enabled) |
| `child_summary` | `str \| None` | The final response the child returned to the parent |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"`, `"error"`, or `"timeout"` |
| `duration_ms` | `int` | Wall-clock time spent running the child, in milliseconds |

**Fires:** In `tools/delegate_tool.py`, after `ThreadPoolExecutor.as_completed()` drains all child futures. Firing is marshalled to the parent thread so hook authors don't have to reason about concurrent callback execution.

**Return value:** Ignored.

**Use cases:** Logging orchestration activity, accumulating child durations for billing, writing post-delegation audit records.

**Example — log orchestrator activity:**

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
With heavy delegation (e.g. orchestrator roles × 5 leaves × nested depth), `subagent_stop` fires many times per turn. Keep your callback fast; push expensive work to a background queue.
:::

---

### `subagent_batch_complete`

Fires **once per `delegate_task` invocation**, after every per-child `subagent_stop` has fired. The payload aggregates the whole batch so you can "notify me when the swarm is done" without de-duping N per-child notifications. Single-task delegations still fire the event once — don't gate your logic on `child_count > 1` unless you explicitly want to skip the single-task case.

**Callback signature:**

```python
def my_callback(
    parent_session_id: str,
    child_count: int,
    completed_count: int,
    failed_count: int,
    errored_count: int,
    interrupted_count: int,
    timeout_count: int,
    total_duration_ms: int,
    children: list[dict],
    **kwargs,
):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_session_id` | `str` | Session ID of the delegating parent agent |
| `child_count` | `int` | Number of children scheduled (equals the length of `children`) |
| `completed_count` | `int` | Children that returned usable output (`status == "completed"`) |
| `failed_count` | `int` | Children that hit max iterations with no summary (`status == "failed"`) |
| `errored_count` | `int` | Children whose worker thread raised an uncaught exception (`status == "error"`) |
| `interrupted_count` | `int` | Children interrupted mid-run (`status == "interrupted"`) |
| `timeout_count` | `int` | Children terminated by the per-child hard timeout (`status == "timeout"`) |
| `total_duration_ms` | `int` | Wall-clock time from first child spawn to last child exit, including hook dispatch |
| `children` | `list[dict]` | Per-child `{"task_index", "role", "status", "duration_ms", "summary"}` records |

**Invariant:** `completed_count + failed_count + errored_count + interrupted_count + timeout_count == child_count`. No "unknown" bucket.

**Fires:** In `tools/delegate_tool.py`, on the parent thread, immediately after the per-child `subagent_stop` loop completes.

**Return value:** Ignored.

**Use cases:** Single-post Slack notifications for batch completion, swarm-level observability dashboards, accumulator-style billing or SLA tracking.

**Example — Slack notification when a batch finishes:**

```python
import logging
logger = logging.getLogger(__name__)

def notify_batch(parent_session_id, child_count, completed_count,
                 failed_count, errored_count, interrupted_count,
                 timeout_count, total_duration_ms, **kwargs):
    logger.info(
        "BATCH parent=%s n=%d ok=%d fail=%d err=%d interrupt=%d timeout=%d "
        "duration_ms=%d",
        parent_session_id, child_count, completed_count,
        failed_count, errored_count, interrupted_count,
        timeout_count, total_duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_batch_complete", notify_batch)
```

---

### `pre_gateway_dispatch`

Fires **once per incoming `MessageEvent`** in the gateway, after the internal-event guard but **before** auth/pairing and agent dispatch. This is the interception point for gateway-level message-flow policies (listen-only windows, human handover, per-chat routing, etc.) that don't fit cleanly into any single platform adapter.

**Callback signature:**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `event` | `MessageEvent` | The normalized inbound message (has `.text`, `.source`, `.message_id`, `.internal`, etc.). |
| `gateway` | `GatewayRunner` | The active gateway runner, so plugins can call `gateway.adapters[platform].send(...)` for side-channel replies (owner notifications, etc.). |
| `session_store` | `SessionStore` | For silent transcript ingestion via `session_store.append_to_transcript(...)`. |

**Fires:** In `gateway/run.py`, inside `GatewayRunner._handle_message()`, immediately after `is_internal` is computed. **Internal events skip the hook entirely** (they are system-generated — background-process completions, etc. — and must not be gate-kept by user-facing policy).

**Return value:** `None` or a dict. The first recognized action dict wins; remaining plugin results are ignored. Exceptions in plugin callbacks are caught and logged; the gateway always falls through to normal dispatch on error.

| Return | Effect |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | Drop the message — no agent reply, no pairing flow, no auth. Plugin is assumed to have handled it (e.g. silent-ingested into the transcript). |
| `{"action": "rewrite", "text": "new text"}` | Replace `event.text`, then continue normal dispatch with the modified event. Useful for collapsing buffered ambient messages into a single prompt. |
| `{"action": "allow"}` / `None` | Normal dispatch — runs the full auth / pairing / agent-loop chain. |

**Use cases:** Listen-only group chats (only respond when tagged; buffer ambient messages into context); human handover (silent-ingest customer messages while owner handles the chat manually); per-profile rate limiting; policy-driven routing.

**Example — drop unauthorized DMs silently without triggering the pairing code:**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**Example — rewrite an ambient-message buffer into a single prompt on mention:**

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

## Shell Hooks

Declare shell-script hooks in your `cli-config.yaml` and Hermes will run them as subprocesses whenever the corresponding plugin-hook event fires — in both CLI and gateway sessions. No Python plugin authoring required.

Use shell hooks when you want a drop-in, single-file script (Bash, Python, anything with a shebang) to:

- **Block a tool call** — reject dangerous `terminal` commands, enforce per-directory policies, require approval for destructive `write_file` / `patch` operations.
- **Run after a tool call** — auto-format Python or TypeScript files that the agent just wrote, log API calls, trigger a CI workflow.
- **Inject context into the next LLM turn** — prepend `git status` output, the current weekday, or retrieved documents to the user message (see [`pre_llm_call`](#pre_llm_call)).
- **Observe lifecycle events** — write a log line when a subagent completes (`subagent_stop`) or a session starts (`on_session_start`).

Shell hooks are registered by calling `agent.shell_hooks.register_from_config(cfg)` at both CLI startup (`hermes_cli/main.py`) and gateway startup (`gateway/run.py`). They compose naturally with Python plugin hooks — both flow through the same dispatcher.

### Comparison at a glance

| Dimension | Shell hooks | [Plugin hooks](#plugin-hooks) | [Gateway hooks](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| Declared in | `hooks:` block in `~/.hermes/config.yaml` | `register()` in a `plugin.yaml` plugin | `HOOK.yaml` + `handler.py` directory |
| Lives under | `~/.hermes/agent-hooks/` (by convention) | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| Language | Any (Bash, Python, Go binary, …) | Python only | Python only |
| Runs in | CLI + Gateway | CLI + Gateway | Gateway only |
| Events | `VALID_HOOKS` (incl. `subagent_stop`) | `VALID_HOOKS` | Gateway lifecycle (`gateway:startup`, `agent:*`, `command:*`) |
| Can block a tool call | Yes (`pre_tool_call`) | Yes (`pre_tool_call`) | No |
| Can inject LLM context | Yes (`pre_llm_call`) | Yes (`pre_llm_call`) | No |
| Consent | First-use prompt per `(event, command)` pair | Implicit (Python plugin trust) | Implicit (dir trust) |
| Inter-process isolation | Yes (subprocess) | No (in-process) | No (in-process) |

### Configuration schema

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300
      async: false               # Optional; see "Async hooks" below

hooks_auto_accept: false                          # See "Consent model" below
hooks_async_pool_size: 10                         # Range [1, 100]
hooks_async_shutdown_grace_seconds: 5             # Range [0, 60]
```

Event names must be one of the [plugin hook events](#plugin-hooks); typos produce a "Did you mean X?" warning and are skipped. Unknown keys inside a single entry are ignored; missing `command` is a skip-with-warning. `timeout > 300` is clamped with a warning.

### JSON wire protocol

Each time the event fires, Hermes spawns a subprocess for every matching hook (matcher permitting), pipes a JSON payload to **stdin**, and reads **stdout** back as JSON.

**stdin — payload the script receives:**

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

`tool_name` and `tool_input` are `null` for non-tool events (`pre_llm_call`, `subagent_stop`, session lifecycle). The `extra` dict carries all event-specific kwargs (`user_message`, `conversation_history`, `child_role`, `duration_ms`, …). Unserialisable values are stringified rather than omitted.

**stdout — optional response:**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Silent no-op — any empty / non-matching output is fine:
```

Malformed JSON, non-zero exit codes, and timeouts log a warning but never abort the agent loop.

### Worked examples

#### 1. Auto-format Python files after every write

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

The agent's in-context view of the file is **not** re-read automatically — the reformat only affects the file on disk. Subsequent `read_file` calls pick up the formatted version.

#### 2. Block destructive `terminal` commands

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

#### 3. Inject `git status` into every turn (Claude-Code `UserPromptSubmit` equivalent)

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

Claude Code's `UserPromptSubmit` event is intentionally not a separate Hermes event — `pre_llm_call` fires at the same place and already supports context injection. Use it here.

#### 4. Log every subagent completion

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

#### 5. Summarise a whole subagent batch in one Slack post

`subagent_batch_complete` fires exactly once per `delegate_task` call — whether the delegation fanned out to one worker or twenty — after every per-child `subagent_stop` has fired. Use it when you want "notify me when the whole swarm is done" without de-duping N per-child notifications.

```yaml
hooks:
  subagent_batch_complete:
    - command: "~/.hermes/agent-hooks/slack-swarm-summary.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/slack-swarm-summary.sh
payload=$(cat)
total=$(echo       "$payload" | jq -r '.extra.child_count')
ok=$(echo          "$payload" | jq -r '.extra.completed_count')
failed=$(echo      "$payload" | jq -r '.extra.failed_count + .extra.errored_count')
interrupted=$(echo "$payload" | jq -r '.extra.interrupted_count')
timed_out=$(echo   "$payload" | jq -r '.extra.timeout_count')
duration=$(echo    "$payload" | jq -r '(.extra.total_duration_ms / 1000)')
msg="Batch finished in ${duration}s: ${ok}/${total} ok, ${failed} failed, ${interrupted} interrupted, ${timed_out} timed out"
curl -sS -X POST -H 'Content-Type: application/json' \
     -d "{\"text\": \"$msg\"}" "$SLACK_WEBHOOK_URL"
echo '{}'
```

Contrast with `log-orchestration.sh` above: that fires **per child** via `subagent_stop`. `subagent_batch_complete` is the single-firing aggregate variant — each child's `status`, `duration_ms`, `role`, and (truncated) `summary` rides under `extra.children[]`, and the five counters (`completed_count`, `failed_count`, `errored_count`, `interrupted_count`, `timeout_count`) partition the full status keyspace so their sum always equals `extra.child_count`.

Don't gate your logic on `child_count > 1` unless you explicitly want to skip single-task delegations — single-task calls still fire the event once.

**Payload shape (stdin).** `parent_session_id` is surfaced as the top-level `session_id`; everything else lives under `extra`:

```json
{
  "hook_event_name": "subagent_batch_complete",
  "tool_name": null,
  "tool_input": null,
  "session_id": "parent-session-id",
  "cwd": "/home/user/project",
  "extra": {
    "child_count": 3,
    "completed_count": 2,
    "failed_count": 0,
    "errored_count": 0,
    "interrupted_count": 1,
    "timeout_count": 0,
    "total_duration_ms": 5000,
    "children": [
      {"task_index": 0, "role": null, "status": "completed",
       "duration_ms": 1200, "summary": "..."},
      {"task_index": 1, "role": null, "status": "completed",
       "duration_ms": 1800, "summary": "..."},
      {"task_index": 2, "role": null, "status": "interrupted",
       "duration_ms": 500, "summary": null}
    ]
  }
}
```

### Async hooks

Add `async: true` to a hook entry and Hermes runs the subprocess in a background thread pool; the agent loop doesn't wait. Useful when a hook posts to a webhook, hits an antivirus daemon, or writes to a remote audit log — work where the 100–500 ms round-trip shouldn't block every tool call.

```yaml
hooks:
  post_tool_call:
    - command: "~/.hermes/agent-hooks/audit-slack.sh"
      async: true
      timeout: 120    # background subprocess is killed if it exceeds this
```

Async hooks are fire-and-forget observers. The subprocess runs, its stdout is logged at `DEBUG`, and the return value is discarded — there's no Claude-Code-style "async output re-injection" yet.

**Three-tier compatibility.** Events are partitioned by what async semantics mean for them:

| Tier | Events | Behaviour |
|------|--------|-----------|
| ✅ Accepted | `post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`, `on_session_*`, `subagent_stop`, `subagent_batch_complete` | Registers silently; hook runs on a worker thread. |
| ⚠️ Warned | `pre_tool_call` | Registers with a warning. Any `{"action": "block"}` directive returned by the hook is **logged and discarded** because the tool has already executed by the time the subprocess completes. Use only for observational work (audit logs per tool invocation). |
| ❌ Rejected at config-parse | `pre_llm_call`, `transform_tool_result`, `transform_terminal_output` | `async: true` logs an `ERROR` and the entry is **not registered**. These events only propagate through their return value; async would make the hook 100% useless. |

Startup messages for the rejected tier are printed by `register_from_config`; check the CLI/gateway log if a hook you configured doesn't appear in `hermes hooks list`.

**Config keys.**

- `hooks_async_pool_size` (default `10`, clamped to `[1, 100]`) — maximum number of concurrent background subprocesses. A bounded semaphore gates submissions: when every slot is busy, additional firings are dropped with a `WARN` log rather than queued unboundedly. Bump this up if you see "pool saturated" warnings and your hardware can handle the fan-out.
- `hooks_async_shutdown_grace_seconds` (default `5`, clamped to `[0, 60]`) — time to wait for async subprocesses to exit after `SIGTERM` during shutdown. Survivors are `SIGKILL`'d. Raise if your hooks need longer to flush output, lower if you want Ctrl-C to come back faster.

**Backpressure is visible.** Under sustained load exceeding `hooks_async_pool_size`, firings are silently dropped (with a `WARN`). This is by design — unbounded queuing would accumulate thousands of pending firings under a chatty `post_tool_call` hook and then lose most of them at shutdown. For guaranteed delivery, either raise the pool size, use a sync hook, or write to a durable queue from inside the hook script itself.

**Consent is on the command, not sync-vs-async.** If you already approved `~/.hermes/agent-hooks/audit.sh` as a sync hook, flipping `async: true` in config doesn't re-prompt — same binary, same security surface. The allowlist keys on `(event, command)`, not on the sync/async flag.

**SIGINT and shutdown behaviour.**

- **CLI mode:** Ctrl-C terminates tracked subprocesses immediately via an installed SIGINT handler, then chains to the original handler so `KeyboardInterrupt` propagation still works. Expect sub-100 ms Ctrl-C latency even with hooks mid-flight.
- **Gateway mode:** SIGINT flows through the gateway's asyncio-native shutdown handler, which calls `shutdown_async_hooks(...)` as one teardown step. Worst-case latency is `hooks_async_shutdown_grace_seconds` (default 5s). Acceptable because gateway processes are rarely Ctrl-C'd interactively.

In either mode, async subprocesses that don't exit within the grace window are `SIGKILL`'d and may leave partial work. **Write idempotent async hooks** for external state changes (webhook posts, log appends). If the Python process is `SIGKILL`'d directly (no grace window runs), subprocesses become orphans owned by init — not recoverable at the process level.

**Claude-Code compatibility note.** Hermes declares async at **config time**, not at runtime. If a sync hook's stdout contains `{"async": true}` (the Claude-Code runtime marker), the bridge logs a warning and treats it as a no-op — lets users copy-paste Claude Code hook scripts without confusion but doesn't silently change semantics.

### Consent model

Each unique `(event, command)` pair prompts the user for approval the first time Hermes sees it, then persists the decision to `~/.hermes/shell-hooks-allowlist.json`. Subsequent runs (CLI or gateway) skip the prompt.

Three escape hatches bypass the interactive prompt — any one is sufficient:

1. `--accept-hooks` flag on the CLI (e.g. `hermes --accept-hooks chat`)
2. `HERMES_ACCEPT_HOOKS=1` environment variable
3. `hooks_auto_accept: true` in `cli-config.yaml`

Non-TTY runs (gateway, cron, CI) need one of these three — otherwise any newly-added hook silently stays un-registered and logs a warning.

**Script edits are silently trusted.** The allowlist keys on the exact command string, not the script's hash, so editing the script on disk does not invalidate consent. `hermes hooks doctor` flags mtime drift so you can spot edits and decide whether to re-approve.

### The `hermes hooks` CLI

| Command | What it does |
|---------|--------------|
| `hermes hooks list` | Dump configured hooks with matcher, timeout, and consent status |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | Fire every matching hook against a synthetic payload and print the parsed response |
| `hermes hooks revoke <command>` | Remove every allowlist entry matching `<command>` (takes effect on next restart) |
| `hermes hooks doctor` | For every configured hook: check exec bit, allowlist status, mtime drift, JSON output validity, and rough execution time |

### Security

Shell hooks run with **your full user credentials** — same trust boundary as a cron entry or a shell alias. Treat the `hooks:` block in `config.yaml` as privileged configuration:

- Only reference scripts you wrote or fully reviewed.
- Keep scripts inside `~/.hermes/agent-hooks/` so the path is easy to audit.
- Re-run `hermes hooks doctor` after you pull a shared config to spot newly-added hooks before they register.
- If your config.yaml is version-controlled across a team, review PRs that change the `hooks:` section the same way you'd review CI config.

### Ordering and precedence

Both Python plugin hooks and shell hooks flow through the same `invoke_hook()` dispatcher. Python plugins are registered first (`discover_and_load()`), shell hooks second (`register_from_config()`), so Python `pre_tool_call` block decisions take precedence in tie cases. The first valid block wins — the aggregator returns as soon as any callback produces `{"action": "block", "message": str}` with a non-empty message.
