---
sidebar_position: 6
title: "Event Hooks"
description: "Run custom code at key lifecycle points — log activity, send alerts, post to webhooks"
---

# Event Hooks

Hermes has four hook systems that run custom code at key lifecycle points:

| System | Registered via | Runs in | Use case |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/` | Gateway only | Logging, alerts, webhooks |
| **[Plugin hooks](#plugin-hooks)** | `ctx.register_hook()` in a [plugin](/user-guide/features/plugins) | CLI + Gateway | Tool interception, metrics, guardrails |
| **[Shell hooks](#shell-hooks)** | `hooks:` block in `~/.hermes/config.yaml` pointing at shell scripts | CLI + Gateway | Drop-in scripts for blocking, auto-formatting, context injection |
| **[Outbound webhooks](#outbound-webhooks)** | `hooks.outbound:` list in `~/.hermes/config.yaml` | CLI + Gateway | Push signed lifecycle events to external HTTP endpoints — CI, dashboards, other agents |

Hook callback errors are isolated and logged rather than crashing the agent. Hooks are not all passive: directive/control hooks can change flow, transforms can replace content, and a shell `pre_tool_call` hook can block or fail closed.

## Gateway Event Hooks

Gateway hooks fire automatically during gateway operation (Telegram, Discord, Slack, WhatsApp, Teams) without blocking the main agent pipeline.

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
| `session:compress` | Context compression completed for a session | `platform`, `session_id`, `old_session_id` (empty when compacted in place), `in_place` (bool — `true` = transcript compacted on the same id, `false` = rotated from `old_session_id`), `compression_count` |
| `agent:start` | Agent begins processing a message | `platform`, `user_id`, `chat_id`, `thread_id` (forum-topic / thread root id; empty when not in a thread), `chat_type` (`"dm"` \| `"group"` \| `"forum"`; empty if unknown), `session_id`, `message` (truncated to 500 chars) |
| `agent:step` | Each iteration of the tool-calling loop | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Agent finishes processing | same keys as `agent:start`, plus `response` (truncated to 500 chars) |
| `reaction:added` | An emoji reaction was added to a message the bot can see (Slack adapter currently). Requires the `reactions:read` scope + the `reaction_added` bot event subscription; the bot must be a member of the channel. | `platform`, `reaction`, `user_id`, `item_user_id`, `item_type`, `channel_id`, `message_ts`, `team_id`, `event_ts`, `raw_event` |
| `reaction:removed` | An emoji reaction was removed from a message the bot can see. Requires the `reaction_removed` bot event subscription. | same shape as `reaction:added` |
| `command:*` | Any slash command executed | `platform`, `user_id`, `command`, `args` |

#### Wildcard Matching

Handlers registered for `command:*` fire for any `command:` event (`command:model`, `command:reset`, etc.). Monitor all slash commands with a single subscription.

:::tip Threaded replies
A handler posting a follow-up message into the same Telegram forum topic should include `message_thread_id=int(thread_id)` when `chat_type == "forum"` and `thread_id` is non-empty.
:::

### Examples

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

### Tutorial: BOOT.md — Run a Startup Checklist on Every Gateway Boot

A popular pattern from the community: drop a Markdown checklist at `~/.hermes/BOOT.md`, and have the agent run it once every time the gateway starts. Useful for "on every boot, check overnight cron failures and ping me on Discord if anything failed," or "summarize the last 24h of deploy.log and post it to Slack #ops."

This tutorial shows how to build it yourself as a user-defined hook. Hermes does not ship a built-in BOOT.md hook — you wire up exactly the behavior you want.

#### What we're building

1. A file at `~/.hermes/BOOT.md` with natural-language startup instructions.
2. A gateway hook that fires on `gateway:startup`, spawns a one-shot agent with your gateway's resolved model/credentials, and runs the BOOT.md instructions.
3. A `[SILENT]` convention so the agent can opt out of sending a message when there's nothing to report.

#### Step 1: Write your checklist

Create `~/.hermes/BOOT.md`. Write it as if you were giving instructions to a human assistant:

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, summarize them for Discord #ops (the hook delivers your final response to its configured target).
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same report.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

The agent sees this as part of its prompt, so anything you can describe in plain language works — tool calls, shell commands, sending messages, summarizing files.

#### Step 2: Create the hook

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

The two key lines:

- `_resolve_gateway_model()` reads the gateway's currently-configured model.
- `_resolve_runtime_agent_kwargs()` resolves provider credentials the same way a normal gateway turn does — including API keys, base URLs, OAuth tokens, and credential pools.

Without these, a bare `AIAgent()` falls back to built-in defaults and will 401 against any non-default endpoint.

#### Step 3: Test it

Restart the gateway:

```bash
hermes gateway restart
```

Watch the logs:

```bash
hermes logs --follow --level INFO | grep boot-md
```

You should see `Running BOOT.md (N chars)` followed by either `boot-md completed: ...` (summary of what the agent did) or `boot-md completed (nothing to report)` when the agent replied with an exact silence token such as `[SILENT]`.

Delete `~/.hermes/BOOT.md` to disable the checklist — the hook stays loaded but silently skips when the file isn't there.

#### Extending the pattern

- **Schedule-aware checklists:** key off `datetime.now().weekday()` inside BOOT.md's instructions ("if it's Monday, also check the weekly deploy log"). The instructions are free-form text, so anything the agent can reason about is fair game.
- **Multiple checklists:** point the hook at a different file (`STARTUP.md`, `MORNING.md`, etc.) and register separate hook directories for each.
- **Non-agent variant:** if you don't need a full agent loop, skip `AIAgent` entirely and have the handler post a fixed notification directly via `httpx`. Cheaper, faster, and has no provider dependency.

#### Why this isn't a built-in

An earlier version of Hermes shipped this as a built-in hook and silently spawned an agent with bare defaults on every gateway boot. That surprised users with custom endpoints and made the feature invisible to users who didn't know it was running. Keeping it as a documented pattern — built by you, in your hooks directory — means you see exactly what it does and opt in by writing the files.

### How It Works

1. On gateway startup, `HookRegistry.discover_and_load()` scans `~/.hermes/hooks/`
2. Each subdirectory with `HOOK.yaml` + `handler.py` is loaded dynamically
3. Handlers are registered for their declared events
4. At each lifecycle point, `hooks.emit()` fires all matching handlers
5. Errors in any handler are caught and logged — a broken hook never crashes the agent

:::info
Gateway hooks only fire in the **gateway** (Telegram, Discord, Slack, WhatsApp, Teams). The CLI does not load gateway hooks. For hooks that work everywhere, use [plugin hooks](#plugin-hooks).
:::

## Plugin Hooks

[Plugins](/user-guide/features/plugins) can register hooks that fire in **both CLI and gateway** sessions. These are registered programmatically via `ctx.register_hook()` in your plugin's `register()` function.

For plugin packaging and registration details, see
the [Plugins guide](/docs/user-guide/features/plugins).

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

**General rules for all hooks:**

- Callbacks receive **keyword arguments**. Always accept `**kwargs` for forward compatibility.
- Callback exceptions are logged and skipped; later callbacks continue.
- If a Python plugin callback on a **timeout-bounded** hook (hot-path observers such as `post_tool_call` / `pre_llm_call`, plus the policy hook `pre_tool_call`) **blocks** longer than `plugins.hook_callback_timeout` (default 30s, set `0` to disable, max 600), it is abandoned without joining the worker so the agent loop continues. Timed-out or still-running `pre_tool_call` callbacks **fail closed** (block the tool); other bounded hooks fail open (skip). Hooks with a documented caller-thread contract (`subagent_stop`) are never moved onto a timeout worker. Shell hooks keep their own per-entry `timeout`.
- The catalog below is descriptive: **observers** ignore returns, **transforms** accept the first valid string replacement, and **directive/control** hooks consume documented return shapes. Plugin middleware is a separate registry and surface, not another hook category.
- Correlation fields such as `turn_id`, `api_request_id`, `task_id`, `session_id`, and `api_call_count` are hook-specific and may be absent. Treat IDs as opaque.
- Runtime event-name validity comes from `hermes_cli.plugins.VALID_HOOKS`. `hermes hooks list` lists configured shell/outbound hooks, not every available event; `hermes hooks test <event>` reports the valid set only when an invalid event is supplied.

### Cache-safe system prompt sections

Plugins that need durable, always-on guidance can register a bounded system
prompt section instead of injecting the same text through `pre_llm_call` on
every turn:

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

The contract is deliberately narrow:

- IDs are global, stable, 1–128 character lowercase identifiers using only
  letters, numbers, `.`, `_`, and `-`. Duplicate IDs are rejected.
- `after_memory` is the only placement anchor. Sections are sorted by ID,
  rendered after memory/profile context and before session metadata; plugins
  cannot reorder or replace core prompt content.
- A callable receives a read-only mapping with `session_id`, `model`,
  `provider`, `platform`, `profile_name`, and `cwd`. It runs **once for a new
  session**. Its rendered bytes are frozen on compression and recovered from
  the already-persisted full system prompt after a process restart/resume;
  plugin state is not re-read for an existing session.
- `max_chars` is capped at 4,000 characters. All plugin sections together,
  including their audit headings, are capped at 8,000 characters and 32
  sections. Empty, non-string, oversized, aggregate-over-budget, or raising
  sections are skipped with a warning; prompt construction continues.
- Every accepted section is named in the prompt and logged at session start
  with its plugin, position, and character count.

Use `pre_llm_call` for truly dynamic per-turn context. There is intentionally
no plugin environment-hints hook in this contract: changing cwd, branch, or
other environment data must not silently mutate a session's cached prompt.
Such a hook needs a concrete consumer and the same frozen/resume-safe semantics
before it can be added.

### Shipped plugin-hook catalog

Payload fields below are the exact event-specific fields supplied by each call site. For backward compatibility, `PluginManager` also adds `telemetry_schema_version="hermes.observer.v1"` to every plugin-hook callback. That legacy envelope marker does not mean all hook payloads share one semantic schema; new versioned contracts belong to their concrete event or capability family.

| Hook | Category | Exact timing and return behavior | Explicit payload fields | Privacy / sensitivity |
|---|---|---|---|---|
| [`pre_tool_call`](#pre_tool_call) | Directive/control | Once before execution; first valid `block` or `approve` directive wins, and `modify` returns are shallow-merged into the tool arguments. | `tool_name`, `args`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace` | Raw arguments may contain user content, paths, commands, or secrets. |
| `post_tool_call` | Observer | After blocked, error, or successful result; return ignored. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message`, `middleware_trace` | Result/error text may contain arbitrary tool or user content and secrets. |
| `transform_tool_result` | Transform | After `post_tool_call`, before conversation append; first string replaces the result. | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message` | Exposes the full model-bound result and arguments. |
| `transform_terminal_output` | Transform | After bounded foreground process capture, before final output limiting; first string replaces output. | `command`, `output`, `returncode`, `task_id`, `env_type` | Command/output may contain credentials. |
| `pre_transcription` | Transform | Fired by the STT dispatcher after provider resolution and before any backend (built-in, command-type, or plugin-registered) is invoked; dict results are applied in registration order, last-writer-wins per field (`prompt`, `language`, `model`; `file_path` is read-only). | `file_path`, `provider`, `model`, `language`, `prompt`, `source` | The final prompt is uploaded to the configured STT provider with the audio — keep secrets out of hook returns. |
| `pre_llm_call` | Directive/control | Once per turn before the loop; all valid string/`{"context": ...}` returns are joined and injected into the user message. | `session_id`, `task_id`, `turn_id`, `user_message`, `conversation_history`, `is_first_turn`, `model`, `platform`, `parent_session_id`, `sender_id` | Full user message and conversation history. |
| `post_llm_call` | Observer | Successful, non-interrupted turn finalization; return ignored. | `session_id`, `task_id`, `turn_id`, `user_message`, `assistant_response`, `conversation_history`, `model`, `platform` | Full prompt, response, and history. |
| `transform_llm_output` | Transform | Before `post_llm_call` and final delivery; first non-empty string replaces the response. | `response_text`, `session_id`, `model`, `platform` | Full final assistant text. |
| `pre_verify` | Directive/control | At the bounded edited-code verify gate; first valid continue/block-stop directive keeps the turn going. | `session_id`, `platform`, `model`, `coding`, `attempt`, `final_response`, `changed_paths` | Draft response and changed paths. |
| `pre_api_request` | Observer | Per provider attempt, immediately before the request; return ignored. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `user_message`, `conversation_history`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `retry_count`, `request_messages`, `message_count`, `tool_count`, `approx_input_tokens`, `request_char_count`, `max_tokens`, `started_at`, `middleware_trace`, `request` | High sensitivity: legacy `user_message`, `conversation_history`, and `request_messages` are intentionally raw; prefer sanitized `request`. |
| `post_api_request` | Observer | After normalized provider success; return ignored. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `finish_reason`, `message_count`, `response_model`, `response`, `usage`, `assistant_message`, `assistant_content_chars`, `assistant_tool_call_count` | Sanitized `response` is available, but raw normalized `assistant_message` may contain model/user content; `usage` is accounting data. |
| `api_request_error` | Observer | On each failed provider attempt; return ignored. | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `status_code`, `retry_count`, `max_retries`, `retryable`, `reason`, `error`, `request` | Error text may contain provider/user data; `request` is intended to be sanitized. |
| `on_stream_start` | Observer | Dispatched when a streaming LLM response begins; delivered off the token path via a host-owned bounded queue with one worker per callback; return ignored. | `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Identifiers and routing metadata only. |
| `on_stream_delta` | Observer | Dispatched per normalized streaming text delta via the bounded observer queue; a stalled callback drops only its own oldest events; return ignored. | `delta`, `kind` (`text` or `reasoning`), `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Delta text is raw model output; reasoning deltas require the `plugins.stream_reasoning_deltas` opt-in. |
| `on_stream_end` | Observer | Dispatched when a streaming response finishes or errors, after the stream closes; return ignored. | `final_text`, `finished`, `error`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Full assembled response text; error text may include provider data. |
| `on_interim_message` | Observer | Dispatched when a mid-loop assistant message is surfaced before the final answer (streaming or non-streaming); return ignored. | `text`, `already_streamed`, `turn_id`, `iteration`, `session_id`, `model`, `provider`, `surface` | Full interim assistant text. |
| `transform_api_error_classification` | Transform | On each failed provider attempt, at the top of the built-in classifier; all callbacks run, then the first dict with a valid `reason` wins (run-all-then-pick-first), and skipped valid results log a runtime warning. Python plugins only. | `provider`, `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages` | `error_message` and `error_body` may contain raw provider/user data. |
| `on_session_start` | Observer | First turn of a new session; return ignored. | `session_id`, `model`, `platform` | Identifiers and routing metadata only. |
| `on_session_end` | Observer | Canonically at each turn finalization; CLI/TUI exits have additional reduced legacy shapes. Return ignored. | Canonical: `session_id`, `task_id`, `turn_id`, `completed`, `failed`, `interrupted`, `turn_exit_reason`, `model`, `platform`; exit paths may add `reason`/`api_request_id` and omit fields. | IDs, model/platform, and outcome; canonical payload has no message body. |
| `on_session_finalize` | Observer | CLI/TUI/gateway teardown through `finalize_session`; gateway shutdown or expiry may finalize without a reset. Return ignored. | Surface-dependent `session_id`, `platform`, optionally `reason`, `old_session_id`, `new_session_id` | Session and routing identifiers. |
| `on_session_reset` | Observer | CLI/TUI session boundary and gateway after the replacement session exists; return ignored. | CLI: `session_id`, `platform`, `reason`; TUI: `session_id`, `platform`; gateway: those plus `reason`, `old_session_id`, `new_session_id` | Session and routing identifiers. |
| `on_skill_lifecycle` | Observer | After an authoritative skill-usage state change; return ignored. | `action`, `skill_name`, `provenance`, `task_id`, `session_id`, `use_count`, `reused`, `reuse_after_patch` | Exposes the local skill name and provenance. |
| `subagent_start` | Observer | Child constructed and about to run; return ignored. | `parent_session_id`, `parent_turn_id`, `parent_subagent_id`, `child_session_id`, `child_subagent_id`, `child_role`, `child_goal` | Child goal may contain user/project content. |
| `subagent_stop` | Observer | Child exit; return ignored. | `parent_session_id`, `parent_turn_id`, `child_session_id`, `child_role`, `child_summary`, `child_status`, `tool_call_history`, `duration_ms` | Summary and redacted tool-history metadata may reveal project structure. |
| `pre_gateway_dispatch` | Directive/control | Incoming non-internal message before auth/pairing/dispatch; first valid `skip`, `rewrite`, `allow`, or `authorize` controls flow. | `event`, `gateway`, `session_store` | Extremely privileged in-process objects expose inbound user/routing data and host handles. |
| `gateway_platform_event` | Observer | After the gateway's profile-scoped authorization succeeds, when a supported platform-native event is normalized at the gateway boundary (Telegram: reactions, message edits; Discord: message edits/deletes, thread created/renamed); return ignored. | `platform`, `event_type`, `payload` (event-type-specific dict — see the per-event contracts below) | Normalized plain-dict envelope only; raw SDK objects, adapter handles, and bot clients are never exposed. |
| `pre_command` | Observer | Recognized slash command about to be dispatched, before the handler runs, on CLI and gateway cold-path dispatch; return ignored in v1 (directive-shaped dicts are logged at debug). Gateway running-agent intercept commands (`/stop`, `/approve` during an active run) are deliberately excluded — control-plane escape hatches must stay outside plugin reach. | `surface` (`"cli"` \| `"gateway"`), `command` (canonical name), `alias_used`, `args_raw`, `session_key`, `platform` | `args_raw` may contain user content or secrets typed after the command. |
| `pre_approval_request` | Observer | Before prompted or smart approval; return ignored. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id` | Command may contain secrets; smart observer preparation force-redacts, but surfaces do not all have identical redaction. |
| `post_approval_response` | Observer | After a decision, timeout, or gateway notification failure; return ignored. | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id`, `choice`; smart path may add `decided_by` | Same command sensitivity plus decision metadata. |
| `kanban_task_claimed` | Observer | After claim commit, in dispatcher process before worker spawn; return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id` | Board/task/profile/assignee identifiers. |
| `kanban_task_completed` | Observer | After completion and cleanup, usually in worker process; return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `summary` | Summary may contain project/user content. |
| `kanban_task_blocked` | Observer | After a blocked transition; the dependency-wait path fires before its transaction exits. Return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `reason` | Reason may contain project/user content. |
| `on_kanban_worker_spawned` | Observer | After `spawn_fn` returns and the worker PID is persisted; runs inside the dispatch lock, keep callbacks fast. Return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `workspace_path` | `workspace_path` is a filesystem path and may reveal project layout or usernames. |
| `on_kanban_worker_exited` | Observer | Tick-derived: after `detect_crashed_workers` reclaims a dead-PID task and the reclaim commits. Return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status` | Identifiers and exit metadata only. |
| `on_kanban_worker_stale_claim` | Observer | After a TTL-expired claim is reclaimed; live-PID extensions don't fire. Return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `worker_pid`, `heartbeat_stale`, `retry_status` | Identifiers and claim metadata only. |
| `on_kanban_task_updated` | Observer | After a committed task-field write outside the claim/complete/block lifecycle (assign, overrides, dashboard editors). Return ignored. | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `changed_fields` | `changed_fields` carries field names only, never values; the named title/body values in the board DB may contain user/project content. |
| `on_kanban_dispatch_tick` | Observer | Once per dispatcher tick, strictly after the dispatch lock is released; idle and contended ticks fire too. Return ignored. | `board`, `profile_name`, `dry_run`, `outcome`, `result` | `result` is the tick's `DispatchResult` and carries task ids, assignees, and workspace paths. |

---

### Streaming output hooks

These observer-only hooks let plugins consume streaming LLM output for telemetry, live dashboards, or TTS pipelines without changing the response. They are delivered through host-owned bounded queues with one background worker per registered callback, so plugin callbacks never run inline on the token path. If one callback stalls, only that callback's queue can fill and drop its oldest pending observer event; other observers continue receiving events independently.

Register them like any other plugin hook:

```python
def on_delta(delta, kind, model, provider, **kwargs):
    if kind == "text":
        print(delta, end="", flush=True)

def register(ctx):
    ctx.register_hook("on_stream_delta", on_delta)
```

Common fields for all four hooks:

| Parameter | Type | Description |
|-----------|------|-------------|
| `turn_id` | `str` | Opaque turn identifier, when available |
| `iteration` | `int` | Current API-call/tool-loop iteration |
| `session_id` | `str` | Current Hermes session id |
| `model` | `str` | Active model identifier |
| `provider` | `str` | Active provider name |
| `surface` | `str` | Calling surface, e.g. `cli`, `discord`, `telegram` |

Additional fields:

| Hook | Extra fields |
|------|--------------|
| `on_stream_start` | none |
| `on_stream_delta` | `delta: str`, `kind: "text" | "reasoning"` |
| `on_stream_end` | `final_text: str`, `finished: bool`, `error: str | None` |
| `on_interim_message` | `text: str`, `already_streamed: bool` |

`on_interim_message` can also fire after a non-streaming response, so registering only that hook does not force a provider call onto streaming transport.

Reasoning deltas are not exposed to plugins by default. Opt in explicitly:

```yaml
plugins:
  stream_reasoning_deltas: true
```

Return values are ignored. To keep the stream fast, callbacks should enqueue their own work and return quickly. Exceptions are logged and do not stop the stream.

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

**Return value — block or require approval:**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
# or
return {"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}
```

The first valid directive wins (Python plugins registered first, then shell hooks). `block` requires a non-empty `message` and short-circuits the tool with that text as the error returned to the model. `approve` escalates the call to the existing human-approval gate; `message` and `rule_key` are optional, and denial, timeout, or gate error fails closed. Other return values are ignored, so existing observer-only callbacks keep working unchanged.

**Return value — rewrite the tool's arguments:**

```python
return {"action": "modify", "args": {"new_string": "fixed content"}}
```

The returned `args` dictionary is shallow-merged over the original tool arguments before the tool executes. Multiple `modify` hooks accumulate — each hook's keys are merged into one accumulated dict built from the original args, so hook A changing `path` and hook B changing `content` both survive. If two hooks modify the same key, the later hook wins.

Shell hooks also accept the Claude Code-compatible format:

```json
{"decision": "modify", "tool_input": {"new_string": "fixed content"}}
```

Both formats are normalized internally to `{"action": "modify", "args": {...}}`.

If a `pre_tool_call` callback exceeds `plugins.hook_callback_timeout` (or is still running from a previous timed-out fire), Hermes **fails closed**: the tool is blocked with a timeout message rather than proceeding without a policy decision.

**Use cases:** Logging, audit trails, tool call counters, blocking dangerous operations, rate limiting, per-user policy enforcement, argument sanitization, path rewriting, injecting default parameters.

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

Fires **once per turn**, before the tool-calling loop begins. All valid callback returns are aggregated in plugin order and injected into the current turn's user message.

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

The clean user-message `content` remains unchanged. For replay and prompt-cache stability, Hermes may persist the exact API-bound message, including plugin-injected context, in the row's `api_content` sidecar.

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

### `pre_verify`

Fires **once per turn when the agent edited code**, just before it finishes (after the built-in verify-on-stop guard). This is a user/plugin policy gate: a callback can keep the agent going — run a check, defer it, tidy the diff — instead of letting it stop.

Hermes' shipped verification guidance is not a default `pre_verify` hook. It is appended to the evidence-based verify-on-stop nudge when edited code lacks fresh verification evidence, so it does not create a second default continuation path. Set `agent.verify_guidance: false` to keep that built-in evidence nudge terse.

**Callback signature:**

```python
def my_callback(session_id: str, platform: str, model: str, coding: bool,
                attempt: int, final_response: str, changed_paths: list, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Unique identifier for the current session |
| `platform` | `str` | Where the session is running (`"cli"`, `"telegram"`, …) |
| `model` | `str` | The model identifier |
| `coding` | `bool` | Whether the turn is in the coding posture (in a code workspace) — scope your hook on this |
| `attempt` | `int` | How many times this turn has already been nudged (0 on the first) — self-throttle on this |
| `final_response` | `str` | The answer the agent is about to deliver |
| `changed_paths` | `list` | Files the agent edited this turn (sorted, always non-empty here) |

Scope a hook to the coding context by checking `coding` and make it one-shot with `attempt` (shell hooks read both from `.extra`), the same way a `pre_tool_call` hook scopes on `tool_name` — so you can register several `pre_verify` hooks, each firing only where it should.

**Fires:** In `agent/conversation_loop.py`, at the point the agent would accept a final answer, immediately after the verify-on-stop check — but only when the agent edited code this turn and at least one `pre_verify` hook is registered.

**Return value — keep the agent going:**

```python
return {"action": "continue", "message": "Run the formatter on your changes, then finish."}
```

The `message` is appended as a synthetic user turn and the loop runs again. The Claude-Code Stop shape (`{"decision": "block", "reason": "..."}`, where blocking the stop means *keep going*) is accepted too. A directive with no message — or any other return — lets the turn finish.

**Bounded:** consecutive continue directives in one turn are capped by `agent.max_verify_nudges` (default 3), so a hook that always says continue can never trap the loop. The attempted answer is kept in history but not surfaced to the user while the agent is being nudged.

**Make it idempotent:** the hook re-fires after each nudge, so gate on `attempt` (`if attempt: return None`) — otherwise it just nudges until the bound is hit.

**Use cases:** defer tests/lints during creative iteration, require green checks for certain paths, block "done" until a changelog entry exists, run a project-specific verification checklist.

**Example — defer checks on creative UI work, scoped + one-shot:**

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

For standing guidance that should shape the built-in missing-evidence nudge, use `agent.verify_guidance`. For broader coding posture rules that don't need to *gate* verification, prefer `agent.coding_instructions` in `config.yaml` — it rides the coding brief and costs no extra turn.

---

### `transform_api_error_classification`

Fires once per failed API call, at the top of `agent/error_classifier.classify_api_error()`, before the built-in pipeline. Provider plugins use it to own their provider's error quirks without core patches. It is behavior-changing (transform family): the returned classification drives retry, compression, credential rotation, and fallback routing.

Callbacks receive the parsed error context as kwargs — `provider` (self-scope on this), `model`, `status_code`, `error_type`, `error_code`, `error_message`, `error_body`, `error`, `approx_tokens`, `context_length`, `num_messages`. Return `None` to decline, or a dict to claim the error:

```python
return {"reason": "model_not_found",   # required: a FailoverReason name
        "retryable": False, "should_fallback": True}  # optional recovery-hint overrides
```

Dispatch is run-all-then-pick-first: every callback runs, failures are isolated, and the first valid result in registration order wins (valid-but-losing results log a runtime warning). Invalid dicts and unknown reasons are skipped, so a broken plugin can never break classification.

**Privacy:** `error_message` and `error_body` may carry unredacted provider data. **Python plugins only** — shell registrations are refused at config parse with a warning.

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

Fires when the CLI or gateway **tears down** an active session — for example, when the user runs `/new`, the gateway GC'd an idle session, or the CLI quit with an active agent. Use it to flush state tied to the outgoing session ID. On gateway reset, the replacement session already exists before this callback runs.

**Callback signature:**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` or `None` | The outgoing session ID. May be `None` if no active session existed. |
| `platform` | `str` | `"cli"` or the messaging platform name (`"telegram"`, `"discord"`, etc.). |

**Fires:** In CLI/TUI teardown and in gateway reset, shutdown, or idle-expiry paths. Gateway shutdown and expiry can finalize without a matching `on_session_reset`.

**Return value:** Ignored.

**Use cases:** Persist final session metrics before the session ID is discarded, close per-session resources, emit a final telemetry event, drain queued writes.

---

### `on_session_reset`

Fires at a CLI or TUI session boundary, or when the gateway **swaps in a new session key** for an active chat. This lets plugins react to cleared conversation state without waiting for the next `on_session_start`.

**Callback signature:**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | The new session's ID (already rotated to the fresh value). |
| `platform` | `str` | `"cli"`, `"tui"`, or the messaging platform name. |
| `reason` | `str`, optional | Present on CLI and gateway reset paths. |
| `old_session_id` | `str`, optional | Gateway-only outgoing session ID. |
| `new_session_id` | `str`, optional | Gateway-only replacement session ID. |

**Fires:** CLI supplies `session_id`, `platform`, and `reason`; TUI supplies `session_id` and `platform`; gateway adds `reason`, `old_session_id`, and `new_session_id` after allocating the replacement key. On gateway reset, the order is: create and persist the replacement → `on_session_finalize(old_id)` → `on_session_reset(new_id)` → `on_session_start(new_id)` on the first inbound turn.

**Return value:** Ignored.

**Use cases:** Reset per-session caches keyed by `session_id`, emit "session rotated" analytics, prime a fresh state bucket.

---

See the **[Build a Plugin guide](/developer-guide/plugins)** for the full walkthrough including tool schemas, handlers, and advanced hook patterns.

---

### `subagent_start`

Fires **once per child agent** after `delegate_task` has constructed the child `AIAgent` and before that child is run. Whether you delegate a single task or a batch of three, this hook fires once for each child.

This hook is specific to delegation/subagent lifecycle. It is not a universal "before any agent invocation" gate for gateway, CLI, cron, batch, MoA, or other runner-originated agent executions.

**Callback signature:**

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_session_id` | `str \| None` | Session ID of the delegating parent agent. |
| `parent_turn_id` | `str` | Turn ID of the parent agent turn that requested delegation, if available. |
| `parent_subagent_id` | `str \| None` | Parent subagent ID when this child was spawned by another subagent; `None` for top-level parent agents. |
| `child_session_id` | `str \| None` | Session ID allocated for the child agent. |
| `child_subagent_id` | `str` | Stable subagent ID used by delegation observability and controls. |
| `child_role` | `str` | Effective child role after delegation policy is applied, for example `"leaf"` or `"orchestrator"`. |
| `child_goal` | `str` | Delegated goal/prompt that the child agent will execute. |

**Fires:** In `tools/delegate_tool.py`, inside `_build_child_agent()`, after the child `AIAgent` has been constructed and annotated with subagent identity metadata, and before `_run_single_child()` runs the child.

**Return value:** Ignored. This is an observer hook only; returning a value does not block or mutate the child agent run.

**Use cases:** Logging subagent creation, mapping parent/child session relationships, tracking nested delegation trees, emitting pre-run audit records, pre-allocating per-child observability resources.

**Example — log subagent creation:**

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
`subagent_start` is useful for delegation observability, but it is not a blocking policy hook. To block delegation before a child is built, use [`pre_tool_call`](#pre_tool_call) to block the `delegate_task` tool call.
:::

---

### `subagent_stop`

Fires **once per child agent** after `delegate_task` finishes. Whether you delegated a single task or a batch of three, this hook fires once for each child. Dispatch is serialised on the parent thread after child futures drain, and each Python callback body runs on that same caller thread (not on a timeout worker).

**Callback signature:**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                tool_call_history: list[dict], duration_ms: int, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_session_id` | `str` | Session ID of the delegating parent agent |
| `child_role` | `str \| None` | Orchestrator role tag set on the child (`None` if the feature isn't enabled) |
| `child_summary` | `str \| None` | The final response the child returned to the parent |
| `child_status` | `str` | `"completed"`, `"failed"`, `"interrupted"`, or `"error"` |
| `tool_call_history` | `list[dict]` | Ordered metadata-only tool calls: `tool_name`, bounded `tool_input`, `input_bytes`, `output_bytes`, and `status`; raw inputs and outputs are excluded |
| `duration_ms` | `int` | Wall-clock time spent running the child, in milliseconds |

**Fires:** In `tools/delegate_tool.py`, after `ThreadPoolExecutor.as_completed()` drains all child futures. `invoke_hook("subagent_stop", ...)` is marshalled to the parent thread so authors don't see child-pool re-entrancy, and callbacks stay on that caller thread.

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
| `{"action": "authorize"}` | Skip the platform allowlist check and dispatch normally. Use when the hook itself is the authorization layer — e.g. an identity/access plugin that already authenticated the sender and may have rewritten `source.user_id` to a canonical identity that is not (and should not need to be) in the platform allowlist. |

**Use cases:** Listen-only group chats (only respond when tagged; buffer ambient messages into context); human handover (silent-ingest customer messages while owner handles the chat manually); per-profile rate limiting; policy-driven routing; plugin-managed identity and access control (`authorize`).

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

### `gateway_platform_event`

Fires for supported platform-native events only **after** the gateway's normal, profile-scoped authorization check succeeds. The callback receives plain dictionaries; raw SDK objects, adapter handles, bot clients, and callback contexts are never part of this stable contract.

Telegram message reactions were the first supported event; message edits, deletes, and thread lifecycle events followed:

```python
def on_platform_event(platform, event_type, payload, **kwargs):
    if platform == "telegram" and event_type == "reaction":
        print(payload["chat_id"], payload["message_id"], payload["emojis"])
    elif event_type == "message_edited":
        print(platform, payload["chat_id"], payload["message_id"], payload["text"])

def register(ctx):
    ctx.register_hook("gateway_platform_event", on_platform_event)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `platform` | `str` | Stable platform id (`"telegram"`, `"discord"`). |
| `event_type` | `str` | Event-local contract id (see the table below). |
| `payload` | `dict` | Event-type-specific fields, documented per event type below. |

Every payload is additive and event-specific; there is no monolithic gateway payload version. All ids are strings; missing/unavailable fields are `None`, never guessed. Malformed events and events whose source cannot be authorized are dropped (fail closed). A transient Telegram Application rebuild re-registers the observer together with the core handlers.

**Per-event payload contracts (v1, additive):**

| `event_type` | Platforms | Payload fields |
|--------------|-----------|----------------|
| `reaction` | telegram | `emojis: list[str]`, `custom_emoji_ids: list[str]`, `chat_id: str`, `message_id: str`, `thread_id: str \| None` (Telegram reaction updates carry no topic id, so currently always `None`). |
| `message_edited` | telegram, discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `text: str \| None` (edited text or caption, bounded; `None` for media-only edits or when uncached), `edited_at: str \| None` (ISO 8601). |
| `message_deleted` | discord | `chat_id: str`, `message_id: str`, `thread_id: str \| None`, `author_id: str \| None`. Discord's delete event does not identify the deleter; the authorized source is the deleted message's author, and uncached deletions never fire. |
| `thread_created` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `name: str \| None`, `owner_id: str \| None`. |
| `thread_renamed` | discord | `thread_id: str`, `parent_chat_id: str \| None`, `old_name: str \| None`, `new_name: str`. Fired only when the name actually changed; other thread updates (archive, slowmode, tags) are dropped. Discord's thread-update event carries no actor, so the thread owner is the authorized source. |

The bot's own progressive message edits (streaming) never fire `message_edited` on Discord — bot-authored events are dropped at the fire-site.

This hook is observer-only: it does **not** add raw-event access or adapter access. **Raw SDK payload access is deliberately not shipped** — adapter SDK objects change shape without notice and would become un-evolvable API surface; where genuinely needed it requires its own explicit capability (`gateway.raw_events`) with a "no stability guarantee" label and its own design (tracked in #64228). For *acting* on a platform (adding a reaction, renaming a thread), use the capability-gated `ctx.platform_actions` facade documented in the [plugins guide](plugins.md#platform-actions) — it is gated off by default behind the `gateway.platform_actions` capability. `PluginContext.dispatch_tool()` can only call tools registered in the tool registry; `send_message` is intentionally not registered there (its transport is reserved for explicit CLI, cron, kanban, and MCP delivery paths). A future outbound-delivery contract must first provide stable delivered content/handles across all adapters; this slice does not pre-register an inert `gateway_message_delivered` hook.

---

### `pre_approval_request`

Fires before an approval decision is requested. It covers prompted surfaces—interactive CLI, Ink TUI, gateway platforms, and ACP clients—and `approvals.mode=smart` decisions made without a human prompt (`surface="smart"`). In smart mode, the hook runs before the auxiliary LLM is called.

This is the right place to wire a custom notifier — for example, a macOS menu-bar app that pops an allow/deny notification, or an audit log that records every approval request with context.

**Callback signature:**

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | Terminal command or `execute_code` script being assessed. Smart and gateway payloads are redacted before observer dispatch. Smart observer redaction is mandatory even when `security.redact_secrets` is disabled; if redaction fails, smart hooks are skipped. |
| `description` | `str` | Human-readable reason(s) the command is flagged (combined when multiple patterns match) |
| `pattern_key` | `str` | Primary pattern key that triggered the approval (e.g. `"rm_rf"`, `"sudo"`) |
| `pattern_keys` | `list[str]` | All pattern keys that matched |
| `session_key` | `str` | Session identifier, useful for scoping notifications per-chat |
| `surface` | `str` | `"cli"` for interactive CLI/TUI prompts, `"gateway"` for async platform approvals, or `"smart"` for auxiliary-LLM auto approve/deny decisions |

**Return value:** ignored. Hooks here are observer-only; they cannot veto or pre-answer the approval. Use [`pre_tool_call`](#pre_tool_call) to block a tool before it reaches the approval system.

**Use cases:** Desktop notifications, push alerts, audit logging, Slack webhooks, escalation routing, metrics.

**Example — desktop notification on macOS:**

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

Fires after a prompted or smart approval decision, after a prompt times out, or when the gateway cannot deliver the approval notification. Notification failure emits `choice="notify_failed"` before any approval decision exists.

**Callback signature:**

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

Same kwargs as `pre_approval_request`, plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `choice` | `str` | Prompted surfaces use `"once"`, `"session"`, `"always"`, `"deny"`, `"timeout"`, or `"notify_failed"`; smart decisions use `"smart_approve"` or `"smart_deny"` |
| `decided_by` | `str` | `"aux_llm"` for smart decisions; absent on prompted surfaces |

**Return value:** ignored.

**Use cases:** Close the matching desktop notification, record the final decision in an audit log, update metrics, roll forward a rate limiter.

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `pre_transcription`

Fires inside the STT dispatcher (`tools.transcription_tools.transcribe_audio`) **after** the provider has been resolved and **before** any backend is invoked, whether that backend is built-in, a `type: command` provider, or a plugin-registered provider. Lets a plugin steer the transcription request itself instead of only observing the transcript afterwards.

**Callback signature:**

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Absolute path to the audio file about to be transcribed. Read-only. |
| `provider` | `str` | Resolved STT provider (`local`, `groq`, `openai`, `mistral`, `xai`, `elevenlabs`, `deepinfra`, `local_command`, a command provider name, or a plugin provider name). |
| `model` | `str \| None` | Model resolved so far, or `None` when the backend default applies. |
| `language` | `str \| None` | Language from the provider's config section, or `None`. |
| `prompt` | `str \| None` | The static [`stt.prompt`](/user-guide/configuration#transcription-prompt-vocabulary-hints) value, or `None`. |
| `source` | `str \| None` | Caller surface label (`gateway`, `voice_mode`, …). Observability only, not used for dispatch. |

**Return value:** a `dict` with any of `"prompt"`, `"language"`, `"model"` mapped to strings, or `None` to leave the request unchanged. Non-string values, unknown keys, and `file_path` are ignored (`file_path` attempts are logged as a warning). Results are applied in **registration order, last-writer-wins per field**, on top of the `stt.prompt` config value. Returning `""` for `prompt` clears the configured prompt for that request.

**Use cases:** Inject a per-user or per-chat vocabulary list before the audio is uploaded, force `language` from the caller's locale, downgrade `model` for long recordings, route noisy sources to a different model.

```python
VOCAB = "Hermes, Teknium, Nous Research, kanban"

def add_vocab(provider, prompt, source, **kwargs):
    if source != "gateway":
        return None
    return {"prompt": f"{prompt}. {VOCAB}" if prompt else VOCAB}

def register(ctx):
    ctx.register_hook("pre_transcription", add_vocab)
```

Not every backend accepts a prompt. `local` maps it to faster-whisper's `initial_prompt`; `openai`, `groq`, `mistral`, and `deepinfra` send it as `prompt`; `xai`, `elevenlabs`, `local_command`, and `type: command` providers log at DEBUG and transcribe without it. See the [provider support table](/user-guide/configuration#transcription-prompt-vocabulary-hints) for the full matrix and the privacy boundary. Hook-plumbing errors are fail-open: the dispatch continues with the unmodified request.

---

### `transform_tool_result`

Fires **after** a tool returns and **before** the result is appended to the conversation. Lets a plugin rewrite ANY tool's result string — not just terminal output — before the model sees it.

**Callback signature:**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs) -> str | None:
```

The full payload also includes `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, and `error_message`. `result` is the final result returned by tool dispatch; it and `args` can contain arbitrary user/tool content and secrets.

**Return value:** The first `str` replaces the result (including an empty string); `None` leaves it unchanged.

**Use cases:** Redact organization-specific PII from `web_extract` output, wrap long JSON tool responses in a summary header, inject retrieval-augmented hints into `read_file` results, rewrite `delegate_task` subagent reports into a project-specific schema.

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

Applies to every tool. For terminal-only rewriting see `transform_terminal_output` below — it is narrower, runs before `transform_tool_result`, and its replacement is still subject to the terminal tool's final output limit.

---

### `transform_terminal_output`

Fires inside the `terminal` tool after foreground process capture has already been bounded by the environment, and before the final output limit. It lets plugins replace the captured stdout/stderr; the replacement is still subject to the final output limit.

**Callback signature:**

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | The shell command that produced the output. |
| `output` | `str` | Combined stdout/stderr after bounded process capture. |
| `returncode` | `int` | Process return code. |
| `task_id` | `str` | Effective task identifier, or an empty string. |
| `env_type` | `str` | Execution-environment type. |

**Return value:** First `str` replaces the output; `None` leaves it unchanged. Command and output can contain credentials or other sensitive data.

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

Pairs with `transform_tool_result`, which runs afterward for every tool, including `terminal`.

---

### `transform_llm_output`

Fires **once per turn** after the tool-calling loop completes and the model has produced a final response, **before** that response is delivered to the user (CLI, gateway, or programmatic caller). Lets a plugin rewrite the assistant's final text using classical-programming methods — no extra inference tokens burned on SOUL flavor text or a skill-driven transform.

**Callback signature:**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_text` | `str` | The assistant's final response text for this turn. |
| `session_id` | `str` | Session ID for this conversation (may be empty for one-shot runs). |
| `model` | `str` | Model name that produced the response (e.g. `anthropic/claude-sonnet-4.6`). |
| `platform` | `str` | Delivery platform (`cli`, `telegram`, `discord`, …; empty when unset). |

**Return value:** Non-empty `str` to replace the response text, `None` or empty string to leave it unchanged. **First non-empty string wins** when multiple plugins register. Unlike the tool and terminal transforms, an empty string is not accepted as a replacement.

**Use cases:** Apply a personality/vocabulary transform (pirate-speak, Spongebob), redact user-specific identifiers from the final text, append a project-specific signature footer, enforce a house style guide without burning tokens on SOUL instructions.

When CLI streaming is enabled, an append-only transform is printed after the
streamed body. A transform that replaces the response is printed in full after
the streamed body, labeled as a post-stream transformation, so replacement
content is never silently lost.

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

The hook is guarded on a non-empty, non-interrupted response — it will not fire on stop-button interrupts or empty turns. Exceptions are logged as warnings and do not break agent execution.

### API-request observer hooks

#### `pre_api_request`

Fires for each provider attempt immediately before sending it. This is observer-only. The legacy `user_message`, `conversation_history`, and `request_messages` fields are raw and intentionally unsanitized for compatibility; new consumers should prefer the sanitized `request` envelope.

#### `post_api_request`

Fires after a provider response has been normalized successfully. This is observer-only. Prefer the sanitized `response`; `assistant_message` is the raw normalized message, and `usage` contains accounting data.

#### `api_request_error`

Fires for a failed provider attempt with status/retry timing, an `error` object, and sanitized `request`. This is observer-only. Error messages may still contain provider or user data.

### `on_skill_lifecycle`

Fires after an authoritative skill-usage state change. It is observer-only and exposes the local `skill_name`, provenance, correlation IDs, usage count, and reuse flags.

### Kanban lifecycle observers

#### `kanban_task_claimed`

Fires after the claim commit in the dispatcher process, immediately before worker spawn.

#### `kanban_task_completed`

Fires after completion and cleanup, usually in the worker process. Its `summary` can contain project or user content.

#### `kanban_task_blocked`

Fires after a normal blocked transition. The dependency-wait path invokes it before that write transaction exits. Its `reason` can contain project or user content.

All three kanban hooks are observer-only and carry `task_id`, `profile_name`, `board`, `assignee`, and `run_id`; completed adds `summary`, and blocked adds `reason`.

### Kanban worker-lifecycle, task-mutation, and dispatch observers

Five additional observers (RFC #58548) extend the kanban family. All are observer-only, fire after the relevant transaction commits, and short-circuit on `has_hook` — with no subscriber, dispatch behavior is unchanged. Task-scoped hooks carry the same common fields as the hooks above.

- **`on_kanban_worker_spawned`** — after `spawn_fn` returns and the worker PID is persisted. Adds `worker_pid` (may be `None`) and `workspace_path`. Runs inside the dispatch lock; keep callbacks fast.
- **`on_kanban_worker_exited`** — tick-derived, when `detect_crashed_workers` reclaims a dead-PID task. Adds `worker_pid`, `exit_kind`, `exit_code`, `outcome`, `retry_status`.
- **`on_kanban_worker_stale_claim`** — when a TTL-expired claim is reclaimed; live-PID extensions don't fire. Adds `worker_pid`, `heartbeat_stale`, `retry_status`.
- **`on_kanban_task_updated`** — after a committed task-field write outside the claim/complete/block lifecycle (`assign_task`, model/reasoning overrides, dashboard editors). Adds `changed_fields` — field names only, never values.
- **`on_kanban_dispatch_tick`** — once per dispatcher tick, strictly after the dispatch lock is released, including idle and lock-contended ticks. Payload: `board`, `profile_name`, `dry_run`, `outcome`, `result`.

---

## Shell Hooks

Declare shell-script hooks in your `~/.hermes/config.yaml` and Hermes will run them as subprocesses whenever the corresponding plugin-hook event fires — in both CLI and gateway sessions. No Python plugin authoring required.

Use shell hooks when you want a drop-in, single-file script (Bash, Python, anything with a shebang) to:

- **Block or modify a tool call** — reject dangerous `terminal` commands, enforce per-directory policies, require approval for destructive `write_file` / `patch` operations, or rewrite arguments (sanitize paths, inject defaults) before the tool runs.
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
      fail_closed: <bool>        # Optional; default false. pre_tool_call only.
                                 # `failClosed` also accepted (Cursor/Claude Code compat)

hooks_auto_accept: false         # See "Consent model" below
```

Event names must be one of the [plugin hook events](#plugin-hooks); typos produce a "Did you mean X?" warning and are skipped. Unknown keys inside a single entry are ignored; missing `command` is a skip-with-warning. `timeout > 300` is clamped with a warning. `fail_closed: true` on an event other than `pre_tool_call` warns and is ignored (only blocking-capable events can fail closed).

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

// Modify a pre_tool_call — rewrite tool args before dispatch:
{"action": "modify", "args": {"new_string": "fixed content"}}         // Hermes-canonical
{"decision": "modify", "tool_input": {"new_string": "fixed content"}} // Claude-Code style

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Keep the agent going at the verify gate (pre_verify); both shapes accepted:
{"action": "continue", "message": "Run the formatter, then finish."}
{"decision": "block",  "reason":  "Run the formatter, then finish."}

// Silent no-op — any empty / non-matching output is fine:
```

Malformed JSON, non-zero exit codes, and timeouts log a warning but never abort the agent loop.

### Exit code 2 = block (Claude Code / Cursor compatible)

A `pre_tool_call` hook that exits with code **2** blocks the tool call even when its stdout carries no block JSON. The block message is resolved in priority order:

1. stdout block JSON (`reason` / `message`), when present;
2. the first 400 characters of stderr;
3. a generic `"Blocked by shell hook."` default.

So the simplest possible blocking hook is:

```bash
#!/usr/bin/env bash
echo "policy violation: rm -rf is not permitted" >&2
exit 2
```

For events whose block directive is not honored (everything except `pre_tool_call`), exit 2 is treated like any other non-zero exit: a warning is logged and stdout is still parsed.

### Fail-open vs fail-closed

By default shell hooks **fail open**: a spawn error, timeout, or unparseable stdout logs a warning and the action proceeds. That is the right default for observability hooks — but wrong for security gates. A crashed secret-scanner must not silently allow the tool call it was supposed to vet.

Set `fail_closed: true` (or `failClosed: true`, the Cursor/Claude Code spelling) on a `pre_tool_call` entry to invert that:

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal|write_file|patch"
      command: "~/.hermes/agent-hooks/secret-scan.sh"
      timeout: 10
      fail_closed: true
```

With `fail_closed: true`, each of these now **blocks** the tool call with `hook <command> failed closed: <reason>`:

| Failure | Fail-open (default) | `fail_closed: true` |
|---------|--------------------|--------------------|
| Command not found / not executable | warn, proceed | **block** |
| Timeout | warn, proceed | **block** |
| Non-JSON stdout (e.g. a stack trace) | warn, proceed | **block** |
| Clean exit, valid no-op JSON (`{}`) | proceed | proceed |

`fail_closed` only applies to blocking-capable events (`pre_tool_call` today); setting it on any other event logs a warning at config-parse time and is ignored. `hermes hooks test` reflects these semantics — the `parsed` line shows exactly the block shape the dispatcher would receive.

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

### Consent model

Each unique `(event, command)` pair prompts the user for approval the first time Hermes sees it, then persists the decision to `~/.hermes/shell-hooks-allowlist.json`. Subsequent runs (CLI or gateway) skip the prompt.

Three escape hatches bypass the interactive prompt — any one is sufficient:

1. `--accept-hooks` flag on the CLI (e.g. `hermes --accept-hooks chat`)
2. `HERMES_ACCEPT_HOOKS=1` environment variable
3. `hooks_auto_accept: true` in `~/.hermes/config.yaml`

Non-TTY runs (gateway, cron, CI) need one of these three — otherwise any newly-added hook silently stays un-registered and logs a warning.

**Script edits are silently trusted.** The allowlist keys on the exact command string, not the script's hash, so editing the script on disk does not invalidate consent. `hermes hooks doctor` flags mtime drift so you can spot edits and decide whether to re-approve.

#### Manual allowlisting

Manual allowlisting is useful for non-TTY or service-account deployments where an operator cannot answer the first-use prompt interactively. The allowlist file is `~/.hermes/shell-hooks-allowlist.json`, and the expected format is an `approvals` array. Each approval records the hook `event` and the exact `command` string:

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

The command string must match the configured hook command exactly. A path-keyed object with a `sha256` field is not the expected format and will not approve the hook. Verify manual entries with `hermes hooks list`.

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

## Outbound Webhooks

Outbound webhooks are the push-side mirror of the [inbound webhook platform](/user-guide/messaging/webhooks): inbound webhooks wake Hermes when the world changes; outbound webhooks tell the world when Hermes does something. Configure a list of HTTP endpoints and the lifecycle events they care about, and Hermes POSTs a signed JSON payload to each endpoint whenever a matching event fires — no polling on the receiving end.

Typical uses:

- Notify a CI system or dashboard when an agent turn finishes (`on_session_end`)
- Track subagent completions across a fleet (`subagent_stop`)
- Feed tool activity into external monitoring (`post_tool_call` with a `matcher`)
- Wake *another* Hermes instance: point the URL at that instance's inbound webhook

### Configuration

Add a `hooks.outbound:` list to `~/.hermes/config.yaml`:

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

Any event from the plugin-hook set is valid (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `subagent_start`, `subagent_stop`, ...). Malformed entries warn and are skipped — a broken webhook never crashes the agent. Changes take effect on the next CLI session / gateway restart.

Secrets: prefer `secret_env` (the name of an environment variable, typically set in `~/.hermes/.env`) over an inline `secret:` literal, so the config file stays free of credentials. Entries without a secret are delivered unsigned (flagged as `UNSIGNED` by `hermes hooks list`).

### Wire format

Each firing POSTs a JSON body with the same top-level shape as shell hooks' stdin, plus delivery metadata. `profile` names the Hermes profile that emitted the event (`"default"` outside profiles), so receivers behind a multiplexed gateway can tell profiles apart:

```json
{
  "hook_event_name": "on_session_end",
  "profile": "default",
  "tool_name": null,
  "tool_input": null,
  "session_id": "sess_abc123",
  "cwd": "/home/user/project",
  "extra": {"completed": true, "interrupted": false, "model": "...", "platform": "cli"},
  "delivery_id": "3f2c9a...",
  "timestamp": "2026-07-22T14:00:00Z"
}
```

Headers:

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Hermes-Event` | The hook event name |
| `X-Hermes-Delivery` | Unique id per delivery — same value as `delivery_id` in the body |
| `X-Hermes-Signature-256` | `sha256=<hex>` — HMAC-SHA256 of the raw body, GitHub-style; only present when a secret is configured |

Verify the signature exactly as you would a GitHub webhook:

```python
import hashlib, hmac

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

Because `delivery_id` and `timestamp` live **inside the signed body**, a verified receiver also gets replay protection for free:

- **Dedupe** on `delivery_id` (or the matching `X-Hermes-Delivery` header) — remember recently seen ids and skip duplicates. Hermes retries failed deliveries once, so the same id can legitimately arrive twice.
- **Reject stale events** by checking `timestamp` against your clock with a tolerance window (5 minutes is the common default). An attacker replaying a captured request can't forge a fresh timestamp without the secret.

### Delivery semantics

- **Fire-and-forget, off the hot path.** Events are serialized and queued instantly; a single background thread performs the HTTP POSTs. A slow or dead endpoint can never stall a tool call or an agent turn.
- **Notify-only.** Unlike shell hooks, outbound webhooks cannot block tool calls or inject context — the response body is ignored. They observe, never steer.
- **Bounded retries.** Connection errors and 5xx responses are retried once with backoff; 4xx responses are not retried (the receiver said the request itself is wrong). Failures are logged and dropped — delivery is best-effort, not guaranteed.
- **Redirects are never followed.** A 3xx response is treated as a misconfiguration and logged — following a redirected POST would silently drop the signed payload. Point the `url` at the final endpoint.
- **Bounded queue.** If the queue backs up (dead endpoint, event storm), new events are dropped with a warning rather than consuming unbounded memory.
- **No consent prompt.** Outbound targets execute no code on your machine — they receive data at a URL you configured. `HERMES_SAFE_MODE=1` still skips registration, same as plugins and shell hooks. Note that payloads include tool inputs and event metadata, so only point targets at endpoints you trust, and prefer `https://`.

`hermes hooks list` shows configured outbound targets alongside shell hooks, including whether each target is signed.
