---
sidebar_position: 3
title: "Agent Loop Internals"
description: "Detailed walkthrough of AIAgent execution, API modes, tools, callbacks, and fallback behavior"
---

# Agent Loop Internals

The public agent interface is still `run_agent.py`'s `AIAgent` class, but the
main per-turn loop now lives in `agent/conversation_loop.py`. `AIAgent.chat()`
is the simple interface: it calls `AIAgent.run_conversation()` and returns only
the final text response. `AIAgent.run_conversation()` sets conversation and
accounting context, then forwards into
`agent.conversation_loop.run_conversation(agent, ...)`.

Inside `agent/conversation_loop.py`, the loop builds the turn context, calls the
model, decides whether the response contains tool calls or final text, executes
tools when needed, and finalizes the turn. Tool-call batches pass through
`AIAgent._execute_tool_calls(...)`, which delegates sequential, concurrent, or
segmented execution to `agent/tool_executor.py`. Registry-dispatched tool calls
pass through `model_tools.handle_function_call(...)`. Agent-runtime tools, such
as `session_search` and `delegate_task`, are handled inline by
`agent/tool_executor.py` and do not reach `handle_function_call(...)`.

## Core Responsibilities

The agent loop is split across a few primary layers:

- `run_agent.py` defines the stateful `AIAgent` facade, public entry points,
  and helper methods used by the loop.
- `agent/conversation_loop.py` contains the main per-turn API/tool loop.
- `agent/turn_context.py` builds the per-turn context consumed by the loop.
- `agent/tool_executor.py` executes tool-call batches sequentially,
  concurrently, or in planned segments.
- `model_tools.py` dispatches individual tool calls to registered Hermes tools.

## Two Entry Points

```python
# Simple interface — returns final response string
response = agent.chat("Fix the bug in main.py")

# Full interface — returns dict with messages, metadata, usage stats
result = agent.run_conversation(
    user_message="Fix the bug in main.py",
    system_message=None,           # auto-built if omitted
    conversation_history=None,      # auto-loaded from session if omitted
    task_id="task_abc123"
)
```

`chat()` is a thin wrapper around `run_conversation()` that extracts the
`final_response` field from the result dict. `AIAgent.run_conversation()` is
also a facade method: after setting conversation and accounting context, it
forwards to `agent.conversation_loop.run_conversation(agent, ...)`.

## API Modes

Hermes supports several API execution modes, resolved from provider selection,
explicit args, and base URL heuristics:

| API mode | Used for | Client type |
|----------|----------|-------------|
| `chat_completions` | OpenAI-compatible endpoints (OpenRouter, custom, most providers) | `openai.OpenAI` |
| `codex_responses` | OpenAI Codex / Responses API | `openai.OpenAI` with Responses format |
| `anthropic_messages` | Native Anthropic Messages API | `anthropic.Anthropic` via adapter |
| `bedrock_converse` | AWS Bedrock Converse-compatible Claude calls | Bedrock runtime client |
| `codex_app_server` | Codex subprocess/app-server runtime | Codex app-server adapter |

The mode determines how messages are formatted, how tool calls are structured,
how responses are parsed, and how caching/streaming works. Standard model-call
modes converge on the same internal message format (OpenAI-style
`role`/`content`/`tool_calls` dicts) before and after API calls;
`codex_app_server` is an alternate runtime path that bypasses the standard
API/tool loop for the turn.

**Mode resolution order:**
1. Explicit `api_mode` constructor arg (highest priority)
2. Provider-specific detection (e.g., `anthropic` provider → `anthropic_messages`)
3. Base URL heuristics (e.g., `api.anthropic.com` → `anthropic_messages`)
4. Default: `chat_completions`

## Turn Lifecycle

A useful way to navigate the current loop is:

```text
AIAgent.chat(...)
    ↓
AIAgent.run_conversation(...)
    ↓
agent.conversation_loop.run_conversation(agent, ...)
    ↓
build_turn_context(...)
    ↓
API/tool loop
    ↓
model response
    ├─ tool calls → AIAgent._execute_tool_calls(...) → agent/tool_executor.py
    └─ final text → finalize_turn(...)
```

Inside the API/tool loop, Hermes prepares the provider request, optionally
applies compression and prompt-caching markers, calls the model through the
configured runtime path, normalizes the response, and either executes tool calls
or finalizes the assistant's text response.

### Message Format

All messages use OpenAI-compatible format internally:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

Reasoning content (from models that support extended thinking) is stored in `assistant_msg["reasoning"]` and optionally displayed via the `reasoning_callback`.

### Message Alternation Rules

The agent loop enforces strict message role alternation:

- After the system message: `User → Assistant → User → Assistant → ...`
- During tool calling: `Assistant (with tool_calls) → Tool → Tool → ... → Assistant`
- **Never** two assistant messages in a row
- **Never** two user messages in a row
- **Only** `tool` role can have consecutive entries (parallel tool results)

Providers validate these sequences and will reject malformed histories.

## Interruptible API Calls

API requests are made through the interruptible API-call helpers. The loop uses
`_interruptible_streaming_api_call(...)` when streaming is enabled and supported,
and falls back to `_interruptible_api_call(...)` otherwise. Both paths monitor
interrupt state so the agent can stop waiting on a provider response:

```text
┌────────────────────────────────────────────────────┐
│  Main thread                  API thread           │
│                                                    │
│   wait on:                     HTTP POST           │
│    - response ready     ───▶   to provider         │
│    - interrupt event                               │
│    - timeout                                       │
└────────────────────────────────────────────────────┘
```

When interrupted (user sends new message, `/stop` command, or signal):
- The API thread is abandoned (response discarded)
- The agent can process the new input or shut down cleanly
- No partial response is injected into conversation history

## Tool Execution

### Batch Planning

When the model returns tool calls, the conversation loop passes the assistant
message to `AIAgent._execute_tool_calls(...)`. That method plans the batch
before execution:

- **Single tool call** → sequential execution.
- **Multiple parallel-safe tool calls** → concurrent execution.
- **Multiple sequential or barrier tool calls** → sequential execution.
- **Mixed batches** → segmented execution, preserving required ordering while
  parallelizing safe runs.

The sequential, concurrent, and segmented implementations live in
`agent/tool_executor.py`.

### Execution Flow

```text
assistant_message.tool_calls
    ↓
AIAgent._execute_tool_calls(...)
    ↓
agent/tool_executor.py
    ├─ agent-runtime tools → handled inline
    └─ registry-dispatched tools
           ↓
       model_tools.handle_function_call(...)
           ↓
       tools/registry.py dispatch
```

For registry-dispatched tools, `model_tools.handle_function_call(...)` routes
the call through `tools/registry.py` and returns the tool result. The executor
appends results from either execution path to the conversation as tool messages.

### Agent-Level Tools

Some agent-runtime tools are handled inline by `agent/tool_executor.py` instead
of passing through `model_tools.handle_function_call(...)`:

| Tool | Why intercepted |
|------|--------------------|
| `todo` | Reads/writes agent-local task state |
| `memory` | Writes to persistent memory files with character limits |
| `session_search` | Queries session history via the agent's session DB |
| `delegate_task` | Spawns subagent(s) with isolated context |

These tools modify agent state directly and return synthetic tool results without going through the registry.

## Callback Surfaces

`AIAgent` supports platform-specific callbacks that enable real-time progress in the CLI, gateway, and ACP integrations:

| Callback | When fired | Used by |
|----------|-----------|---------|
| `tool_progress_callback` | Before/after each tool execution | CLI spinner, gateway progress messages |
| `thinking_callback` | When model starts/stops thinking | CLI "thinking..." indicator |
| `reasoning_callback` | When model returns reasoning content | CLI reasoning display, gateway reasoning blocks |
| `clarify_callback` | When `clarify` tool is called | CLI input prompt, gateway interactive message |
| `step_callback` | After each complete agent turn | Gateway step tracking, ACP progress |
| `stream_delta_callback` | Each streaming token (when enabled) | CLI streaming display |
| `tool_gen_callback` | When tool call is parsed from stream | CLI tool preview in spinner |
| `status_callback` | State changes (thinking, executing, etc.) | ACP status updates |

## Budget and Fallback Behavior

### Iteration Budget

The agent tracks iterations via `IterationBudget`:

- Default: 500 iterations (configurable via `agent.max_turns`)
- Each agent gets its own budget. Subagents get independent budgets capped at `delegation.max_iterations` (default 50) — total iterations across parent + subagents can exceed the parent's cap
- At 100%, the agent stops and returns a summary of work done

### Fallback Model

When the primary model fails (429 rate limit, 5xx server error, 401/403 auth error):

1. Check `fallback_providers` list in config
2. Try each fallback in order
3. On success, continue the conversation with the new provider
4. On 401/403, attempt credential refresh before failing over

The fallback system also covers auxiliary tasks independently — vision, compression, and web extraction each have their own fallback chain configurable via the `auxiliary.*` config section.

## Compression and Persistence

### When Compression Triggers

- **Preflight** (before API call): If conversation exceeds 50% of model's context window
- **Gateway auto-compression**: If conversation exceeds 85% (more aggressive, runs between turns)

### What Happens During Compression

1. Memory is flushed to disk first (preventing data loss)
2. Middle conversation turns are summarized into a compact summary
3. The last N messages are preserved intact (`compression.protect_last_n`, default: 20)
4. Tool call/result message pairs are kept together (never split)
5. A new session lineage ID is generated (compression creates a "child" session)

### Session Persistence

After each turn:
- Messages are saved to the session store (SQLite via `hermes_state.py`)
- Memory changes are flushed to `MEMORY.md` / `USER.md`
- The session can be resumed later via `/resume` or `hermes chat --resume`

## Key Source Files

| File | Purpose |
|------|---------|
| `run_agent.py` | Defines the `AIAgent` facade, public entry points, state, and helper forwarders |
| `agent/conversation_loop.py` | Main per-turn conversation loop |
| `agent/turn_context.py` | Builds per-turn context before the loop runs |
| `agent/tool_executor.py` | Executes tool-call batches sequentially, concurrently, or in planned segments |
| `agent/prompt_builder.py` | System prompt assembly from memory, skills, context files, personality |
| `agent/context_engine.py` | ContextEngine ABC — pluggable context management |
| `agent/context_compressor.py` | Default engine — lossy summarization algorithm |
| `agent/prompt_caching.py` | Anthropic prompt caching markers and cache metrics |
| `agent/auxiliary_client.py` | Auxiliary LLM client for side tasks (vision, summarization) |
| `model_tools.py` | Tool schema collection and individual tool-call dispatch |

## Related Docs

- [Provider Runtime Resolution](./provider-runtime.md)
- [Prompt Assembly](./prompt-assembly.md)
- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
- [Tools Runtime](./tools-runtime.md)
- [Architecture Overview](./architecture.md)
