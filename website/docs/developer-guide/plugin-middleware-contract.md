---
sidebar_position: 16
title: "Plugin Middleware Contract"
description: "Stable observer hooks and middleware surfaces for Hermes plugins"
---

# Plugin Middleware Contract

Hermes exposes a backend-neutral middleware contract for plugins that need to
observe or carefully participate in agent execution. The implementation calls
into this contract from several runtime files, but the supported integration
surface is the contract in `hermes_cli.middleware` and `hermes_cli.plugins`.

## Contract Families

Hermes has two extension families:

| Family | Purpose | May change execution? |
|---|---|---|
| Observer hooks | Lifecycle telemetry for sessions, turns, API calls, tools, approvals, and subagents. | No, except documented legacy return shapes. |
| Middleware | Request rewrites and execution wrappers around API/tool calls. | Yes, when explicitly registered. |

Observer payloads include `telemetry_schema_version = "hermes.observer.v1"`.
Middleware payloads also include
`middleware_schema_version = "hermes.middleware.v1"`.

## Observer Hooks

Observer hooks are registered with `ctx.register_hook(name, callback)`. Callback
exceptions are logged and ignored so observability does not break the agent.

Important hook groups:

| Group | Hooks |
|---|---|
| Session | `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset` |
| Turn | `pre_llm_call`, `post_llm_call` |
| API/LLM | `pre_api_request`, `post_api_request`, `api_request_error` |
| Tool | `pre_tool_call`, `post_tool_call`, `transform_tool_result` |
| Approval | `pre_approval_request`, `post_approval_response` |
| Subagent | `subagent_start`, `subagent_stop` |

Correlation fields are stable across hook groups when available:

- `session_id`
- `task_id`
- `turn_id`
- `api_request_id`
- `tool_call_id`
- `parent_session_id`
- `child_session_id`
- `parent_subagent_id`
- `child_subagent_id`

`pre_tool_call` may return `{"action": "block", "message": "..."}` to block a
tool. When a block wins, Hermes emits `post_tool_call` with `status="blocked"`
and matching IDs so observers can close spans.

`pre_llm_call`, `transform_llm_output`, `transform_terminal_output`, and
`transform_tool_result` have existing behavior-changing return contracts. All
other observer hook return values are ignored.

## Middleware

Middleware is registered with `ctx.register_middleware(kind, callback)`.

Supported middleware kinds:

| Kind | Callback purpose |
|---|---|
| `tool_request` | Rewrite tool arguments before `pre_tool_call` and execution. |
| `tool_execution` | Wrap the real tool callback. |
| `api_request` | Rewrite provider API kwargs before `pre_api_request` and dispatch. |
| `api_execution` | Wrap the real provider API call. |

Request middleware returns a replacement payload:

```python
def register(ctx):
    def add_trace_id(**kw):
        args = dict(kw["args"])
        args["trace_id"] = kw["tool_call_id"]
        return {"args": args, "source": "trace-plugin"}

    ctx.register_middleware("tool_request", add_trace_id)
```

API request middleware uses the `request` key instead:

```python
return {"request": new_api_kwargs, "source": "provider-rewriter"}
```

Execution middleware receives `next_call` and must call it to continue:

```python
def register(ctx):
    def around_tool(**kw):
        result = kw["next_call"](kw["args"])
        return result

    ctx.register_middleware("tool_execution", around_tool)
```

Middleware exceptions are logged and skipped. This keeps the default runtime
fail-open unless a plugin intentionally returns a documented blocking or
replacement result.

## Payload Guarantees

For request middleware and downstream hooks, Hermes passes both the original
and effective payload when a rewrite surface is active:

- Tools: `original_args`, `args`
- API calls: `original_request`, `request`
- Middleware changes: `middleware_trace`

Observer hooks always receive the effective payload in the legacy field name
(`args` or `request`) so existing plugins keep working. New integrations that
need exact provenance should inspect both original and effective fields.

## Compatibility

The contract is backend-neutral. A plugin may map it to Langfuse, NeMo-Flow,
OpenTelemetry, local JSONL, or a custom runtime. NeMo-Flow Adaptive execution
intercepts should use middleware, not passive observer hooks, because
intercepts intentionally participate in the execution path.
