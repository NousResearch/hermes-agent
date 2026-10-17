---
title: Public Subagent Lifecycle API
sidebar_label: Subagent lifecycle API
---

# Public Subagent Lifecycle API

Plugins can launch and supervise fresh Hermes child sessions without importing
`tools.delegate_tool`, gateway internals, TUI state, or `AIAgent` fields.
The service resolves its parent from the current agent turn, so it works in
CLI, gateway, non-interactive, and kanban-worker sessions. Launching outside an
active agent turn fails closed with `No active Hermes parent session`.
The facade also works when a plugin runs under `plugins.isolation: host`.

Requests accept `goal`, `context`, `role`, `model`, `provider`, `allowed_toolsets`,
`blocked_tools`, `working_directory`, `parent_session_id`, `correlation_id`,
`metadata`, `timeout_seconds`, `images`, and `output_schema`, subject to the restrictions below.

```python
from agent.subagent_lifecycle import SubagentLaunchRequest

def launch_review(ctx):
    # Call from a plugin tool or hook while an agent turn is active.
    service = ctx.subagent_lifecycle
    handle = service.launch(SubagentLaunchRequest(
        goal="Review this change for regressions.",
        context="Only inspect the supplied repository.",
        role="leaf",
        correlation_id="review-42",
        allowed_toolsets=("file",),
    ))
    # Persist handle.to_dict() if desired.
    if service.wait(handle, timeout_seconds=2).timed_out:
        return handle.to_dict()
    return service.result(handle)
```

`SubagentHandle` is serializable and carries a versioned, opaque capability.
Requests and handles may also arrive as mappings, including plugin-host wire records.
Pass a handle back to `status`, `wait`, `cancel`, `steer`, `result`, or `reconnect`; malformed
or forged handles return `UNKNOWN`/`UNKNOWN_HANDLE` and cannot access a child.

`steer(handle, text)` queues text for the child's next iteration boundary and
returns `True` while the child accepts steering. It returns `False` for empty
text, unknown or forged handles, handles owned by another parent session, or a
child that has finished or stopped accepting steering.

The stable states are `PENDING`, `STARTING`, `RUNNING`, `SUCCEEDED`, `FAILED`,
`INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED`, and `UNKNOWN`.

`cancel(handle, reason=...)` is cooperative: it asks the child agent to
interrupt at its next safe boundary and returns `CANCEL_REQUESTED`; it never
claims completion until `wait` or `result` observes a terminal state. Terminal
results are immutable, idempotent, bounded to 32k characters, omit transcripts
and hidden reasoning, and include a stable result hash.

This API is lifecycle-managed asynchronous execution. Child construction and
completion use the same host-owned path as `delegate_task`, including parent
tool-resolution restoration, memory notification, serialized `subagent_stop`
hooks, resource cleanup, and child-cost rollup. It does not change the
synchronous `delegate_task` tool, batch delegation, or its gateway/TUI display.
The initial implementation retains metadata and terminal results in-process for
one hour.
After a process restart, `reconnect` returns `RECONNECT_UNAVAILABLE` and never
starts a replacement child. Running Python threads also cannot survive process
exit; callers must treat those handles as interrupted by process exit.

Requests are fail-closed: goal/context/metadata sizes are capped, unknown or
parent-broadening toolsets are rejected (plugin-registered toolsets count as
known), and per-tool blocks, working-directory overrides, and per-launch
timeouts are explicitly rejected until Hermes can
support them without weakening isolation. Use `allowed_toolsets` to narrow a
child; Hermes's existing unsafe-tool block remains enforced.

## Routing

Children follow the `delegation` config (`provider`, `model`, `base_url`,
`api_key`, `api_mode`, `request_overrides`) exactly as `delegate_task` children do.
The request's `model` overrides the configured model. Its optional `provider`
names a configured provider for this launch, with precedence request >
delegation config > parent; the selected provider supplies its own credential
bundle. An unresolvable provider raises `SubagentLifecycleError` before launch.
Core does not expose `provider` to the model: the plugin chooses it, for example
from operator config.

## Images and structured output

`images` and `output_schema` are the same optional task inputs `delegate_task` accepts, validated by the same
helpers and rejected with `SubagentLifecycleError` before any child is built. Both default to `None`; requests
without them behave exactly as before.

- `images`: up to 8 local file paths, `http(s)` URLs, or `data:image/` URLs (a single string is accepted as a
  one-entry list). They ride on the child's **first user turn** through the native image routing: a
  vision-capable child receives image content parts, any other child receives `[Image attached …]` hints for
  `vision_analyze`. Unreadable sources degrade to the text-only goal with a logged warning, as in `delegate_task`.
- `output_schema`: a JSON Schema object (a JSON-encoded string is also accepted, as in `delegate_task`).
  Hermes validates it with the native schema helper and appends the `OUTPUT CONTRACT` block after the caller's
  `context`. The existing 32000-character limit applies to the caller's `context`, not the appended contract;
  no additional schema-specific or combined-size cap is imposed. The child still uses its normal model/context
  handling. The schema is copied at launch; later mutation by the caller does not affect the child. Hermes
  validates the child's final answer and, on a miss, gives that same child exactly one correction turn carrying
  the validation errors; no second child is launched.

When an `output_schema` was supplied, `SubagentResult.structured_payload` is the validation receipt, and only
these keys ever appear (bounded to 10 errors, 1000 characters each):

| key | meaning |
| --- | --- |
| `schema_valid` | `true` when the (possibly repaired) final answer validates |
| `schema_errors` | validation errors, present only when `schema_valid` is `false` |
| `schema_note` | present only when `schema_valid` is `false` and a summary exists |

A persistent miss does not fail the run: the state stays `SUCCEEDED`, `summary` is the child's raw, unvalidated
final answer, and `schema_valid` is `false`. Launches without a schema leave `structured_payload` as `None`.
