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
`metadata`, `timeout_seconds`, and `fork_context=False`, subject to the restrictions below.
`fork_context=True` requires an enabled extension implementing `format_fork_history`.
The extension chooses the inherited history. Ordinary launches remain fresh by default.

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
The native tool schema does not expose `provider` to the model. Plugins can select
it from operator config or expose per-task selectors through a delegation extension.

The lifecycle API does not consult `delegation.backend`: the plugin that launches a child already chooses
its route. To change how `delegate_task` routes the model's subagents, write a
[delegation provider plugin](delegation-provider-plugin.md).

## Retained conversations (opt-in)

The same host service also offers explicit retained-conversation operations:
`launch_session(goal=..., context=None, title=None, fork_context=None,
task_parameters=None, background=False)`,
`resume_session(session_id, message, model=None, background=False)`,
`list_sessions(query=None, limit=20)`, `observe_session(session_id, limit=20)`,
`close_session(session_id)`, `steer_session(session_id, message)`, and `stop_session(session_id)`. Optional arguments are keyword-only. These return
normal delegation result dictionaries; refusals raise `SubagentLifecycleError`.
Background execution still requires a host capable of delivering later results.

Launch routing uses the session's active [delegation backend](delegation-provider-plugin.md).
Pass its declared fields through `task_parameters`, for example
`launch_session(goal="Review the design", task_parameters={"tier": "deep"})` for a
backend declaring `tier`. Core-owned task fields cannot be supplied through that
mapping. There is no separate retained-spawn model/provider override: selection
policy and schema belong to the backend. Backend route errors refuse only the
unroutable task, as in ordinary delegation.
On resume, only the model may change within the saved provider. Changes to delegation
defaults do not retarget an existing child; credentials refresh from its saved route.

Core defaults to fresh context, including retained launches. Extension registration
and retention alone do not enable forking. A plugin may return `fork_context=True`
from `child_options` to opt in by default. An explicit per-task boolean takes
precedence over that preference. `launch_session(fork_context=None)` omits the
selection, using the plugin preference if present or otherwise fresh context.
The separate run-handle `SubagentLaunchRequest` retains its fresh default. Internal Python
`delegate_task` callers such as `/review` do not inherit the extension's retention or
fork defaults; those defaults apply to model-facing calls and `launch_session`.
A fork seeds the child's own history once; resume does not copy subsequent parent turns.

Retained operations currently require an in-process parent with a native SessionDB.
Retained conversations currently support model/provider routes within that profile. A
`delegation.backend` route selecting an explicit profile is refused for retention; ordinary
nonretained profile delegation remains available.
Managed compute workers using RuntimeSessionStore remain unsupported: upstream
child-registration restrictions stay enforced. This is separate from the existing
run-handle facade and requires owner-mediated storage operations before those
workers can retain conversations.

Conversation methods use stable session IDs stored in the parent's profile
database. The existing immutable run handles and their reconnect/retention
contract are unchanged. Use `launch_session` when future continuation is needed.
The parent conversation owns the child across reconstruction and compression;
another conversation cannot adopt it. Each accepted activation has exclusive
admission and uses the normal transcript turn lease. Spawn and resume both consume
the one-shot child budget; observation and control do not.

Idle children have no AIAgent instance and no automatic expiry. Explicit close
retires them without deleting history. Retained history has no feature-specific storage
cap and can grow without bound. Normal session deletion still applies, including
opt-in `sessions.auto_prune` after `retention_days` for ended sessions; deleting a
required transcript makes continuation unavailable, never silently fresh.
If a timeout leaves a worker alive, its retained conversation is permanently closed
to reuse before deferred cleanup. Observation and idempotent close remain available;
start a new child for further work. A late worker exit cannot reopen it. A timeout
whose worker has already settled may leave an idle, resumable conversation.
Process loss ends the active run without replaying it; the saved conversation can
be continued after the recorded process is proven dead. This is not durable
execution or crash recovery.


## Delegation extensions

`ctx.register_subagent_extension(extension)` accepts an instance of the public
`agent.subagent_extension.SubagentExtension` contract. It requires the same operator
consent as replacing a built-in tool: `plugins.entries.<plugin>.allow_tool_override: true`.
There is one active extension per profile; normal registration leases restore its
predecessor when unloaded. The callbacks receive plain data, never live agents:

- `enabled()` determines whether the extension participates.
- `extend_schema(schema)` receives a deep copy of the dynamic `delegate_task` schema
  and returns its model-facing description/parameters. Existing native fields should be preserved.
- `handle_action(action, request)` returns a JSON tool result or `None` for native
  dispatch. The active caller is bound to `ctx.subagent_lifecycle` for the callback.
  Spawn remains host-owned. Plugin-owned parameters travel through both native
  registry and inline model dispatch without adding fields to core.
- `child_options(task)` returns optional `retain`, `review_skills`, and `fork_context`
  booleans and `title`. The host evaluates and validates options once per task before
  constructing any children. `fork_context` supplies a default only; an explicit task
  choice wins. Retention does not imply forking. A retained child uses its existing conversation on continuation. Review
  remains host-budgeted, skills-only, cancellable, and subject to normal eligibility.
- `format_fork_history(messages, task=...)` chooses the initial history for a new child.
  It receives a deep copy of the unformatted conversation boundary, including prior
  tool exchanges, reasoning and the unfinished spawning round. Ephemeral recovery
  scaffolding is excluded because it cannot become durable history. `content` holds
  durable conversation content; optional `api_content` sidecars may include ephemeral
  memory-prefetch or request-hook additions. The host normalizes recognized compaction
  handoffs to `_compressed_summary=True`. The companion projects `content` only.
  The host caches
  the source boundary, not the formatted output: each task receives its own copy and
  may use a different format. The task is the spawn task dictionary (or the run-handle
  request fields for `launch`). This method has no default format; an explicit fork
  without an enabled formatter fails before child construction. Fresh launches and
  continuations do not invoke it.
  Return a JSON-compatible list of message dictionaries with valid roles/content and
  complete tool-call/result pairs. Incomplete spawning calls must be removed or
  represented as ordinary context. Core validates the result before constructing any
  child in the batch and appends an assistant boundary if the history ends with a
  user row, preventing role repair from merging an inherited request into the next
  assignment. This boundary is persisted with the history. Core strips private host
  bookkeeping except compaction metadata, and
  persists the inherited history in the child's own session. It does not choose which
  conversation turns, tool exchanges or reasoning to retain. Plugins must account for
  provider-specific message compatibility when changing routes. The hook controls
  history only; it does not replace core's child instructions, select identity policy,
  or override tool definitions. System/developer messages in the returned history do
  not replace the host-built child prompt. Preserving the parent's complete request
  prefix would require a separate prompt-assembly design; this hook alone does not
  provide it or guarantee prompt-cache reuse.
- `format_assignment(message, session_id=..., timestamp=...)` formats the new assignment.
  It cannot change persisted parentage or acquire control over an unrelated child.

The host owns retention storage, ownership, provider reconstruction, activation
admission, live controls, transcript safety, and teardown even after plugin unload.
Raw conversation-service records contain routing identity and ownership metadata;
plugins own bounded presentation and should avoid exposing unnecessary fields to models.
Observation preserves compact assistant `tool_calls` entries (`name`, optional
`target` from `path`/`file_path`, and `command_preview`) and stored `tool_name`
on tool results. It never returns full tool arguments or reasoning. Tool results
are capped at 1,500 characters; assistant prose can use the remaining window.
The default 12,000-character budget counts content and serialized preview metadata
(excluding the session record and message envelope). Call metadata is capped at
1,000 characters per message, with names capped at 80 and targets/commands at 200.
Long previews preserve head and tail with an omitted-character marker; very small
remaining budgets use an ellipsis. `truncated` reports shortened previews or omitted
messages. Preview text is not guaranteed to remain valid JSON. Observation does
not modify transcripts, renew activity, or activate the child.

Existing rows need no migration when policy moves into a plugin.

This extension currently requires `plugins.isolation: in_process`; host-isolated
registration fails explicitly because the synchronous activation-context callback
contract has not been implemented across the process boundary. There are no private
agent objects, runtime monkeypatches, or plugin-name checks in the host integration.

Observation applies the ordinary child-result redaction policy to prose and tool
previews before truncation; vault values are always scrubbed. Listing and transcript
search use read connections, without holding an admission writer transaction.
