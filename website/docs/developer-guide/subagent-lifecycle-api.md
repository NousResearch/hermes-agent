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
Pass it back to `status`, `wait`, `cancel`, `result`, `message`, `resume`, or `reconnect`; malformed
or forged handles return `UNKNOWN`/`UNKNOWN_HANDLE` and cannot access a child.
Treat the capability as private: do not put complete handles in public logs or
execution evidence. Durable storage retains a capability digest, not the bearer
value. Knowing a worker or run ID alone does not grant another session access.

The stable states are `PENDING`, `STARTING`, `RUNNING`, `SUCCEEDED`, `FAILED`,
`INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED`, and `UNKNOWN`.

`cancel(handle, reason=...)` is cooperative: it asks the child agent to
interrupt at its next safe boundary and returns `CANCEL_REQUESTED`; it never
claims completion until `wait` or `result` observes a terminal state. Terminal
results are immutable, idempotent, bounded to 32k characters, omit transcripts
and hidden reasoning, and include a stable result hash.
An explicit reconciliation may add an audit annotation to a durable run; it does
not rewrite the original execution summary or turn an unknown effect into a
verified success.

This API is lifecycle-managed asynchronous execution. Child construction and
completion use the same host-owned path as `delegate_task`, including parent
tool-resolution restoration, memory notification, serialized `subagent_stop`
hooks, resource cleanup, and child-cost rollup. It does not change the
synchronous `delegate_task` tool, batch delegation, or its gateway/TUI display.
When the active parent has profile-scoped `SessionDB` state, worker conversations,
run state, message queues, and completion acknowledgments are retained there.
The in-process registry supplies live executor references; it is not the durable
source of truth. Callers without durable state retain the legacy in-process
behavior and cannot assume their handles survive process exit.

## Profiles, messages, and subsequent runs

Pass `profile="analyst"` in `SubagentLaunchRequest` to select a user-defined worker
profile. Provider/model/effort overrides remain subject to the user's routing
policy. See [worker profiles](../user-guide/features/worker-profiles.md) for
configuration and [worker architecture](worker-orchestration.md) for enforcement.

```python
handle = service.launch(SubagentLaunchRequest(
    goal="Check the supplied synthetic calculation.",
    profile="analyst",
))
message = service.message(handle, "Also explain your units.")
terminal = service.wait(handle, timeout_seconds=2)
if terminal.completed:
    next_run = service.resume(handle, "Now check the same calculation in reverse.")
```

`worker_id` identifies the retained conversation; `run_id` changes for each
assignment. Messages have durable IDs and delivery state. Submission is not proof
that the worker has consumed the message: acknowledgment follows a conversation
checkpoint. Resume rechecks the current provider, credentials, tools, and policy;
it does not grant the previous run's capabilities indefinitely.

## Restart and uncertain actions

Python threads do not survive process exit. Reconnection reads durable state
without launching a replacement. Expired execution leases are reconciled to an
interrupted state, and an old executor cannot overwrite a replacement run's
checkpoints. Resume starts a new linked run using Hermes-owned conversation state;
it does not require a resumable provider-side session.

A tool action interrupted before its result checkpoint has an uncertain outcome.
The service refuses automatic replay. The parent must reconcile the side effect
before resuming; it must not assume the action failed. Internal completion
acknowledgments survive restart, while external message transports retain their
own delivery guarantees.

For a retained plugin handle, the explicit reconciliation form is:

```python
next_run = service.resume(
    handle,
    "Continue with the next assignment; do not repeat the previous operation.",
    reconcile_uncertain=True,
    reconciliation_disposition="confirmed_applied",
    reconciliation_note="Checked the external operation by its reference; it completed.",
)
```

Use `confirmed_not_applied` or `accepted_unknown_no_replay` when that accurately
describes the caller's decision. The latter preserves the fact that the external
outcome is unknown. A nonempty note is required. This records the decision and
affected tool-call identities without replaying the effect; the new run still
passes current-authority validation. The parent tool exposes reconciliation and
resume as separate actions.

Requests are fail-closed: goal/context/metadata sizes are capped, unknown or
parent-broadening toolsets are rejected. Request-level `blocked_tools` is a tuple
of exact tool names and narrows the effective grant alongside `allowed_toolsets`
and the selected profile. Working-directory overrides and request-level timeout
fields retain explicit unsupported errors; select a worker profile for its
supported context and execution policy. Hermes's existing unsafe-tool block
remains enforced.
