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
Pass it back to `status`, `wait`, `cancel`, `result`, or `reconnect`; malformed
or forged handles return `UNKNOWN`/`UNKNOWN_HANDLE` and cannot access a child.

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
parent-broadening toolsets are rejected, and per-tool blocks, working-directory
overrides, and per-launch timeouts are explicitly rejected until Hermes can
support them without weakening isolation. Use `allowed_toolsets` to narrow a
child; Hermes's existing unsafe-tool block remains enforced.

## Guided model routing

`delegate_task`'s task dict (the model-tool schema) accepts `routing_role` /
`routing_mode` / `routing_requirements` / `routing_policy_id` to resolve a route through the shared
guided-routing selector and policy store also used by Kanban and MoA (see
[Delegation → Guided model routing](../user-guide/features/delegation.md#guided-model-routing-opt-in-per-task)).

`SubagentLaunchRequest` exposes the same `routing_role`, `routing_mode`, `routing_policy_id`, and
`routing_requirements` fields. The host-owned service uses the delegated-child
constructor and shared selector, rather than interpreting `model` as an unrestricted
provider override. Under managed authority, a `model` preference must agree with the
selected route.

`routing_mode="shadow"` is observational: it records a recommendation but leaves the
legacy child constructor and route unchanged. `SubagentHandle.routing_receipt_id` exposes
the enforced decision id; `routing_shadow_receipt_id` exposes an observational decision.
Neither field contains prompts or credentials.

A managed parent, including a managed Kanban worker or a managed MoA aggregator's
tool-executing agent, passes its authority ceiling to nested lifecycle launches.
Omitting routing fields does not remove that ceiling. Children cannot replace the
parent's provenance, widen its role/policy, or admit routes outside its receipted
selection and eligible alternatives. Unsupported custom constructors fail closed.
Tool restrictions, spawn depth, and the lifecycle's existing ownership still apply.

Provide positive input and output/tool-growth reserve estimates in the requirements.
Unknown or zero estimates block selection; a last-mile assembled-input check is
performed again before transmission. This API does not approve a roster or activate
another profile's credentials.
