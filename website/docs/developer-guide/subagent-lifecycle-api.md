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

## Required work declared by a routing policy

Use `ctx.subagent_lifecycle.require(request)` from a `pre_llm_call` hook when a
user-selected policy makes a child result a prerequisite for completing the
current turn. It takes the same `SubagentLaunchRequest` and returns the same
handle as `launch`. The host starts the child directly; the parent model does
not have to call `delegate_task`.

The parent continues its normal agent loop with its existing tools, memory and
skills. Hermes supplies the child's handle through the current user message's
context sidecar, so the parent can collect additional information or steer the
child. When the parent attempts a text-only answer, Hermes waits for the required
children, records their results as a runtime-originated assistant/tool batch,
and gives the parent another model call to consume them. A pre-result completion
candidate is discarded; its text deltas and stream-end text are withheld from
display, TTS and stream hooks. Tool activity remains visible.

This is opt-in through the policy's call to `require`. Ordinary `launch` and
model-issued delegation retain their existing behavior. No new model tool or
tool-schema field is added. Sent message prefixes and the system prompt remain
unchanged; result receipts are appended.

### Example: explicitly selected worker

An ordinary native plugin can bind a user-selected task without a gateway
interceptor. Save these files in the active Hermes home's
`plugins/required-worker/` directory:

```yaml title="plugin.yaml"
name: required-worker
version: 1.0.0
```

```python title="__init__.py"
from agent.subagent_lifecycle import SubagentLaunchRequest


def register(ctx):
    def route(user_message, parent_session_id, turn_id, **kwargs):
        # This policy applies only to explicitly tagged, top-level requests.
        # A child must not recursively reapply its parent's routing policy.
        if parent_session_id or not isinstance(user_message, str):
            return
        if not user_message.startswith("@worker "):
            return
        ctx.subagent_lifecycle.require(SubagentLaunchRequest(
            goal=user_message[len("@worker "):],
            context="Return your findings, changes, validation and unresolved issues.",
            model=ctx.get_config("worker_model"),
            correlation_id=f"required:{turn_id}",
        ))

    ctx.register_hook("pre_llm_call", route)
```

Enable the plugin in `config.yaml` and select a model supported by the child's
inherited provider. `worker_model` is a plugin-relative setting:

```yaml
plugins:
  enabled: [required-worker]
  entries:
    required-worker:
      settings:
        worker_model: your-worker-model
```

For example, `@worker Review the proposed API migration` starts required work
even if the main model never emits a delegation tool call. Supply relevant
background in `context`, as with ordinary delegation; this API does not copy the
parent's full conversation. Reuse the handle with the existing status and
cancellation methods. The parent can use `delegate_task` steering for the active
child. Multiple `require` calls declare independently running prerequisites.

### Completion and failure contract

- Requirements are declared before the first parent model request. Calling
  `require` later, or outside an active Hermes turn, raises
  `SubagentLifecycleError`. The parent's delegation toolset must be enabled;
  a policy cannot widen the parent's capabilities or recursively bypass a leaf
  child's delegation restriction.
- A launch/validation failure is retained even if the hook catches its exception.
  Hermes ends the turn as failed before executing the main model's task.
- A successful requirement needs a terminal `SUCCEEDED` result with a nonempty
  summary, followed by a parent response after that result entered its context.
  Budget exhaustion before that response is incomplete, even if the child
  succeeded. Domain correctness and acceptance still belong to the parent and
  the task's validation process.
- Child failure, interruption or an unconsumed result produces
  `completed: false`, `failed: true` and
  `failure_reason: "required_delegation_incomplete"`. Each requirement appears in
  `required_delegations` with its child identity, state, result hash and
  `result_consumed` flag. Provider/model fields describe the launch configuration;
  they are not independent verification of a provider's internal execution.
- Requirements belong to the current turn. Parent cancellation, budget exhaustion
  or an exceptional exit requests cancellation of outstanding required children.
  Cancellation remains cooperative: a request is not proof that a child has
  stopped. Existing iteration/timeout controls still apply; the result wait also
  honors the parent's configured run budget.

The guarantee starts when a policy calls `require`. Natural-language task
classification, a policy that fails before declaring its requirement, and
cross-turn/restart recovery are separate concerns. This API manages native Hermes
children; selecting a model is not an external CLI integration. It requires the
Hermes agent loop, so whole-turn `codex_app_server` execution is rejected. Tasks
without a declared requirement keep the normal Hermes behavior.
