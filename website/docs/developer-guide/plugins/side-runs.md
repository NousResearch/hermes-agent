---
title: Authenticated plugin side runs
---

# Authenticated plugin side runs

The gateway can run a fresh, saved conversation on behalf of an authenticated plugin
slash command. The host owns execution, delivery, cancellation, and approval routing.
A concrete consumer is the standalone configurable model-routes plugin; no model-route
presets or third-party integration are built into the host.

Feature-detect `ctx.supports_side_runs is True` before registering handlers that require
this capability. Older hosts must receive ordinary one-argument handlers which explain
that the capability is unavailable. There is no release-number or global plugin-API
version check.

```python
def register(ctx):
    if getattr(ctx, "supports_side_runs", False) is not True:
        ctx.register_command("example-run", lambda args: "A compatible host is required.")
        return

    def run(args, *, context=None):
        if context is None:
            return "Use an authenticated gateway chat."
        config = ctx.get_config("route", {})
        session_id = context.start_side_run(args, config)
        return f"Started {session_id}"

    ctx.register_command("example-run", run, args_hint="<prompt>",
                         argument_mode="text", busy_policy="noninterrupting")
```

`register_command` adds an optional keyword-only `busy_policy`. The default `None`
preserves the existing busy-input queue/steer behavior. Explicit `reject` declines inline;
`noninterrupting` passes both adapter and runner busy guards inline,
without interrupting or claiming the parent session. The adapter reads policy in the
owning/routed profile before handing the event to its authenticated message handler.
A noninterrupting handler must not change the parent's model, history, or cached prompt.

The optional named `context` parameter is signature-inspected. Legacy one-argument
handlers are still called with only `raw_args`; CLI/TUI handlers without a gateway
context should decline side-run launch. Context is created only after source admission
and slash authorization; internal synthetic events cannot launch contextual commands.
Plugin handler failures are consumed as sanitized command errors, never model prompts.
Do not launch from `pre_gateway_dispatch`, which runs before authentication.

## Invocation context

- `context.source`: a copy of the full trusted `SessionSource`, including transport
  metadata. It is not reconstructed from plugin-supplied IDs.
- `context.start_side_run(prompt, config) -> session_id`: admits bounded work, registers
  host cleanup with the plugin's unload ledger, and returns its fresh session ID.
- `context.cancel_side_run(session_id) -> bool`: requests interruption only when the
  invocation's owner and plugin match the active child. A task ID is not authority.

Config requires explicit `provider` and `model`. Optional fields are `tools`, `reasoning`,
`max_iterations`, `max_tokens`, and `run_budget_seconds`. See
`hermes_cli/plugin_side_runs.py::SideRunConfig` for the validated bounds. Unknown fields
are rejected. Canonical identifiers are validated without trimming or case normalization.
`tools=[]` means no tools; `tools=None` requests all available toolsets.
Toolset selection is not a security boundary. The gateway service caps concurrent runs
with `gateway.side_runs.max_concurrent` (default 3, 1–10), rejecting excess work.

The host uses `resolve_runtime_provider(..., strict=True)`, verifies the resolved
provider/model and constructor identities, and supplies `fallback_model=[]`. Canonical
provider names or configured custom identities are required; an alias resolving to a
different identity fails closed. Provider request overrides that can replace the primary
model/route, isolated history/input, selected tools, output limits, or reasoning are rejected
at either the top level or in `extra_body`. External-agent command/ACP transports, virtual MoA routes,
and Codex app-server are refused because their tool/approval loops are not host-owned.
Native app-server promotion is disabled on the child. No parent route resolver or smart
classifier is called.

## Isolation, approvals, and persistence

Each worker receives its source profile's config and secret ContextVars, a new session
ID and gateway tool/approval key, and `conversation_history=None`. The parent ID is
provenance. The durable independent-branch marker prevents parent compression/resume
from selecting a child, and prevents resume from loading parent history. Child ownership
and effective configuration are stored in the real SessionDB. Runtime fallback is empty;
context files, memory loading, and background review are skipped.

The durable listing key remains the initiating chat's key, separate from the child-only
execution/approval key, so `/sessions` can discover the child. Unnamed children follow
the existing `include_unnamed` listing option; ownership filtering still precedes display.

Gateway resume/listing checks compare the child's initiating owner, platform, profile,
chat/thread, and scope before shared-room or administrator exceptions, including Matrix.
An active child cannot be resumed until it ends. Explicitly resuming a finished child
changes the caller's active session by request; later turns use ordinary gateway policies.

Every child registers its own approval callback. Plain-text prompts route
`/approve side:<request-id>` and `/deny side:<request-id>` to the exact pending request,
after source/slash authorization and ownership checks. A stale ID never falls through
to a parent's FIFO approval. Hosts restricting slash commands must grant the owner
`approve` and `deny` as well as the initiating plugin command; owning a child does not
override an administrator's slash policy. Missing permission fails closed.
Missing delivery, timeout, cancellation, and unload deny
pending work. Isolated execution forces manual approval and masks process/session YOLO;
existing explicit permanent approvals remain host policy.

Cancellation sets the agent's cooperative interrupt flag. The asyncio supervisor shields
the synchronous executor worker and retains capacity until its cleanup actually ends.
Timeout/unload/shutdown request cancellation; none claims to forcibly kill a blocked OS
or provider operation. Worker-side SQLite finalization precedes executor completion, so
normal shutdown's executor-quiesce/skip-close policy also protects child writes.
Side supervisors join the existing detached-worker ledger for active-work reporting,
drain, interruption and idle/suspend checks. During host shutdown a cancelled supervisor
may stop waiting after the host's bounded drain/settle windows; a still-running sync
worker owns its final cleanup and the host retains its database handle.

The final answer is sent directly with the source's reply anchor and adapter metadata.
`_interim_send=True` keeps this independent message from sealing a simultaneous parent
native stream. Delivery is best-effort; transcripts remain available when transport fails.
The host does not automatically resume interrupted work after a restart.

## Limits

The explicit route guarantee applies to the primary model. Auxiliary compression,
vision, tool-internal LLM calls, and other auxiliary tasks use their own configuration.
A runtime deadline and per-request output limit are not monetary or total-spend limits.
The host checks the assembled primary request after provider transformations, including
thinking/headroom expansion and retry boosts. Incompatible reasoning/output settings
are rejected rather than silently changing reasoning or sending an enlarged cap.
Transports that omit the explicit wire cap (including Codex subscription Responses)
cannot serve bounded side runs; ordinary unbounded agents are unchanged. Native Gemini's
later thinking expansion is also checked. Trusted request middleware can still mutate
requests after assembly: this is not a sandbox against arbitrary installed plugin code.
Tools use the same bot/service/profile credentials and filesystem permissions.

The parent's actual cached prompt and tool schemas are retained. The legacy process-global
`model_tools._last_resolved_tool_names` can change when any new agent selects tools;
this capability does not promise immutable process-global diagnostics/defaults.

Tests cover real discovery, adapter/runner authentication and busy dispatch, SQLite
lineage/ownership, fake-transport AIAgent concurrency, profile secrets, approval request
ownership, and cooperative cancellation. All tests use the canonical `scripts/run_tests.sh`.
