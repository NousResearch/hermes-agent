# Native plugin idle continuation boundary

`on_session_idle` is an additive, synchronous PluginManager hook. It lets a
plugin continue a conversation after its own durable, owner-bound work finishes
(for example a consented setup job the model asked for).
It is not a generic job scheduler, event mailbox, process notification, or bot DM.
Only the TUI/Desktop native backend emits it in this slice.

## Host contract

The existing session notification poller offers the owning profile's discovered
plugins the normal session identity keyword fields plus
`submit(message, *, terminal_callback) -> bool`. Callbacks run on the poller's
thread and must be short: no installation, network operation, waiting for a model,
or retaining `submit`. The capability expires when hook dispatch returns and
rejects use from another thread. Normal scoped hook registration/unload applies.

`submit` checks the captured runtime identity/source/transport again and claims
`running` under the history lock. Running, closing, finalized, queued user work,
auto-continuation, absent agents and released leases defer it. The existing
`_run_prompt_submit` remains the final cross-process session-ownership admission
and executes the ordinary native turn, including history, profile scopes, stream
events, cancellation, crash marker and terminal callback. There is no mid-loop
injection, prompt/toolset rebuild, historical-message edit or bot author spoof.

False means no turn started and a plugin may leave its receipt pending. True means
started, **not completed**. The submit capability captures the existing queued-prompt
cancellation generation when claiming the turn and passes it to native admission.
If an explicit Stop wins that boundary, it raises `concurrent.futures.CancelledError`:
no turn started, and the consumer must record cancellation rather than retry pending
work. Native rejection and notification cleanup release `running` only while that
captured generation still owns the claim. A stale refusal or exception must not
mark a newer ordinary turn idle. Other thrown exceptions can be ambiguous. The terminal
callback reports `settled`, `failed` or `cancelled`; it runs on the turn thread.
A plugin must commit its durable terminal receipt before returning. A failed
receipt write leaves the existing non-auto-replay crash marker in place.

This does not change `PluginContext.inject_message`, its CLI semantics, gateway
registration or `allow_gateway_injection`. The consumer must require a real
request and its own domain consent; enabling a plugin is not proof of a request.

## Consumer guidance

- Use the existing `on_session_identity` publication for trusted native ownership
  provenance and remember it by durable owner, never by the foreground chat.
  Model-tool dispatch supplies only session/task IDs, so provenance cannot be taken
  from extra handler kwargs.
- Record continuation intent only for an explicit request (for example a tool or
  target-resolver result that says setup is required), never for manual commands,
  status reads or ordinary host tools. Retire it on session finalize.
- Store the intent with the owner, profile home, source/surface, stored/runtime IDs
  and a generation fence, **not a prompt, command or click**.
- Before submitting, recheck permission, disablement, generation and service
  lifetime under the plugin's own guards. Send a short readiness fact asking the
  model to reevaluate its **current** task and continue only if still needed,
  never to replay an action.
- Keep setup outcome and turn outcome separate. Write an `uncertain` state before
  native admission, restore `pending` on an explicit False, and replace it with the
  terminal receipt. A crash or lost terminal receipt stays uncertain and is **never
  automatically retried** (at-most-once attempts, not exactly-once effects).

## Qualification

`tests/tui_gateway/test_plugin_idle_continuation.py` exercises the host boundary
with a neutral in-process hook: one admitted native turn per idle poll, deferral
for busy sessions, and refusal of a `submit` retained past the hook.
