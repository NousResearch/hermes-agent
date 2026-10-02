---
title: "Gateway Request Lifecycle"
description: "Profile-scoped request controls, guarded interim delivery, and final formatting"
---

# Gateway request capability

A profile plugin can narrate an authorized gateway request while it waits in a queue or runs.
The gateway owns identity, authorization, admission, cancellation, output readiness, and transport.
A plugin supplies text and request-local policy. No new model tool or business permission is added.
This capability is ephemeral and has no restart recovery or global preference storage.

Register optional native hooks through `register(ctx)`:

```python
def register(ctx):
    ctx.register_hook("gateway_request_lifecycle", on_request)
    ctx.register_hook("gateway_request_control", on_control)
    ctx.register_hook("gateway_request_final", format_final)
    ctx.register_hook("gateway_request_tool", observe_tool)
```

The supported handle is the `request` supplied by these hooks. Plugins do not call runner methods,
replace core methods, or construct request contexts. Profiles without a subscriber retain existing
message dispatch. Callback failures use the native plugin dispatcher's isolation and timeout policy.

## Admission and observations

`gateway_request_lifecycle(request, stage)` accepts sync or async callbacks. Admission occurs after
authorization and correlated approval/clarification handling, before preparation or either gateway
busy queue. The same event retains its handle when executed later. `request.facts` provides:

- `request_id`, `session_key`, `profile_home`, `runtime_profile`, `transport_profile`
- `platform`, `chat_id`, `thread_id`, `requester_id`, `message_id`, `text`
- `admitted_at`: monotonic admission time, including queued time

Facts originate in the gateway's trusted source and routing identity, never ingress metadata. The
facts object is frozen; the runtime replaces it when a retained queue event absorbs a revision.
`request.run_generation` is `None` while queued and binds when execution starts. The generation and
receiving adapter belong to this request; an ambient profile switch does not alter them. Callbacks
and delayed sends re-enter the original runtime profile scope.

Stages are `admitted`, `running`, `control`, `revised`, `final_ready`, `completed`, `failed`,
`cancelled`, `superseded`, `merged`, and `rejected`. `final_ready` and other terminal stages close
interim publication synchronously, before asynchronous lifecycle notification or final delivery.
Lifecycle notifications after admission are scheduled on the admission loop. Observe the `stage`
argument for the transition and `request.active` for whether publication is still allowed.
Terminal contexts leave the live registry, while their event-bound handle remains available for
transport receipts. `delivered` and `delivery_failed` are independent final transport observations.

`request.state` is an initially empty, plugin-owned dictionary for this request only. Do not store
credentials, raw staff records, or global preferences there. Timer tasks started by the plugin
should cancel on a terminal stage. The runtime guard also rejects delayed callbacks after closure.
Queue merges supersede the old narration owner and transfer the revised handle to the retained
queue event, preserving the earliest admission clock for the same trusted requester. Existing
adapter merge paths can combine distinct requesters in a shared session; when both events have
live request handles, the runtime keeps those distinct owners in the existing FIFO before merging.
Both remain independently narratable and execute with their original requester and delivery target.
This exception affects subscribed requests only; no-subscriber merging stays unchanged. A replacement
closes the displaced request. Successful steer or
redirect transfers run ownership; an actual interrupt closes obsolete narration before interruption.

## Interim publication and receipts

`await request.send_interim(text)` publishes using the captured receiving adapter, chat, reply
anchor, and thread metadata, always with `_interim_send=True`. It returns `True` only when the
transport returns `success=True`; a refused/failed/obsolete send returns `False`. Call it on the
admission loop, including from a timer created by an async admission callback. No agent needs to
exist. The API does not authorize contacting another destination or performing business actions.

Publication rechecks active status and run generation immediately before invoking `adapter.send`
under the same grant lock used to close publication. That invocation is the irrevocable initiation
boundary; wire bytes or observed delivery can happen later. A final-ready transition that wins
before invocation suppresses the send. Already initiated transport work cannot be recalled. The
lock is never held while awaiting transport I/O.

`request.interim_receipts` contains immutable receipts with monotonic `initiated_at`,
`delivered_at`, and `success`. A suppressed attempt produces no receipt; transport failure has no
`delivered_at`. `request.final_ready_at` records readiness independently of transport.
`request.final_receipts` uses the same shape for ledgered text, inline replies, queued text, and
confirmed streamed queued finals. Initiation is `None` for a stream already confirmed when the
queued delivery boundary observes it. These are process-local facts, with no telemetry export.
Media-only delivery and later durable-ledger redelivery are outside these text receipt measurements;
consumers must leave unavailable delivery timings unknown.

## Pure controls

`gateway_request_control(control, requests)` runs after existing correlated prompt replies and
before queue, steer, or interrupt dispatch in both ingress paths. `control` contains text,
message ID, reply-to message ID, and trusted requester ID. `requests` is a tuple of live handles
matching the requester, runtime profile/home, receiving adapter, platform, chat, thread, and
session. A shared conversation is insufficient to establish requester ownership.

Return `None` to preserve normal handling. A pure control returns:

```python
return {"request_id": target.facts.request_id, "state": {"detail": "expanded"}}
```

The runtime rechecks the selected live candidate, applies its state update, emits `control`, and
consumes the message without another model turn or business-tool execution. The consumer owns
phrase recognition, explicit reply-anchor correlation, and ambiguity decisions. A plain control
requires an unambiguous request. A reply to settled work must not retarget another current request.
Mixed business corrections and ordinary questions must fall through. No terminal handle is a
candidate. The generic core contains no language-specific detail phrases or profile policy.

## Tool evidence and final formatting

`gateway_request_final(request, response)` returns an optional string. It receives the latest
request state at the output boundary before final-ready closure and stream sealing. It can format
the answer using already acquired evidence. It must not run business tools, mutate the cached
system prefix or transcript, inject a synthetic user turn, or replay the control as steering.
The final string is shared by streaming and non-streaming delivery. No subscriber preserves the
original response.

`gateway_request_tool(request, stage, tool_name, args, result=None)` observes registry handler
invocation. `started` occurs inside the tool-execution middleware's dispatch closure, after tool
request middleware and existing pre-dispatch guards. A pre-tool veto or execution-middleware block
emits no execution observation. `completed` carries the returned result; it does **not** establish
success or authorize a public completion claim. Handler-level authorization refusals and error
results still require validation. `failed` means invocation raised. Agent-inline tools and catalog
bridge reads are outside this seam; it adds no model tool, permissions or tool schema.

The turn binds a ContextVar observer factory. Each invocation resolves the currently active trusted
request for the session/generation at **start**, then retains that exact handle through completion.
A successful correction/redirect can bind later checks to its replacement request; a late old check
cannot publish into the new owner. Tool-worker context propagation preserves profile and generation.
Synchronous observers run under the original request profile and the existing plugin timeout policy.
They should validate/project findings promptly and schedule guarded sends on the admission loop.
The Tito consumer captures that loop during admission; it reads latest detail when publication runs.

`args` and `result` are private tool data, potentially containing staff records and credentials.
Never copy raw fields into public messages or persistent state. Consumers must project validated
public findings and retain missing fields as unknown. A permitted start supports only checking;
returned employee rows do not prove availability, hours, acceptance, outreach or assignment.

Before the final formatter runs, `request.state["failure"]` is set to `provider` for failed model
results or `execution` for a request-bound escaping execution exception. Early provider/credential
resolution failures set `provider_resolution` before their existing reply reaches final formatting. These safe failure kinds
never contain raw exception text. Normal failure delivery retains the existing failed flag and
stream error behavior, while a consuming formatter can close with acquired facts and unresolved
checks. Without a request subscriber, escaping exceptions keep their original behavior.
`TurnContext.request_context` remains the optional owning handle for downstream turn callbacks.

Related upstream [PR 130904](https://github.com/NousResearch/hermes-agent/pull/130904) proposes an
idle post-admission consuming hook for durable plugin work. This capability supplies ongoing
request ownership, both busy guards, queued timing, and guarded interim delivery. It does not
introduce another generic idle handler for durable work.
