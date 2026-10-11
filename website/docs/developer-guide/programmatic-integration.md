---
sidebar_position: 8
title: "Programmatic Integration"
description: "Three protocols for driving hermes-agent from external programs: ACP, the TUI gateway JSON-RPC, and the OpenAI-compatible HTTP API"
---

# Programmatic Integration

Hermes ships three protocols for driving the agent from external programs — IDE plugins, custom UIs, CI pipelines, embedded sub-agents. Pick the one that matches your transport and consumer.

| Protocol | Transport | Best for | Defined by |
|----------|-----------|----------|------------|
| **ACP** | JSON-RPC over stdio | IDE clients (VS Code, Zed, JetBrains) that already speak the [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) | `acp_adapter/` |
| **TUI gateway** | JSON-RPC over stdio (or WebSocket) | Custom hosts that want fine-grained control of sessions, slash commands, approvals, and streaming events | `tui_gateway/server.py` |
| **API server** | HTTP + Server-Sent Events | OpenAI-compatible frontends (Open WebUI, LobeChat, LibreChat…) and language-agnostic web clients | `gateway/platforms/api_server.py` |

All three drive the same `AIAgent` core. They differ only in wire format and which set of features they expose.

---

## ACP (Agent Client Protocol)

`hermes acp` starts a stdio JSON-RPC server speaking ACP. Used in production by VS Code (Zed Industries' ACP extension), Zed, and any JetBrains IDE with an ACP plugin.

Capabilities exposed: session creation, prompt submission, streaming agent message chunks, tool-call events, permission requests, session fork, cancel, and authentication. Tool output is rendered into ACP `Diff`/`ToolCall` content blocks the IDE understands.

Full lifecycle, event bridge, and approval flow: [ACP Internals](./acp-internals).

```bash
hermes acp                  # serve ACP on stdio
hermes acp --check          # verify ACP dependencies and adapter imports
hermes acp --setup          # interactive provider/model setup for ACP terminal auth
```

---

## TUI Gateway JSON-RPC

`tui_gateway/server.py` is the protocol the Ink TUI (`hermes --tui`) and the embedded dashboard PTY bridge talk to. Any external host can speak the same protocol over stdio (or WebSocket via `tui_gateway/ws.py`).

### Method catalog (selected)

```
prompt.submit           prompt.background       session.steer
session.create          session.list            session.active_list
session.activate        session.close           session.interrupt
session.history         session.compress        session.branch
session.title           session.usage           session.status
clarify.lock            config.set / config.get commands.catalog
client.capabilities     gateway.capabilities    ping
command.resolve         command.dispatch        cli.exec
reload.mcp              reload.env              process.stop
delegation.status       subagent.interrupt      subagent.steer
spawn_tree.save / list / load
terminal.resize         clipboard.paste         image.attach
```

`session.active_list`, `session.activate`, and `session.close` are the process-local live-session controls used by the TUI session switcher. Use `session.list` / `/resume` for saved transcript discovery; use the active-session methods only for sessions that are currently open in the TUI gateway process.

Within one authenticated gateway, resuming or activating a live session attaches another event subscriber rather than replacing the previous connection. Streaming and terminal events go to all attached clients; disconnecting one client does not end a session another client is viewing. Existing submit exclusivity and configured busy-input policy remain in force. Attached clients can steer the session's subagents; browser-controller results still require the connection that registered that controller. This does not enable independent gateway processes to write the same session, nor does it imply durable prompt admission across an owner restart.

### Model overrides on `session.create`

`session.create` accepts per-session `model` / `provider` overrides. A pair the provider cannot serve (`model: gpt-5.5` with `provider: anthropic`, or with no `provider` when the profile's configured provider is Anthropic) is refused up front with JSON-RPC code `-32602` instead of minting a session whose first turn fails at the provider; `error.data` carries `model`, `provider` and up to five `suggestions` from that provider's catalog, and `error.message` repeats them. The check is offline and only refuses names Hermes knows belong elsewhere: custom endpoints (`custom`, `custom:<name>`), aggregators (OpenRouter, Nous, …), models in the provider's own family that the curated list has not caught up with, and names no catalog lists are all accepted as before.

### Rewinding history on `prompt.submit`

A rewind / edit / regenerate is a `prompt.submit` that drops part of the stored transcript before running the new turn. Because that write is a destructive rewrite of the session's durable rows, the gateway honors it only when the client states its intent:

| Parameter | Meaning |
|-----------|---------|
| `truncate_before_user_ordinal` | Zero-based index of the user turn to cut at. Everything from that turn onward is dropped. Display-only timeline rows (`display_kind`) are not counted. Must be a real integer — a JSON boolean is refused with code `4004`. |
| `truncate_before_row_id` | Integer SQLite row ID (`messages.id` / `row_id`) of the target user turn to cut at. Preferred durable address. When both ordinal and row ID are provided, gateway verifies they match (returning `4030` on mismatch). An unknown/stale row ID is refused with `4018` — it does **not** fall back to the ordinal. |
| `confirm_truncate` | Required whenever an ordinal, message ID, or row ID is sent. Declares that this submit really is a rewind, not an ordinary send that happens to carry leftover parameters. Sending it without a target is refused with code `4004`. |
| `confirm_empty_truncate` | Additionally required when the cut would leave the transcript empty (ordinal `0`). |
| `confirm_deep_truncate` | Send it when the cut would archive more than one user turn (anything deeper than regenerating or editing the last turn), after asking the user. This release logs an unconfirmed deep cut and still runs it, so older clients keep working. The next release will refuse it with code `4033` (`error.data` carries `archived_messages` and `archived_user_turns`, and nothing is written). |

A truncation parameter without `confirm_truncate` is refused with code `4004` or `4029` and nothing is written. Hosts that implement rewind must set the flag at the moment the user asks for it, and must never keep truncation parameters in state across ordinary submits. Prefer `truncate_before_row_id` (from resume `row_id` / `_row_id`) over ordinals; keep the ordinal as a back-compat / optimistic-row path only when no durable id is available yet.

A truncating submit is never absorbed by the busy-input policy. While a turn is still running, an ordinary `prompt.submit` is steered, redirected, or queued (`display.busy_input_mode`), but a rewind / edit / regenerate refuses with code `4009` (`session busy`) instead — queueing it would drop the history cut and run the edit as a plain follow-up after the un-edited turn. Hosts call `session.interrupt` and retry the same submit until it lands; the Desktop app does this automatically, so editing a message while Hermes is still thinking stops the live turn and reruns from the edited prompt.

On a successful truncating submit against a durable session, the `prompt.submit` result additionally carries `survivor_user_row_ids` — the fresh post-rewrite row IDs of the surviving user turns, in visible-user-ordinal order. The rewrite re-inserts the kept prefix as new rows, so every row ID the host cached before the rewind is stale afterward; rebind cached IDs from this list (a `null` entry means that turn has no durable ID — drop the cached one) or the next rewind targeting an older surviving turn will be refused with `4018`.

### Events streamed back

`message.delta`, `message.complete`, `tool.start`, `tool.generating`, `tool.complete`, `gateway.ready`, `request.cancel`, plus session lifecycle and error events.

### Server→client requests (questions the agent asks you)

Approvals, clarify questions, sudo/secret prompts, vault unlock, MCP setup and the desktop read/act bridges are **JSON-RPC requests from the gateway to the client**, not events. The frame carries a string id, and the client answers with a normal JSON-RPC response bearing the same id:

```
← {"jsonrpc":"2.0","id":"srq-7","method":"approval","params":{"session_id":"…","request_id":"…","command":"rm -rf build","description":"…"}}
→ {"jsonrpc":"2.0","id":"srq-7","result":{"choice":"once"}}
```

Methods: `approval` → `{choice}`; `clarify` → `{answers}` keyed by question id, or `{}` to cancel (`clarify.lock` locks one answer early; `null` skips that question); `sudo`, `secret`, `vault.code`, `vault.unlock_prompt` → `{value}`; `connection` → `{settled_by, targets}` (the `manage_connections` card: one outcome per target); `terminal.read`, `window.read`, `preview.act`, `tour` → `{value}` (JSON text). Respond with a JSON-RPC error (`-32601`) for a method your host does not implement so the agent fails fast instead of waiting out the timeout.

**Advertise that you answer them (breaking for existing WebSocket integrations).** Once per connection, after `gateway.ready`, call `client.capabilities` with `{"server_requests": true}`; the result lists the request methods this backend may send. A WebSocket client that never does is treated as a build that predates server→client requests: the gateway fails every such request for it immediately (the agent sees the same "no answer" an error response produces; an approval is withdrawn, not denied) instead of stalling for the full deadline. There is no grace path — a third-party WebSocket client that answered `clarify`/`approval`/`sudo`/… before this change but never sends `client.capabilities` now has every such request refused until it adds the one call. A session with no client attached is not affected — its open questions wait in `open_requests` for the reconnect replay. The stdio TUI, the desktop app and the dashboard advertise through the shared `JsonRpcRequestChannel`.

When the gateway withdraws a question (timeout, interrupt, answered from another surface) it emits `request.cancel` `{ id, method, reason }`; clear only the matching prompt. `session.resume` / `session.activate` results and `session.events.since` carry `open_requests` — the still-open frames — so a reconnecting client re-renders (and can still answer) them.

### Authenticated creation provenance

`session.create` and `session.branch_stored` may return `creation_binding`:
`session_id`, `stored_session_id`, `authenticated_owner`, `runtime_incarnation`
and `profile_store_scope`. The owner is the gateway-authenticated
`<provider>:<user id>`, not the model provider. Anonymous/legacy transports and
unbounded identity values receive no binding. The incarnation is minted for the
new live record independently of its short runtime ID and process replay epoch;
a new record created through these methods mints another incarnation even under
the same runtime ID. Other creation/resume paths do not acquire this provenance.

The scope token names that runtime's captured resolved profile and `state.db`
location without exposing a host path. It is an opaque per-origin identifier,
not a lookup key for comparing independently created sessions or proving the
physical database file has not been replaced. It does not grant authentication
or access. No extra durable store is introduced.

An idempotent creation retry returns the original binding only to the same
authenticated creator requesting the same resolved profile/store scope. The
binding never restamps origin from a later compression segment, profile or login
on the mutable runtime. Consequently its original `stored_session_id` may differ
from a retry's current top-level `stored_session_id`; a client must not interpret
that as permission to follow a descendant. Each response carries a fresh copy.

Creation metadata alone does not prove current membership or authorize recovery.
Legacy `session.activate` is unchanged and does not compare these fields. Clients
must not send invented preconditions to it or adopt a runtime from discovery.

### Conditional subscription prerequisite

`session.activate_bound` accepts `{session_id, expected_binding, profile?}`.
`expected_binding` must contain **all five** creation-binding fields as bounded
strings; unknown, missing, wrong-type or oversized values fail with `4000`.
The gateway derives the caller principal from its authenticated transport. A
well-formed mismatch, anonymous/foreign/dead peer, unavailable record, or unproven
runtime fails with `4007`, without a session snapshot or fallback. An unavailable
named profile retains the normal `4064` profile error.

This bounded contract supports **unbuilt idle creation records and their original
healthy ready local engines**, including an ongoing turn with an active local engine
lease holder. The first engine attachment captures an immutable engine object,
store object and identity revision for that creation record. Attachment retries
cannot refresh this witness. A removed/replaced engine, changed revision (even
if the segment ID changes back), changed engine/store scope, failed/interrupted
turn or unready/unproven engine refuses. Original ready compute-host engines can
also qualify through the child-authoritative protocol below. An unbuilt record
routed to a child still refuses: activation never starts, resumes or rebuilds an
engine to manufacture proof. Any attempted child lifetime permanently invalidates
local eligibility, even after a failed send or parent-mirror ABA. A ready local
engine stays local under the existing routing policy.
An accepted prompt still waiting for its engine turn lease is not certified as a
healthy running engine.

`AIAgent` inherits the local boundary in `agent/session_identity.py` through its
persistence mixin. Segment assignments take `session_identity_guard()` and
advance `session_identity_revision` only when the ID changes. Compression child
publication/adoption and persistence-tip adoption hold the same reentrant guard
over DB work and engine state updates. Post-lease-admission reload now also
holds this guard over presence checks, tip resolution, adoption and transcript
loading; the potentially long lease acquisition wait stays outside it. Ordinary
reads retain their historical behavior. Duck-typed non-engine callers retain
legacy behavior and provide no proof of this boundary.

Conditional activation tries that guard without waiting while holding the gateway
guards, then compares the original engine/store/revision and both engine and
gateway segment IDs before subscription. It retains the engine guard through
receipt capture. A busy engine transition refuses instead of waiting on callbacks
that might need gateway locks. A healthy turn continues on its existing worker
and lease, with existing subscribers retained; activation starts no execution.

The parent mirror alone supplies no certificate of compute-host identity.
Complete recovery, durable request outcomes and downstream
supported-release/pin qualification remain separate gates. Method presence or a
subscription receipt alone must not be interpreted as writable recovery support.

For a supported record, registry object membership, creator, requested and current
resolved store, original exact stored segment, incarnation and scope are compared
under resume/history/registry/transport guards before subscription. Copying origin
metadata onto a replacement record cannot pass. Gateway compression key writes
now use history_lock; compute-host key adoption already uses it. Existing
subscribers remain attached, and only a successful transport attach adds a viewer
and cancels orphan reap. Repeating the same successful request preserves one
subscriber membership and starts no execution.

Success returns `{attached: true, accepted_binding: {...}}`, captured within that
comparison/subscription boundary. It contains no mutable history, inflight state
or pending requests. The receipt certifies that historical cut, not future
connection liveness, complete recovery or input authority. Conditional operations
below revalidate identity independently at use. Never automatically replay prompts or responses or fall
back to legacy activation/cold resume after refusal or a lost receipt.

### Compute-host authority

The public envelope and accepted binding are the same for local and host engines.
Private pipe tokens are never client credentials. The first deliberate ordinary
host turn carries the immutable creation origin and pins the actual child boot.
Only that first child record can capture the original engine, database object,
identity revision and exact segment. Failed capture is retained as a tombstone;
ordinary dispatch, replacement, restart and away/back identity changes cannot
recertify it. A new authenticated creation can qualify in a new child lifetime.

Activation queries only the existing pinned child. Two private reservation
workers compare the complete binding, owner, resolved profile/store, original
record/engine/revision and strict readable durable tip under the child's
resume/history/registry/transport guards and nonblocking engine guard. A prepared
reservation holds that cut for at most five seconds pending commit. Settling,
failed, interrupted, missing or changed authority refuses; a healthy running
engine keeps its worker and lease.

The parent separately compares its original record, owner, scope, segment and
child boot. It installs a tentative delivery sink, queues commit inside that cut,
and waits outside gateway locks. The child commits one logical subscription at
its held identity cut; the parent publishes membership only after receiving the
matching receipt and revalidating its own cut and the actual peer attachment.
Tentative sinks do not count as client liveness and cannot suppress orphan
reaping. Their events are dropped, so recovery must retain explicit loss/gap evidence.
Refusal exposes no snapshot or tentative events and leaves parent viewers and
orphan-reap state unchanged. Only the tentative sink is removed; existing peers
are retained. Abandoned reservations expire; unused logical child subscriptions
are released asynchronously. Child membership is also leased: a child-clock lease
lasts at most 15 seconds without fresh parent confirmation. Every two seconds the
child challenges the parent with a boot-pinned snapshot of subscription tokens.
Only exact peers still registered on their original parent records can be confirmed.
A single bounded private response renews that snapshot; its deadline is measured
from challenge issuance, so delayed/replayed replies cannot extend expired authority.
A confirmed peer omitted from a fresh response is revoked; the first commit has only
its bounded initial lease while the parent publishes membership. Direct detach release
is a prompt cleanup hint, not the lifetime authority. A dropped/full/unavailable writer
cannot strand live authority indefinitely: expiry makes the logical peer dead before
any transport-lock cleanup, removes its retained token and frees the 64-member cap.
Physical transport removal follows asynchronously without restoring concurrent peers.
Expiry never cancels or retries an accepted operation; reactivation independently
compares identity and obtains a new subscription if its old lease has expired. Repeating an accepted activation on the same peer
reuses its child membership token. A peer-only activation gate serializes
concurrent retries without holding gateway guards. Repeating after a lost receipt can reconcile
membership, but cannot prove history or operation outcomes.

Conditional pipe sends use a bounded eight-packet queue and capture the exact
process object: blocked I/O holds no parent gateway lock, and queued controls can
never migrate to a replacement process. Reservations and subscriptions are private and boot-pinned; commits are never
automatically retried. The child limits retained subscriptions to 64.
RPC relay frames carry the actual boot internally. New-lifetime frames revoke
old conditional delivery membership before forwarding; internal tags are stripped
before client delivery. Queries never start or respawn a host.

All four conditional operations use the same child-held reservation and compare
accepted membership and identity again through the immediate handler. Prompt
admission additionally carries its witness through both production worker hops
and the durable lease wait. After acquiring the lease, while retaining the engine
identity guard, it obtains a generation-pinned parent admission decision for that
specific pending turn, then verifies the exact engine/store/revision and strict
tip before execution. Changed/retired/cancelled parent authority refuses; the
newly acquired lease is released. Successful admission consumes the witness
before model or delegated work. An accepted user row can remain after refusal;
it proves neither execution nor completion and must not be replayed.

A missing commit acknowledgment or partial pipe failure returns `5019` for an
operation whose outcome is unconfirmed. The caller must not infer refusal or retry
that operation. `4007` remains enforceable identity/unavailability refusal before
operation commit; `4009` covers busy plain-text submission and reservation capacity.
An activation with an unconfirmed child receipt grants no parent authority; an
exact activation retry is reconciliation, never operation replay. These controls
do not add cold attachment, request revision CAS, durable outcomes or complete
recovery. Supported upstream publication/capability and downstream qualification
remain independent gates.

### Conditional operations

After accepting `session.activate_bound`, the connection enters conditional mode.
`session.invoke_bound` requires `{session_id, expected_binding, profile?, operation}`
and a live subscription installed for that exact binding on the calling peer.
Its closed, discriminated operation catalog is:

```json
{"method":"prompt.submit","text":"a deliberate new prompt"}
{"method":"session.interrupt"}
{"method":"request.answer","id":"srq-...","result":{"answer":"reviewed answer"}}
{"method":"clarify.lock","request_id":"srq-...","question_id":"q1","answer":"chosen"}
```

The envelope returns `{operation_result: ...}` with the underlying operation's
result. Missing, malformed, oversized or unsupported operation fields return
`4000`. Identity, owner, scope, incarnation, engine revision/store, subscription
or pending-request ownership mismatch returns `4007` before mutation; there is
no discovery, durable-ID fallback, engine bootstrap or replay. Unreadable durable
lineage is refused, using the strict form of the existing resume-tip lookup.
Expired/missing request IDs also refuse; absence is not evidence an earlier answer
was accepted. Request identity lookup and settlement share the request registry
lock. This does not add revision-CAS for editable request params or outcome lookup.

An original ready local engine or a child-authoritatively qualified original
ready compute-host engine can execute an operation. Prompts are plain
text deliberate new **idle** turns without staged attachments; busy input returns
`4009` and cannot steer, interrupt, queue or persist a follow-up. Rewinds, media
staging, slash commands, subagent controls and other mutators are outside this
catalog. Conditional mode is sticky for the connection, including after removal
or replacement of its runtime. Legacy mutators and raw JSON-RPC response frames
cannot bypass the envelope. The gateway permits the documented informational
history/replay/list reads, creation and capability/heartbeat calls; their data does
not grant authority. Other clients retaining their legacy mode keep their behavior.

The operation holds the gateway comparison guards and a nonblocking engine guard
through the existing immediate handler's mutation. Creation-record history and
resume locks are reentrant so those handlers participate in this cut. The envelope
runs in the RPC pool; it never waits for an engine build, a model turn or a host RPC
under the cut. A prompt's worker carries an immutable engine/store/revision witness
across both worker hops and the durable lease wait. After durable admission, the
engine checks that witness and exact readable stored tip before adoption/execution;
it releases a newly acquired lease on refusal and cannot carry that unadmitted input
into a later turn. Successful admission consumes the witness before model work or
background threads, so legitimate delegated engines do not inherit the parent pin. The model loop may compress normally within a turn already
admitted against the original identity. Later operations must still refuse the
old binding after rotation. Admission is not a delivery/completion guarantee: a
prompt accepted before worker refusal can retain its original submit row/error;
never retry it automatically or infer a provider executed it.

Compute-host support uses the child-authoritative identity/subscription and
operation-admission protocol described above. The four operation envelopes work
against both qualified topologies; parent mirrors or process liveness alone do
not certify a host engine. Unbuilt/unproven host records, replacement/restarted
lifetimes and changed engine revisions still refuse. Clients must independently
qualify the supported contract and quarantine superseded connection generations
before consuming it. Full recovery and request/outcome settlement remain separate.

### Rebuilding the in-flight turn on reconnect

`session.resume` / `session.activate` results carry `inflight` — the turn still running (or the retained failed one) that history does not hold yet: `user`, `assistant` streamed so far, `streaming`, mid-turn `corrections`, and error fields. When the turn was started by the gateway rather than typed by a person (a background-process completion, an async delegation result, a hidden scaffolding prompt) `inflight` also carries the same `display_kind` / `display_metadata` the persisted `messages` row will get, so a client renders the live prompt exactly as it will render history after the turn lands — a `process_complete` timeline marker with `display_metadata.display_text`, nothing at all for `hidden`. Both fields are absent for genuine user input; never infer origin from the prompt text (a user quoting a marker string is still a user).

### Pi-style RPC mapping

Every command in the Pi-mono RPC spec ([issue #360](https://github.com/NousResearch/hermes-agent/issues/360)) has a TUI-gateway equivalent:

| Pi command | Hermes equivalent |
|------------|-------------------|
| `prompt` | `prompt.submit` (or ACP `session/prompt`) |
| `steer` | `session.steer` |
| `follow_up` | `prompt.submit` queued after current turn |
| `abort` | `session.interrupt` |
| `set_model` | `command.dispatch` for `/model <provider:model>` (mid-session, persistent) |
| `compact` | `session.compress` |
| `get_state` | `session.status` |
| `get_messages` | `session.history` |
| `switch_session` | `session.resume` |
| `fork` | `session.branch` |
| `ui_request` / `ui_response` | server→client requests `clarify` / `sudo` / `secret` / `approval` answered by JSON-RPC response frames |

---

## OpenAI-Compatible API Server

`gateway/platforms/api_server.py` exposes hermes over HTTP for any client that already speaks the OpenAI format. Useful when you want a web frontend, a curl-driven CI runner, or a non-Python consumer.

Endpoints:

```
POST /v1/chat/completions        OpenAI Chat Completions (streaming via SSE)
POST /v1/responses               OpenAI Responses API (stateful)
POST /v1/runs                    Start a run, returns run_id (202)
GET  /v1/runs/{id}               Run status
GET  /v1/runs/{id}/events        SSE stream of lifecycle events
POST /v1/runs/{id}/approval      Resolve a pending approval
POST /v1/runs/{id}/steer         Inject mid-run guidance at the next tool boundary
POST /v1/runs/{id}/stop          Interrupt the run
GET  /v1/capabilities            Machine-readable feature flags
POST /v1/browser-control/register Register a browser controller
GET  /v1/browser-control/ws       Browser-controller WebSocket
GET  /v1/models                  Lists hermes-agent
GET  /api/model/options          Provider-aware picker inventory
GET  /health, /health/detailed
```

Setup, headers (`X-Hermes-Session-Id`, `X-Hermes-Session-Key`), and frontend wiring: [API Server](../user-guide/features/api-server).

Browser extensions can opt into the disabled-by-default controller protocol to
drive the exact browser session that opened the Hermes conversation. The API
and dashboard transports share one principal-bound broker and one explicit
capability allowlist; see [Browser-extension control](../user-guide/features/api-server#browser-extension-control).

### Model catalog surfaces

The OpenAI-compatible API intentionally keeps `GET /v1/models` minimal: it is
the compatibility endpoint frontends expect, not the full Hermes provider/model
picker catalog.

If an external control plane needs Hermes' curated provider rows, per-model
pricing, or capability hints, use one of the authenticated picker surfaces:

- API server REST: `GET /api/model/options` with the API-server bearer key
- Dashboard backend REST: `GET /api/model/options` with `X-Hermes-Session-Token`
- TUI gateway RPC: `model.options`

Those surfaces share the same payload builder and the same custom-provider
probe policy:

- Normal open: probe only the current custom provider so offline saved
  endpoints do not stall the picker.
- Explicit refresh (`refresh=1` or `refresh: true`): bust the provider-model
  cache and probe all saved custom providers so live catalogs repopulate fully.

Use `/v1/models` for OpenAI-client compatibility. Use `/api/model/options` or
`model.options` when you are building a Hermes-aware model picker.

`POST /v1/runs/{id}/steer` is the HTTP equivalent of Hermes `/steer`: it does not create a new user turn or immediately rewrite the assistant output already in flight. Instead, the text is appended to the live run and becomes visible to the agent after the next tool boundary, so it can course-correct without discarding the current tool-calling loop.

`/v1/runs/{id}/steer` is only accepted while the run status is `running`. Queued, approval-paused, stopping, cancelled, failed, and completed runs return `409 run_not_accepting_steer`, even if the server still retains internal agent references during cooperative shutdown.

A `200` (and the `run.steered` event) means the text was **queued**, not that the agent consumed it. If a steer lands after the agent's final response — with no later tool boundary to deliver it at — the undelivered text is returned as `pending_steer` on the terminal event (`run.completed`, `run.failed`, or `run.cancelled`) and run status, so the client can replay it as the next user turn instead of losing it.

#### Terminal run status

The terminal status of a run is derived from how the agent's turn actually ended, and the terminal event name always matches it (`run.<status>`):

| Turn outcome | Status | Terminal event | Flags on the event / status |
|---|---|---|---|
| Final answer produced | `completed` | `run.completed` | `completed: true` |
| Interrupted (`/stop`, or an interrupt inside the agent) | `cancelled` | `run.cancelled` | `completed: false`, `interrupted: true`, `turn_exit_reason` naming the issuer — `interrupted_by_user` for a human stop, `interrupted_by_system(<issuer>)` / `interrupted_during_api_call(<issuer>)` when a watchdog (e.g. `cron_inactivity_watchdog`, `turn_liveness_watchdog`, `gateway_inactivity_watchdog`, `session_turn_lease_lost`) ended the turn |
| Provider/agent failure | `failed` | `run.failed` | `completed: false`, `error` |
| Gateway shut down while the run was active (`/v1/runs` only) | `interrupted` | `run.interrupted` | `error: "Gateway shutdown interrupted the run."` — recorded before the agent is interrupted and never overwritten by the turn's late result; a client `/stop` still settles as `cancelled` |
| Ended without finishing (iteration budget, truncated or partial reply) | `failed` | `run.failed` | `completed: false`, `partial` when applicable, `turn_exit_reason` (e.g. `max_iterations_reached(60/60)`), `output` with any fallback text |

A run is never reported as `completed` with `completed: false` or `partial: true` in the same payload. The same rule applies to `/api/sessions/{id}/chat/stream`, whose `assistant.completed` payload carries the real `completed` / `partial` / `interrupted` flags and whose terminal event is `run.completed`, `run.failed`, or `run.cancelled`.

---

## Which one should I use?

- **You're writing an IDE plugin and the IDE already speaks ACP** → ACP. Zero protocol work on the IDE side.
- **You're writing a custom desktop / web / TUI host and want every Hermes feature** (slash commands, approvals, clarify, multi-agent, session branching) → TUI gateway JSON-RPC.
- **You want any OpenAI-compatible frontend, a language-agnostic HTTP client, or curl-driven automation** → API server.
- **You want a Python in-process embed without a subprocess** → import `run_agent.AIAgent` directly. See [Agent Loop](./agent-loop).

---

## Model hot-swapping

Mid-session model switching works on every surface — it's the `/model` slash command under the hood.

- **CLI / TUI:** `/model claude-sonnet-4` or `/model openrouter:anthropic/claude-sonnet-4.6`
- **TUI gateway RPC:** `command.dispatch` with `{"command": "/model claude-sonnet-4"}`
- **ACP:** the IDE sends the slash command as a prompt; the agent dispatches it
- **API server:** include a `model` field in the request body

Provider-aware resolution (the same model name picks the right format for whatever provider you're on) is built in. See `hermes_cli/model_switch.py`.

---

## A note on `--mode rpc`

Hermes does not have a `--mode rpc` flag. The three protocols above already cover the use cases — ACP for IDE-protocol clients, the TUI gateway for stdio JSON-RPC hosts, and the API server for HTTP. If you find a real gap that none of them fill, open an issue with the concrete consumer you're building.
