# Shared-session input and execution observations

Attached clients already share assistant events through the session fan-out transport. Version 1 also exposes starting inputs, visible corrections, and volatile input correlation. These are additive observations of existing work; clients must consume them to display the new information.

## Identities

`gateway.ready.payload.shared_session` advertises `{"version":1,"socket_id":"S1"}`. The server creates a new socket ID on every connection. This is separate from the Desktop registry's existing `connection_id` (box identity). Neither grants authority.

Turn-bound events have optional `params.turn`:

```json
{"id":"T1","source":{"kind":"connection","socket_id":"S1"}}
```

An execution ID identifies a runner invocation. It is created at inline admission or compute dispatch and carried across the compute-host boundary. A proven failed dispatch that falls back inline reuses it. It is not a durable message ID, admission receipt, or exactly-once guarantee.

`source.kind` is `connection`, `mixed`, or `unknown`; only `connection` has `socket_id`. It describes the starting envelope. Queue merges combine contributor provenance; later corrections do not rewrite it. Socket provenance is independent of the existing internal per-turn author channel. Missing context has unknown source: there is no session-wide fallback. Child-session events do not borrow the parent's execution.

Clients may send `submission_ref` on `prompt.submit`, `session.steer`, and `session.redirect`: 1–64 printable ASCII characters. Invalid references are ignored without changing work. Mint and retain a reference locally **before sending**, so a lost reply does not lose all correlation. Reusing a reference does not deduplicate requests: every accepted RPC occurrence receives a fresh server input ID.

Scope these identities and state revisions by backend replay epoch and runtime session. A matching locally retained reference is evidence about a submitted input; socket equality identifies the current connection. Socket inequality proves neither another device nor another person. Ownership remains unknown without positive evidence.

## RPC replies

Existing `status` strings remain unchanged. Replies add:

```json
{"status":"streaming","submission":{"input_id":"I1","ref":"R1","disposition":"starting"}}
```

Dispositions are `starting`, `queued`, `merged`, `absorbed`, `steered`, `redirected`, or `unresolved`. Omit the reference when absent or invalid. A correction reply includes `submission.turn` when its target is established. Scheduling may place start before or after the reply. A successful `session.steer` keeps legacy `status:"queued"`; its disposition distinguishes steering from next-turn queueing.

Errors preserve established fields and, when an input ID exists, add correlation at `error.data.submission`. Pre-start error events carry `inputs` and `inputs_complete`. Early validation and ownership refusals before accepting an occurrence need not mint an ID. A callout reporting acceptance after its target ends is `unresolved`; do not guess its target or automatically resubmit it.

## Events and display

One `message.start` precedes assistant output for each started execution:

```json
{"method":"event","params":{"type":"message.start","session_id":"s","turn":{"id":"T1","source":{"kind":"connection","socket_id":"S1"}},"payload":{"input":{"role":"user","text":"inspect the build"},"inputs":[{"id":"I1","ref":"R1"}],"inputs_complete":true}}}
```

`input` is the existing canonical transcript projection, or null when there is no displayable input. Expanded skills expose their invocation; hidden scaffolding is omitted. `inputs` identifies constituent occurrences. `inputs_complete` is independent of visibility and becomes false when evidence is missing or evicted. Hidden executions still carry IDs with `input:null`.

Accepted visible corrections add `message.input` with the same execution descriptor and this payload:

```json
{"kind":"redirect","input":{"role":"user","text":"use debug mode","display_kind":"steer"},"inputs":[{"id":"I2","ref":"R2"}],"inputs_complete":true,"offset":27}
```

`kind` is steer or redirect; projection uses existing steer-row rules. Offset counts Python string code points in assistant text and is only a placement hint. Input IDs distinguish identical corrections at the same offset.

Direct steer/redirect requests must declare `input_visibility:"visible"` to receive this new live text observation. Missing or unknown visibility retains legacy behavior without a new live text broadcast. `input_visibility:"hidden"` uses the hidden prompt path: queue while busy, start normally while idle. Hidden input does not interrupt unrelated execution. Desktop's hidden-input helper accepts both queued and idle streaming replies. Ordinary prompt.submit remains visible unless its existing hidden kind or projection says otherwise.

Queued work merges only within the same visibility kind. Visible and hidden envelopes remain separate, preserving visible words in durable history. Existing model-text merging and sanitization otherwise remain in force.

Terminal `message.complete` payloads retain old fields and add all retained input IDs associated with the execution, including accepted corrections. Delta, interim, and terminal publication is serialized for the captured session record. A callback finishing slow rendering after the terminal cannot publish or mutate the snapshot. Post-terminal housekeeping is unscoped; later goal/queue work has its own execution.

## Resume and positive outcomes

All live resume paths expose the same projection, including lazy sessions without a persisted row and profile-scoped sessions. Alongside legacy user, assistant, and streaming fields, in-flight state contains `turn` when known, canonical `input`, `inputs`, and `inputs_complete`. Deferred agent initialization can expose input IDs before execution identity exists. Legacy user coercion remains compatible except that a null display projection suppresses hidden text.

Visible corrections appear in `inflight.input_observations`, using their event payloads, with `input_observations_complete`. Existing corrections and correction_offsets remain intact. Legacy untyped correction text does not become a new typed observation.

`session.info.payload.submission_state` and `session.resume.result.submission_state` expose the queue owner's authoritative volatile state:

```json
{"revision":18,"queued":[{"inputs":[{"id":"I3","ref":"R3"}],"inputs_complete":true}],"queued_complete":true,"outcomes":[{"revision":17,"input":{"id":"I2","ref":"R2"},"disposition":"cancelled","reason":"queue_cleared"}],"outcomes_truncated_before_revision":null}
```

This queue list contains identity metadata only. The existing head queued projection separately gains canonical input and occurrence fields; its legacy user string is blank for hidden work and an invocation for expanded skills. Compute children do not own the gateway queue: the parent replaces their submission state and publishes settled state, including idle clears.

Queue removal records a positive `cancelled`, `absorbed`, or `failed_before_start` outcome. Ambiguous resolution can record `unresolved`. Absorbed duplicates can include `into_inputs` and `turn` when established. A partially sanitized occurrence keeps its ID while any text survives. Draining into execution is not cancellation. Completion retains a `terminal` outcome with execution and terminal status, so a client missing start and completion can find positive evidence on resume.

Input descriptors and outcomes are each bounded to 256 records and 64 KiB of serialized metadata. Envelope sidecars discard evicted descriptors without discarding work; queue and observation lists expose completeness flags. Outcome eviction advances `outcomes_truncated_before_revision`. Missing or truncated evidence, including after restart, means uncertainty. Never infer cancellation, model consumption, or persistence from absence.

## Reconciliation and limits

1. Retain a local reference before sending. Each input ID is an occurrence; retrying may create several occurrences sharing one reference.
2. Resume the stored session and buffer events during hydration. Reconcile in-flight and submission state with existing `session.events.since` replay. Reset watermarks on epoch changes; refresh history and snapshots after truncation.
3. Deduplicate starts by execution ID and corrections by input ID. Prefer canonical input; null suppresses a bubble. Apply state revisions monotonically within their scope. Outcomes settle only their named occurrence.
4. Reconcile provisional display against canonical history after settlement. Text equality and observation IDs do not supply persisted row IDs. A terminal proves execution ended, not durable admission of a particular visible message.

The old draft session_mirroring, scalar origin, and watching fields are not advertised. Ignoring clients retain their existing behavior and do not automatically gain shared-input display. This contract does not add frontend rendering, presence discovery, watcher permissions, stable device identity, durable admission, cross-backend recovery, or generic compute-host steering. Existing control authority remains. Ambiguous compute sends do not provide exactly-once semantics.

New observation metadata stays outside model messages, persisted history, system prompts, tool definitions, and memory. Existing fan-out, replay bounds, profile routing, and transcript projection remain the supporting infrastructure.
