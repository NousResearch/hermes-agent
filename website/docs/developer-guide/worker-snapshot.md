---
title: "Delegate Worker Snapshot"
description: "subagent.snapshot: recovering native delegate_task worker state after a client reconnects"
---

# Delegate Worker Snapshot (`subagent.snapshot`)

> **Audience:** Desktop / mobile / TUI client authors and gateway maintainers
> **Source files:** `tools/worker_roster.py`, `tui_gateway/worker_snapshot.py`, `tui_gateway/methods_subagents.py`

`subagent.list` shows children that are live in the serving process right now. After
a reconnect, a resume, a compute-host hop or a backend restart, a client also needs to
know which delegated workers it started are still running and which already finished.
`subagent.snapshot` answers that from a small durable, profile-local roster.

## Scope

This is **not a global worker inventory**. `scope: "native_delegate_task"` and
`coverage: "admitted_since_upgrade"` cover native `delegate_task` admissions made by a
backend that has this feature: synchronous and background batches, children waiting for
construction or an executor slot, and nested descendants. Argument validation and
credential preflight are not workers. Work started before the upgrade, slash workers,
external agents, OS processes and cron jobs are not listed.

Admission is written before child construction and dispatch. A storage failure rejects
the admission instead of launching an untracked child. Each child inherits its profile
home and root conversation key from its parent's admission, never from a client-supplied
key. `parent_run_id` identifies the exact nested parent even if a public subagent ID is
reused. Compression lineage (read through the owning profile's `SessionDB`) keeps
workers visible after the root conversation compresses; explicit forks are excluded.

Only the interpreter that admitted a run can attest that it is active. When turns run
in a compute host, the gateway asks the existing host over the supervisor pipe
(`workers` / `workers.ack`); it never starts a host to answer a read. A persisted
non-terminal row with no positive attestation is reported as `unknown` — never inferred
dead from a PID or a clock. Disconnecting never stops work or fabricates a result.

## Wire contract

Negotiate per connection: `gateway.capabilities` returns `worker_snapshot_v1: true`.
Older servers omit the field (treat as `false`) and clients keep their existing
`subagent.list` behaviour; older clients never call the method.

```json
{"method": "subagent.snapshot", "params": {"session_id": "<runtime session id>"}, "id": 1}
```

Profile, home and root key are resolved from the transport-owned runtime session.
Foreign or missing session authority: error `4001`. Storage or lineage read failure:
error `5036` — never a successful empty list.

Result:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `snapshot_epoch` | random nonce of the serving interpreter |
| `snapshot_seq` | monotonically increasing within one epoch; never compare across epochs |
| `session_key` | current stored session key |
| `scope` | `"native_delegate_task"` |
| `coverage` | `"admitted_since_upgrade"` |
| `owner_available` | whether the attached compute owner answered this read |
| `workers[]` | `{run_id, owner_id, subagent_id, status, version, goal?, parent_id?, parent_run_id?, delegation_id?, started_at?}` |

`run_id` is immutable and random per admission; key recovered rows by it, not by
`subagent_id` (which starts as `pending-…` and is replaced when the child is built).
`owner_id` is an interpreter nonce, not a PID. `version` orders persisted transitions.
`goal` is a display label truncated to 160 characters.

States:

- `queued` — admitted, not yet claimed for execution (includes construction).
- `running` — the owner claimed the run; not a claim about progress.
- `waiting` — the worker is blocked in a synchronous join on its own native children
  (including the inline fallback of a background batch). Counted, so nested joins
  restore `running` only after the last one returns.
- `completed` / `failed` / `cancelled` — runtime verdict. A queued child cancelled
  before start never executes. Timeout is `failed`.
- `ended` — cleanup without a classified result.
- `unknown` — durable active record whose owner cannot currently attest it.

Terminal states latch: later cleanup cannot overwrite a classified result, and a
parent's completion never settles its descendants. If a terminal write is lost, the
active attestation is dropped too, so the row reads `unknown` rather than running forever.

An empty `workers` list means no retained admissions in this scope — not "no work
anywhere". Clients should keep `unknown` rows visible, drop active claims on disconnect
or snapshot failure (terminal evidence can be kept), and not merge unversioned
`subagent.*` events into these rows by public ID; use them as refresh hints.

## Storage and retention

A profile-local `worker-roster.sqlite` next to `state.db`; `state.db` and transcripts
are untouched. Rows hold IDs, timestamps and the 160-character goal label only — no
context, output or credentials. On each new admission, terminal rows written by this
version are pruned to at most 2000 per profile and 30 days after completion. Active and
`unknown` rows are never pruned.
