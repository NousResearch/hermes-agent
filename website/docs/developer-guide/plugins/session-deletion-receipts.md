---
title: "Session deletion receipts"
description: "Explicit deletion intent and transactionally prepared routing identities for plugins"
---

# Session deletion receipts

An external plugin can pair a stored conversation with a resource it owns. A
post-commit notification alone cannot safely drive that resource's deletion:
the session's routing rows are already gone, retention is not a user request,
and Desktop's backend and a messaging gateway can be different processes.

`SessionDB` therefore prepares **minimal routing identities inside the deletion
transaction**, after its active-write checks and snapshot fences, and persists
that preparation as a committed receipt. A plugin reads receipts through public
methods, not private tables. No plugin code runs under the SQLite write lock.
This is a proposed generic preparation/read surface, not a vendor cleanup engine
or a synchronous pre-delete callback.

## Provenance and scope

`delete_session` and `delete_sessions` accept an additive `deletion_origin`
keyword, defaulting to `"unknown"`. Only these two exact values prepare receipts:

| Origin | Set by core caller | Receipt reason / surface |
|---|---|---|
| `user_rest` | Dashboard single and selected-ID bulk REST deletion | `explicit_user` / `dashboard_rest` |
| `user_rpc` | Desktop/TUI backend `session.delete` RPC | `explicit_user` / `session_rpc` |

These values are authored by the actual handlers, not accepted from REST/RPC
payloads. They describe local caller intent, **not external authorization**.
Plugins are trusted code; the Python keyword is not an access-control boundary.
An absent, unknown or unrecognized origin does not authorize external cleanup.
Retention/prune, empty sweeps (including a sweep requested through REST), draft
compensation and profile-move deletion do not prepare receipts. Other explicit
user interfaces are not yet instrumented; do not infer intent from missing rows.

The receipt contains:

- `operation_id`: fresh opaque operation identity.
- `reason`, `surface`: exact provenance above.
- `store_id`: canonical path of the database actually deleted from.
- `profile_home`: that database's canonical parent directory, never the ambient
  launch profile. For a custom database path this is a store directory, not a
  claim that core has discovered a named profile.
- `identities`: sorted exact removed rows, each with `id`, `source`, `session_key`,
  `chat_id`, `chat_type`, `thread_id`, `parent_session_id` (nullable where unknown).

The set includes recursive delegate cascades and compression-chain links when
that delete uses chain expansion. Surviving branches and unknown/skipped IDs are
excluded. These are routing identifiers, not proof that the plugin owns the
resource. No messages, titles, summaries, model config, origin JSON, user names,
credentials or file contents enter the receipt. Routing identifiers and local
paths are still private metadata: do not transmit them to telemetry by default.

## Public API

```python
from pathlib import Path
from hermes_state import SessionDB

# Use the explicitly selected owning profile, not a cached launch-profile home.
with SessionDB(Path(profile_home) / "state.db", read_only=True) as db:
    rows = db.list_session_deletion_receipts(after_sequence=cursor, limit=100)
    # rows: [{"sequence": int, "deletion": {...}}, ...], ordered by sequence
    receipt = db.get_session_deletion_receipt(operation_id)  # dict or None
```

`after_sequence` must be nonnegative; `limit` must be between 1 and 1000. The
sequence belongs to one database; never share a cursor between profiles/stores.
The read-only API requires a writer to have reconciled the current schema first.
A missing receipt is not proof of a successful deletion.

## Durable backend-to-gateway handoff

A gateway-side plugin can consume receipts even if it was not loaded in the
Desktop backend. Core does not execute callbacks, launch a worker or call any
external API. Use a **plugin-owned durable ledger**:

1. Read a bounded batch from the exact owning store with the plugin's saved
   per-store cursor. Verify `reason == "explicit_user"`, the expected surface,
   and exact `store_id`/`profile_home` against the selected store. A copied or
   moved store's old receipts must not silently authorize work for a new profile.
2. In ONE transaction in the plugin's own ledger, insert pending work keyed by
   `operation_id` (deduplicate), preserve only necessary identifiers, and advance
   that store's cursor. Do not advance the cursor before pending work is durable.
3. The gateway worker verifies its opt-in configuration and independently proven
   resource ownership. Unknown routing, ownership or profile means no external
   deletion. Pending work can then be retried according to plugin policy.
4. Record external completion in the plugin ledger. Crash recovery can repeat an
   action that succeeded remotely before its local completion was saved: use
   idempotent APIs where possible. **No exactly-once external action guarantee.**

The cross-process test runs the actual `session.delete` RPC against a temporary
`SessionDB`, exits the backend immediately after commit, then starts separate
consumer processes which stage a plugin-owned SQLite ledger through these public
reads. A subsequent consumer restart resumes its durable cursor without adding a
second pending record. This tests handoff/deduplication, not an external service.

## Transaction and notification semantics

Receipt preparation and row deletion commit together. A failed transaction rolls
both back. Missing IDs, guarded rows and rejected snapshot fences produce no
receipt; bulk operations record only rows actually eligible for deletion.
Deletion's existing return values and wire results are unchanged. A failure to
persist required receipt evidence rolls back an explicit delete; it is not
silently ignored.

This surface is independent of the process-local, post-commit `on_session_delete`
observer proposed in [PR #124988](https://github.com/NousResearch/hermes-agent/pull/124988).
It neither duplicates that observer nor changes its fail-open, ignored-return,
non-veto semantics. If that observer is available it may be used as a wake-up
hint; durable confirmation is the receipt, never the hint. Integration with the
unmerged observer is not claimed here.

## Retention and draft design decision

Receipts currently have no automatic expiry. Administrative code can call
`db.forget_session_deletion_receipts(through_sequence=N)` on a writer **only after
ALL registered consumers have durably staged the range**, or when the owner
explicitly accepts dropping undelivered work. It returns the number removed and
never reuses sequence numbers. This is not a per-plugin acknowledgement API:
one consumer must not erase another's unread evidence. Forgotten receipts cannot
be replayed. Database restore/replacement requires reconciling plugin cursors;
this is not a distributed queue with a generation/recovery protocol.

This follow-up remains draft pending maintainer agreement on core receipt
storage and retention/consumer coordination. An arbitrary synchronous preparation
hook was deliberately not wired into a live delete transaction: callbacks could
re-enter the same store or perform external actions before a rollback. If a
callback is required beyond this transactionally prepared identity surface, its
constrained writer/rollback contract needs a separate design decision first.
