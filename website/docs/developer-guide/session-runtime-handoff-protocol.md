# Session Runtime Ownership and Turn Handoff Protocol

## Status and boundary

This document defines an **additive SQLite compatibility foundation** and a **direct-test-only PostgreSQL 18 adapter**. Neither is selected by a live runtime, neither authorizes Redis, and neither authorizes hosted-room retries. Existing `session_turn_leases` remains the local runtime path.

A future consumer may route to this protocol only when it carries the exact ownership receipt through admission, transcript publication, finalization, and every external-effect boundary.

## Identity and authoritative time

An owner is `(installation_id, host, process_generation)`. A PID alone is invalid cross-machine identity. The runtime namespace and session ID are trusted routing inputs, not caller-selected tenant SQL.

SQLite stores compatibility evidence using its transaction and local clock. PostgreSQL must use a single server-time transaction/CAS for acquire, renewal, expiry takeover, turn transition, and fence-guarded acknowledgement. Optional Redis may notify/wake waiters only; it is never an authority or fence source.

## State machine

`session_runtime_owners` holds one row per `(namespace, session_id)` with an increment-only `fence`.

- acquire absent row → fence `1`;
- same owner acquire → renews without changing fence;
- foreign non-expired owner → no receipt;
- expired foreign owner → replaces identity, increments fence, and marks older `running` turns `indeterminate`;
- renew/release require the exact identity and fence.

`session_runtime_turns` is keyed by `(namespace, session_id, turn_id)`.

- `begin` requires the current unexpired receipt and creates `running`; same receipt duplicate is idempotent;
- takeover converts prior-fence `running` to `indeterminate`;
- a settled turn requires a verified receipt payload; storage success alone is not an external-effect receipt;
- stale fences cannot begin or resolve turns; an indeterminate turn is not implicitly retried. A new turn ID is required after explicit recovery assessment.

Browser/CUA, approval waits, and other process-local capabilities are explicitly excluded from handoff. They must fail closed on destination recovery.

## PostgreSQL 18 direct-test evidence

`PostgreSQLStateStore` exposes the same direct ownership methods through the existing explicit StateStore factory, but no runtime consumer calls them. Migration 18 creates the tenant-local owner/turn catalog under the existing per-tenant migration advisory lock. Ownership operations additionally take a transaction-scoped advisory lock keyed by trusted tenant schema, namespace, and session, then read `clock_timestamp()` on the server before applying the fence-guarded transition.

The PG18 direct suite covers two independent adapter instances contending on an absent row, same-owner restart, abandoned-owner expiry takeover, `running → indeterminate`, stale renew/release/begin/settle rejection, receipt-required explicit settlement, terminal duplicate-payload rejection, namespace isolation, release fence preservation, pooled `search_path` reset, migration/catalog drift rejection, and injected transaction rollback recovery. It deliberately does not alter SQLite behavior.

## Remaining local PostgreSQL sandbox / runtime work

1. Define and implement a backend-neutral consumer that carries the exact receipt through admission, transcript publication, finalization, and every external-effect acknowledgement.
2. Prove that complete vertical slice in a local PG18 sandbox, including destination startup/recovery and all process-local capability fail-closed paths.
3. Produce a separately authorized runtime selection, import/rollback, and operational cutover plan. Do not introduce Redis authority, fallback, or hosted-room retry.

## Required runtime completion plan

1. Add a backend-neutral protocol/factory and migrate the SQLite implementation plus one complete consumer vertical slice.
2. Require the PostgreSQL receipt on every durable transcript, compaction, finalization, and acknowledgement write in that consumer.
3. Add end-to-end recovery evidence for all external-effect boundaries and process-local capability rejection.
4. Only after those tests and a separately authorized cutover plan may runtime selection be considered. Redis signals, live configuration, and hosted-room work remain outside this protocol.
