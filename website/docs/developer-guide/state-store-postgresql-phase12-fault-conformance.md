# Phase 12 — PG18 fault injection and multi-process conformance

Phase 12 is **direct PostgreSQL adapter evidence** for implemented contracts. It does not select PostgreSQL at runtime and does not claim a runtime cutover, SQLite fallback, or full SessionDB parity.

## Fixture and execution

`tests/integration/test_postgresql_phase12_fault_harness.py` runs against the loopback-only Compose PG18 fixture (`pgvector/pgvector:pg18`) with a generated tenant schema per test. Every spawned actor uses the `spawn` start method and constructs its own PostgreSQL adapter/pool; no connection is inherited from a parent process. Parent-owned cleanup drops every generated schema after child exit.

All outcomes are classified in assertions as **committed**, **rolled back**, **indeterminate**, or **rejected**. There is no automatic retry in the harness. The tests use database server time, PostgreSQL advisory locks, process events, and direct controlled lease expiry rather than timeout-based ownership conclusions.

## Capability-scoped evidence

| Implemented capability | Fault/concurrency proof | Durable postcondition |
|---|---|---|
| Fenced session ownership and turns | Two independent processes contend for one ownership key. A separate owner process begins a turn and is sent `SIGKILL`; the supervisor expires its server-side lease and a successor takes it. | Exactly one contention receipt commits; loser is rejected. The killed owner’s `running` turn becomes `indeterminate`, stale receipt settlement is rejected, and the successor may settle once only with receipt data. This does **not** assert that a killed process survived. |
| Message + activity and token-usage atomic writes | PostgreSQL `BEFORE INSERT` triggers inject a server-side message fault and a `session_model_usage` fault. | Both public operations raise and roll back their coupled rows/counters. The same single-connection pool subsequently serves a successful write. |
| Pool exhaustion / cleanup | A pool of size one has its only checkout deliberately held while a second public append waits. | The second write cannot complete while the lease is held; it commits after event-governed release with no alternate backend or retry path. |
| Delivery receipt fence | An independent delivery-ledger process claims a receipt, then is `SIGKILL`ed. The test expires that lease in PostgreSQL and a new adapter sweeps it. | The stale receipt completion is rejected; exactly one replacement receipt commits and can deliver. Gateway routing remains unported. |
| Search catalog repair | A spawned psycopg process owns the tenant maintenance advisory lock while the adapter attempts repair. | The adapter reports `already_running` and health reports in-progress. This tests the PG GIN maintenance lock, not SQLite FTS semantics. |
| Concurrent namespace isolation | Three spawned stores write the same logical session id to three generated trusted tenant schemas. | Every actor commits only its own payload and each schema read sees only its own row. This is physical schema isolation evidence; existing configuration tests separately cover canonical home/profile-derived schema selection. |
| Backup and import interruption | A migration-only disposable SQLite source is interrupted at the importer’s documented `messages` injection point; a `pg_dump` command runner returns a deterministic interruption before a backup can be established. | Import records a durable `failed` manifest and no imported messages; backup produces no success manifest. SQLite is used only as the explicitly supplied offline import source, never as selected-PG fallback. |

## Deliberate exclusions and remaining gaps

- The harness does not route the PostgreSQL adapters through CLI, gateway, cron, TUI, ACP, browser/CUA, hosted, or full async-delegation runtime paths. Those consumers remain unported or explicitly fail closed when PostgreSQL is selected.
- It does not claim process-wide pooling: each process has an intentionally independent adapter/pool.
- It does not make a process-death availability claim. `SIGKILL` evidence only establishes the observed durable rows after the child is dead and a different actor takes over.
- It does not import gateway delivery state or assert SQLite FTS equivalence. PostgreSQL search is generated `tsvector` + GIN and fails closed while its catalog is unhealthy.

## Commands

```bash
# Serial file gate
HERMES_TEST_WORKERS=1 scripts/run_tests.sh tests/integration/test_postgresql_phase12_fault_harness.py -v --tb=short

# Deliberate parallel file gate (the harness remains schema-isolated)
HERMES_TEST_WORKERS=4 scripts/run_tests.sh \
  tests/integration/test_postgresql_phase12_fault_harness.py \
  tests/integration/test_postgresql_session_runtime_ownership.py \
  tests/integration/test_postgresql_delivery_ledger.py -v --tb=short
```

A green Phase 12 gate is conformance evidence for the table above only. It is not authorization to change default configuration, operate `state.db`, or claim a PostgreSQL runtime cutover.
