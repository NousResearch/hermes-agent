# PostgreSQL compression-rotation acceptance gate

## Status

PostgreSQL now exposes the narrowly-scoped
`atomic-compression-rotation-v1` adapter: migration v20 creates a durable,
tenant-local publication receipt and the publisher uses the existing
server-clock fenced compression lease. It atomically creates the child and
handoff, closes the parent, and records the receipt; duplicate receipt reads
are idempotent and no indeterminate request is automatically replayed. It does
not create a SQLite fallback or open `state.db`. Gateway, cron, TUI, ACP,
hosted, async-delegation, and normal selected-PG runtime routing remain
unported and fail closed.

## SQLite oracle → PG18 executable matrix

| SQLite observable contract | PG18 evidence |
|---|---|
| Parent closes only with one live child | `test_pg18_rotation_protocol_matches_sqlite_oracle_and_idempotency` |
| Summary + live tail exactly once; duplicate retry has one child | same test; durable request receipt |
| prompt/model/config/title/lineage/visibility/watermark copied | `_assert_exact_oracle_snapshot` |
| activity/cooldown/fallback/ineffective reset; generation increments | `_assert_exact_oracle_snapshot` |
| every pre-commit database exception leaves no child/message/receipt | `test_pg18_transaction_phase_database_fault_rolls_back_and_reopens` |
| post-commit acknowledgement loss is committed, not replayed | `test_pg18_commit_returned_fault_is_durably_committed_and_reopenable` |
| stale owner/fence rejected and server clock permits successor | `test_pg18_spawned_owners_server_clock_fence_stale_rejection_and_namespace_isolation` |
| process death before commit classifies rolled back; supervisor uses new receipt | `test_pg18_sigkill_during_real_transaction_is_rolled_back_without_replay` |
| server backend termination at each critical SQL phase rolls back and reopens | `test_pg18_terminate_backend_at_every_critical_phase_reopens_cleanly` |
| tenant schemas cannot cross-read/write the same parent ID | spawned-owner namespace-isolation cell |
| real fake-provider `AIAgent` API + no SQLite fallback | `tests/agent/test_postgresql_compression_rotation_oracle.py` plus selected-PG CLI trap |

`tests/integration/postgresql_rotation_protocol.py` is test-only controlled
instrumentation.  It creates its tables only inside an
`OwnedPostgreSQLTestTarget` UUID schema, uses real PostgreSQL transactions and
`clock_timestamp()`, exposes backend PID at phase boundaries, and is never
available to production imports.  A real adapter must replace—not call—it.

## Phase 7 gate

Phase 7 is passed only when the SQLite oracle drives real `AIAgent` compression
with its deterministic provider and confirms exact child metadata, prompt,
model config, title, lineage, visibility, activity/cooldown/counters,
watermark/generation, messages, reopening tip, and idempotent receipt behavior.
It also requires selected-PG refusal before mutation with an empty `state.db`
open trap.

## Phase 12 gate

Phase 12 is passed only on PostgreSQL 18+ with an owned UUID schema per test:

- transaction exception, `pg_terminate_backend`, and SIGKILL cases have a
  fresh durable audit;
- all named phases are covered (`lease-revalidated`, `child-row-inserted`,
  `handoff-inserted`, `parent-close-issued`, `commit-returned`);
- spawned/fenced owners use server time; stale owner acknowledgement is
  rejected; a successor uses a new receipt rather than replay;
- serial and parallel invocations use independent UUID schemas.

## Phase 13 gate

`atomic-compression-rotation-v1` is advertised only by the production adapter.
The broader selected-PG runtime remains unavailable until its other consumers
have equivalent contracts; a SQLite fallback is always a failure.

## Commands

```bash
scripts/run_tests.sh tests/agent/test_postgresql_compression_rotation_oracle.py -v --tb=short
# ``pyproject.toml`` normally excludes integration tests.  Clear that default
# explicitly so all 14 parametrized rotation-acceptance cases are selected.
HERMES_TEST_WORKERS=1 scripts/run_tests.sh -o "addopts=" -m integration tests/integration/test_postgresql_compression_rotation_acceptance.py -v --tb=short
HERMES_TEST_WORKERS=2 scripts/run_tests.sh -o "addopts=" -m integration tests/integration/test_postgresql_compression_rotation_acceptance.py -v --tb=short
# These three fixture families are intentionally unmarked; do not apply
# ``-m integration`` or pytest will deselect them all.
HERMES_TEST_WORKERS=1 scripts/run_tests.sh -o "addopts=" tests/integration/test_postgresql_phase12_fault_harness.py tests/integration/test_postgresql_compression_coordination.py tests/integration/test_postgresql_cli_session_store.py -v --tb=short
```

## Full selected-PG manifest

`scripts/validate_selected_postgresql_manifest.py` fixes the complete 19-file
selected-PG gate. It includes the bounded public AIAgent/CLI and desktop-reaction
routes, runtime readiness/config/factory checks, rotation/coordination/Phase12, runtime
ownership, delivery, SQLite import, operations, marker-validated target safety,
StateStore parity, search grammar, and contextual-session contracts.

```bash
python3 scripts/validate_selected_postgresql_manifest.py --mode serial
python3 scripts/validate_selected_postgresql_manifest.py --mode parallel
```

The parallel mode runs four non-overlapping partitions with one worker and zero
file retries per partition, then requires their summed file/test totals to equal
the serial manifest. The manifest always clears `addopts` because the repository
default excludes integration tests. It never broad-selects `-m integration`:
that selection can include unrelated ACP/aiohttp optional-dependency suites,
while several owned PostgreSQL fixture families are deliberately unmarked and
would instead be deselected. Every PostgreSQL integration path uses the
marker-validated owned UUID target fixture and does not mutate preexisting
named-volume data.
