# Async-Delegation Ledger — PostgreSQL Routing Audit

Audit brief for routing `tools/async_delegation.py`'s durable ledger through a
selected PostgreSQL state-store backend, mirroring the delivery-ledger precedent
(`gateway/delivery_ledger_postgresql.py` + `gateway/delivery_ledger_adapter.py` +
a `selected → adapter / else legacy module` seam). Read-only: this document makes
no code changes.

## 1. Module under audit

`tools/async_delegation.py` (~1040 lines). One module owns both the **durable
ledger** (SQLite over shared `state.db`, table `async_delegations`) and the
**in-memory registry + dispatch orchestration** (executor, stale monitor, event
publication). The two surfaces must be treated separately: only the durable
ledger needs a PostgreSQL path; the in-memory registry is process-local and
backend-independent.

## 2. Durable ledger surface — DB-touching functions

Every function below reads or writes `async_delegations` through `_transaction()`.
These are the seam points.

| Function | Operation | SQL shape |
|---|---|---|
| `_persist_dispatch(record)` | write | `INSERT OR REPLACE` a `running` dispatch |
| `_prune_durable_records()` | write | `DELETE` delivered/terminal rows beyond retention/caps |
| `_persist_completion(event, result)` | write | `UPDATE` state/completed_at/event_json/result_json, `delivery_state='pending'` |
| `record_unit_child(delegation_id, entry)` | read+write (best-effort) | `SELECT result_json` + `UPDATE` partial results |
| `recover_abandoned_delegations()` | read+write | `SELECT` running/finalizing + `UPDATE` → `unknown` |
| `restore_undelivered_completions(target_queue)` | read+write + queue | `recover` + `SELECT` pending completions + `UPDATE` drop/replay + `queue.put` |
| `mark_completion_delivered(delegation_id)` | write | `UPDATE delivery_state='delivered'` |
| `claim_completion_delivery(delegation_id, claim_id)` | read+write | `SELECT delivery_state` + `UPDATE` claim (CAS on `delivery_claim`/`delivery_claimed_at`) |
| `release_completion_delivery(delegation_id, claim_id)` | write | `UPDATE` cap→dropped or release claim |
| `defer_completion_delivery(delegation_id, claim_id)` | write | `UPDATE` release claim, `attempts=MAX(0,attempts-1)` |
| `drop_completion_delivery(delegation_id, claim_id)` | write | `UPDATE delivery_state='dropped'`, clear claim |
| `complete_completion_delivery(delegation_id, claim_id)` | write | `UPDATE delivery_state='delivered'` where `delivery_claim=?` |
| `get_durable_delegation(delegation_id)` | read | `SELECT` 8 columns → dict |

Internal helpers used only by the above: `_connect()`, `_initialize_schema()`,
`_transaction()`, `_update_delivery(sql, params)`, `_recovered_results(task, result_json, error)`.

## 3. Pure / in-memory surface (NO PostgreSQL path needed)

| Function | Kind | Notes |
|---|---|---|
| `_capture_routing_origin()` | env read | snapshot `scope_id`/`user_id`/`user_name` |
| `_new_delegation_id()` | pure | uuid |
| `active_count()`, `active_task_count()` | in-memory | registry scan |
| `has_live_for_session(...)`, `_session_records(...)` | in-memory | registry scan |
| `_prune_completed_locked()`, `_current_origin_session_id()` | in-memory | registry/env |
| `is_interim_delegation_event(evt)` | pure | |
| `claim_event_delivery / release_event_delivery / complete_event_delivery / _event_delivery` | delegation | route to the DB claim functions (which seam) |
| `_dispatch*`, `_finalize`, `_push_completion_event`, `list_async_delegations`, `interrupt_*`, `push_task_failure_notice`, stale monitor | orchestration | call `_persist_dispatch`/`_persist_completion` internally (which seam) |
| `active_for_session(...)` | in-memory | plugin-compat block |

`claim_event_delivery`/`release_event_delivery`/`complete_event_delivery` and the
`_dispatch`/`_finalize` orchestration are **not** seam points themselves: they
delegate to the DB functions listed in §2, which route.

## 4. Schema DDL (canonical, `hermes_state_common.SCHEMA_SQL`)

```sql
CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id        TEXT PRIMARY KEY,
    origin_session       TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id    TEXT,
    state                TEXT NOT NULL,
    dispatched_at        REAL NOT NULL,
    completed_at         REAL,
    updated_at           REAL NOT NULL,
    event_json           TEXT,
    result_json          TEXT,
    delivery_state       TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts    INTEGER NOT NULL DEFAULT 0,
    delivered_at         REAL,
    owner_pid            INTEGER,
    owner_started_at     INTEGER,
    task_json            TEXT,
    delivery_claim       TEXT,
    delivery_claimed_at  REAL,
    origin_session_id    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
```

PostgreSQL type mapping: `REAL` → `DOUBLE PRECISION`, `INTEGER` → `INTEGER`,
`TEXT` → `TEXT`, `owner_started_at` → `BIGINT`.

## 5. Connection, locking & transaction model (SQLite)

- `_connect()` builds `state.db` path from `get_hermes_home()`; `open_db(wal=False)`
  with `_initialize_schema` (durability barriers + `reconcile_state_schema`).
- Every DB operation opens a short-lived connection via `_transaction()` and runs
  under the module-global `_DB_LOCK` (thread mutex).
- `owner_pid` / `owner_started_at` stamp process ownership; `delivery_claim` /
  `delivery_claimed_at` / `delivery_attempts` implement cross-consumer claim CAS
  (5-minute claim staleness window in `claim_completion_delivery`).

## 6. Fail-closed gate (must remain)

`_connect()` calls `require_legacy_state_db_runtime(home=path.parent)`, which
raises `PostgreSQLRuntimeActivationError` for a selected-PG profile **before**
`state.db` is touched. This is the backstop: after the seam is added, a selected-PG
profile routes away from `_connect()` to the adapter, but the gate stays as
defense-in-depth for any path that still reaches SQLite. It is NOT removed.

## 7. Callers

| Caller | Functions used | DB-touching? |
|---|---|---|
| `gateway/run_notifications.py` | `claim_event_delivery`, `restore_undelivered_completions`, `_DURABLE_CLAIM_OPS` (`drop/release/defer/complete_completion_delivery`) | yes |
| `tools/process_registry.py` | `restore_undelivered_completions` | yes |
| `tools/delegate_tool_dispatch.py` | `_new_delegation_id`, `record_unit_child` | `record_unit_child` yes |
| `hermes_cli/cli_process_notifications.py` | `claim_event_delivery`, `complete_event_delivery` | yes |
| `hermes_cli/quiet_single_query.py` | `claim_event_delivery`, `complete_event_delivery` | yes |
| `tui_gateway/session_notifications.py` | `claim_event_delivery`, `complete_event_delivery`, `release_event_delivery` | yes (NOT owned by this slice) |
| `tools/cronjob_tools.py` | `dispatch_async_delegation`, `_current_origin_session_id` | via dispatch |
| `tools/kanban_tools.py` | `_current_origin_session_id` | no |
| `acp_adapter/server.py` | `interrupt_for_session` | no |
| `tui_gateway/session_lifecycle.py` | `interrupt_for_session`, `has_live_for_session` | no |
| `gateway/wake.py` | (comment/display metadata only) | no |
| `evals/*` (non-runtime) | `get_durable_delegation`, `dispatch_async_delegation` | yes (test-only) |

## 8. Proposed architecture (mirrors delivery-ledger, does NOT copy it)

1. **Dedicated adapter** — `tools/async_delegation_ledger_postgresql.py`:
   `PostgreSQLAsyncDelegationLedger`, tenant-scoped schema
   (`hermes_async_delegation` root / `hermes_async_delegation_tenant_<sha256>`
   for named profiles), migration + catalog validation, PG18 minimum, same method
   surface as §2. Imports pure helpers (`_recovered_results`, `_ROUTING_KEYS`,
   constants, liveness) from `tools.async_delegation`; reimplements SQL + event
   shaping.
2. **Selected resolver** — `tools/async_delegation_ledger_adapter.py`:
   `selected_async_delegation_ledger()` returns `None` for SQLite (legacy module
   path) or a cached configured PG ledger; raises typed on missing/unreachable DSN
   (never degrades to SQLite). Plus a `SqliteAsyncDelegationLedger` compat wrapper
   and `open_configured_async_delegation_ledger` factory for tests.
3. **Internal runtime seam** — in `tools/async_delegation.py`: a module-level
   `_selected_async_delegation_ledger()` helper; each §2 function routes to the
   adapter when non-`None`, else keeps the unchanged SQLite body. This is the
   single choke point (same position `require_legacy_state_db_runtime` occupies
   today), so no caller file changes.

The capability stays **missing** (`async-delegation-ledger-routing` remains in
`state_store_runtime_readiness.missing_capabilities`) until L0 E2E flips it.

## 9. Residual notes / risks

- `restore_undelivered_completions` and `recover_abandoned_delegations` embed
  process-liveness (`_pid_exists`, `get_process_start_time`) and event shaping;
  the adapter reuses the pure helpers and re-expresses only SQL + dict building.
- `record_unit_child` is best-effort (swallows exceptions); the seam preserves
  that contract for both backends.
- The dispatch orchestration (`_persist_dispatch` → `_prune_durable_records`,
  `_persist_completion`) routes internally, so `dispatch_async_delegation*`,
  `_finalize`, and `_push_completion_event` need no caller changes.
