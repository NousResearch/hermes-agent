# State-store architecture (incremental replacement)

## Status

This is the approved target architecture. The first functional session/message slice now provides a backend-selected factory, an unchanged SQLite adapter, and a PostgreSQL 14+ integration implementation with a bounded internal pool, migration metadata, and capability probe. SQLite remains the default and existing installations continue to use it unchanged. PostgreSQL cannot be selected for production until phases 6–13 have supplied complete schema, consumer, migration, rollback, and runtime evidence.

## Boundary

All state consumers will depend on a backend-neutral `StateStore` contract and factory, not `sqlite3.Connection`, file paths, SQLite pragmas, or FTS5 commands. The contract owns transaction lifecycle, schema head, health, sessions/messages/search, operational gateway state, migration, and change feed behavior. A backend may expose implementation-private connection types only beneath that boundary.

The existing `SessionDB` API is the compatibility facade during the migration. `SqliteStateStore` preserves its current behavior and becomes the default backend. `PostgresStateStore` is a first-class selectable backend with an explicit optional client dependency and capability probe; it will reach production readiness only when all current SessionDB and gateway raw-opener operations have moved through the contract.

## Configuration

The core `state_store` section is deliberately a new top-level core section, not a plugin or observability namespace:

```yaml
state_store:
  backend: sqlite # sqlite (default) | postgresql
  postgresql:
    dsn_env: HERMES_STATE_STORE_POSTGRES_DSN
    connect_timeout_seconds: 10
    pool_max_size: 8
```

`backend` is profile-local because `HERMES_HOME` is profile-local. Configuration is resolved by the existing `load_config()` path, including its profile scope, deep merge, managed overlay, and `${ENV}` secret expansion. The DSN itself is never persisted in `config.yaml`; `dsn_env` names one profile-scoped secret. A missing, malformed, or empty secret fails closed only when `backend: postgresql` is selected. SQLite ignores PostgreSQL-only configuration.

There is no global-to-profile inheritance beyond Hermes’s existing profile configuration semantics. A root profile and each named profile resolve their own config and consequently their own backend target. The PostgreSQL target must derive a stable tenant/database/schema identity from the resolved profile identity, never merely from a PID or a file path.

## PostgreSQL topology and semantics

The minimum supported server is PostgreSQL 14; the integration fixture targets PostgreSQL 18 with pgvector image `pgvector/pgvector:pg18`; `pg_trgm` and `vector` are capability-probed extensions. `vector` is available for future semantic indexes but is not a substitute for FTS semantics. `pg_trgm` is optional and its absence must select a documented bounded fallback rather than silently claim CJK/substring equivalence.

Hermes uses one PostgreSQL instance. The active `HERMES_HOME` is resolved once with the canonical `profile_name_for_home()` resolver; **every** profile, including the installation-root/default profile, maps to a deterministic `hermes_state_store_tenant_<32 lowercase SHA-256 hex>` schema derived from its canonical home and canonical profile identity. These schemas are logical namespaces and accidental-mixing protection, not mandatory product authorization barriers. Default calls remain profile-local. An explicit user-directed target-profile resolver may deliberately select and read that profile's namespace. `hermes_state_store_slice` is a historic fixed-schema cutover boundary, not a default compatibility route: if it contains historic objects for the default profile, selected PostgreSQL fails closed and requires formal reinitialization/cutover before hashed-tenant bootstrap. Historic fixed-schema rows are never attributed, backfilled, or silently reused.

### Namespace routing is not authorization

Profile identity and `profile_id` may be persisted as durable routing or audit metadata, but do not grant or deny access. Application-level resolvers select a default namespace from trusted runtime context and must accept an explicit target profile only through the canonical resolver; user-controlled profile text, row values, caller metadata, config fragments, and DSN text never become SQL identifiers. Every pool checkout resets a quoted, derived schema search path, and migrations/ledgers use namespace-specific advisory locks. These are SQL-injection and accidental-selector/mix-up defenses, including on pooled connections; they do not prohibit an explicitly requested cross-profile content read.

Database roles, grants, RLS, or per-schema ACLs may be added later as an operator deployment hardening choice, but they are not a required Hermes profile model and must not contradict application-level explicit cross-profile reads. Shared hosted-room coordination belongs in a root/global namespace rather than a profile schema because a room may intentionally span profiles.

## Bounded connection pool and pgBouncer

The PostgreSQL store uses a hand-rolled bounded idle-queue pool (a `queue.LifoQueue` of warm connections capped at `pool_max_size`) instead of `psycopg_pool`. The deciding property is per-operation self-description: every checkout re-pins the connection's `search_path` to this store's quoted tenant schema before any statement runs, every checkout ends in an explicit `commit()` or `rollback()`, and nothing else survives between borrowers — no leaked open transaction, no accumulated session state, no advisory lock held past the transaction that took it. A borrower therefore always starts from a known state on a connection whose identity is fully re-established, which is what makes the unqualified table references inside transactions safe. Because a checkout is the only unit of concurrency, pool exhaustion is plain queue backpressure (callers block on the idle queue) rather than overflow creation, and `close()` is a synchronous lock-ordered drain with no background threads to outlive the process or race shutdown — `psycopg_pool`'s maintainer/reconnect workers would add exactly the kind of ambient lifecycle the store deliberately refuses.

The same discipline is what an external pooler must preserve. Under pgBouncer **transaction** pooling, server connections rotate per transaction and session-level state does not follow the client; pgBouncer tracks only `client_encoding`, `DateStyle`, `TimeZone`, `standard_conforming_strings`, and `application_name` by default. Because the store's per-checkout `SET search_path` is session-level, transaction pooling is compatible **only when the operator adds `search_path` to `track_extra_parameters`** (pgBouncer 1.20+); then every transaction again lands with the tenant search path pinned. Without it, the re-pin lands on a different server connection than the statements and unqualified references fail immediately (`relation … does not exist`) rather than silently resolving elsewhere — loud, but an outage. pgBouncer **session** pooling is compatible without extra configuration, since the `SET` persists for the server session exactly as it does on the store's own pool.

Two further notes for operators fronting pgBouncer: no pooled connection carries `LISTEN`/`NOTIFY`, prepared statements, or advisory locks that outlive their transaction today, so there is no other session state for a pooler to preserve; and the future change-feed design (durable sequence plus `LISTEN`/`NOTIFY` wake-up) must use a dedicated session outside any transaction-pooled path, because a listener registration cannot survive connection rotation.

Browser/YOLO/CUA approval capabilities are separate from state-content routing. One-time approvals, browser-session ownership, in-memory waiters, and CUA lifecycle remain process-local and fail closed; neither a cross-profile read nor a shared hosted-room row confers a durable approval capability.

Critical writes use PostgreSQL transactions with explicit retry classification. Fencing uses installation UUID + host identity + process generation + monotonically increasing lease generation; PID-only checks and SQLite file locks are not portable. A durable change sequence plus `LISTEN/NOTIFY` wake-up provides notification; observers must catch up by sequence after reconnect.

## Migration and rollback

The PostgreSQL session/message slice has one schema-evolution authority: a programmatic Alembic bootstrap driven only by the trusted runtime DSN connection and trusted tenant schema. Its exact linear chain is **v25 immutable historic core baseline -> v26 non-topic sqlite import manifest head** (`state_store_v25_core` → `state_store_v26_sqlite_import`). **v26 is NOT historic session-topic v26.** This core branch creates neither `session_topics` nor `messages.topic_id`; that DDL is reserved exclusively for the separate session-topic branch and must not be backported into the core migration chain. Under the tenant-scoped advisory transaction lock and PostgreSQL 14 gate, a completely empty tenant receives that full chain and its tenant-local `alembic_version` table. A legacy `schema_migrations` relation, any other pre-existing tenant relation (including a test marker), malformed/nonexact Alembic metadata, or any catalog mismatch fails closed before application writes; no stamping, custom-ledger replay, self-repair, historic-root-schema exception, or SQLite fallback is allowed. The same read-only semantic validator runs after bootstrap and in operations doctor: it verifies the full v1–v25 table/column set plus the v26 `sqlite_import_manifests` relation, PostgreSQL types, nullability, defaults and generated expressions, primary/unique/foreign-key columns and FK validation state, and every non-primary index's method, key columns, predicate, uniqueness, readiness/liveness, and validity. The immutable core baseline deliberately excludes all follow-on session-segmentation objects and related message foreign keys/indexes. Integration fixtures retain the production tenant as an empty schema and keep their ownership capability only in a separately UUID-derived, fixture-owned companion schema. v8 adds bounded read-only compression-lineage root/tip, model/display projection, prefix, and resume-guard APIs; it does not claim SessionDB's complete replay transformation, rewrite, compaction, FTS, or runtime-consumer parity. v9 adds content-addressed system-prompt snapshots (`system_prompts`, SHA-256 session reference, validated FK, read/set/clear and orphan collection). v10 adds PostgreSQL atomic model-usage persistence through the shared `TokenUsageTransport`: summary counters, billing route metadata, and six-dimensional (`session_id`, model, provider, base URL, mode, task) `session_model_usage` attribution rows migrate and validate transactionally.

Migration is offline and single-authority: quiesce the selected SQLite profile/root writer, capture a consistent SQLite backup including WAL, import canonical rows in FK order, restore sequence high-water marks, rebuild derived indexes, verify counts/invariants/search, and only then select PostgreSQL. No dual write is permitted.

Pre-write rollback switches back only after preserving the PostgreSQL candidate. After PostgreSQL has accepted writes, rollback requires a verified reverse export or restore of a PostgreSQL backup to a separately selected SQLite candidate; changing config alone is prohibited because it would lose post-cutover writes. Existing live databases are out of scope for this branch run.

## DeliveryLedger PostgreSQL adapter (task 3b9561)

A dedicated `PostgreSQLDeliveryLedger` exists for direct adapter conformance tests. It has a separate tenant schema (`hermes_delivery_ledger…`), its own `delivery_schema_migrations` catalog and advisory migration lock, and uses receipt/fence transitions with server-time leases. `adapter_profile` is persisted delivery-target metadata only, never a tenant selector. The factory in `gateway.delivery_ledger_adapter` can open SQLite or PostgreSQL adapters for tests.

This does **not** route live gateway consumers, add runtime configuration, import `state.db`, or authorize cutover. SQLite remains the gateway runtime. A production cutover still requires consumer routing, an audited offline import, operational rollback/export evidence, and full gateway delivery acceptance; a database state transition is not proof a platform sent a message.

## Implementation phases

1. Configuration, factory, capability probing, and SQLite compatibility (current earliest slice).
2. Backend-neutral SessionDB operations and complete canonical PostgreSQL schema.
3. PostgreSQL search and all gateway/hosted-room operational consumers.
4. Native recovery/doctor/change-feed behavior. The bounded sandbox-only PostgreSQL doctor/status plus logical schema backup, manifest, and disposable restore verification are implemented; runtime command routing, production recovery, and change-feed behavior remain unported.
5. Resumable migration, rollback, and real sandbox conformance.
