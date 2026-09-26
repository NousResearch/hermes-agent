# PostgreSQL State Store v14 profile tenant acquisition evidence

## Tenant policy

`open_state_store()` resolves the active `HERMES_HOME` once at the PostgreSQL acquisition boundary. It uses `hermes_constants.profile_name_for_home(get_hermes_home())`, the runtime canonical resolver: the root home is `default`; a named profile is a validated `root/profiles/<name>` home. Caller session metadata, `active_profile`, config strings, and DSN text do not select a tenant. This v14 evidence records the original profile-acquisition boundary; the current routing and head contract below supersede its historical v14 ledger wording.

The root/default tenant and every named profile get `hermes_state_store_tenant_<32 lowercase SHA-256 hex>` derived from canonical home plus canonical profile name. The profile name is never interpolated into an SQL identifier. `hermes_state_store_slice` is historic fixed-schema state only: it is never accepted as a runtime tenant route, and historic default-profile objects there fail closed pending formal reinitialization/cutover.

Every tenant schema has its own Alembic ledger at the current non-topic head `state_store_v26_sqlite_import`, reached through `state_store_v25_core` → `state_store_v26_sqlite_import`. Tenant-specific advisory locks serialize bootstrap. All table references are schema-qualified. A pool checkout sets the fixed quoted `search_path` for that store before use, including an idle connection returned after a hostile or accidental `SET search_path`. The chain contains no `session_topics` or `messages.topic_id`; topic DDL is reserved for the separate session-topic branch.

## PG18 evidence

`tests/integration/test_postgresql_state_store_slice.py` opens root/default plus two resolved named profile homes against the PG18 fixture. It proves same session IDs do not cross-read, each acquired schema differs, per-tenant migration succeeds, and pool checkout restores search path. It concurrently opens the named tenant three times and observes one schema head. Existing atomic rollback and concurrent lifecycle tests remain exercised by the same PG18 suite.

## Compatibility and remaining cutover gaps

The old fixed schema is never selected for any profile, including root/default. It is a fail-closed cutover boundary, not a compatibility route, backfill, or attribution claim. Operators that previously pointed one or more Hermes roots/profiles at that fixed schema must explicitly audit and complete formal reinitialization/cutover before selecting PostgreSQL; this change deliberately does not guess ownership or reuse those rows.

No runtime consumer/configuration is switched, no SQLite state is migrated/deleted, and no production database is touched. Remaining gaps include complete SessionDB/gateway coverage, profile-ownership backfill tooling, operator provisioning/privileges, recovery/search/FTS parity, verified SQLite↔PostgreSQL cutover and rollback, and production selection.