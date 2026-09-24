# PostgreSQL State Store v14 profile tenant acquisition evidence

## Tenant policy

`open_state_store()` resolves the active `HERMES_HOME` once at the PostgreSQL acquisition boundary. It uses `hermes_constants.profile_name_for_home(get_hermes_home())`, the runtime canonical resolver: the root home is `default`; a named profile is a validated `root/profiles/<name>` home. Caller session metadata, `active_profile`, config strings, and DSN text do not select a tenant.

The root/default tenant deliberately retains `hermes_state_store_slice` as its compatibility schema. A named profile gets `hermes_state_store_tenant_<32 lowercase SHA-256 hex>` derived from its canonical home plus canonical profile name. The profile name is never interpolated into an SQL identifier. The PostgreSQL adapter accepts only that generated form or the fixed root compatibility schema.

Every tenant schema has its own v1–v13 migration ledger. Tenant-specific advisory locks serialize bootstrap. All table references are schema-qualified. A pool checkout sets the fixed quoted `search_path` for that store before use, including an idle connection returned after a hostile or accidental `SET search_path`.

## PG18 evidence

`tests/integration/test_postgresql_state_store_slice.py` opens root/default plus two resolved named profile homes against the PG18 fixture. It proves same session IDs do not cross-read, each acquired schema differs, per-tenant migration succeeds, and pool checkout restores search path. It concurrently opens the named tenant three times and observes one schema head. Existing atomic rollback and concurrent lifecycle tests remain exercised by the same PG18 suite.

## Compatibility and remaining cutover gaps

The old fixed schema is root/default-only; it is never selected for a named profile. This is a compatibility route for prior single-root installations, not a backfill or attribution claim for historical rows. Operators that previously pointed multiple Hermes roots/profiles at the same fixed schema must explicitly audit and export/import those rows before selecting PostgreSQL for named profiles; this change deliberately does not guess ownership.

No runtime consumer/configuration is switched, no SQLite state is migrated/deleted, and no production database is touched. Remaining gaps include complete SessionDB/gateway coverage, profile-ownership backfill tooling, operator provisioning/privileges, recovery/search/FTS parity, verified SQLite↔PostgreSQL cutover and rollback, and production selection.