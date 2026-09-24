# PostgreSQL rollback path (PostgreSQL → SQLite)

This is the reverse direction of the sanctioned offline migration. It documents how an
operator returns a profile from `state_store.backend: postgresql` to SQLite without
losing PostgreSQL-era writes. It adds no new tooling: every step composes the already
implemented sandbox operations (Phase 10), the SQLite portability import surface, and
the configuration switch semantics proven by the switch round-trip evidence.

Rollback is a data movement procedure plus a config switch — in that order. Switching
the configuration alone is never a rollback: both backends keep their own state, and a
bare switch abandons every write the PostgreSQL tenant accepted after cutover.

## When to roll back

- **Before cutover** (PostgreSQL has accepted no writes): switch back freely. The
  PostgreSQL candidate schema is preserved untouched; no export is required.
- **After PostgreSQL has accepted writes**: a verified reverse export is mandatory
  before the switch. Changing config alone is prohibited because it would lose
  post-cutover writes (`state-store-architecture.md`, "Migration and rollback").
- **Unplanned rollback** (PostgreSQL unavailable, extension loss, catalog drift):
  the store fails closed by design; the same export-then-switch procedure applies
  once the server is reachable again. There is no SQLite fallback under a
  PostgreSQL selection.

`tests/integration/test_postgresql_state_store_switch_roundtrip.py` pins the
isolation contract both directions: after switching back, the SQLite file never
contains PostgreSQL-era writes, the PostgreSQL tenant never contains SQLite-era
writes, and the pre-switch SQLite prefix is preserved byte-for-byte.

## Step 1 — Quiesce and export from PostgreSQL

Quiesce every writer against the selected profile's tenant schema first. The backup
contract otherwise fails closed (`quiesced=True` must be asserted by the caller):

1. Run native doctor and require a clean report — it proves loopback reachability,
   server version, extension availability, migration ledger heads, table inventory,
   FK-orphan checks, and search health. See
   `state-store-postgresql-phase10-operations.md` for the full sanitized contract.
2. Capture the logical backup with
   `PostgreSQLSandboxOperations.backup(backup_root, quiesced=True)`. It produces a
   custom-format `pg_dump` of the trusted state schema (plus the delivery-ledger
   schema when present) and an atomically published manifest with archive
   checksum, tenant catalog heads, row counts, and invariants. No DSN is recorded.
3. Prove the archive with `restore_and_verify()` into its generated disposable
   database. A rollback is only as good as its last verified restore.

The logical backup is the recovery artifact of record. For the SQLite import in
step 2, also export the supported session/message slices through the store's
bounded read APIs (`get_session`, `get_message_records`,
`list_session_summaries`) into `export_all`-shaped payloads — the same
session-dict shape `SessionDB.export_session()` emits.

## Step 2 — Import into a SQLite candidate

This is the reverse direction of the Phase 11 sandbox importer
(`state-store-postgresql-phase11-sqlite-import.md`): instead of moving a
consistent SQLite snapshot into a fresh PostgreSQL tenant, the exported payloads
are adopted by a **separate, disposable SQLite candidate** — never the active
`state.db`:

- `SessionDB.import_sessions(payloads)` (or `import_foreign_history` for a single
  adopted conversation) validates size, shape, and types, skips already-present
  ids, re-attaches parents only when they exist, and resets live runtime state.
- Derived indexes are never imported: SQLite rebuilds its FTS natively from the
  canonical rows, exactly as PostgreSQL's generated `search_document` is rebuilt
  through its own catalog in the forward direction.
- Process-local objects (gateway routing, leases, async-delegation rows) are not
  portable and are not synthesized. The fail-closed policy mirrors Phase 11:
  unsupported populated values reject the import rather than truncating it.

Verify imported counts and invariants against the Phase 10 manifest before
proceeding: session/message counts must match, parents must resolve, and every
imported id must be readable back.

## Step 3 — Switch the configuration

Select SQLite again by setting `state_store.backend: sqlite`, or by removing the
`state_store` block entirely (SQLite is the default):

```yaml
state_store:
  backend: sqlite # postgresql section may be deleted; dsn_env is never persisted
```

A SQLite selection ignores a stale `postgresql` block rather than validating it,
but delete it anyway so no reader mistakes the profile for a PostgreSQL tenant.
The switch is profile-local: each profile resolves its own backend, so rolling
back one profile never touches another profile's selection.

## Step 4 — Verify

1. Open the store under the SQLite selection and confirm it is a
   `SqliteStateStore` — the factory must never silently degrade (a PostgreSQL
   selection with a missing secret fails closed instead of falling back).
2. Resume smoke: read back a session imported in step 2, append one message, and
   confirm it is visible in a fresh open of the same database.
3. Re-run the count/invariant comparison from step 2 against the live file.
4. Keep the PostgreSQL tenant. Rollback does not drop it; it remains the durable
   archive of the PostgreSQL era, still inspectable with native doctor and
   restorable from the step 1 backup. Reversing the rollback means repeating this
   procedure in the forward direction — there is no dual-write and no merge.

## Fail-closed notes

- **A switch is not a migration.** Both backends keep their own state; the
  round-trip test proves writes never cross the selection boundary in either
  direction.
- **No dual-write.** After the switch, writes go to SQLite only. The PostgreSQL
  tenant is frozen at its last pre-quiesce state.
- **No config-only rollback after cutover.** Post-cutover writes exist only in
  PostgreSQL; losing them requires nothing more than an unexported switch.
- **The importer refuses the active `state.db`** by design; the import target is
  always a separately supplied candidate file that is then moved into place by
  the operator, mirroring the forward-direction sandbox rules.
- **Runtime activation boundaries still apply.** Gateway routing, cron, TUI/API,
  ACP, hosted rooms, and async delegation remain fail-closed surfaces during and
  after rollback; see `state-store-postgresql-phase13-runtime-readiness.md`.
