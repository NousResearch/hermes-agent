# Phase 11 — Offline SQLite-to-PostgreSQL sandbox rehearsal

This phase adds a **sandbox-only command** for importing the currently supported StateStore slices. It does not add a runtime PostgreSQL selector, a SQLite handoff path, dual-write, or a cutover procedure.

```bash
python -m postgresql_state_store_sqlite_import \
  --source /absolute/disposable/source.db \
  --snapshot-root /absolute/disposable/rehearsal-output \
  --dsn 'postgresql://…/hermes_state_store_test' \
  --evidence /absolute/disposable/rehearsal-output/import-manifest.json
```

The standalone importer has no `--schema` option. It refuses the active default `HERMES_HOME/state.db`; the source must be explicitly supplied, and the importer allocates its own newly created marker-owned PostgreSQL tenant schema. `hermes state-store sqlite-import` follows the same allocation rule while resolving the selected profile's PostgreSQL secret through the maintenance boundary. Neither command accepts an operator-selected destination schema, and neither ever imports into the selected runtime tenant or a historic `hermes_state_store_slice` schema.

Each allocated target reaches the current non-topic head `state_store_v26_sqlite_import` through the only supported linear chain: **v25 immutable historic core baseline -> v26 non-topic sqlite import manifest head** (`state_store_v25_core` → `state_store_v26_sqlite_import`). **v26 is NOT historic session-topic v26**: it creates no `session_topics` relation and no `messages.topic_id` column. Session-topic DDL belongs only to the separate session-topic branch.

## Mapping and fail-closed policy

`source_object_mapping_manifest()` emits machine-readable classifications.

| SQLite object | Classification | Rehearsal action |
|---|---|---|
| `system_prompts`, `sessions`, `messages`, `session_model_usage`, `conversation_generations`, `session_runtime_owners`, `session_runtime_turns` | canonical-supported | Import in FK order with explicit JSON, boolean, timestamp, ID, and identity-sequence conversion. Runtime ownership records remain direct-test-only; this does not route runtime ownership. |
| `messages_fts*` | derived-rebuildable | Never import rows. PostgreSQL's generated `search_document` and GIN index are validated/rebuilt through its own catalog. SQLite FTS equivalence is not claimed. |
| `schema_version`, `state_meta`, `gateway_*`, `compression_locks`, `session_turn_leases`, `async_delegations` | intentionally process-local/non-migrated | Must be empty. Populated rows fail before target import. This includes the SQLite async-delivery state; PostgreSQL delivery ledger is independently owned and not synthesized from it. |
| Unrepresented `sessions` fields and `messages.display_order/display_identity` | canonical-unimplemented | Must retain their default/null values. Any populated value rejects import rather than truncating it. |

Unknown SQLite domain objects fail before PostgreSQL target writes. SQLite internal `sqlite_sequence` is excluded as storage metadata. FTS shadow tables are never imported.

## Snapshot, resume, and recovery

The importer creates a consistent read-only SQLite snapshot with `sqlite3.Connection.backup()`, which includes committed WAL content. It fingerprints that snapshot, records source counts/schema and target pre-import counts in the durable `sqlite_import_manifests` table, and imports in one PostgreSQL transaction.

`SQLitePostgreSQLSandboxImporter.import_source()` is the direct-library primitive: its caller supplied an ownership-validated target, so an interrupted import rolls back PostgreSQL rows, marks the durable manifest `failed`, and leaves that caller-owned isolated target available for an explicit retry or `target.drop()`. `import_into_allocated_target()` is the allocation helper: it marker-validates and drops its fresh target on an import failure. A command-line invocation has no returned ownership capability, so after a successful import both CLI entry points marker-validate and drop the target before emitting status `complete`; their JSON includes `cleanup.status: "dropped"` and the former `target_schema` only as audit evidence. If that final cleanup does not commit, the CLI reports failure rather than a successful rehearsal. A retry is allowed only when the same snapshot fingerprint is supplied and all supported target tables are empty. A completed retry is idempotent only when the destination counts still match the manifest. Changed source snapshots and target drift fail closed.

Before a rehearsal is considered recoverable, run native PG doctor, then `PostgreSQLSandboxOperations.backup(..., quiesced=True)` and `restore_and_verify()` into its generated disposable database. That logical backup is the sandbox rollback/recovery proof; it is not a reverse migration or a production rollback.

## Evidence

Real PG18 integration coverage in `tests/integration/test_postgresql_state_store_sqlite_import.py` proves:

- supported-object counts, foreign-key invariants, preserved message IDs/order, identity high-water, generated search document, durable import manifest, and doctor;
- WAL-safe source snapshot capture, source fingerprint mismatch, target drift, populated-target rejection, and idempotent retry;
- injected interruption rollback followed by resume;
- FTS exclusion, populated non-migrated/missing-map rejection, and unimplemented canonical-column rejection;
- logical backup plus isolated disposable restore verification.

Full runtime parity remains blocked by unimplemented canonical fields/tables and process-local protocols. PostgreSQL sandbox runtime selection must remain disabled until every selected runtime consumer and full canonical state surface has an approved mapping and equivalent ownership/search contracts.
