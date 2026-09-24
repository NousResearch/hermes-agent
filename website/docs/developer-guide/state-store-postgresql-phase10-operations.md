# PostgreSQL sandbox operations — Phase 10 evidence

## Scope and boundary

This is an isolated PG18 reliability slice for the already implemented PostgreSQL tenant schemas. It does **not** select PostgreSQL for a runtime, change default/live configuration, access `state.db`, migrate SQLite data, or alter the named-volume fixture contents outside test-created tenant schemas and disposable restore databases.

`postgresql_state_store_operations.PostgreSQLSandboxOperations` is the backend-native API seam. It receives a trusted state-store schema derived by the existing tenant resolver; it does not accept a profile name or caller-provided schema. An optional already-trusted dedicated delivery-ledger schema is included only when supplied by its owner.

## Doctor/status contract

`doctor()` is sanitized: it returns no DSN or credentials. It proves loopback reachability, PostgreSQL server version, required extension availability, migration ledger values, required-table inventory/counts, message and usage FK-orphan checks, generated-document/GIN search health, active ownership-lease count, and delivery-ledger v1/catalog/count when present. A missing extension, catalog drift, unhealthy search index, or orphan invariant fails closed.

## Logical backup and restore contract

1. The caller must assert `quiesced=True`; backup otherwise fails closed.
2. `pg_dump` creates a custom-format logical dump only for the trusted state schema and, when present, the trusted delivery-ledger schema. Passwords are not placed in command argv.
3. The manifest is atomically published after the archive. It contains the archive byte count/SHA-256/format, server and tool versions, tenant catalog heads, row counts, search/lease invariants, extension requirements, and delivery-ledger state. It contains no DSN.
4. Restore rejects malformed manifests, unsafe archive paths, wrong tenant schemas, byte/checksum mismatch, invalid target names, and pre-existing targets.
5. Restore creates a fresh `hermes_state_restore_<uuid>` database only. It never uses `--clean` or `--create`, pre-creates only the trusted schemas and manifest-required extensions in that disposable database, restores with `pg_restore --exit-on-error --single-transaction`, then re-runs the non-mutating catalog doctor and compares manifest migration/count/search/invariant/delivery-ledger state.
6. The successful target is dropped by default. `keep_restored_target=True` is explicit diagnostic retention; it remains an isolated generated-name database. This provides the reversible sandbox proof, not a production rollback claim.

## PG18 acceptance evidence

`tests/integration/test_postgresql_state_store_operations.py` proves on the loopback named-volume PG18 fixture:

- doctor covers state search, active lease, `pg_trgm`/`vector`, and dedicated delivery-ledger catalog/count;
- a schema-only logical dump round-trips state and ledger into a new database, validates the manifest contract, and leaves the pre-existing restore-database set unchanged after cleanup;
- missing extension, no-quiesce backup, pre-existing restore target, and tampered manifest all fail closed without accepting a restore candidate.

## Sandbox readiness and rollback plan

The sandbox is ready for operator-invoked native inspection and disposable restore verification only. Before any future cutover, a separate approved task must establish offline writer quiescence, a consistent SQLite source snapshot including WAL, FK-order import, sequence restoration, derived-index rebuild, cross-backend semantic checks, consumer routing, and a reverse-export rollback path. No dual write, default-state overwrite, or config-only rollback is authorized by this evidence.
