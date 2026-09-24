# PostgreSQL test-target fixture safety inventory

`tests/integration/postgresql_test_target.py` is the only allowed implementation
of target allocation, ownership-marker validation, or schema teardown. Its
fixture allocates a UUID-derived schema, stores an opaque marker capability, and
revalidates that marker immediately before destructive teardown.

## Migrated family

- `test_postgresql_compression_rotation_acceptance.py`
  - `rotation_schema` now obtains the schema from `postgresql_test_target`.
  - StateStore receives that schema at construction.
  - v21 catalog fault injection goes through `OwnedPostgreSQLTestTarget.execute`.
  - PG v19 coordination and v20 atomic-rotation catalog contracts remain covered by the rotation gate.
- `test_postgresql_state_store_slice.py` (28 tests)
  - Every StateStore factory is routed to a fixture-owned UUID schema; root and named profile acquisitions obtain separately owned UUID targets through the same helper.
  - Fixture reset, historical seed, catalog drift, index rebuild, and data fault injection use marker-validated target operations.
  - Database-wide optional-extension mutation was removed from this family; it never changes shared named-volume state.
- `test_postgresql_session_runtime_ownership.py` (5 tests)
  - Runtime ownership stores are constructed against the owned schema; expiry, invalid-row rollback, and index-drift injection use the target helper.
- `test_postgresql_delivery_ledger.py` (6 tests)
  - The ledger uses `postgresql_delivery_target`; dedicated migration/catalog and index-drift coverage use its marker-validated target.
- `test_postgresql_cli_session_store.py` (8 PostgreSQL lifecycle/public-consumer tests; 1 legacy SQLite control)
  - CLI and AIAgent lifecycle factories receive the owned schema at construction; direct post-close schema deletion is removed.
  - The inline public `session_search` request/response path exercises CJK/grammar, discovery health, anchored scroll, read, browse, and explicit tenant-local rebuild under the `state.db` trap.
- `test_postgresql_owned_family_safety.py` (2 tests)
  - A pre-existing shared sentinel schema/catalog record survives StateStore, runtime, and delivery migration construction unchanged.
  - AST inventory rejects direct dangerous `cursor.execute` DDL/DML in the four migrated files. Explicit exemptions are catalog/data reads and the store-pool `SET/SHOW search_path` reset contract.

## Static inventory status

`test_postgresql_owned_family_safety.py` statically inventories every migrated PostgreSQL
integration family (state-store, runtime ownership, ledger, CLI, rotation acceptance,
compression coordination, Phase12 fault harness, SQLite import, and operations). It
rejects direct dangerous `cursor.execute` DDL/DML and permits only documented reads and
store-pool `search_path` lifecycle checks.

**Remaining unsafe PostgreSQL integration callsites: 0.**

The v20 production adapter is covered by the owned rotation acceptance target. Broader
selected-PG routing remains fail-closed; no test path opens SQLite as fallback.
