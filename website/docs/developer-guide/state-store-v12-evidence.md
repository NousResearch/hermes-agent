# PostgreSQL State Store v12 model/config lifecycle evidence

## Bounded contract

Migration v12 is a sequential catalog-validation checkpoint for the already-created `sessions.model_config jsonb`, system-prompt foreign key, and v10 usage-route columns. It adds no DDL. The StateStore exposes the direct model/config lifecycle slice:

- `update_session_meta` drains queued token usage before replacing valid JSON config and only fills a missing model;
- `update_session_model` drains pre-switch usage, atomically clears `browser_model_lock`, persists model/provider config, and invalidates the prompt snapshot;
- `patch_session_model_config` shallow-merges JSON under a row lock, with `None` deleting a key and an empty config stored as SQL NULL;
- `get_session_model_config_value` is tolerant of absent/malformed config; and
- `update_session_billing_route` drains pre-switch usage, updates the current billable route, and invalidates the prompt snapshot.

The SQLite adapter delegates to the canonical `SessionDB` methods. PostgreSQL normalizes JSON internally and relies on its transaction context manager to roll back failed JSON serialization without a partial write.

## Live verification

On 2026-09-12 04:19:26 CST, against the configured local PostgreSQL 18 test service with `vector` and `pg_trgm`:

```text
scripts/run_tests.sh tests/integration/test_postgresql_state_store_fixture.py tests/integration/test_postgresql_state_store_slice.py tests/agent/test_token_usage_transport.py
22 passed, 0 failed
```

The PG18 slice includes fresh and v2/v5 upgrade migration paths, ledger `[1..12]`, catalog drift rejection, SQLite/PostgreSQL differential model/config JSON/null/merge behavior, token flush-before-switch ordering, prompt invalidation, billable-route persistence, and failed JSON-serialization rollback.

```text
uv lock --check
uv run ruff check state_store.py state_store_postgresql.py tests/integration/test_postgresql_state_store_slice.py
python3 -m py_compile state_store.py state_store_postgresql.py
git diff --check
```

All four checks passed.

## Explicitly excluded

No runtime consumer or configuration is switched to PostgreSQL. This does not port session cwd/git publication, profile ownership/backfill, tool-name persistence, browser-runtime locks/YOLO helpers, compression/rewind writes, gateway routing/handoff, search/FTS, recovery/import/cutover/rollback, tenancy, or production selection.