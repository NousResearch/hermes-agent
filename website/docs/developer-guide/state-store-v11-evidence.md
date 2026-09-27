# StateStore v11 conversation-generation lifecycle evidence

## Bounded contract

v11 extends the backend-neutral `StateStore` with `end_session`, recoverable-close `promote_to_session_reset`, and source-qualified `latest_conversation_boundary` lifecycle parity. SQLite remains the compatibility implementation; PostgreSQL is still an opt-in bounded slice and no runtime consumer/configuration cutover is made here.

A reset-eligible end advances generation only when its session-row end stamp is actually written. PostgreSQL uses one transaction: `UPDATE ... RETURNING source, session_key` enforces first-end-reason-wins and supplies the exact peer identity for the generation upsert before commit. Duplicate/stale ends therefore cannot increment. Promotion can update only a live or recoverably ended row and advances in that same transaction. Compression, ordinary close reasons, missing rows, and unkeyed rows do not advance.

Migration v11 creates `conversation_generations(source, session_key, generation)` with composite primary key and deliberately no session foreign key. The durable row is never deleted by session deletion/pruning, preventing a retired source/key generation from being reissued after an ABA sequence. Catalog validation requires its exact `text`/`text`/non-null `bigint default 0` columns, ordered composite primary key `(source, session_key)`, and no foreign key; fresh and upgrade ledgers are linear through `[1..11]`.

## Executed evidence

- `scripts/run_tests.sh tests/integration/test_postgresql_state_store_slice.py` — 16 passed against live PostgreSQL 18; includes SQLite/PostgreSQL lifecycle differential, reset/recoverable promotion, duplicate and stale end behavior, unkeyed/source separation, raw session deletion ABA survival, transaction rollback injection for both end and promotion, and two independently pooled PG StateStore instances released simultaneously for close/promotion plus concurrent duplicate reset.
- `scripts/run_tests.sh tests/agent/test_declared_conversation_scope.py tests/gateway/test_session_store_expiry_finalized.py` — 58 passed (55 conversation-scope, 3 promotion lifecycle).
- `scripts/run_tests.sh tests/integration/test_postgresql_state_store_slice.py -k 'token_usage_transport_parity or token_usage_delta_rolls_back_summary_when_attribution_fails'` — 2 passed, preserving v10 model-usage transport and rollback regression coverage.
- `uv lock --check`, `uv run --extra dev ruff check state_store.py state_store_postgresql.py tests/integration/test_postgresql_state_store_slice.py`, `python3 -m py_compile state_store.py state_store_postgresql.py tests/integration/test_postgresql_state_store_slice.py`, and `git diff --check` — passed.

## Remaining scope and exact blocker

No global runtime consumer has been switched to PostgreSQL; full SessionDB/gateway surface, FTS/search, lease/recovery, import/reverse rollback, multi-process/multi-host semantics, and runtime cutover remain out of scope.

`tests/gateway/test_api_server_declared_conversation.py` was attempted with the mandated runner but could not collect because the active test environment lacks `aiohttp` (`ModuleNotFoundError`); it is an environment dependency failure, not a claimed pass. No dependency or configuration was changed to mask it.
