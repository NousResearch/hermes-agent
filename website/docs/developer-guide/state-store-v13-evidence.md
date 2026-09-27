# PostgreSQL State Store v13 cwd/Git publication evidence

## Selected bounded group

Direct caller/schema/test audit selected `cwd` plus asynchronous Git repository publication as the next independently closed group. It owns `cwd`, `git_branch`, `git_repo_root`, and the monotonic `git_metadata_generation` fence. Profile ownership/backfill was excluded because it requires an explicit PostgreSQL tenant/profile acquisition boundary and cross-profile runtime policy. Browser route locks/YOLO were excluded because persistent runtime policy is separate from leases and process-local approval ownership.

The backend-neutral StateStore now exposes `update_session_cwd` and `publish_session_git_metadata`. SQLite delegates unchanged to SessionDB. PostgreSQL v13 adds `git_branch text` and `git_metadata_generation bigint NOT NULL DEFAULT 0`, validates both against the catalogs on every open, and keeps the migration ledger linear through v13.

## Semantics

`update_session_cwd` atomically increments the generation for every claim. A changed cwd or explicit replacement clears/replaces Git identity, while a same-cwd claim only writes non-empty captured values. `publish_session_git_metadata` writes only non-empty Git values and requires the exact `(session_id, cwd, generation)` claim. Therefore a delayed A probe cannot overwrite a newer A→B→A claim, and a failed newer probe does not erase established metadata.

## Live verification

At 2026-09-12 04:35:15 CST, against the local loopback-only PostgreSQL 18 test service:

```text
scripts/run_tests.sh tests/state/test_session_git_metadata_generation.py tests/integration/test_postgresql_state_store_slice.py
38 passed, 0 failed
```

The execution covers existing SQLite generation behavior and live SQLite↔PG18 differential claim/publish behavior, fresh and v2/v5 upgrade migration paths through ledger `[1..13]`, PostgreSQL concurrent claims, stale publication rejection, move-clearing, catalog default drift rejection, and retained token/model lifecycle regressions in the integration slice.

```text
python3 -m py_compile state_store.py state_store_postgresql.py tests/integration/test_postgresql_state_store_slice.py
git diff --check
```

Both checks passed.

## Explicitly excluded

No runtime consumer or configuration is switched to PostgreSQL. This does not port profile ownership/backfill or tenant isolation, browser route-lock/YOLO policy, process-local approval/CUA lifecycle, leases/fencing/heartbeats, tool-name persistence, compression/rewind, gateway routing/handoff, search/FTS, recovery/import/cutover/rollback, or production selection.
