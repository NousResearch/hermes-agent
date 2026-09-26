# StateStore v7 active-message record evidence

## Selected contract

v7 ports the **active appendable message-record** group through the backend-neutral `StateStore`:

- full append fields exposed by `SessionDB.append_message` (`hermes_state_messages.py:279-306`), including structured content, tool calls, token/finish/reasoning/Codex fields, platform id, observation/effect flags, compression-summary flag, API-content sidecar, display kind/metadata, and caller timestamp;
- physical identity ordering (`id`, never timestamp) and active-row reads (`hermes_state_messages.py:765-810`);
- empty and multi-row atomic batch append (`hermes_state_messages.py:340-362`).

`MessageRecord`, `append_message_record`, `append_message_records`, and `get_message_records` are additive. The compatibility `append_message`/`get_messages` projection is unchanged.

## PostgreSQL migration

Migration v7 adds the corresponding nullable/defaulted `messages` columns to the existing locked, catalog-validated ledger. Fresh and v1/v2/v5-ledger upgrade coverage verifies the linear `[1, 2, 3, 4, 5, 6, 7]` ledger and required columns. No state.db was opened, migrated, or deleted.

## Live parity evidence

`tests/integration/test_postgresql_state_store_slice.py` opens real PostgreSQL 18 and SQLite stores. It verifies full representative field round-trip, null content, supplied timestamps, decoded structured content/JSON, identity order, and unknown-session batch rollback for both adapters.

Focused execution completed 2026-09-12 02:40:18 CST:

```text
scripts/run_tests.sh tests/integration/test_postgresql_state_store_fixture.py tests/integration/test_postgresql_state_store_slice.py tests/hermes_state/test_append_messages_batch.py tests/hermes_state/test_get_messages_include_compacted.py tests/test_state_store_config.py tests/test_state_store_factory.py
43 passed, 0 failed
```

## Remaining raw SQLite transcript paths (not claimed by v7)

- Rewrite/archive/compaction, leases, counters, display generations, reactions, and resume assembly remain in `hermes_state_messages.py:181-236, 340-362, 486-618, 765-1015, 1265-1300`.
- SQLite schema/FTS/search/repair infrastructure remains in `hermes_state_schema.py`, `hermes_state_fts.py`, `hermes_state_search.py`, `hermes_state_repair.py`, and `hermes_state_compression.py`.
- Direct operational SQLite consumers remain in `agent/transcript_repair.py`, `hermes_cli/session_recovery.py`, `hermes_cli/session_lost_and_found.py`, `tools/session_search_tool.py`, and `hermes_cli/backup.py`.

Therefore v7 does **not** claim compaction/rewind/lineage/transcript display or OpenAI conversation projection, FTS/search, counters/lease semantics, global consumer cutover, SQLite-to-PostgreSQL import, or production PostgreSQL selection parity.

# StateStore v8 resume-projection evidence

## Selected contract

v8 ports the read-only compression-lineage resume projection: compression root/tip identity, full lineage display projection, tip-only model projection, ancestor display prefix, and bounded resume count/guard. The PostgreSQL adapter preserves active-only model rows and display rows where `active OR compacted`; it excludes rewind-only rows and deduplicates repeated carried-forward rows by the current SQLite display key preference (active, then newest row).

This is deliberately not compaction or rewind mutation: those retain SQLite-only lease, watermark, counter, model-config-patch, FTS, and atomic publication behavior.

## PostgreSQL migration and live evidence

Migration v8 adds catalog-validated `messages_resume_projection (session_id, active, id)` to the sequential ledger. Fresh bootstrap, reopen, and v2/v5 upgrade paths now reach `[1, 2, 3, 4, 5, 6, 7, 8]`. Live PostgreSQL 18 and SQLite differential coverage verifies root/tip lineage, model/display split, ancestor-only prefix, count semantics, and guard rejection.

Focused execution:

```text
scripts/run_tests.sh tests/integration/test_postgresql_state_store_fixture.py tests/integration/test_postgresql_state_store_slice.py tests/hermes_state/test_display_projection_parity.py tests/hermes_state/test_resolve_resume_session_id.py tests/test_state_store_config.py tests/test_state_store_factory.py
36 passed, 0 failed
```

## Remaining raw SQLite gaps

- `archive_and_compact`, `rewind_to_message`, compression leases, turn leases, counter repair, display-order backfill, and rotation publication remain `SessionDB`-only.
- Full OpenAI replay fidelity still depends on `SessionMessagesMixin._rows_to_conversation`, including sanitization, persisted-marker stamping, exact clone/carrier handling, and alternation repair; v8 is not a global consumer cutover.
- FTS/search, import/cutover/rollback, recovery, profile tenancy, and production PostgreSQL selection remain unported.
