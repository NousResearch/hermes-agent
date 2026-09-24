# StateStore v9 system-prompt snapshot evidence

## Selected canonical group

v9 ports the independently coherent **content-addressed system-prompt snapshot** group. Direct source evidence is `hermes_state.py::_store_system_prompt` and `_delete_unreferenced_system_prompts`; `hermes_state_sessions.py::update_system_prompt`; and the runtime cache restore/write path in `agent/conversation_loop.py:643,662`. The group is bounded to SHA-256 identity, byte-preserving prompt content, session reference/clear, and orphan collection. It does not mutate transcripts, leases, counters, watermarks, FTS, or conversation generations.

Compaction/rewind/rotation was rejected for this slice: `hermes_state_messages.py::_check_transcript_write_guards`, `_bump_session_counters`, `_bump_conversation_generation`, and `archive_and_compact` couple it to turn/compression leases, counter repair, watermarks, carried-forward rows, FTS triggers, and atomic publication. v8 remains read-only projection only.

## Contract and migration

`StateStore` now exposes `set_system_prompt(session_id, system_prompt)` and `get_system_prompt(session_id)`. `SqliteStateStore` delegates unchanged to `SessionDB.update_system_prompt` and its hydrated session read.

PostgreSQL migration v9 creates `system_prompts(hash text primary key, prompt text not null)`, adds `sessions.system_prompt_hash`, and validates the `sessions_system_prompt_hash_fkey` foreign key. Writes SHA-256 hash and prompt in one transaction, set the session reference, and collect orphan rows in that transaction. Reopen validates every v1-v9 catalog contract; fresh and historical v2/v5 ledgers migrate linearly to `[1,2,3,4,5,6,7,8,9]`.

## Verified parity and failure behavior

The live PG18/SQLite differential test proves shared content is read byte-for-byte through both APIs, clearing one reference preserves the other, and setting a prompt for an absent session does not expose a session prompt. Fresh/reopen and upgrade tests assert v9 ledger/catalog/table/FK state. The direct SQLite dedupe suite validates original replacement, deletion, import, and compression-child collection behavior without changing legacy code.

## Exact incomplete raw SQLite scope

Unported canonical state remains: complete session lifecycle/read/list/delete/import/export and model-config patches; usage queue/coalescing, billing, `session_model_usage`; messages rewrite/archive/compaction/rewind/reactions; compression/turn leases, counters, watermarks, generations; FTS/search/repair; gateway routing, delivery, hosted rooms, policy and driver state; file/WAL/backup/recovery/doctor; profile tenancy, SQLite import/cutover/reverse rollback, consumer cutover, multi-process/multi-host crash semantics, and production PostgreSQL selection. No live `state.db`, configuration, remote, or deployment was changed.
