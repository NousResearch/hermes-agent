# Memory sync snapshot repair — 2026-09-10

Implementation worker handoff; no commits, push, deployment, restart, credential reads, or production DB/config writes. Only the two authorized worktrees were edited. Main owns review and production acceptance.

## Change and contract

`MemoryManager.sync_all` now synchronously prepares an immutable completed-turn snapshot before its existing background submission for providers opting into `sync_turn_snapshot_version = 1`. Default providers retain the existing full-history list and signature compatibility. The snapshot binds enqueue-time session/current profile home and serializes only the last user-led completed text turn: at most 128 messages, 64,000 content characters per message, and 1,000,000 serialized characters. Full nested tool calls and actual persistence markers are retained without modifying live messages, API history, prompts, or cache state. Unsupported context passes None; ordinary capture continues. No new tool, queue, scheduler or registry.

The eimemory provider opts in, decodes the concrete host snapshot type and independently verifies exact committed IDs/content/tool structure, session ownership and current profile with read-only bounded SQL. Historical allowance requires that host object with matching payload/session/home, all original rows active and contiguous, and the immediately following active row (if any) a user boundary. Caller dictionaries/flags retain strict current-tail checks. Bodies are read only for exact IDs; boundary metadata remains limited to 129 rows. Snapshot type provenance is an in-process contract, not authentication against arbitrary Python execution; source/ACL/RPC checks remain mandatory and unchanged.

Unsupported multimodal, partial, oversized, missing or failed-persistence evidence is never promoted. Interrupted turns are excluded by the existing real AIAgent entry-point guard. Delayed turns whose rows have been deactivated/rewritten, whose session binding changes, or whose active profile changes fail closed. Supporting-context diagnostics report rejection; ordinary capture continues. This does not promise replay after compression/session switch.

## Files

Host: `agent/memory_manager.py`, `agent/memory_provider.py`, new `agent/memory_sync_snapshot.py`, new `tests/agent/test_memory_sync_snapshot.py`, this audit.

Corresponding eimemory changes: `eimemory/adapters/hermes/provider_core.py`, `eimemory/adapters/hermes/durable_handoff.py`, `tests/test_hermes_durable_handoff.py`, `docs/audit/memory-core-v1-repair.md`. Existing unrelated worktree changes were preserved.

## Commands and results

All commands prefixed `rtk proxy`. No real host imports were skipped. Existing host venv supplies dependencies; source imports resolve to the isolated host worktree. HOME and HERMES_HOME are temporary for tests, with a clean environment. Local RPC means in-process `EIBrainRPCBridge`, not network transport.

Integration command prefix:

```sh
env -i PATH=/usr/bin:/bin HOME=/tmp HERMES_SOURCE=/home/darrow/tmp/hermes-memory-sync-snapshot PYTHONPATH=/home/darrow/tmp/eimemory-core-v1-worktree TZ=UTC LANG=C.UTF-8 /home/darrow/.hermes/hermes-agent/venv/bin/python -m pytest -q
```

- Before implementation, prefix plus `/home/darrow/tmp/eimemory-core-v1-worktree/tests/test_hermes_durable_handoff.py -k host_manager --tb=short`: **3 failed, 19 deselected** (3.64s). List clear, nested arguments mutation and later persisted user each lost evidence.
- After implementation, same file without selection: **22 passed** (3.44s).
- Expanded final affected run: prefix plus `/home/darrow/tmp/eimemory-core-v1-worktree/tests/test_hermes_durable_handoff.py /home/darrow/tmp/eimemory-core-v1-worktree/tests/test_same_turn_project_context.py /home/darrow/tmp/eimemory-core-v1-worktree/tests/test_hermes_adapter.py --tb=short`: **128 passed** (8.57s). Includes host-to-RPC sensitive-support rejection, real failed transaction rollback, invalid boolean/numeric markers/IDs, profile/session/content/tool/boundary denials and ordinary capture.
- Added two-completed-turn enqueue contract test: prefix plus `/home/darrow/tmp/eimemory-core-v1-worktree/tests/test_hermes_durable_handoff.py -k two_queued --tb=short`: **1 passed, 32 deselected** (1.28s). Both associations use their own actual committed IDs.

Host command:

```sh
env -i PATH=/usr/bin:/bin HOME=/tmp HERMES_PYTHON=/home/darrow/.hermes/hermes-agent/venv/bin/python bash scripts/run_tests.sh tests/agent/test_memory_sync_snapshot.py tests/agent/test_memory_provider.py
```

Existing provider file: **73 passed** (4.9s). New fixture initially failed registration because its stub lacked `get_tool_schemas`; corrected stub and designated legacy provider builtin to obey single-external-provider rule. Re-run same host command with only `tests/agent/test_memory_sync_snapshot.py`: **6 passed** (3.7s). Includes immutable decoding, legacy compatibility, unsupported context, unchanged markers/history and actual AIAgent interrupted-entry guard. After the final provider type annotation update, the complete two-file host command passed again: **79 passed, 0 failed** (5.4s). No full suite run. `git diff --check` passed in both worktrees.

## Remaining acceptance

Main must review the host/plugin pair, commit/version/deploy and perform scoped production verification. No production acceptance, socket transport test, LLM conversation replay or historical backfill is claimed. The host and plugin changes must ship together to obtain queue-time snapshots; older hosts retain the plugin current-tail behavior. Legacy providers intentionally keep their original behavior unless they opt in.
