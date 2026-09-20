# Kanban supervision red-contract evidence

Captured: 2026-09-20T08:57:58+02:00
Baseline: `main` at `7a1e78e411884e1e50f41630b0a0b80e6c490e7f` (local branch was 3 commits ahead and 1584 behind `origin/main`; no rebase or product-code change was performed).
Scope: tests and evidence only; no deploy, service restart, or production-container action.

## Implemented contract

Kanban publication now validates every explicitly requested skill against the
assignee's effective profile before writing any task, link, event, or run. The
dispatcher repeats this check for legacy/external rows and records a terminal
`spawn_failed` attempt instead of claiming or spawning them. Creator and root
lineage are durable task fields. Dependency blocks cannot make an orchestrator
wait on its own descendant, and the dashboard projects `running` only while the
PID, process-start fingerprint, and heartbeat all prove a fresh worker; otherwise
it reports `recovering`. Queued, dependency-wait, and failure states are likewise
explicit. Existing atomic reclaim, retry limit, and sticky breaker behavior
remain the source of truth. No deployment or service restart is part of this fix.

## Executable contract

`tests/hermes_cli/test_kanban_supervision_contract_e2e.py` contains seven boundary regressions:

1. CLI creation with an explicit unknown skill must fail before publication and leave task/link/event/run counts unchanged. The error must name the assignee profile and bad skill.
2. A legacy or externally inserted row with an unavailable explicit skill must be rejected by the dispatcher as a concrete ended `spawn_failed` run, never claimed/spawned or left `running`.
3. The graph remains acyclic and an orchestrator cannot block waiting on its own descendant; it completes so the child is promoted. Self-edge and cycle rejection are checked transactionally.
4. A dead PID is reclaimed, retry accounting is bounded, and the terminal breaker remains sticky.
5. A stale heartbeat is reclaimed after the configured threshold.
6. A recycled PID/fingerprint mismatch is reclaimed without signalling the unrelated live process.
7. A continuation inherits the goal session and exposes durable root/continuation lineage plus dependency-wait projection.

The worker tests use real SQLite transactions and dispatcher functions. Skill publication uses a real `python -m hermes_cli.main kanban create` subprocess. Status projection imports and calls the actual dashboard plugin serializer.

## Old-main red baseline

Command:

    scripts/run_tests.sh tests/hermes_cli/test_kanban_supervision_contract_e2e.py --tb=short

Observed result:

    collected 7 items
    FFFFFFF
    7 failed in 2.27s

Failure boundaries:

- CLI returned exit 0 and persisted a ready task containing `contract-skill-that-does-not-exist`.
- Dispatcher called the spawn function for that invalid legacy task.
- `block_task(..., kind="dependency")` accepted an orchestrator waiting on its descendant.
- Dashboard projection reported raw `running` for dead PID, stale-heartbeat, and PID-reuse rows.
- Dashboard projection had no `operational_status`, `root_task_id`, or `continuation_of` fields.

The first three graph assertions (self-edge, reverse-edge cycle, unchanged edge count) execute before the descendant-block assertion, so existing cycle guards are preserved while the missing orchestration guard remains red.

## Root-cause code paths

- `hermes_cli/kanban.py:_cmd_create` passes `args.skills` directly to `kanban_db.create_task`; no profile-effective skill resolution runs before the transaction.
- `hermes_cli/kanban_db.py:create_task` only calls `_normalize_task_skills`, which validates syntax/toolset confusion but not existence or disabled/effective availability. It then inserts task, links, and events in one transaction.
- `hermes_cli/kanban_db_dispatch.py:_dispatch_lane_task` checks only `profile_exists`, claims the card, resolves workspace, and calls the spawn function. It does not defensively validate `claimed.skills`.
- `hermes_cli/kanban_db.py:block_task` only interprets dependency as an incomplete parent. A parent that names its descendant in free-text has no graph guard and is re-kinded to sticky `needs_input` when it has no incomplete parents.
- `hermes_cli/kanban_db_dispatch.py:_worker_alive`, `_reclaim_dead_workers`, `detect_stale_running`, and `kanban_db.release_stale_claims` already contain substantial PID/fingerprint and reclaim machinery; the missing contract is the read projection before the next sweep.
- `plugins/kanban/dashboard/plugin_api.py:_task_dict` serializes the stored dataclass status verbatim and adds only age/latest summary. It performs no live PID/fingerprint/heartbeat validation and exposes no operational status.
- `hermes_cli/kanban_db_graph.py:inherit_creator_origin` copies session/subscription provenance but stores no durable creator/root lineage column; the creator id exists only in the `created` event payload.

## Existing-suite signal

Command:

    scripts/run_tests.sh tests/hermes_cli/test_kanban_worker_pid_fingerprint.py tests/hermes_cli/test_kanban_block_kinds.py tests/hermes_cli/test_kanban_creator_origin.py tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py --tb=short

Observed: 11 passed, 2 failed. PID fingerprint (4), creator origin (3), and reclaim lock guard (2) tests passed. Two pre-existing block-kind tests failed because current `link_tasks` now rejects linking a running child; this is baseline drift independent of the new test file and should be reconciled in the implementation phase.

## Hyper-V read-only watch

`t_3ef8d5a0` was queried through the active Kanban API and returned `task ... not found`. No task, worker, continuation, process, container, or service was stopped or mutated. The implementer should repeat this read-only lookup against the board/home that owns the Hyper-V task if that board is supplied by the orchestrator.
