---
sidebar_position: 30
title: "Agent runtime primitives (proposal)"
description: "Proposal: finish consolidating claim/lease/settle and wake paths on the unified gateway's session queue, and share timing helpers"
---

# Agent runtime primitives (proposal)

Status: **proposal for review**. Nothing here is merged. The PRs below are the code you can review; this page is the map.

Updated after the unified gateway runtime (#106742) landed. That PR already consolidated most of what an earlier version of this page proposed, so this version describes what is left.

## Where main is now

The unified gateway gives every profile one durable, fenced queue of work for each session, in `state.db`:

- **`session_admissions`** (`hermes_state_runtime.admit/claim/settle_session_input`) is a per-session FIFO.
  - Admission is keyed on an exact identity (principal, target session, request id), so a repeat returns the existing row.
  - Every claim carries the owner epoch and a generation.
  - Work whose outcome is uncertain becomes `unknown` and is never replayed.
- **`worker_executions`** holds fenced records for cron, child, compute and kanban workers. Each record has an owner epoch, a generation, and adoption by digest.
- **Every wake source enters through `authority.admit_automation`:**
  - process completions (`gateway/session_automation.admit_process_notice`)
  - async-delegation completions
  - Bot Chat DMs (`gateway/session_bot`)
  - relay deliveries
  - cron
- **A failed DM is retried once,** by the owner (`session_bot._maybe_retry`, gated by `tools.bot_failure_reasons.result_retry_action`). The per-lane retry copies are gone.

A completion that the receiving turn cannot confirm goes back to pending instead of being marked delivered (#136225). The rule throughout is: settle only after the effect is known.

## What is still duplicated

Two job systems still run their own claim, lease and settle code beside the session queue:

| Subsystem | Storage | Claim / lease | Crash handling |
|---|---|---|---|
| Kanban | `kanban.db` (`tasks`, `task_runs`, `task_events`) | `claim_lock` + `claim_expires` (900 s), run-id fence | PID + boot-epoch fingerprint; reclaim pauses on a missing or ambiguous owner receipt in `session_admissions` |
| Hosted rooms | `shared-state.db` (`hosted_room_driver_*`) | authority epoch + `lease_generation` + per-task execution/cancel generations | an expired holder's work becomes `indeterminate`, never re-run |

Each of them, and `session_admissions`, has the strongest version of a different primitive:

| Primitive | Best implementation | Why |
|---|---|---|
| Fencing | `session_admissions`, rooms | owner epoch + generation on every claim; rooms add separate execution and cancel generations |
| Idempotent admission | `session_admissions` | an exact identity returns the existing row |
| Never replay uncertain work | `session_admissions`, rooms, cron `delivery_queue` | `unknown` / `indeterminate` |
| Crash detection | Kanban | a PID + boot-epoch + start-time fingerprint defeats PID recycling |
| Failure limits | Kanban | one funnel; rate-limit and infrastructure failures are not counted |

### Gaps still open on main

- **Background-process completions** sit in an in-memory `queue.Queue` until they are admitted. A gateway restart in that window loses them. The process checkpoint (`processes.json`) restores watchers, not completions that are waiting to be sent.
- **The Kanban notifier** rewinds its cursor when a send fails. A crash between the claim and the send still skips that range.
- **The per-profile turn lock** (`bot_relay.acquire_turn_lock`) is a no-op on Windows (no `fcntl`).
- **The explicit stdin peer route** in `tools/bot_mode_dm._run_delivery` runs its child with no timeout. Local DMs no longer take this path; they are admitted through the profile authority.
- **Child `hermes` launchers** still differ in how they resolve the executable, which env vars they scrub, and which cwd they use. This needs re-auditing against the new runtime before any fix.

## Target model

1. **One fenced store for work.** Kanban's claim core and the hosted-room driver move onto the gateway's fenced store, `worker_executions` or a sibling table, behind their existing APIs. They keep their own workflow parts: Kanban's statuses, links and board UI; rooms' per-room FIFO and discussion planning. Kanban's fingerprint and failure-limit funnel move with them so that every consumer gets them.
2. **Every wake source goes through `session_admissions`.** It already does. What remains is making process completions durable *before* admission, so that nothing is held only in memory.
3. **Shared timing helpers.** `poll_until` and `unlink_files_older_than` replace hand-written wait and sweep loops.

Out of scope: multi-host work distribution, and any user-visible change to Kanban or Bots.

## PR stack

No step deletes code until the behaviour it replaces is pinned by a test that passes on main.

| # | PR | Kind | What it does | Status |
|---|---|---|---|---|
| 1 | #135282 | refactor | `poll_until` and `unlink_files_older_than`; about 30 hand-written wait/sweep loops moved onto them (call sites +129 / −244) | ready for review |
| 2 | #136006 | tests | 6 behaviour pins: async-delegation lease/caps/replay age, Kanban stale-run writes, the relay stale-file sweep | ready for review |
| 3 | #136007 | draft | `agent/work_ledger.py`, a generic claim/lease/settle core aimed at Kanban and rooms. Not wired. Moves Kanban's process fingerprint to `hermes_cli/process_identity.py` | draft: decide whether it should exist (question 1) |

Closed after #106742 landed: #136005 (its two retry copies were replaced by `session_bot._maybe_retry`) and #136008 (`session_admissions` is the durable wake queue it proposed).

### After the drafts are agreed

1. **Durable process completions:** write a completion to `state.db` before it enters the in-memory queue, and admit it from there. Fixes the restart loss.
2. **Rooms onto the shared store:** the room driver is the smaller consumer, so it is the cheapest one to move.
3. **Kanban's claim core onto the shared store,** behind its existing API, after rooms.
4. **Launcher fixes,** each small and each a behaviour change: a timeout on the stdin peer route; one `hermes` resolver; aligned env scrubbing.

## Risks

- **Kanban is the largest and most-used subsystem.** It moves last, behind its existing API.
- **Behaviour drift during migration.** Mitigated by the pin PR plus the existing suites. Every migration PR keeps them green without editing them.
- **Two fenced stores.** If #136007 ships as a separate ledger, there are two. That is the main thing to decide before any wiring.

## Open questions for the team

1. **Ledger or `worker_executions`?** Should Kanban and the room driver move onto the gateway's existing `worker_executions`, extended as needed, or onto a separate generic ledger (#136007)? The first gives one store but couples Kanban to the gateway schema. The second gives a cleaner API but two fenced stores.
2. **Expired-lease settle.** Should a holder past its expiry still be able to settle until reclaim runs (Kanban today), or does expiry close the lease at once (rooms and `session_admissions` today)?
3. **Failure classes.** Should every reclaim count as an attempt, or should a settle carry a reason so that rate-limit and infrastructure failures don't count (Kanban's model)?
4. **Durable completions.** Should a completion be written to `session_admissions` straight from the process watcher, or to a small pre-admission table that the gateway drains?
