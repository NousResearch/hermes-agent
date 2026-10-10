---
sidebar_position: 30
title: "Agent runtime primitives (proposal)"
description: "Proposal: one work ledger, one agent inbox and shared launch/timing helpers under Kanban, Bots, rooms, cron delivery and async delegation"
---

# Agent runtime primitives (proposal)

Status: **proposal for review**. Nothing here is merged. The PR stack below is the reviewable code; this page is the map.

## Problem

Hermes has grown five separate ways for agents to hand work to each other and wake each other up. Each one re-implements the same small set of primitives (claim, lease, heartbeat, crash reclaim, idempotent settle, wake) with its own storage and its own edge-case bugs.

| Subsystem | Storage | Claim / lease | Crash handling | Lines |
|---|---|---|---|---|
| Kanban | `kanban.db` (`tasks`, `task_runs`, `task_events`) | `claim_lock` + `claim_expires` (900 s), run-id fence | PID + boot-epoch fingerprint, terminate-then-release, failure limit | ~11.6k |
| Hosted rooms | `shared-state.db` (`hosted_room_driver_*`, `hosted_room_events`) | authority epoch + `lease_generation` + per-task execution/cancel generations | expired holder's work becomes `indeterminate`, never re-run | ~4.6k |
| Async delegation | `state.db` `async_delegations` | delivery-only claim, 300 s lease, 8 attempts | owner PID + start time; replay for 48 h | 1.2k |
| Bot DMs (`message_agent`) | `bot_live_delivery/*.json` mailbox + temp DM files | file lock; claims never expire | none for a stuck claim | 1.3k |
| Bot relay | `bot_relay/{outbox,claimed,replies}` dirs | atomic rename; one re-offer | re-offer once, then `delivery_timeout` | 0.7k |
| Cron → Bot Chat | `cron/deliveries.db` + `bot_chat_pending/*.json` | owner-fenced UPDATE | dead owner → `unknown`, never replayed | 0.6k |

Plus a sixth wake path, background-process completions, which lives only in an in-memory `queue.Queue` and is lost on restart.

Every primitive exists in at least three places, and each subsystem has the best version of a different one:

| Primitive | Best existing implementation | Why |
|---|---|---|
| Fencing | Hosted rooms | monotonic generations; separate execution/cancel generations make cancel-vs-complete races decidable |
| Idempotent settle | Hosted rooms | same `settlement_id` replays the original result; a different one raises |
| Admission | Hosted rooms | `task_id` + payload digest, conflict on drift (async delegation uses `INSERT OR REPLACE`, which overwrites) |
| Crash detection | Kanban | PID + boot-epoch + start-time fingerprint defeats PID recycling |
| Failure limits | Kanban | one funnel, non-counted classes (rate limit, infra), immediate trip on config errors |
| Delivery protocol | Async delegation | claim / complete / release / defer / drop with an attempt budget |
| Never replay uncertain work | Cron `delivery_queue`, rooms | dead owner → `unknown` / `indeterminate` |

### Known gaps found by the audit

These fall out of the duplication, not out of any one bug:

- A DM record stuck in `claimed` is never recovered (mailbox claims never expire, and nothing sweeps them). Same for `bot_chat_pending`.
- Background-process completions are in memory only; a restart loses them.
- Kanban's notifier advances its cursor before sending, so a crash between the two loses that range.
- The CLI marks an async-delegation completion delivered before its turn runs.
- The per-profile delivery turn lock (`bot_relay.acquire_turn_lock`) is a no-op on Windows and covers only delivery turns.
- The local `message_agent` turn has no timeout; a hung child hangs the background runner forever (the relay and cron lanes cap at 600 s).
- Seven launchers build `hermes` child processes and disagree in 18 places (executable resolution order, which env vars are scrubbed, cwd). Most look accidental.

## Target model

Three primitives, each with one implementation. Kanban, Bots, rooms, cron delivery and delegation become *policies and views* over them instead of owning their own queues.

**1. Work ledger** (`agent/work_ledger.py`). A durable unit of work: admit → claim (lease with a monotonic generation) → renew → settle, plus crash reclaim. It takes the strongest version of each primitive from the table above. Every state change appends to an event log in the same transaction. Per-kind policy decides whether an expired holder's work is retried (counted against a limit) or marked `indeterminate`.

**2. Agent inbox** (`agent/agent_inbox.py`). One durable per-profile queue of "things that should cause a turn". Items carry a source (user, dm, relay, cron, process, delegation, kanban) and a lane:

- `interactive` (a human) > `agent` (teammate DMs, relay) > `background` (completions, cron, kanban).
- A burst of items with the same coalesce key becomes one turn.
- Recovery policy is per source: DMs and cron never replay uncertain work; relay and delegation get one more try; process and kanban events re-queue.
- Today's preemption rule is kept: only a human interrupts, and only a background turn. Teammate DMs and background work never interrupt.

**3. Shared launch and timing helpers.** One retry policy for child-process turns. Later, one `hermes` executable resolver and one env builder. `poll_until` / `unlink_files_older_than` replace hand-written wait and sweep loops.

Out of scope for now: multi-host work distribution, and changing any user-visible behaviour of Kanban or Bots.

## Migration map

| Today | Becomes | Stays private to the subsystem |
|---|---|---|
| Kanban `tasks.claim_lock` / `claim_expires` / `current_run_id` | ledger `holder_id` / `expires_at` / `lease_generation` | workflow statuses, links, comments, attachments, board UI, dispatcher |
| Kanban `consecutive_failures` / `max_retries` | ledger `attempts` / `max_attempts` + per-kind policy | exit-code taxonomy |
| Rooms `hosted_room_driver_tasks` + lease rows | ledger units + lease generation | authority epoch, per-room FIFO, discussion planning |
| Async delegation `async_delegations` delivery columns | inbox item (source `delegation`) | stall monitor, child results |
| `bot_live_delivery` mailbox | inbox item (source `dm`, lane `agent`) | live-owner pinning to lease + compression tip |
| Relay outbox/claimed/replies dirs | inbox item (source `relay`) | Desktop transport, roster |
| Cron `deliveries.db` + `bot_chat_pending` | inbox item (source `cron`) | platform send for non-Bot-Chat targets |
| `process_registry.completion_queue` | inbox item (source `process`), now durable | watch patterns, output capture |
| Kanban notifier cursor | inbox item (source `kanban`, coalesced per subscription) | subscription management |

## PR stack

Order is lowest risk first. No step deletes code until the behaviour it replaces is pinned by a test that passes on main.

| # | PR | Kind | What it does | Status |
|---|---|---|---|---|
| 1 | #135282 | refactor | `poll_until` and `unlink_files_older_than`; ~30 hand-written wait/sweep loops moved onto them (call sites +127 / −251) | ready for review |
| 2 | #136005 | refactor | `run_turn_with_retry`: one retry policy for child-process delivery turns; cron uses the shared Bot Chat argv | ready for review |
| 3 | #136006 | tests | 13 behaviour pins for relay re-offer/expiry, DM mailbox claims, async-delegation lease/caps, kanban stale-run writes; lists the ~15 invariants already pinned elsewhere | ready for review |
| 4 | #136007 | draft | work ledger: schema, operations, tests, migration map. Not wired. Moves kanban's process fingerprint to `hermes_cli/process_identity.py` (re-exported, no caller change) | draft — review the shape |
| 5 | #136008 | draft | agent inbox + scheduler: schema, lanes, coalescing, per-source recovery, stale-claim watchdog. Not wired. Stacked on #135282 | draft — review the shape |

### After the drafts are agreed

Each is a separate PR, in this order:

1. **Launcher fixes** (behaviour changes, each small): timeout on the local `message_agent` turn; one `hermes` resolver honouring `HERMES_BIN` and refusing Windows batch shims; align env scrubbing across delivery launchers; pin the relay child's cwd to `HERMES_HOME`.
2. **Inbox, first consumers**: process completions (fixes restart loss), then async-delegation delivery, then the DM mailbox (adds stuck-claim recovery), then relay and cron Bot Chat delivery. Each one deletes its old queue in the same PR.
3. **Ledger, first consumer**: async-delegation work records, then the room driver (not yet wired in production, so cheapest to move), then Kanban's claim/lease core behind its existing API.
4. **Rooms**: member turns go through each member's inbox, which removes the room lease/liveness code and lets a human DM reach a member mid-room.

## Risks

- **Kanban is the largest and most-used subsystem.** It moves last, behind its existing API, and only after the ledger has two other consumers.
- **Behaviour drift during migration.** Mitigated by the pin PR (#136006) plus the existing suites; every migration PR must keep them green without editing them.
- **One more SQLite file or a shared one.** Open question below; both work, the choice affects backup and multiplex profiles.
- **The drafts may be the wrong shape.** That is why they are drafts and not wired: changing them now costs nothing.

## Open questions for the team

1. **Expired-lease settle.** Should a holder past `expires_at` still be able to settle until reclaim runs (Kanban today), or does expiry close the lease immediately (rooms today)?
2. **Failure classes.** Should `attempts` count every reclaim, or should settle carry a reason so rate-limit and infra failures don't count (Kanban's model)?
3. **Where the inbox lives.** A table in each profile's `state.db`, or a separate `inbox.db`? It also needs retention/tombstones (the `delivery_queue` model).
4. **Typed payloads.** Should inbox payloads be a typed union per source, or opaque JSON with per-source parsers?
5. **Wake signal.** Replace the 0.5–5 s pollers with one notify mechanism in the first inbox PR, or later? Should preemption read the configured `busy_input_mode` instead of hard-coding today's default?
6. **Delivery vs work.** Async delegation's delivery protocol maps onto the inbox, its work record onto the ledger. Is splitting one subsystem across both primitives acceptable?
