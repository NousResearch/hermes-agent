# Kanban worker lanes

A **worker lane** is a class of process that the kanban dispatcher can route tasks to. Each lane has an identity (the assignee string), a spawn mechanism, and a contract for what it must do with the task once spawned.

This page is the contract. It exists for two audiences:

- **Operators** picking which lanes to wire into a board (which profiles to create, which assignees to use).
- **Plugin / integration authors** wanting to add a new lane shape (a CLI worker that wraps Codex / Claude Code / OpenCode, a containerised review worker, a non-Hermes service that pulls tasks via the API).

If you're writing the worker code itself — the agent that runs *inside* a lane — the kanban lifecycle and reference details are injected into the worker's system prompt automatically (the `KANBAN_GUIDANCE` block in [`agent/prompt_builder.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py)).

## The hierarchy

```text
Hermes Kanban  =  canonical task lifecycle + audit trail
Worker lane    =  implementation executor for one assigned card
Reviewer       =  human or human-proxy that gates "done"
GitHub PR      =  upstreamable artifact (optional, for code lanes)
```

Hermes Kanban owns lifecycle truth — `ready` → `running` → `review` / `blocked` / `done` / `archived`. Worker lanes execute work but never own that truth; everything they do flows back through the kanban kernel via the `kanban_*` tools (or, for non-Hermes external workers, via the API). Reviewers gate the transition from "code change written" to "task done."

## What a lane provides

To be a kanban worker lane, an integration must provide three things:

### 1. An assignee string

The dispatcher matches `task.assignee` against either a Hermes profile name (the default lane shape) or a plugin-registered worker lane (see [Plugin worker lanes](#plugin-worker-lanes) below). Tasks whose assignee doesn't resolve are left on `ready` with a `skipped_nonspawnable` event so a board operator can fix them; they are not silently dropped or executed by an arbitrary fallback.

### 2. A spawn mechanism

For Hermes profile lanes, the dispatcher's `_default_spawn` runs `hermes -p <assignee> chat -q <prompt>` (or the equivalent module form when the `hermes` shim isn't on `$PATH`) inside the task's pinned workspace, with these env vars set:

| Variable | Carries |
|---|---|
| `HERMES_KANBAN_TASK` | the task id the worker is operating on |
| `HERMES_KANBAN_DB` | absolute path to the per-board SQLite file |
| `HERMES_KANBAN_BOARD` | board slug |
| `HERMES_KANBAN_WORKSPACES_ROOT` | root of the board's workspace tree |
| `HERMES_KANBAN_WORKSPACE` | absolute path to *this* task's workspace |
| `HERMES_KANBAN_RUN_ID` | the current run's id (for the lifecycle gate) |
| `HERMES_KANBAN_CLAIM_LOCK` | the claim lock string (`<host>:<pid>:<uuid>`) |
| `HERMES_PROFILE` | the worker's own profile name (for `kanban_comment` author attribution) |
| `HERMES_TENANT` | tenant namespace, if the task has one |

For plugin lanes, the plugin's `KanbanWorkerLane.spawn(claim)` starts the work. It returns a child PID for crash detection, `None` for work that runs inside the plugin's own process, or `NOT_HANDLED` to give the task back to the native spawn. See [Plugin worker lanes](#plugin-worker-lanes).

### Descendant process scope

A task assignment belongs to the dispatcher worker, not to every program it starts.
Hermes subprocess helpers carry a non-owner fence into shells, execution kernels,
cron deliveries, hooks, language servers, and ordinary stdio MCP servers. Later
children remain fenced even when a script removes the inherited task ID: CLI and
tool mutations are rejected, rather than treating that script as an orchestrator.
Board/database routing and workspace paths are retained. Descendants can read an
existing board without running schema migrations; its owner must initialize it.
The fence is scoped to the lineage's board root (the marker's value is that root, plus the
dispatcher-pinned `HERMES_KANBAN_DB`): a descendant that works against a different Kanban
home — a test or reproduction under a scratch `HERMES_HOME` — gets a normal read-write board.

The dispatcher explicitly grants a newly assigned worker its own scope. The managed
Hermes-tools MCP endpoint can likewise act for its supervising worker, while the
executor's ordinary shell children remain fenced. Workers may only perform lifecycle
handoffs and attach files to their assigned task; `unblock` remains orchestrator-only.
Cross-task comments and follow-up task creation retain their existing behavior.

Integration authors spawning code should use
`agent.delegation_context.delegated_child_subprocess_env` at the actual spawn, after
merging environment overrides. It preserves the caller's credential/profile policy.
This is cooperative runtime scoping, **not OS confinement**: it does not prevent
arbitrary code from deliberately erasing lineage metadata or opening SQLite directly.

### 3. A lifecycle terminator

Every claim must end in exactly one of:

- `kanban_complete(summary=..., metadata=...)` — task succeeds, status flips to `done`.
- `kanban_request_review(summary=..., metadata=..., reviewer=...)` — same-card implementation is complete and enters first-class review; status flips to `review`. The dispatcher loads the bundled `sdlc-review` skill unless `kanban.review_dispatch` is disabled. A reviewer approves with `kanban_complete`, returns actionable rework with `kanban_request_changes`, or escalates a genuine external blocker with `kanban_block`.
- `kanban_block(reason=...)` — task waits for human input, status flips to `blocked`. The dispatcher respawns when `kanban_unblock` runs.
- The worker process exits without a tool call. The kernel reaps it and emits `crashed` (PID died) or `gave_up` (consecutive-failure breaker tripped) or `timed_out` (max_runtime exceeded). This is the failure path; healthy workers don't end here.

The kanban kernel enforces that exactly one of these terminates each run. A worker that calls neither and exits normally is treated as crashed.

## Outputs and the review handoff

For code-changing tasks, pick the review model encoded by the task graph:

- **Same-card review:** call `kanban_request_review(summary=..., metadata=..., reviewer=...)`. The task enters `review` without touching block recurrence accounting. The dispatcher claims it with the bundled `sdlc-review` skill by default. The reviewer approves with `kanban_complete`, calls `kanban_request_changes(reason=...)` to close the review run and route the task back to its original implementer, or blocks only for a genuine external escalation.
- **Pre-created downstream review/QA/release card:** `kanban_show` lists child IDs; inspect those cards with `kanban_show(task_id=...)` before choosing the terminal action. When a child is the downstream review/QA/release phase, call `kanban_complete` on the implementation phase. It cannot promote until this parent is `done`/`archived`. Do not additionally request same-card review and never sticky-block the parent with `review-required:` — either choice strands or duplicates the downstream lane.
- **Human-only boards:** set `kanban.review_dispatch: false`. A task can then remain in `review` until a human approves it or uses `reopen-review`/the dashboard to return it to `ready`/`todo`.

Both review models carry their structured handoff on the lifecycle transition itself. Do not place secrets, tokens, or raw PII in `summary` or `metadata`; run rows are durable.

The injected `KANBAN_GUIDANCE` covers both graph shapes, `kanban_complete`, the same-card review loop, and `kanban_block` for genuine blockers.

## Logs and audit trail

The dispatcher writes per-task worker stdout/stderr to `<board-root>/logs/<task_id>.log`. Logs are auditable from kanban metadata:

- `task_runs` rows carry the `log_path`, exit code (where available), summary, and metadata.
- `task_events` rows carry every state transition (`promoted`, `claimed`, `heartbeat`, `completed`, `blocked`, `review_requested`, `changes_requested`, `review_reopened`, `gave_up`, `crashed`, `timed_out`, `reclaimed`, `claim_extended`).
- `kanban_show` returns both, so a reviewer (or a follow-up worker) reading the task gets the full history without needing dashboard access.

The dashboard renders run history with summaries, metadata blocks, and exit-status badges. CLI users can run `hermes kanban tail <task_id>` to follow live, or `hermes kanban runs <task_id>` for the historical attempt list.

## Existing lane shapes

### Hermes profile lane (default)

The shape every kanban worker takes today: the assignee is a profile name, the dispatcher spawns `hermes -p <profile>`, the worker gets the `KANBAN_GUIDANCE` system-prompt block injected automatically, and uses the `kanban_*` tools to terminate the run. No setup beyond defining the profile.

When you create profiles for your fleet, choose names that match the *role* you want the orchestrator to route to. The orchestrator (when there is one) discovers your profile names via `hermes profile list` — there's no fixed roster the system assumes (the orchestrator side of the contract is part of the injected `KANBAN_GUIDANCE`).

### Orchestrator profile lane

A specialisation of the profile lane: an orchestrator is a Hermes profile whose toolset includes `kanban` but excludes `terminal` / `file` / `code` / `web` for implementation. Its job is decomposing a high-level goal into child tasks via `kanban_create` + `kanban_link` and stepping back. The orchestrator skill encodes the anti-temptation rules.

## Plugin worker lanes

A plugin adds a lane with `ctx.register_kanban_worker_lane(lane)`, where `lane` is a `hermes_cli.kanban_worker_lanes.KanbanWorkerLane`. Use it for a non-Hermes CLI worker (Codex CLI, Claude Code CLI, OpenCode CLI, a local model runner), or for a platform plugin that runs Kanban tasks inside the gateway as its own turns.

```python
from hermes_cli.kanban_worker_lanes import NOT_HANDLED, KanbanWorkerLane


class CodexLane(KanbanWorkerLane):
    name = "codex"
    assignees = ("codex-*",)   # exact names or shell-style globs
    boards = ()                # empty: every board

    def accepts(self, task, *, board, lane):
        return lane == "ready"   # optional: leave review runs to the native spawn

    def spawn(self, claim):
        return start_codex(claim.task, claim.workspace)   # the child's PID


def register(ctx):
    ctx.register_kanban_worker_lane(CodexLane())
```

### Routing

- **Match.** A lane takes a task when the task's assignee matches `assignees`, the board slug matches `boards`, and `accepts(task, board=..., lane=...)` returns True. An empty `assignees` or `boards` matches every value, but a lane must name at least one assignee or board. A lane may name assignees that are also Hermes profiles: on the boards it names, it runs their tasks, and the same profiles keep the native spawn on every other board.
- **Order.** Lanes are tried in name order. Hermes claims the task first, then calls `spawn`. A `spawn` that returns `NOT_HANDLED` passes the claimed task to the next matching lane, then to the native spawn. A `spawn` that raises is a failed spawn: the claim is released and the attempt counts toward the failure breaker.
- **Lane kind.** `claim.lane` is `"ready"` for ordinary work and `"review"` for a review run (`kanban.review_dispatch`).
- **What Hermes keeps.** The claim, the run record, the caps (`max_spawn`, `max_in_progress`, `max_in_progress_per_profile`), the respawn guard, retries and the breaker apply to lane tasks as to native ones. `spawn` runs while the dispatcher holds the board's tick lock, so it must start the work and return.
- **Where lanes apply.** A lane runs only in a process that loaded its plugin, normally the gateway's embedded dispatcher (`kanban.dispatch_in_gateway: true`, the default). A dispatcher that did not load the plugin, for example a standalone `hermes kanban daemon`, leaves the lane's non-profile tasks in `ready`. `kanban.dispatch_profiles` limits native profile claims only. A gateway that serves several profiles sees the lanes of every profile it serves, and each lane's code runs under the Hermes home of the profile that registered it. A plugin under `plugins.isolation: host` cannot register a lane.

### The claim and in-process work

`spawn` receives a `WorkerLaneClaim`: `task` (with `current_run_id` and `claim_lock`), `workspace`, `board`, `lane`, `run_id` and `claim_lock`.

A lane that starts a child process returns its PID. Crash detection and the live-worker claim extension then follow that PID, as for a native worker, and the child ends its run through the `kanban_*` tools or the CLI.

A lane that runs the work inside its own process (for example a gateway platform plugin that runs the task as one of its turns) returns `None` and keeps the claim. The work reports through it, from any thread:

| Call | Effect |
|---|---|
| `claim.heartbeat(note=None)` | Records progress and extends the claim. |
| `claim.complete(summary, result=None, metadata=None)` | Ends the run as `done` (approves a review run). |
| `claim.request_review(summary, metadata=None, reviewer=None)` | Ends the run and moves the card to `review`. |
| `claim.block(reason, kind=None)` | Ends the run as `blocked`. |
| `claim.fail(error)` | Ends the run as a failed attempt: releases the claim, counts toward the breaker, and lets the dispatcher retry. |

Every call is fenced on the run id. A call for a run that Hermes already reclaimed, or that another run replaced, changes nothing and returns False. With no PID, the claim TTL is the liveness signal: heartbeat at least once per `DEFAULT_CLAIM_TTL_SECONDS` (15 minutes; `HERMES_KANBAN_CLAIM_TTL_SECONDS`). A run that stops heartbeating, for example because the gateway restarted, is reclaimed after the TTL and counted as a failure.

The issue that asked for this is [#70547](https://github.com/NousResearch/hermes-agent/issues/70547). The earlier Codex-specific proposal is [#19931](https://github.com/NousResearch/hermes-agent/issues/19931) and the closed-not-merged PR [#19924](https://github.com/NousResearch/hermes-agent/pull/19924).

## Failure modes the dispatcher handles

So lane authors don't have to reimplement these:

- **Stale claim TTL** — a worker that claims and then never heartbeats / completes / blocks gets reclaimed after `DEFAULT_CLAIM_TTL_SECONDS` (15 min default) — but only if the worker process has actually died. A plugin lane run with no PID is reclaimed when its heartbeats stop. A live worker (slow model spending 20+ min in one tool-free LLM call) gets the claim *extended* instead of killed; only a dead PID is reclaimed.
- **Crashed worker** — a worker whose host-local PID has vanished is detected by `detect_crashed_workers` and reaped; the task increments `consecutive_failures` and may auto-block when the breaker trips.
- **Run-level retry** — when a task is retried (post-block, post-crash, post-reclaim), the worker can use the `expected_run_id` parameter on terminating tools to fail fast if its own run was already superseded.
- **Per-task max runtime** — `task.max_runtime_seconds` hard-caps wall-clock time per run, regardless of PID liveness. Catches genuinely-deadlocked workers that the live-PID extension would otherwise keep running.
- **Stranded-task detection** — a ready task whose assignee never produces a claim within `kanban.stranded_threshold_seconds` (default 30 min) shows up in `hermes kanban diagnostics` as a `stranded_in_ready` warning. Severity escalates to error at 2x the threshold and critical at 6x. Catches typo'd assignees, deleted profiles, and down external worker pools in one signal — identity-agnostic, no per-board allowlist to curate.
- **Dead attempt behind an unresolved owner receipt** — a worker killed by a restart can leave an `owner_admitted` receipt the dispatcher cannot resolve, and reclaims used to be refused on that ground alone, forever: the card stayed `running`, held its profile's `kanban.max_in_progress_per_profile` slot, and the alarms named the cards queued behind it rather than the cause. The refusal now lapses as soon as nothing can be executing the attempt — the owner's process is gone or recycled, no heartbeat inside `DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`, and every pid the run recorded (`worker_pid`, `spawned`, `worker_bound`) is dead — including runs that never wrote a `worker_bound` event. Anything unprovable (a live pid, a missing fingerprint, no recorded pid, another host's lock) still holds the claim. A card that survives even that shows up in `hermes kanban diagnostics` as `stale_running`, escalated by age past `kanban.diagnostics.stale_running_grace_seconds` (default 1h).
- **Running with open parents** — a `running` card whose direct parent is not `done`/`archived` (the parent reopened mid-run, or the link predates the running-child refusal) shows a `running_with_open_parents` warning: the dependency gate is not serialising the two runs and `kanban_complete` will be refused until the parent finishes. Read-only; suggests `hermes kanban unlink`.
- **Legacy review dependency deadlock** — a parent sticky-blocked with `review-required:` while one or more direct children remain dependency-gated in `todo` produces an immediate `review_dependency_deadlock` error. The diagnostic is read-only: it suggests completing the finished phase or unlinking the incorrect edge but never removes a user block automatically.

## Related

- [Kanban overview](./kanban) — the user-facing intro.
- [Kanban tutorial](./kanban-tutorial) — walkthrough with the dashboard open.
- [`KANBAN_GUIDANCE`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py) — the worker + orchestrator lifecycle injected into every kanban worker's system prompt.
