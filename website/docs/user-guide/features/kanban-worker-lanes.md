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

The dispatcher matches `task.assignee` against either a Hermes profile name (the default lane shape) or a registered non-spawnable identifier (the plugin lane shape — see [Adding an external CLI worker lane](#adding-an-external-cli-worker-lane) below). Tasks whose assignee doesn't resolve are left on `ready` with a `skipped_nonspawnable` event so a board operator can fix them; they are not silently dropped or executed by an arbitrary fallback.

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

For non-Hermes lanes (registered via a plugin), the plugin supplies its own `spawn_fn` callable that gets `task`, `workspace`, and `board` and returns an optional pid for crash detection.

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

## Adding an external CLI worker lane

Wiring a non-Hermes CLI tool (Codex CLI, Claude Code CLI, OpenCode CLI, a local coding-model runner, etc.) as a kanban worker lane is *not yet a paved path*. The dispatcher's spawn function is pluggable (`spawn_fn` is a parameter on `dispatch_once`), and a plugin could register its own `spawn_fn` for a non-Hermes assignee, but the surrounding integration work — wrapping the CLI's exit code into `kanban_complete` / `kanban_block` calls, mapping the CLI's workspace/sandbox conventions onto the dispatcher's `HERMES_KANBAN_WORKSPACE` env, handling auth and per-CLI policy — is still per-integration design work.

If you're considering adding a CLI lane, open an issue describing the specific CLI and the workflow you're trying to enable. The contract above is the constraints any such lane must satisfy; the implementation shape (one plugin per CLI vs a generic CLI-runner plugin parameterised by config) is open.

The historical issue for this is [#19931](https://github.com/NousResearch/hermes-agent/issues/19931) and the closed-not-merged Codex-specific PR [#19924](https://github.com/NousResearch/hermes-agent/pull/19924) — those describe the original architecture proposal but didn't land a runner.

## Failure modes the dispatcher handles

So lane authors don't have to reimplement these:

- **Stale claim TTL** — a worker that claims and then never heartbeats / completes / blocks gets reclaimed after `DEFAULT_CLAIM_TTL_SECONDS` (15 min default) — but only if the worker process has actually died. A live worker (slow model spending 20+ min in one tool-free LLM call) gets the claim *extended* instead of killed; only a dead PID is reclaimed.
- **Crashed worker** — a worker whose host-local PID has vanished is detected by `detect_crashed_workers` and reaped; the task increments `consecutive_failures` and may auto-block when the breaker trips.
- **Run-level retry** — when a task is retried (post-block, post-crash, post-reclaim), the worker can use the `expected_run_id` parameter on terminating tools to fail fast if its own run was already superseded.
- **Per-task max runtime** — `task.max_runtime_seconds` hard-caps wall-clock time per run, regardless of PID liveness. Catches genuinely-deadlocked workers that the live-PID extension would otherwise keep running.
- **Stranded-task detection** — a ready task whose assignee never produces a claim within `kanban.stranded_threshold_seconds` (default 30 min) shows up in `hermes kanban diagnostics` as a `stranded_in_ready` warning. Severity escalates to error at 2x the threshold and critical at 6x. Catches typo'd assignees, deleted profiles, and down external worker pools in one signal — identity-agnostic, no per-board allowlist to curate.
- **Legacy review dependency deadlock** — a parent sticky-blocked with `review-required:` while one or more direct children remain dependency-gated in `todo` produces an immediate `review_dependency_deadlock` error. The diagnostic is read-only: it suggests completing the finished phase or unlinking the incorrect edge but never removes a user block automatically.

## Guided model routing (opt-in, per task)

Guided model routing lets an operator assign a task to a curated, approved role
(`--routing-role`) instead of a fixed profile/model override, and have the dispatcher
resolve the actual provider/model/reasoning at **claim/start time** — not at card
creation — against a versioned, approved policy. This is off by default: a task with
no `--routing-role` behaves exactly as before, and existing `--model`/`--provider`
overrides are untouched.

**Approved membership is not qualification, and qualification is not availability.**
A route can be a member of the roster (an operator has approved sending work to that
provider/model combination at all) without being qualified for a given role, and a
qualified route can still be temporarily unavailable (auth, quota, outage). The
selector checks all three independently before choosing a route; none of them implies
the others.

```bash
# Opt a task into a routing role. Requirements must be genuine, not fabricated —
# see `hermes kanban create --help` for the exact JSON shape.
hermes kanban create --title "..." --assignee <profile> \
  --routing-role reviewarchitecture \
  --routing-requirements '{"task_class": "cross-component", "required_capabilities": []}'

# Manage the policy a board resolves routing roles against:
hermes kanban routing publish <policy.json> --approval-ref "<who/what approved this>"
hermes kanban routing activate <policy_id> <revision>
hermes kanban routing show [policy_id]            # currently active revision, if any
hermes kanban routing revisions [policy_id]        # every published revision (immutable, never overwritten)
hermes kanban routing receipt <receipt_id>         # a persisted routing decision
hermes kanban routing receipt-for-task <task_id>   # the decision receipted for a task's current attempt
hermes kanban routing revoke <policy_id> [--route-id ID] --reason "..." --approval-ref "..."
hermes kanban routing readmit <policy_id> [--route-id ID] --reason "..." --approval-ref "..."
```

Key properties, so operators know what this does and doesn't do:

- **Publishing is not activating.** `publish` records an immutable, approval-referenced
  policy revision; only `activate` makes one revision the one new claims resolve against.
  A published-but-inactive policy has no effect on any running or queued task.
- **Role, profile, and mandate are preserved.** Routing chooses the model; it does not
  change which profile/toolset a task uses, its review requirements, or its lifecycle
  transitions (`ready`/`running`/`review`/`done`/...).
- **Resolution happens at claim/start, not at card creation**, so a task queued before a
  policy revision or an emergency `revoke` sees the current approval state, never a
  stale one baked in when the card was made.
- **Per-attempt pinning.** Once a task claims and resolves a route, that receipted route
  is what the worker is validated against before it sends any task content; a routine
  policy edit affects new attempts, never an attempt already in flight.
- **`revoke`/`readmit` are distinct from `publish`/`activate`.** `revoke` is the explicit
  emergency path for already-receipted, in-flight attempts (a whole policy or a single
  route); `readmit` is the only way to clear a revocation, and it is never implied by a
  later `publish`/`activate` or automatic/time-based.
- **Every decision is a receipt**, inspectable via `hermes kanban routing receipt` /
  `receipt-for-task` — the selected route, rejection reasons for alternates considered,
  and (once a run starts) the actually observed provider/model identity for comparison.
  Receipts never contain prompts, credentials, or task text.
- **Failure is closed, not silently unmanaged.** A missing/invalid receipt, a stale
  worker build, or a routing decision that can't be resolved fails the attempt through
  the normal Kanban breaker/re-queue path — it does not fall back to an unmanaged
  default route.

This project has not activated any policy for live dispatch. The delegation adapter
(`delegate_task`'s `routing_role`/`routing_requirements`/`routing_policy_id`, see
[Delegation → Guided model routing](./delegation.md#guided-model-routing-opt-in-per-task))
and the MoA reference/aggregator adapter (see
[Mixture of Agents](./mixture-of-agents.md#guided-model-routing-for-moa-slots-opt-in-per-preset))
share this same policy store and selector. The public
[Subagent lifecycle API](../../developer-guide/subagent-lifecycle-api.md#guided-model-routing--not-yet-wired-into-this-public-api)
does not expose an equivalent field yet.

## Related

- [Kanban overview](./kanban) — the user-facing intro.
- [Kanban tutorial](./kanban-tutorial) — walkthrough with the dashboard open.
- [`KANBAN_GUIDANCE`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py) — the worker + orchestrator lifecycle injected into every kanban worker's system prompt.
