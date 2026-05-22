# Kanban worker lanes

A **worker lane** is a class of process that the kanban dispatcher can route tasks to. Each lane has an identity (the assignee string), a spawn mechanism, and a contract for what it must do with the task once spawned.

This page is the contract. It exists for two audiences:

- **Operators** picking which lanes to wire into a board (which profiles to create, which assignees to use).
- **Plugin / integration authors** wanting to add a new lane shape (a CLI worker that wraps Codex / Claude Code / OpenCode, a containerised review worker, a non-Hermes service that pulls tasks via the API).

If you're writing the worker code itself — the agent that runs *inside* a lane — the [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill is the deeper procedural detail.

## The hierarchy

```text
Hermes Kanban  =  canonical task lifecycle + audit trail
Worker lane    =  implementation executor for one assigned card
Reviewer       =  human or human-proxy that gates "done"
GitHub PR      =  upstreamable artifact (optional, for code lanes)
```

Hermes Kanban owns lifecycle truth — `ready` → `running` → `blocked` / `done` / `archived`. Worker lanes execute work but never own that truth; everything they do flows back through the kanban kernel via the `kanban_*` tools (or, for non-Hermes external workers, via the API). Reviewers gate the transition from "code change written" to "task done."

## What a lane provides

To be a kanban worker lane, an integration must provide three things:

### 1. An assignee string

The dispatcher resolves `task.assignee` in this order:

1. registered external worker lane
2. Hermes profile name
3. `skipped_nonspawnable`

This keeps existing profile workers compatible while allowing names such as `codex-fast`, `codex-deep`, and `codex-review` to be registered as external lanes. Unknown terminal/control-plane names still stay in `ready` or `review` and appear in `skipped_nonspawnable` rather than being spawned through a broken fallback.

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

For non-Hermes lanes, the worker lane registry supplies a trusted `spawn_fn` callable that gets `task`, `workspace`, and `board` and returns an optional pid for crash detection. Lanes can be registered from config or by plugins:

```python
def register(ctx):
    ctx.register_worker_lane(
        name="my-cli-worker",
        kind="plugin",
        description="Runs my trusted CLI worker",
        spawn_fn=spawn_my_worker,
        success_policy="block_for_review",
        max_concurrency=1,
    )
```

Plugin lane registration failures are logged and do not stop Hermes startup.

### 3. A lifecycle terminator

Every claim must end in exactly one of:

- `kanban_complete(summary=..., metadata=...)` — task succeeds, status flips to `done`.
- `kanban_block(reason=...)` — task waits for human input, status flips to `blocked`. The dispatcher respawns when `kanban_unblock` runs.
- The worker process exits without a tool call. The kernel reaps it and emits `crashed` (PID died) or `gave_up` (consecutive-failure breaker tripped) or `timed_out` (max_runtime exceeded). This is the failure path; healthy workers don't end here.

The kanban kernel enforces that exactly one of these terminates each run. A worker that calls neither and exits normally is treated as crashed.

## Outputs and the review-required convention

For most code-changing tasks, the work isn't truly *done* the moment the worker finishes — it needs a human reviewer. The kanban kernel doesn't enforce this distinction (a "code-changing task" is fuzzy and forcing block-instead-of-complete on every code worker would break flows where no review is wanted). It's a convention layered on top:

- **Block instead of complete**, with `reason` prefixed `review-required: ` so the dashboard / `hermes kanban show` surfaces the row as awaiting review.
- **Drop structured metadata into a `kanban_comment` first** since `kanban_block` only carries the human-readable `reason`. Comments are the durable annotation channel — every audit-relevant field (changed_files, tests_run, diff_path or PR url, decisions) belongs there.
- **Reviewer either approves and unblocks**, which respawns the worker with the comment thread for follow-ups; or asks for changes via another comment, which the next worker run sees as part of `kanban_show`'s context.

The [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill has worked examples for both `kanban_complete` (truly terminal tasks — typo fixes, docs changes, research writeups) and the `review-required` block pattern.

## Logs and audit trail

The dispatcher writes per-task worker stdout/stderr to `<board-root>/logs/<task_id>.log`. Logs are auditable from kanban metadata:

- `task_runs` rows carry the `log_path`, exit code (where available), summary, and metadata.
- `task_events` rows carry every state transition (`promoted`, `claimed`, `heartbeat`, `completed`, `blocked`, `gave_up`, `crashed`, `timed_out`, `reclaimed`, `claim_extended`).
- `kanban_show` returns both, so a reviewer (or a follow-up worker) reading the task gets the full history without needing dashboard access.

The dashboard renders run history with summaries, metadata blocks, and exit-status badges. CLI users can run `hermes kanban tail <task_id>` to follow live, or `hermes kanban runs <task_id>` for the historical attempt list.

## Existing lane shapes

### Hermes profile lane (default)

The shape every kanban worker takes today: the assignee is a profile name, the dispatcher spawns `hermes -p <profile>`, the worker auto-loads the [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) skill plus the `KANBAN_GUIDANCE` system-prompt block, and uses the `kanban_*` tools to terminate the run. No setup beyond defining the profile.

When you create profiles for your fleet, choose names that match the *role* you want the orchestrator to route to. The LLM decomposer builds its assignee roster from both `hermes profile list` and the worker lane registry; worker lanes use their lane description. There is no fixed roster the system assumes (see the [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) skill for the orchestrator side of the contract).

### Orchestrator profile lane

A specialisation of the profile lane: an orchestrator is a Hermes profile whose toolset includes `kanban` but excludes `terminal` / `file` / `code` / `web` for implementation. Its job is decomposing a high-level goal into child tasks via `kanban_create` + `kanban_link` and stepping back. The orchestrator skill encodes the anti-temptation rules.

## Codex CLI adapter

Codex CLI is the first built-in external adapter. Configure one or more lanes in `config.yaml`:

```yaml
kanban:
  worker_lanes:
    codex-fast:
      type: codex_cli
      model: gpt-5.4-mini
      sandbox: workspace-write
      approval: never
      max_concurrency: 2
      success_policy: block_for_review

    codex-deep:
      type: codex_cli
      model: gpt-5.5
      sandbox: workspace-write
      approval: never
      max_concurrency: 1
      success_policy: block_for_review

    codex-review:
      type: codex_cli
      model: gpt-5.5
      sandbox: read-only
      approval: never
      max_concurrency: 1
      success_policy: block_for_review
```

The adapter runs a Hermes-owned wrapper process, and that wrapper starts Codex with fixed argv:

```text
codex --cd <workspace> --sandbox <sandbox> --ask-for-approval <approval> [--model <model>] exec -
```

The command is not taken from model output and is not an arbitrary shell string. The wrapper passes a small allowlisted environment to Codex rather than forwarding every secret variable.

Each worker instance records the worker lane, kind, task id, run id, worker pid, claim lock, workspace, and model in events and metadata. Codex output is written to the normal worker log (`hermes kanban log <task_id>`).

Operators can inspect the lane roster without interrupting workers:

```bash
hermes kanban worker-lanes --json
```

The dashboard also reads `GET /api/plugins/kanban/worker-lanes` and shows each registered external lane's kind, model, success policy, active/max concurrency, per-status counts, and active task/run/pid instances. This is a bounded status view; it does not read the full Codex session and does not claim, heartbeat, reclaim, or signal running workers.

The wrapper also heartbeats the task and parses these progress formats into `task_events` as `worker_progress`:

```text
o (1) 分析入口
x (2) 修改 dispatcher
```

```text
- [ ] 分析入口
- [x] 修改 dispatcher
```

On success, the default `block_for_review` policy blocks the task with structured evidence instead of marking it `done`:

```text
review-required: Codex completed; Hermes review required
```

The metadata includes bounded output tail, git status, changed files, diff summary, verification commands, and review reason. This is distinct from the `review` column's profile-review dispatch path in current Kanban; Codex lane success hands evidence to Hermes/main-agent review without replaying the full Codex session.

## Skill lane intent

Hermes skills can choose an existing lane directly:

```text
assignee=codex-deep
```

The decomposer and skills may only choose lane names already registered in the roster. Unknown assignees are rewritten to the configured `default_assignee`; model output cannot invent an executable lane by naming it.

Skills may also propose a lane request:

```yaml
worker_lane_request:
  name: codex-long-context
  type: codex_cli
  model: gpt-5.5
  sandbox: workspace-write
  approval: never
  max_concurrency: 1
  success_policy: block_for_review
  reason: "large refactor requiring stronger reasoning"
```

Model output is not trusted execution config. Requests must pass a deterministic validator: type allowlist, model allowlist, sandbox allowlist, approval allowlist, max concurrency cap, fixed command shape, and no arbitrary shell command fields.

Operators can validate a request without enabling it:

```bash
hermes kanban worker-lane-request request.yaml --json
```

Dashboard/plugin clients can use the same validator through:

```text
POST /api/plugins/kanban/worker-lane-requests
```

The dashboard worker-lane panel exposes the same path as a controlled request
form. It lets an operator validate, enable, persist, or replace a Codex lane
request using the allowlisted fields; it does not accept arbitrary shell
commands.

By default the endpoint only validates. Pass `enable=true` to register the
lane for the current process, or `persist=true` to write the sanitized adapter
fields under `kanban.worker_lanes`.

After approval, enable it for the current Hermes process, or persist the sanitized config to `config.yaml`:

```bash
hermes kanban worker-lane-request request.yaml --enable
hermes kanban worker-lane-request request.yaml --persist
```

`--persist` writes only the sanitized adapter fields under `kanban.worker_lanes`; it does not store arbitrary command strings or the model's free-form reason.
For a standalone shell invocation, prefer `--persist` when a later dispatcher process must see the lane; `--enable` is mainly useful for in-process slash/gateway calls.

## Progress queries

Progress queries should read Kanban state, events, logs, and run metadata:

- `hermes kanban progress <task_id> --json`
- `hermes kanban progress <goal_or_root_task_id> --children --json`
- `hermes kanban reviews --json`
- `GET /api/plugins/kanban/tasks/<task_id>/progress`
- `GET /api/plugins/kanban/tasks/<task_id>/progress?children=true`
- `GET /api/plugins/kanban/reviews`
- `hermes kanban show <task_id>`
- `hermes kanban tail <task_id>`
- `hermes kanban log <task_id>`
- `hermes kanban runs <task_id> --json`

These reads do not interrupt a running external worker.

`hermes kanban reviews` lists tasks whose latest run metadata says
`review.required: true`, optionally filtered with `--assignee`, `--tenant`, or
`--lane`. This is the review queue for Codex/external-worker handoffs: it reads
the bounded evidence already written to `task_runs.metadata`, the latest
progress event, and an optional worker-log tail without replaying the complete
Codex session.

Reviewers can close the handoff through the same bounded-evidence path:

```bash
hermes kanban review <task_id> approve --summary "bounded evidence accepted"
hermes kanban review <task_id> request-changes --comment "add a regression test"
```

The dashboard/API equivalent is `POST /api/plugins/kanban/tasks/<task_id>/review`
with `decision=approve` or `decision=request_changes`.

`approve` records the review decision and marks the task done. `request-changes`
records the reviewer comment, emits a review event, and unblocks the task so the
dispatcher can hand the follow-up back to the assigned lane.

Configured orchestrator/main-agent profiles can use the equivalent tools:
`kanban_reviews` for the queue, `kanban_progress` for one task's bounded
snapshot, and `kanban_review` to approve or request changes. These tools are
orchestrator-only; dispatcher-spawned Codex workers do not see them.

Pass `include_children=true` to `kanban_progress` when the task is a goal/root
task and the controller needs a compact status roll-up without interrupting
running workers. The snapshot includes `child_summary` counts and a bounded
`children` list with each related worker task's relationship, status, lane,
latest run state, latest progress checklist, latest heartbeat event,
review-required flag, and verification evidence. For ordinary graphs this
summarizes direct child tasks. For decomposed goals, Hermes also summarizes the
worker tasks recorded in the root task's `decomposed.child_ids` event, because
the current decomposer links those worker tasks as dependencies that wake the
root when complete.

## Goal bridge

The intended `/goal` bridge is:

```text
/goal create "complex objective"
-> create_kanban_task_from_goal(...)
-> orchestrator creates child tasks
-> child tasks use assignee=<lane_name>
-> external lanes execute
-> Hermes reviews Kanban evidence and responds to the user
```

The current `/goal` session-level semantics remain intact. The opt-in bridge is available through Kanban today:

```bash
hermes kanban goal "complex objective" --assignee orchestrator --session <session-id>
hermes kanban goal "complex objective" --assignee orchestrator --decompose
```

`--decompose` runs the existing Kanban decomposer immediately. Its child tasks can use worker lane assignees from the registry, such as `codex-deep`, and the dispatcher later starts those external workers.

## Failure modes the dispatcher handles

So lane authors don't have to reimplement these:

- **Stale claim TTL** — a worker that claims and then never heartbeats / completes / blocks gets reclaimed after `DEFAULT_CLAIM_TTL_SECONDS` (15 min default) — but only if the worker process has actually died. A live worker (slow model spending 20+ min in one tool-free LLM call) gets the claim *extended* instead of killed; only a dead PID is reclaimed.
- **Crashed worker** — a worker whose host-local PID has vanished is detected by `detect_crashed_workers` and reaped; the task increments `consecutive_failures` and may auto-block when the breaker trips.
- **Run-level retry** — when a task is retried (post-block, post-crash, post-reclaim), the worker can use the `expected_run_id` parameter on terminating tools to fail fast if its own run was already superseded.
- **Per-task max runtime** — `task.max_runtime_seconds` hard-caps wall-clock time per run, regardless of PID liveness. Catches genuinely-deadlocked workers that the live-PID extension would otherwise keep running.
- **Stranded-task detection** — a ready task whose assignee never produces a claim within `kanban.stranded_threshold_seconds` (default 30 min) shows up in `hermes kanban diagnostics` as a `stranded_in_ready` warning. Severity escalates to error at 2x the threshold and critical at 6x. Catches typo'd assignees, deleted profiles, and down external worker pools in one signal — identity-agnostic, no per-board allowlist to curate.

## Current limits

- No full Codex event stream integration yet; progress is parsed from wrapper stdout/stderr.
- No approval bridge; configure Codex lanes with controlled approval policy.
- No automatic deep review for large diffs.
- External lane command shapes are adapter-defined, not model-defined.
- Review reads Codex artifacts and bounded metadata, not the full Codex session.

## Related

- [Kanban overview](./kanban) — the user-facing intro.
- [Kanban tutorial](./kanban-tutorial) — walkthrough with the dashboard open.
- [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) — the skill the worker process loads.
- [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) — the orchestrator side.
