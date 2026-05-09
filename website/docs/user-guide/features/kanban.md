---
sidebar_position: 12
title: "Kanban (Multi-Agent Board)"
description: "Durable SQLite-backed task board for coordinating multiple Hermes profiles"
---

# Kanban — Multi-Agent Profile Collaboration

> **Want a walkthrough?** Read the [Kanban tutorial](./kanban-tutorial) — four user stories (solo dev, fleet farming, role pipeline with retry, circuit breaker) with dashboard screenshots of each. This page is the reference; the tutorial is the narrative.

Hermes Kanban is a durable task board, shared across all your Hermes profiles, that lets multiple named agents collaborate on work without fragile in-process subagent swarms. Every task is a row in `~/.hermes/kanban.db`; every handoff is a row anyone can read and write; every worker is a full OS process with its own identity.

### Two surfaces: the model talks through tools, you talk through the CLI

The board has two front doors, both backed by the same `~/.hermes/kanban.db`:

- **Agents drive the board through a dedicated `kanban_*` toolset** — `kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`. The dispatcher spawns each worker with these tools already in its schema; the model reads its task and hands work off by calling them directly, *not* by shelling out to `hermes kanban`. See [How workers interact with the board](#how-workers-interact-with-the-board) below.
- **You (and scripts, and cron) drive the board through `hermes kanban …`** on the CLI, `/kanban …` as a slash command, or the dashboard. These are for humans and automation — the places without a tool-calling model behind them.

Both surfaces route through the same `kanban_db` layer, so reads see a consistent view and writes can't drift. The rest of this page shows CLI examples because they're easy to copy-paste, but every CLI verb has a tool-call equivalent the model uses.

This is the shape that covers the workloads `delegate_task` can't:

- **Research triage** — parallel researchers + analyst + writer, human-in-the-loop.
- **Scheduled ops** — recurring daily briefs that build a journal over weeks.
- **Digital twins** — persistent named assistants (`inbox-triage`, `ops-review`) that accumulate memory over time.
- **Engineering pipelines** — decompose → implement in parallel worktrees → review → iterate → PR.
- **Fleet work** — one specialist managing N subjects (50 social accounts, 12 monitored services).

For the full design rationale, comparative analysis against Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise, and the eight canonical collaboration patterns, see `docs/hermes-kanban-v1-spec.pdf` in the repository.

## Kanban vs. `delegate_task`

They look similar; they are not the same primitive.

| | `delegate_task` | Kanban |
|---|---|---|
| Shape | RPC call (fork → join) | Durable message queue + state machine |
| Parent | Blocks until child returns | Fire-and-forget after `create` |
| Child identity | Anonymous subagent | Named profile with persistent memory |
| Resumability | None — failed = failed | Block → unblock → re-run; crash → reclaim |
| Human in the loop | Not supported | Comment / unblock at any point |
| Agents per task | One call = one subagent | N agents over task's life (retry, review, follow-up) |
| Audit trail | Lost on context compression | Durable rows in SQLite forever |
| Coordination | Hierarchical (caller → callee) | Peer — any profile reads/writes any task |

**One-sentence distinction:** `delegate_task` is a function call; Kanban is a work queue where every handoff is a row any profile (or human) can see and edit.

**Use `delegate_task` when** the parent agent needs a short reasoning answer before continuing, no humans involved, result goes back into the parent's context.

**Use Kanban when** work crosses agent boundaries, needs to survive restarts, might need human input, might be picked up by a different role, or needs to be discoverable after the fact.

They coexist: a kanban worker may call `delegate_task` internally during its run.

## Core concepts

- **Board** — a standalone queue of tasks with its own SQLite DB, workspaces
  directory, and dispatcher loop. A single install can have many boards
  (e.g. one per project, repo, or domain); see [Boards (multi-project)](#boards-multi-project)
  below. Single-project users stay on the `default` board and never see the
  word "board" outside this docs section.
- **Task** — a row with title, optional body, one assignee (a profile name), status (`triage | todo | ready | running | blocked | done | archived`), optional tenant namespace, optional idempotency key (dedup for retried automation).
- **Link** — `task_links` row recording a parent → child dependency. The dispatcher promotes `todo → ready` when all parents are `done`.
- **Comment** — the inter-agent protocol. Agents and humans append comments; when a worker is (re-)spawned it reads the full comment thread as part of its context.
- **Workspace** — the directory a worker operates in. Three kinds:
  - `scratch` (default) — fresh tmp dir under `~/.hermes/kanban/workspaces/<id>/` (or `~/.hermes/kanban/boards/<slug>/workspaces/<id>/` on non-default boards).
  - `dir:<path>` — an existing shared directory (Obsidian vault, mail ops dir, per-account folder). **Must be an absolute path.** Relative paths like `dir:../tenants/foo/` are rejected at dispatch because they'd resolve against whatever CWD the dispatcher happens to be in, which is ambiguous and a confused-deputy escape vector. The path is otherwise trusted — it's your box, your filesystem, the worker runs with your uid. This is the trusted-local-user threat model; kanban is single-host by design.
  - `worktree` — a git worktree under `.worktrees/<id>/` for coding tasks. Worker-side `git worktree add` creates it.
- **Dispatcher** — a long-lived loop that, every N seconds (default 60): reclaims stale claims, reclaims crashed workers (PID gone but TTL not yet expired), promotes ready tasks, atomically claims, spawns assigned profiles. Runs **inside the gateway** by default (`kanban.dispatch_in_gateway: true`). One dispatcher sweeps all boards per tick; workers are spawned with `HERMES_KANBAN_BOARD` pinned so they can't see other boards. After `kanban.failure_limit` consecutive spawn failures on the same task (default: 2) the dispatcher auto-blocks it with the last error as the reason — prevents thrashing on tasks whose profile doesn't exist, workspace can't mount, etc.
- **Tenant** — optional string namespace *within* a board. One specialist fleet can serve multiple businesses (`--tenant business-a`) with data isolation by workspace path and memory key prefix. Tenants are a soft filter; boards are the hard isolation boundary.

## Boards (multi-project)

Boards let you separate unrelated streams of work — one per project, repo,
or domain — into isolated queues. A new install has exactly one board
called `default` (DB at `~/.hermes/kanban.db` for back-compat). Users who
only want one stream of work never need to know about boards; the feature
is opt-in.

Per-board isolation is absolute:

- Separate SQLite DB per board (`~/.hermes/kanban/boards/<slug>/kanban.db`).
- Separate `workspaces/` and `logs/` directories.
- Workers spawned for a task see **only** their board's tasks — the
  dispatcher sets `HERMES_KANBAN_BOARD` in the child env and every
  `kanban_*` tool the worker has access to reads it.
- Linking tasks across boards is not allowed (keeps the schema simple; if
  you really need cross-project refs, use free-text mentions and look
  them up by id manually).

### Managing boards from the CLI

```bash
# See what's on disk. Fresh installs show only "default".
hermes kanban boards list

# Create a new board.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # optional: make it the active board

# Operate on a specific board without switching.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# Change which board is "current" for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban boards show             # who's active right now?

# Rename the display name (the slug is immutable — it's the directory name).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# Archive (default) — moves the board's dir to boards/_archived/<slug>-<ts>/.
# Recoverable by moving the dir back.
hermes kanban boards rm atm10-server

# Hard delete — `rm -rf` the board dir. No recovery.
hermes kanban boards rm atm10-server --delete
```

Board resolution order (highest precedence first):

1. Explicit `--board <slug>` on the CLI call.
2. `HERMES_KANBAN_BOARD` env var (set by the dispatcher when spawning a
   worker, so workers can't see other boards).
3. `~/.hermes/kanban/current` — the slug persisted by `hermes kanban
   boards switch`.
4. `default`.

Slugs are validated: lowercase alphanumerics + hyphens + underscores, 1-64
chars, must start with alphanumeric. Uppercase input is auto-downcased.
Anything else (slashes, spaces, dots, `..`) is rejected at the CLI layer
so path-traversal tricks can't name a board.

### Managing boards from the dashboard

`hermes dashboard` → Kanban tab shows a board switcher at the top as soon
as more than one board exists (or any board has tasks). Single-board users
see only a small `+ New board` button; the switcher is hidden until it
matters.

- **Board dropdown** — pick the active board. Your selection is saved to
  the browser's `localStorage` so it persists across reloads without
  shifting the CLI's `current` pointer out from under a terminal you left
  open.
- **+ New board** — opens a modal asking for slug, display name,
  description, and icon. Option to auto-switch to the new board.
- **Archive** — only shown on non-`default` boards. Confirms, then moves
  the board dir to `boards/_archived/`.

All dashboard API endpoints accept `?board=<slug>` for board scoping. The
events WebSocket is pinned to a board at connection time; switching in
the UI opens a fresh WS against the new board.


## Quick start

The commands below are **you** (the human) setting up the board and creating tasks. Once a task is assigned, the dispatcher spawns the assigned profile as a worker, and from there **the model drives the task through `kanban_*` tool calls, not CLI commands** — see [How workers interact with the board](#how-workers-interact-with-the-board).

```bash
# 1. Create the board (you)
hermes kanban init

# 2. Start the gateway (hosts the embedded dispatcher)
hermes gateway start

# 3. Create a task (you — or an orchestrator agent via kanban_create)
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. Watch activity live (you)
hermes kanban watch

# 5. See the board (you)
hermes kanban list
hermes kanban stats
```

When the dispatcher picks up `t_abcd` and spawns the `researcher` profile, the very first thing that worker's model does is call `kanban_show()` to read its task. It doesn't run `hermes kanban show t_abcd`.

### Gateway-embedded dispatcher (default)

The dispatcher runs inside the gateway process. Nothing to install, no
separate service to manage — if the gateway is up, ready tasks get picked
up on the next tick (60s by default).

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # default
  dispatch_interval_seconds: 60    # default
```

Override the config flag at runtime via `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`
for debugging. Standard gateway supervision applies: run `hermes gateway
start` directly, or wire the gateway up as a systemd user unit (see the
gateway docs). Without a running gateway, `ready` tasks stay where they are
until one comes up — `hermes kanban create` warns about this at creation
time.

Running `hermes kanban daemon` as a separate process is **deprecated**;
use the gateway. If you truly cannot run the gateway (headless host
policy forbids long-lived services, etc.) a `--force` escape hatch keeps
the old standalone daemon alive for one release cycle, but running both
a gateway-embedded dispatcher AND a standalone daemon against the same
`kanban.db` causes claim races and is not supported.

### Idempotent create (for automation / webhooks)

```bash
# First call creates the task. Any subsequent call with the same key
# returns the existing task id instead of duplicating.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### Bulk CLI verbs

All the lifecycle verbs accept multiple ids so you can clean up a batch
in one command:

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## How workers interact with the board

**Workers do not shell out to `hermes kanban`.** When the dispatcher spawns a worker it sets `HERMES_KANBAN_TASK=t_abcd` in the child's env, and that env var flips on a dedicated **kanban toolset** in the model's schema — seven tools that read and mutate the board directly via the Python `kanban_db` layer, same as the CLI does. A running worker calls these like any other tool; it never sees or needs the `hermes kanban` CLI.

| Tool | Purpose | Required params |
|---|---|---|
| `kanban_show` | Read the current task (title, body, prior attempts, parent handoffs, comments, full pre-formatted `worker_context`). Defaults to the env's task id. | — |
| `kanban_complete` | Finish with `summary` + `metadata` structured handoff. | at least one of `summary` / `result` |
| `kanban_block` | Escalate for human input with a `reason`. | `reason` |
| `kanban_heartbeat` | Signal liveness during long operations. Pure side-effect. | — |
| `kanban_comment` | Append a durable note to the task thread. | `task_id`, `body` |
| `kanban_create` | (Orchestrators) fan out into child tasks with an `assignee`, optional `parents`, `skills`, etc. | `title`, `assignee` |
| `kanban_link` | (Orchestrators) add a `parent_id → child_id` dependency edge after the fact. | `parent_id`, `child_id` |

A typical worker turn looks like:

```
# Model's tool calls, in order:
kanban_show()                                     # no args — uses HERMES_KANBAN_TASK
# (model reads the returned worker_context, does the work via terminal/file tools)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (more work)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

An **orchestrator** worker fans out instead:

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → returns {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → returns {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # promotes to ready when both complete
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

The three "(Orchestrators)" tools — `kanban_create`, `kanban_link`, and `kanban_comment` on foreign tasks — are available to every worker; the convention (enforced by the `kanban-orchestrator` skill) is that worker profiles don't fan out and orchestrator profiles don't execute.

### Why tools instead of shelling to `hermes kanban`

Three reasons:

1. **Backend portability.** Workers whose terminal tool points at a remote backend (Docker / Modal / Singularity / SSH) would run `hermes kanban complete` *inside* the container, where `hermes` isn't installed and `~/.hermes/kanban.db` isn't mounted. The kanban tools run in the agent's own Python process and always reach `~/.hermes/kanban.db` regardless of terminal backend.
2. **No shell-quoting fragility.** Passing `--metadata '{"files": [...]}'` through shlex + argparse is a latent footgun. Structured tool args skip it entirely.
3. **Better errors.** Tool results are structured JSON the model can reason about, not stderr strings it has to parse.

**Zero schema footprint on normal sessions.** A regular `hermes chat` session has zero `kanban_*` tools in its schema. The `check_fn` on each tool only returns True when `HERMES_KANBAN_TASK` is set, which only happens when the dispatcher spawned this process. No tool bloat for users who never touch kanban.

The `kanban-worker` and `kanban-orchestrator` skills teach the model which tool to call when and in what order.

### Recommended handoff evidence

`kanban_complete(summary=..., metadata={...})` is intentionally flexible:
the summary is the human-readable closeout, and `metadata` is the
machine-readable handoff that downstream agents, reviewers, or dashboards can
reuse without scraping prose.

For engineering and review tasks, prefer this optional metadata shape:

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

These keys are a convention, not a schema requirement. The useful property is
that every worker leaves enough evidence for the next reader to answer four
questions quickly:

1. What changed?
2. How was it verified?
3. What can unblock or retry this if it fails?
4. What risk is still deliberately left open?

Keep secrets, raw logs, tokens, OAuth material, and unrelated transcripts out of
`metadata`. Store pointers and summaries instead. If a task has no files or
tests, say so explicitly in `summary` and use `metadata` for the evidence that
does exist, such as source URLs, issue ids, or manual review steps.

### The worker skill

Any profile that should be able to work kanban tasks must load the `kanban-worker` skill. It teaches the worker the full lifecycle in **tool calls**, not CLI commands:

1. On spawn, call `kanban_show()` to read title + body + parent handoffs + prior attempts + full comment thread.
2. `cd $HERMES_KANBAN_WORKSPACE` (via the terminal tool) and do the work there.
3. Call `kanban_heartbeat(note="...")` every few minutes during long operations.
4. Complete with `kanban_complete(summary="...", metadata={...})`, or `kanban_block(reason="...")` if stuck.

`kanban-worker` is a bundled skill, synced into every profile during install and
update — there is no separate Skills Hub install step. Verify it is present in
whichever profile you use for kanban workers (`researcher`, `writer`, `ops`,
etc.):

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

If the bundled copy is missing, restore it for that profile:

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

The dispatcher also auto-passes `--skills kanban-worker` when spawning every worker, so the worker always has the pattern library available even if a profile's default skills config doesn't include it.

### Pinning extra skills to a specific task

Sometimes a single task needs specialist context the assignee profile doesn't carry by default — a translation job that needs the `translation` skill, a review task that needs `github-code-review`, a security audit that needs `security-pr-audit`. Rather than editing the assignee's profile every time, attach the skills directly to the task.

**From an orchestrator agent** (the usual case — one agent routing work to another), use the `kanban_create` tool's `skills` array:

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**From a human (CLI / slash command)**, repeat `--skill` for each one:

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**From the dashboard**, type the skills comma-separated into the **skills** field of the inline create form.

These skills are **additive** to the built-in `kanban-worker` — the dispatcher emits one `--skills <name>` flag for each (and for the built-in), so the worker spawns with all of them loaded. The skill names must match skills that are actually installed on the assignee's profile (run `hermes skills list` to see what's available); there's no runtime install.

### The orchestrator skill

A **well-behaved orchestrator does not do the work itself.** It decomposes the user's goal into tasks, links them, assigns each to a specialist, and steps back. The `kanban-orchestrator` skill encodes this as tool-call patterns: anti-temptation rules, a standard specialist roster (`researcher`, `writer`, `analyst`, `backend-eng`, `reviewer`, `ops`), and a decomposition playbook keyed on `kanban_create` / `kanban_link` / `kanban_comment`.

A canonical orchestrator turn (two parallel researchers handing off to a writer):

```
# Goal from user: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # promoted to 'ready' when both researchers complete
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# Optional: add cross-cutting deps discovered later without re-creating tasks
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator` is a bundled skill. It is synced into each profile during
install and update, so there is no separate Skills Hub install step. Verify it is
present in your orchestrator profile:

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

If the bundled copy is missing, restore it for that profile:

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

For best results, pair it with a profile whose toolsets are restricted to board operations (`kanban`, `gateway`, `memory`) so the orchestrator literally cannot execute implementation tasks even if it tries.

## Dashboard (GUI)

The `/kanban` CLI and slash command are enough to run the board headlessly, but a visual board is often the right interface for humans-in-the-loop: triage, cross-profile supervision, reading comment threads, and dragging cards between columns. Hermes ships this as a **bundled dashboard plugin** at `plugins/kanban/` — not a core feature, not a separate service — following the model laid out in [Extending the Dashboard](./extending-the-dashboard).

Open it with:

```bash
hermes kanban init      # one-time: create kanban.db if not already present
hermes dashboard        # "Kanban" tab appears in the nav, after "Skills"
```

### What the plugin gives you

- A **Kanban** tab showing one column per status: `triage`, `todo`, `ready`, `running`, `blocked`, `done` (plus `archived` when the toggle is on).
  - `triage` is the parking column for rough ideas a specifier is expected to flesh out. Tasks created with `hermes kanban create --triage` (or via the Triage column's inline create) land here and the dispatcher leaves them alone until a human or specifier promotes them to `todo` / `ready`. Run `hermes kanban specify <id>` to have the auxiliary LLM expand a triage task into a concrete spec (title + body with goal, approach, acceptance criteria) and promote it to `todo` in one shot; `--all` sweeps every triage task at once. Configure which model runs the specifier under `auxiliary.triage_specifier` in `config.yaml`.
- Cards show the task id, title, priority badge, tenant tag, assigned profile, comment/link counts, a **progress pill** (`N/M` children done when the task has dependents), and "created N ago". A per-card checkbox enables multi-select.
- **Per-profile lanes inside Running** — toolbar checkbox toggles sub-grouping of the Running column by assignee.
- **Live updates via WebSocket** — the plugin tails the append-only `task_events` table on a short poll interval; the board reflects changes the instant any profile (CLI, gateway, or another dashboard tab) acts. Reloads are debounced so a burst of events triggers a single refetch.
- **Drag-drop** cards between columns to change status. The drop sends `PATCH /api/plugins/kanban/tasks/:id` which routes through the same `kanban_db` code the CLI uses — the three surfaces can never drift. Moves into destructive statuses (`done`, `archived`, `blocked`) prompt for confirmation. Touch devices use a pointer-based fallback so the board is usable from a tablet.
- **Inline create** — click `+` on any column header to type a title, assignee, priority, and (optionally) a parent task from a dropdown over every existing task. Creating from the Triage column automatically parks the new task in triage.
- **Multi-select with bulk actions** — shift/ctrl-click a card or tick its checkbox to add it to the selection. A bulk action bar appears at the top with batch status transitions, archive, and reassign (by profile dropdown, or "(unassign)"). Destructive batches confirm first. Per-id partial failures are reported without aborting the rest.
- **Click a card** (without shift/ctrl) to open a side drawer (Escape or click-outside closes) with:
  - **Editable title** — click the heading to rename.
  - **Editable assignee / priority** — click the meta row to rewrite.
  - **Editable description** — markdown-rendered by default (headings, bold, italic, inline code, fenced code, `http(s)` / `mailto:` links, bullet lists), with an "edit" button that swaps in a textarea. Markdown rendering is a tiny, XSS-safe renderer — every substitution runs on HTML-escaped input, only `http(s)` / `mailto:` links pass through, and `target="_blank"` + `rel="noopener noreferrer"` are always set.
  - **Dependency editor** — chip list of parents and children, each with an `×` to unlink, plus dropdowns over every other task to add a new parent or child. Cycle attempts are rejected server-side with a clear message.
  - **Status action row** (→ triage / → ready / → running / block / unblock / complete / archive) with confirm prompts for destructive transitions. For cards in the **Triage** column the row also exposes a **✨ Specify** button that calls the auxiliary LLM (`auxiliary.triage_specifier` in `config.yaml`) to expand the one-liner into a concrete spec (title + body with goal, approach, acceptance criteria) and promote the task to `todo`. The same behaviour is reachable from the CLI (`hermes kanban specify <id>` / `--all`), from any gateway platform (`/kanban specify <id>`), and programmatically via `POST /api/plugins/kanban/tasks/:id/specify`.
  - Result section (also markdown-rendered), comment thread with Enter-to-submit, the last 20 events.
- **Toolbar filters** — free-text search, tenant dropdown (defaults to `dashboard.kanban.default_tenant` from `config.yaml`), assignee dropdown, "show archived" toggle, "lanes by profile" toggle, and a **Nudge dispatcher** button so you don't have to wait for the next 60 s tick.

Visually the target is the familiar Linear / Fusion layout: dark theme, column headers with counts, coloured status dots, pill chips for priority and tenant. The plugin reads only theme CSS vars (`--color-*`, `--radius`, `--font-mono`, ...), so it reskins automatically with whichever dashboard theme is active.

### Architecture

The GUI is strictly a **read-through-the-DB + write-through-kanban_db** layer with no domain logic of its own:

```
┌────────────────────────┐      WebSocket (tails task_events)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     writes call kanban_db.*        │
│  FastAPI router        │     directly — same code path      │
│  plugins/kanban/       │     the CLI /kanban verbs use      │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── append task_events ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```

### REST surface

All routes are mounted under `/api/plugins/kanban/` and protected by the dashboard's ephemeral session token:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | Full board grouped by status column, plus tenants + assignees for filter dropdowns |
| `GET` | `/tasks/:id` | Task + comments + events + links |
| `POST` | `/tasks` | Create (wraps `kanban_db.create_task`, accepts `triage: bool` and `parents: [id, …]`) |
| `PATCH` | `/tasks/:id` | Status / assignee / priority / title / body / result |
| `POST` | `/tasks/bulk` | Apply the same patch (status / archive / assignee / priority) to every id in `ids`. Per-id failures reported without aborting siblings |
| `POST` | `/tasks/:id/comments` | Append a comment |
| `POST` | `/tasks/:id/specify` | Run the triage specifier — auxiliary LLM fleshes out the task body and promotes it from `triage` to `todo`. Returns `{ok, task_id, reason, new_title}`; `ok=false` with a human-readable reason on "not in triage" / no aux client / LLM error is a 200, not a 4xx |
| `POST` | `/links` | Add a dependency (`parent_id` → `child_id`) |
| `DELETE` | `/links?parent_id=…&child_id=…` | Remove a dependency |
| `POST` | `/dispatch?max=…&dry_run=…` | Nudge the dispatcher — skip the 60 s wait |
| `GET` | `/config` | Read `dashboard.kanban` preferences from `config.yaml` — `default_tenant`, `lane_by_profile`, `include_archived_by_default`, `render_markdown` |
| `WS` | `/events?since=<event_id>` | Live stream of `task_events` rows |

Every handler is a thin wrapper — the plugin is ~700 lines of Python (router + WebSocket tail + bulk batcher + config reader) and adds no new business logic. A tiny `_conn()` helper auto-initializes `kanban.db` on every read and write, so a fresh install works whether the user opened the dashboard first, hit the REST API directly, or ran `hermes kanban init`.

### Dashboard config

Any of these keys under `dashboard.kanban` in `~/.hermes/config.yaml` changes the tab's defaults — the plugin reads them at load time via `GET /config`:

```yaml
dashboard:
  kanban:
    default_tenant: acme              # preselects the tenant filter
    lane_by_profile: true             # default for the "lanes by profile" toggle
    include_archived_by_default: false
    render_markdown: true             # set false for plain <pre> rendering
```

Each key is optional and falls back to the shown default.

### Security model

The dashboard's HTTP auth middleware [explicitly skips `/api/plugins/`](./extending-the-dashboard#backend-api-routes) — plugin routes are unauthenticated by design because the dashboard binds to localhost by default. That means the kanban REST surface is reachable from any process on the host.

The WebSocket takes one additional step: it requires the dashboard's ephemeral session token as a `?token=…` query parameter (browsers can't set `Authorization` on an upgrade request), matching the pattern used by the in-browser PTY bridge.

If you run `hermes dashboard --host 0.0.0.0`, every plugin route — kanban included — becomes reachable from the network. **Don't do that on a shared host.** The board contains task bodies, comments, and workspace paths; an attacker reaching these routes gets read access to your entire collaboration surface and can also create / reassign / archive tasks.

Tasks in `~/.hermes/kanban.db` are profile-agnostic on purpose (that's the coordination primitive). If you open the dashboard with `hermes -p <profile> dashboard`, the board still shows tasks created by any other profile on the host. Same user owns all profiles, but this is worth knowing if multiple personas coexist.

### Live updates

`task_events` is an append-only SQLite table with a monotonic `id`. The WebSocket endpoint holds each client's last-seen event id and pushes new rows as they land. When a burst of events arrives, the frontend reloads the (very cheap) board endpoint — simpler and more correct than trying to patch local state from every event kind. WAL mode means the read loop never blocks the dispatcher's `BEGIN IMMEDIATE` claim transactions.

### Extending it

The plugin uses the standard Hermes dashboard plugin contract — see [Extending the Dashboard](./extending-the-dashboard) for the full manifest reference, shell slots, page-scoped slots, and the Plugin SDK. Extra columns, custom card chrome, tenant-filtered layouts, or full `tab.override` replacements are all expressible without forking this plugin.

To disable without removing: add `dashboard.plugins.kanban.enabled: false` to `config.yaml` (or delete `plugins/kanban/dashboard/manifest.json`).

### Scope boundary

The GUI is deliberately thin. Everything the plugin does is reachable from the CLI; the plugin just makes it comfortable for humans. Auto-assignment, budgets, governance gates, and org-chart views remain user-space — a router profile, another plugin, or a reuse of `tools/approval.py` — exactly as listed in the out-of-scope section of the design spec.

## CLI command reference

This is the surface **you** (or scripts, cron, the dashboard) use to drive the board. Workers running inside the dispatcher use the `kanban_*` [tool surface](#how-workers-interact-with-the-board) for the same operations — the CLI here and the tools there both route through `kanban_db`, so the two surfaces agree by construction.

```
hermes kanban init                                     # create kanban.db + print daemon hint
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|dir:<path>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# Bulk verbs — accept multiple ids:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # follow a single task's event stream
hermes kanban watch [--assignee P] [--tenant T]        # live stream ALL events to the terminal
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # worker liveness signal for long ops
hermes kanban runs <id> [--json]                       # attempt history (one row per run)
hermes kanban assignees [--json]                       # profiles on disk + per-assignee task counts
hermes kanban dispatch [--dry-run] [--max N]           # one-shot pass
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # DEPRECATED — standalone dispatcher (use `hermes gateway start` instead)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # per-status + per-assignee counts
hermes kanban log <id> [--tail BYTES]                  # worker log from ~/.hermes/kanban/logs/
hermes kanban notify-subscribe <id>                    # gateway bridge hook (used by /kanban in the gateway)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # what a worker sees
hermes kanban specify [<id> | --all] [--tenant T]      # flesh out a triage-column idea
        [--author NAME] [--json]                       #   into a full spec and promote to todo
hermes kanban gc [--event-retention-days N]            # workspaces + old events + old logs
        [--log-retention-days N]
```

All commands are also available as a slash command in the interactive CLI and in the messaging gateway (see [`/kanban` slash command](#kanban-slash-command) below).

## `/kanban` slash command {#kanban-slash-command}

Every `hermes kanban <action>` verb is also reachable as `/kanban <action>` — from inside an interactive `hermes chat` session **and** from any gateway platform (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, email, SMS). Both surfaces call the exact same `hermes_cli.kanban.run_slash()` entry point that reuses the `hermes kanban` argparse tree, so the argument surface, flags, and output format are identical across CLI, `/kanban`, and `hermes kanban`. You don't have to leave the chat to drive the board.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # flesh out a triage one-liner into a real spec
/kanban specify --all --tenant engineering  # sweep every triage task in one tenant
```

Quote multi-word arguments the same way you would on a shell — `run_slash` parses the rest of the line with `shlex.split`, so `"..."` and `'...'` both work.

### Mid-run usage: `/kanban` bypasses the running-agent guard

The gateway normally queues slash commands and user messages while an agent is still thinking — that's what stops you from accidentally starting a second turn while the first is in flight. **`/kanban` is explicitly exempted from this guard.** The board lives in `~/.hermes/kanban.db`, not in the running agent's state, so reads (`list`, `show`, `context`, `tail`, `watch`, `stats`, `runs`) and writes (`comment`, `unblock`, `block`, `assign`, `archive`, `create`, `link`, …) all go through immediately, even mid-turn.

This is the whole point of the separation:

- A worker blocks waiting on a peer → you send `/kanban unblock t_abcd` from your phone and the dispatcher picks the peer up on its next tick. The blocked worker isn't interrupted — it just stops being blocked.
- You spot a card that needs human context → `/kanban comment t_xyz "use the 2026 schema, not 2025"` lands on the task thread and the *next* run of that task will read it in `kanban_show()`.
- You want to know what your fleet is doing without stopping the orchestrator → `/kanban list --mine` or `/kanban stats` inspects the board without touching your main conversation.

### Auto-subscribe on `/kanban create` (gateway only)

When you create a task from the gateway with `/kanban create "…"`, the originating chat (platform + chat id + thread id) is automatically subscribed to that task's terminal events (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`). You'll get one message back per terminal event — including the first line of the worker's result summary on `completed` — without having to poll or remember the task id.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

Subscriptions auto-remove themselves once the task reaches `done` or `archived`. If you script a create with `--json` (machine output) the auto-subscribe is skipped — the assumption is that scripted callers want to manage subscriptions explicitly via `/kanban notify-subscribe`.

### Output truncation in messaging

Gateway platforms have practical message-length caps. If `/kanban list`, `/kanban show`, or `/kanban tail` produce more than ~3800 characters of output, the response is truncated with a `… (truncated; use \`hermes kanban …\` in your terminal for full output)` footer. The CLI surface has no such cap.

### Autocomplete

In the interactive CLI, typing `/kanban ` and hitting Tab cycles through the built-in subcommand list (`list`, `ls`, `show`, `create`, `assign`, `link`, `unlink`, `claim`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`, `dispatch`, `context`, `init`, `gc`). The remaining verbs listed in the CLI reference above (`watch`, `stats`, `runs`, `log`, `assignees`, `heartbeat`, `notify-subscribe`, `notify-list`, `notify-unsubscribe`, `daemon`) also work — they're just not in the autocomplete hint list yet.

## Collaboration patterns

The board supports these eight patterns without any new primitives:

| Pattern | Shape | Example |
|---|---|---|
| **P1 Fan-out** | N siblings, same role | "research 5 angles in parallel" |
| **P2 Pipeline** | role chain: scout → editor → writer | daily brief assembly |
| **P3 Voting / quorum** | N siblings + 1 aggregator | 3 researchers → 1 reviewer picks |
| **P4 Long-running journal** | same profile + shared dir + cron | Obsidian vault |
| **P5 Human-in-the-loop** | worker blocks → user comments → unblock | ambiguous decisions |
| **P6 `@mention`** | inline routing from prose | `@reviewer look at this` |
| **P7 Thread-scoped workspace** | `/kanban here` in a thread | per-project gateway threads |
| **P8 Fleet farming** | one profile, N subjects | 50 social accounts |
| **P9 Triage specifier** | rough idea → `triage` → `hermes kanban specify` expands body → `todo` | "turn this one-liner into a spec'd task" |

For worked examples of each, see `docs/hermes-kanban-v1-spec.pdf`.

## Multi-tenant usage

When one specialist fleet serves multiple businesses, tag each task with a tenant:

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

Workers receive `$HERMES_TENANT` and namespace their memory writes by prefix. The board, the dispatcher, and the profile definitions are all shared; only the data is scoped.

## Gateway notifications

When you run `/kanban create …` from the gateway (Telegram, Discord, Slack, etc.), the originating chat is automatically subscribed to the new task. The gateway's background notifier polls `task_events` every few seconds and delivers one message per terminal event (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`) to that chat. Completed tasks also send the first line of the worker's `--result` so you see the outcome without having to `/kanban show`.

You can manage subscriptions explicitly from the CLI — useful when a script / cron job wants to notify a chat it didn't originate from:

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

A subscription removes itself automatically once the task reaches `done` or `archived`; no cleanup needed.

## Runs — one row per attempt

A task is a logical unit of work; a **run** is one attempt to execute it. When the dispatcher claims a ready task it creates a row in `task_runs` and points `tasks.current_run_id` at it. When that attempt ends — completed, blocked, crashed, timed out, spawn-failed, reclaimed — the run row closes with an `outcome` and the task's pointer clears. A task that's been attempted three times has three `task_runs` rows.

Why two tables instead of just mutating the task: you need **full attempt history** for real-world postmortems ("the second reviewer attempt got to approve, the third merged"), and you need a clean place to hang per-attempt metadata — which files changed, which tests ran, which findings a reviewer noted. Those are run facts, not task facts.

Runs are also where **structured handoff** lives. When a worker completes a task (via `kanban_complete(...)`) it can pass:

- `summary` (tool param) / `--summary` (CLI) — human handoff; goes on the run; downstream children see it in their `build_worker_context`.
- `metadata` (tool param) / `--metadata` (CLI) — free-form JSON dict on the run; children see it serialized alongside the summary.
- `result` (tool param) / `--result` (CLI) — short log line that goes on the task row (legacy field, kept for back-compat).

Downstream children read the most recent completed run's summary + metadata for each parent. Retrying workers read the prior attempts on their own task (outcome, summary, error) so they don't repeat a path that already failed.

```
# What a worker actually does — a tool call, from inside the agent loop:
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

The same handoff is reachable from the CLI when you (the human) need to close out a task a worker can't — e.g. a task that was abandoned, or one you marked done manually from the dashboard:

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# Review the attempt history on a retried task:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

Runs are exposed on the dashboard (Run History section in the drawer, one coloured row per attempt) and on the REST API (`GET /api/plugins/kanban/tasks/:id` returns a `runs[]` array). `PATCH /api/plugins/kanban/tasks/:id` with `{status: "done", summary, metadata}` forwards both to the kernel, so the dashboard's "mark done" button is CLI-equivalent. `task_events` rows carry the `run_id` they belong to so the UI can group them by attempt, and the `completed` event embeds the first-line summary in its payload (capped at 400 chars) so gateway notifiers can render structured handoffs without a second SQL round-trip.

**Bulk close caveat.** `hermes kanban complete a b c --summary X` is refused — structured handoff is per-run, so copy-pasting the same summary to N tasks is almost always wrong. Bulk close *without* `--summary` / `--metadata` still works for the common "I finished a pile of admin tasks" case.

**Reclaimed runs from status changes.** If you drag a running task off `running` in the dashboard (back to `ready`, or straight to `todo`), or archive a task that was still running, the in-flight run closes with `outcome='reclaimed'` rather than being orphaned. The `task_runs` row is always in a terminal state when `tasks.current_run_id` is `NULL`, and vice versa — that invariant holds across CLI, dashboard, dispatcher, and notifier.

**Synthetic runs for never-claimed completions.** Completing or blocking a task that was never claimed (e.g. a human closes a `ready` task from the dashboard with a summary, or a CLI user runs `hermes kanban complete <ready-task> --summary X`) would otherwise drop the handoff. Instead the kernel inserts a zero-duration run row (`started_at == ended_at`) carrying the summary / metadata / reason so attempt history stays complete. The `completed` / `blocked` event's `run_id` points at that row.

**Live drawer refresh.** When the dashboard's WebSocket event stream reports new events for the task the user is currently viewing, the drawer reloads itself (via a per-task event counter threaded into its `useEffect` dependency list). Closing and reopening is no longer required to see a run's new row or updated outcome.

### Forward compatibility

Two nullable columns on `tasks` are reserved for v2 workflow routing: `workflow_template_id` (which template this task belongs to) and `current_step_key` (which step in that template is active). The v1 kernel ignores them for routing but lets clients write them, so a v2 release can add the routing machinery without another schema migration.

## Event reference

Every transition appends a row to `task_events`. Each row carries an optional `run_id` so UIs can group events by attempt. Kinds group into three clusters so filtering is easy (`hermes kanban watch --kinds completed,gave_up,timed_out`):

**Lifecycle** (what changed about the task as a logical unit):

| Kind | Payload | When |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | Task inserted. `run_id` is `NULL`. |
| `promoted` | — | `todo → ready` because all parents hit `done`. `run_id` is `NULL`. |
| `claimed` | `{lock, expires, run_id}` | Dispatcher atomically claimed a `ready` task for spawn. |
| `completed` | `{result_len, summary?}` | Worker wrote `--result` / `--summary` and task hit `done`. `summary` is the first-line handoff (400-char cap); full version lives on the run row. If `complete_task` is called on a never-claimed task with handoff fields, a zero-duration run is synthesized so `run_id` still points at something. |
| `blocked` | `{reason}` | Worker or human flipped the task to `blocked`. Synthesizes a zero-duration run when called on a never-claimed task with `--reason`. |
| `unblocked` | — | `blocked → ready`, either manually or via `/unblock`. `run_id` is `NULL`. |
| `archived` | — | Hidden from the default board. If the task was still running, carries the `run_id` of the run that was reclaimed as a side effect. |

**Edits** (human-driven changes that aren't transitions):

| Kind | Payload | When |
|---|---|---|
| `assigned` | `{assignee}` | Assignee changed (including unassignment). |
| `edited` | `{fields}` | Title or body updated. |
| `reprioritized` | `{priority}` | Priority changed. |
| `status` | `{status}` | Dashboard drag-drop wrote a status directly (e.g. `todo → ready`). Carries the `run_id` of the run that was reclaimed when dragging off `running`; otherwise `run_id` is NULL. |

**Worker telemetry** (about the execution process, not the logical task):

| Kind | Payload | When |
|---|---|---|
| `spawned` | `{pid}` | Dispatcher successfully started a worker process. |
| `heartbeat` | `{note?}` | Worker called `hermes kanban heartbeat $TASK` to signal liveness during long operations. |
| `reclaimed` | `{stale_lock}` | Claim TTL expired without a completion; task goes back to `ready`. |
| `crashed` | `{pid, claimer}` | Worker PID no longer alive but TTL hadn't expired yet. |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds` exceeded; dispatcher SIGTERM'd (then SIGKILL'd after 5 s grace) and re-queued. |
| `spawn_failed` | `{error, failures}` | One spawn attempt failed (missing PATH, workspace unmountable, …). Counter increments; task returns to `ready` for retry. |
| `gave_up` | `{failures, error}` | Circuit breaker fired after N consecutive `spawn_failed`. Task auto-blocks with the last error. Default N = 5; override via `--failure-limit`. |

`hermes kanban tail <id>` shows these for a single task. `hermes kanban watch` streams them board-wide.

## Out of scope

Kanban is deliberately single-host. `~/.hermes/kanban.db` is a local SQLite file and the dispatcher spawns workers on the same machine. Running a shared board across two hosts is not supported — there's no coordination primitive for "worker X on host A, worker Y on host B," and the crash-detection path assumes PIDs are host-local. If you need multi-host, run an independent board per host and use `delegate_task` / a message queue to bridge them.

## Design spec

The complete design — architecture, concurrency correctness, comparison with other systems, implementation plan, risks, open questions — lives in `docs/hermes-kanban-v1-spec.pdf`. Read that before filing any behavior-change PR.
