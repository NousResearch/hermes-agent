---
sidebar_position: 12
title: "Kanban (Multi-Agent Board)"
description: "Durable SQLite-backed task board for coordinating multiple Hermes profiles"
---

# Kanban — Multi-Agent Profile Collaboration

Hermes Kanban is a durable task board, shared across all your Hermes profiles, that lets multiple named agents collaborate on work without fragile in-process subagent swarms. Every task is a row in `~/.hermes/kanban.db`; every handoff is a row anyone can read and write; every worker is a full OS process with its own identity.

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

- **Task** — a row with title, optional body, one assignee (a profile name), status (`todo | ready | running | blocked | done | archived`), optional tenant namespace.
- **Link** — `task_links` row recording a parent → child dependency. The dispatcher promotes `todo → ready` when all parents are `done`.
- **Comment** — the inter-agent protocol. Agents and humans append comments; when a worker is (re-)spawned it reads the full comment thread as part of its context.
- **Workspace** — the directory a worker operates in. Three kinds:
  - `scratch` (default) — fresh tmp dir under `~/.hermes/kanban/workspaces/<id>/`.
  - `dir:<path>` — an existing shared directory (Obsidian vault, mail ops dir, per-account folder).
  - `worktree` — a git worktree under `.worktrees/<id>/` for coding tasks.
- **Dispatcher** — `hermes kanban dispatch` runs a one-shot pass: reclaim stale claims, promote ready tasks, atomically claim, spawn assigned profiles. Runs via cron every 60 seconds.
- **Tenant** — optional string namespace. One specialist fleet can serve multiple businesses (`--tenant business-a`) with data isolation by workspace path and memory key prefix.

## Quick start

```bash
# 1. Create the board
hermes kanban init

# 2. Create a task
hermes kanban create "research AI funding landscape" --assignee researcher

# 3. List what's on the board
hermes kanban list

# 4. Run a dispatcher pass (dry-run to preview, real to spawn workers)
hermes kanban dispatch --dry-run
hermes kanban dispatch
```

To have the board run continuously, schedule the dispatcher:

```bash
hermes cron add --schedule "*/1 * * * *" \
    --name kanban-dispatch \
    hermes kanban dispatch
```

## The worker skill

Any profile that should be able to work kanban tasks must load the `kanban-worker` skill. It teaches the worker the full lifecycle:

1. On spawn, read `$HERMES_KANBAN_TASK` env var.
2. Run `hermes kanban context $HERMES_KANBAN_TASK` to read title + body + parent results + full comment thread.
3. `cd $HERMES_KANBAN_WORKSPACE` and do the work there.
4. Complete with `hermes kanban complete <id> --result "<summary>"`, or block with `hermes kanban block <id> "<reason>"` if stuck.

Load it with:

```bash
hermes skills install devops/kanban-worker
```

## The orchestrator skill

A **well-behaved orchestrator does not do the work itself.** It decomposes the user's goal into tasks, links them, assigns each to a specialist, and steps back. The `kanban-orchestrator` skill encodes this: anti-temptation rules, a standard specialist roster (`researcher`, `writer`, `analyst`, `backend-eng`, `reviewer`, `ops`), and a decomposition playbook.

Load it into your orchestrator profile:

```bash
hermes skills install devops/kanban-orchestrator
```

For best results, pair it with a profile whose toolsets are restricted to board operations (`kanban`, `gateway`, `memory`) so the orchestrator literally cannot execute implementation tasks even if it tries.

## CLI command reference

```
hermes kanban init                                     # create kanban.db
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|dir:<path>]
                                [--priority N] [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]
hermes kanban complete <id> [--result "..."]
hermes kanban block <id> "<reason>"
hermes kanban unblock <id>
hermes kanban archive <id>
hermes kanban tail <id>                                # follow event stream
hermes kanban dispatch [--dry-run] [--max N] [--json]
hermes kanban context <id>                             # what a worker sees
hermes kanban gc                                       # remove scratch dirs of archived tasks
```

All commands are also available as a slash command in the gateway (`/kanban list`, `/kanban comment t_abc "need docs"`, etc.). The slash command bypasses the running-agent guard, so you can `/kanban unblock` a stuck worker while the main agent is still chatting.

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

## Design spec

The complete design — architecture, concurrency correctness, comparison with other systems, implementation plan, risks, open questions — lives in `docs/hermes-kanban-v1-spec.pdf`. Read that before filing any behavior-change PR.
