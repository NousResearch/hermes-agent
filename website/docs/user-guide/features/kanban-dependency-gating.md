---
sidebar_position: 13
title: "Kanban Dependency Gating"
description: "How Kanban parent links control todo-to-ready promotion"
---

# Kanban Dependency Gating

Kanban dependency gating is the rule that keeps downstream work out of the
dispatcher until its inputs exist. It is implemented with a directed acyclic
graph (DAG): each edge points from a parent task to a child task, and the child
can run only after every parent is complete for dependency purposes.

Use this page when you are designing a multi-agent plan, debugging a task that
is stuck in `todo`, or deciding whether two cards should be linked.

## The Task Graph

Task links are stored as `parent_id -> child_id` rows. Read the arrow as
"the child needs the parent's handoff before it can start."

Good examples:

- `research market -> synthesize findings`
- `write implementation -> review implementation`
- `design schema -> implement API -> write integration tests`

Do not link cards just because the user mentioned them in sequence. Link only
when the child cannot produce useful work without the parent's result.

## Statuses And Gating

Kanban tasks move through seven user-visible statuses:

| Status | Meaning for dependency gating |
|---|---|
| `triage` | Raw idea. It is not eligible for dispatch. Specifying it moves it toward `todo`. |
| `todo` | Waiting for dependency satisfaction, assignment, or normal promotion. |
| `ready` | Eligible for the dispatcher to claim if it has a spawnable assignee. |
| `running` | Claimed by a worker. Downstream children still wait. |
| `blocked` | Waiting for human input or a retry decision. Downstream children still wait. |
| `done` | Normal success state. Children can be promoted when all parents are `done`. |
| `archived` | Removed from the active board. `recompute_ready` treats archived parents as satisfied so stale links do not permanently strand children. |

The important invariant is simple: a task with unfinished parents must not run.
If a child has at least one parent that is not complete for dependency purposes,
the child belongs in `todo`, not `ready`.

## Create Children With Parents

When the graph is known, create parent cards first, capture their returned task
ids, then pass those ids in the child card's `parents` list:

```python
research = kanban_create(
    title="research provider limits",
    assignee="researcher",
)["task_id"]

implementation = kanban_create(
    title="implement provider limit handling",
    assignee="engineer",
    parents=[research],
)["task_id"]
```

A child created with unfinished parents starts in `todo`. When every parent
finishes, the ready recomputation pass promotes the child to `ready`.

Avoid creating a dependent child as an independent card and linking it later.
There is a race window where the dispatcher can claim the child before the
parent link exists.

## Recompute Ready

`recompute_ready` is the maintenance pass that promotes waiting children:

1. It scans tasks in `todo`.
2. For each task, it reads all parent links.
3. If every parent is `done` or `archived`, it updates the child to `ready`.
4. It appends a `promoted` event for each task it moves.

This pass runs in normal lifecycle paths such as dispatch and parent
completion, and it is safe to run repeatedly. If nothing changed, it promotes
nothing.

`recompute_ready` is not a scheduler by itself. A promoted task still needs a
valid assignee and an active dispatcher before work starts.

## Race And Repair Rules

The kernel has guardrails for bad or stale state:

- `create_task(..., parents=[...])` validates that parent ids exist.
- `link_tasks(parent_id, child_id)` rejects self-links and cycles.
- Linking an unfinished parent to a `ready` child demotes the child back to
  `todo`.
- `claim_task` checks parent status again before `ready -> running`; if a racy
  writer made the child `ready` too early, the claim is rejected and the child
  is demoted to `todo`.
- Unblocking a child re-checks parent completion before choosing `ready`.

These checks protect the invariant, but they are not a substitute for creating
the graph correctly. The preferred pattern is still parent-first creation with
`parents=[...]` on the child.

## Anti-Patterns

Avoid these patterns when planning multi-agent work:

- **Prose-only dependency:** writing "wait for T1" in the body but omitting
  `parents=[t1]`. The dispatcher cannot enforce prose.
- **Create all cards as ready, then link later:** a child can start before the
  link exists.
- **Over-linking independent lanes:** static research, source lookup, or docs
  verification can often run in parallel with implementation.
- **Wrong link direction:** `parent_id` is the prerequisite; `child_id` is the
  work that waits.
- **Review as a rerun of the same card:** use a review child task when the
  review needs the implementation handoff.

## Debugging Stuck Tasks

If a task is stuck in `todo`, check these in order:

1. Run `hermes kanban show <task_id>` and inspect its parents.
2. Confirm every parent is actually `done` or intentionally archived.
3. Check for a blocked or running parent that still needs action.
4. Confirm the child has a real assignee before expecting dispatch.
5. If a dependency was removed, use `hermes kanban unlink <parent> <child>` and
   re-check the child; unlinking triggers ready recomputation.

If a task is `ready` but not running, dependency gating is no longer the
blocker. Check assignee availability, dispatcher status, and worker spawn
diagnostics instead.
