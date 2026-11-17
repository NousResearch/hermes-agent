---
title: Interactive task delivery
---

# Interactive task delivery

`hermes work` provides a persistent coordination record for long interactive tasks. It uses the
current profile's SessionDB and existing `terminal` and `delegate_task` interfaces. It is not a
second Kanban scheduler. A work item is scoped to a logical session and a stable ID; verified
compression continuations retain its facts, while branches, resets and delegated sessions remain
separate.

## Claims and evidence

Run `hermes work --help` for the CLI, or load `requesting-code-review` and its task-delivery reference.
The workflow is `init`, `context`, managed delegation, `deliver`, managed `review`, then `advance`
through integration, verification and acceptance. `list` and `show` expose the current facts.

A context begins with `[HERMES_WORK_ITEM_V1]`, followed by one strict JSON line and a blank line.
The reference contains `session_id`, `task_id`, `role` and optional `base` and `files`. This is an
explicit opt-in protocol in the existing context string, not a change to the core tool schema.
The runtime validates the real parent's profile/session, claims the work atomically before
building children, and fences completion with an owner and generation. Ordinary contexts and
same-goal best-of-N delegation keep their existing behavior.

The ledger separates execution, candidate delivery, review, integration, verification and
acceptance. Artifacts are hashed, review reports bind to the current candidate, stale workers
cannot publish under a newer claim, and matching file ownership prevents concurrent managed
implementers from editing the same declared file. This is coordination, not an OS filesystem
sandbox. A claim expiration requires reconciliation instead of silently spawning a replacement.

`advance` records explicit evidence; it does not run tests, deploy code, independently authenticate
a reviewer or decide whether the business outcome is true. The operator/agent must execute the
actual verification first. Verification requires a passing report and exit code zero; acceptance
requires a passing scenario record. A child saying “done” cannot produce acceptance.

Managed review has an initial candidate and two repair candidates. Further automatic repair with
remaining blockers is refused as `needs_decision`, including after restart or issue-ID changes.
Suggestions do not block release; classification and blocking are separate. Reopening a task
preserves its history but cannot erase the review budget. The separate `decision` command requires
an explicit reason and acceptance criterion, archives the former review epoch, invalidates old
execution generations, and starts a new epoch. It is an operator scope/design decision, not an
automatic retry mechanism or proof of human authorization.

## Child completions during a turn

Authenticated asynchronous child completions for canonical, interactive LOCAL in-process parents
can enter the next fresh tool-result row as bounded, explicitly untrusted evidence. The transcript
and exposure receipt commit in one SQLite transaction. No old row or system prompt is changed.
An unread result remains durable on failure or stop, and repeated delivery does not become another
execution. Human inputs and other queued producers are not bypassed.

The exposure receipt is `context_persisted` with `accepted: false`. Full payloads remain available
through `hermes work inbox --session SESSION --admission ID`. This is separate from the legacy
transport acknowledgment, which continues to mean successful admission. Session-wide Stop holds
automatic child-completion dispatch until an authorized viewer sends new input. Without a fresh
tool boundary, or on unsupported execution surfaces, the existing FIFO path remains in use.

## Progress and recovery

The Todo tool retains its existing `summary` and snapshot contracts and adds a separate progress
projection. Desktop and TUI display top-level phases separately from subtasks. A phase counts as
complete only when it is explicitly completed and has no pending/in-progress descendant; completing
all subtasks does not silently approve the parent's integration or acceptance work.

For managed work, Todo completion requires recorded acceptance. Compression restores a bounded
ledger index as reference data alongside the Todo snapshot. Replacing the checklist does not erase
the work item's evidence, owner, or outstanding gate. Permanent session deletion removes that
session's ledger and receipts; ordinary settlement and compression do not.
