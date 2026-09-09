---
title: Saved orchestration workflows
sidebar_label: Saved workflows
---

# Saved orchestration workflows

A saved workflow is a versioned recipe for a finite set of parent-managed
Kanban tasks. It is useful when the same branch, review and join pattern should
run again with new input. Hermes keeps the existing owners: Kanban records task
dependencies and acceptance, while the retained worker service records each
worker conversation, run, checkpoint and effect receipt.

This is a draft implementation. Its focused tests use temporary real Kanban and
WorkerStore databases and hold provider execution at a controlled boundary.
They do not show that the draft is installed, released, or running through a
live provider.

## Actions

Use the existing `kanban_team` tool. With a selected Codex-style interface it is
`team_task`; with a Claude-style interface it is `TeamTask`. All names reach the
same Hermes service and the same permission checks.

| Action | Extra existing capabilities | Result |
| --- | --- | --- |
| `workflow_save` | `kanban_create` | Save an immutable definition version |
| `workflow_list` | `kanban_list` | List visible templates and this session's invocations without changing state |
| `workflow_inspect` | `kanban_show` | Read one exact template version or owned invocation without changing state |
| `workflow_invoke` | `kanban_create`, `delegate_task`, `kanban_heartbeat` | Admit one graph and start its currently ready branches |
| `workflow_pause` | `kanban_block` | Prevent later step claims with an exact control-version check |
| `workflow_resume` | `delegate_task`, `kanban_heartbeat` | Reactivate a paused invocation or restore recorded active work |
| `workflow_cancel` | `delegate_task`, `kanban_block` | Interrupt exact attached runs, then preserve done work and block unfinished work |

The parent must have a stable session and a current board scope. A typed
reference is not authority. Delegated children, dispatcher-owned task workers,
a different parent session and a stale board path are denied before control.

## Save and invoke a graph

This definition runs `left` and `right` independently. `join` cannot start until
both branches have passed review and are accepted.

```json
{
  "action": "workflow_save",
  "definition": {
    "name": "Compare and combine",
    "steps": [
      {
        "key": "left",
        "title": "Inspect the left input",
        "profile": "researcher-a",
        "reviewer": "checker",
        "max_corrections": 1
      },
      {
        "key": "right",
        "title": "Inspect the right input",
        "profile": "researcher-b",
        "reviewer": "checker",
        "max_corrections": 1
      },
      {
        "key": "join",
        "title": "Combine accepted results",
        "profile": "researcher-a",
        "reviewer": "checker",
        "depends_on": ["left", "right"],
        "max_corrections": 1
      }
    ]
  }
}
```

Save returns an immutable reference such as
`workflow_template:<id>@1`. Use a stable admission key for one logical attempt:

```json
{
  "action": "workflow_invoke",
  "template_ref": "workflow_template:<id>@1",
  "admission_key": "proposal-comparison-2026-09-10",
  "input": {"left": "proposal-a", "right": "proposal-b"}
}
```

An identical retry returns the same workflow and task references. Reusing the
key with different template content or input is an error. New input needs a new
admission key, which creates fresh task and run identities.

Step execution uses the normal team actions. After an implementation worker
succeeds, submit it to the reviewer named in the definition, start the review,
then use `accept` or `request_changes`. A rejected review returns to the original
implementation worker with retained context. Once the maximum correction count
is used, another rejection is refused; restarting Hermes does not reset it.
The same immutable correction bound applies to the native
`kanban_request_changes` transition. A reviewed workflow step reaches `done`
only from its exact reviewer claim after Hermes has recorded successful
WorkerStore evidence, so native `kanban_complete` cannot skip the review.

Call `workflow_resume` after accepted prerequisites release another step. It
also restores recorded held work after restart. Hermes schedules a pending held
run, observes a running or terminal attachment, and leaves unresolved external
tool effects behind the existing reconciliation barrier.

## Pause, inspect and cancel

Control writes use the `control_version` returned by invoke or inspect:

```json
{"action":"workflow_pause","workflow_ref":"workflow:<id>","expected_version":1}
```

Pause blocks new claims. Work claimed before the pause may settle. Resume uses
the next returned version:

```json
{"action":"workflow_resume","workflow_ref":"workflow:<id>","expected_version":2}
```

`workflow_list` and `workflow_inspect` do not initialize a missing board,
migrate it, recover a worker or promote a task. Inspection reports task status
separately from invocation control status.

Cancellation records every exact attached execution for unfinished steps before
requesting interrupts, even when a native block or stale-claim recovery already
changed the Kanban task status. Hermes first observes each attachment and sends
an interrupt only while it remains nonterminal. If the interrupt result is uncertain, the workflow remains
`cancelling`; retry with the current control version observes the recorded run
without sending a second interrupt. Once all exact runs have terminal evidence,
Hermes keeps accepted tasks done and sticky-blocks unfinished tasks and the
coordinator. It does not archive them, so a separate task depending on the
cancelled coordinator stays closed.
