---
title: Parent-managed teams
sidebar_label: Parent-managed teams
---

# Parent-managed teams

The team interface connects a parent's retained workers to Kanban tasks. The
parent can assign a task, send guidance, ask a second worker to review the result,
request a correction from the original worker, and accept the reviewed result.
You choose the worker profiles and their providers, models, thinking levels and
tool restrictions. There are no built-in mandatory roles or model families.

This is a draft implementation. Its focused tests use temporary Kanban and
WorkerStore databases and the real lifecycle admission and lease path, while
holding child execution at a synthetic provider boundary. They do not establish
live provider execution, live Bot/room delivery, installation or release. The
[comparison](orchestration-capability-comparison.md) records the separate worker,
interface, discovery and planned workflow evidence.

## Enable the parent's capabilities

Enable the existing `kanban` toolset for the orchestrator profile and retain its
delegation capability. Configure the worker profiles as described in
[Worker profiles](worker-profiles.md). For example, your chosen `researcher` and
`checker` profiles can use different providers; the names below are placeholders
for profiles you create.

The canonical tool is `kanban_team`. With an explicitly selected styled
interface it appears as `team_task` (Codex style) or `TeamTask` (Claude style).
These names call the same Hermes service. Name collisions receive a distinct
alias, and the tool catalog remains fixed during the conversation. The style
does not switch provider or run the vendor's native harness.

Team operations require both the team capability and the relevant existing
native permission:

| Operation | Additional capability | Meaning |
| --- | --- | --- |
| `create` | `kanban_create` | Create a parent-managed task and its dependencies |
| `start` | `delegate_task`, `kanban_heartbeat` | Claim and start the assigned implementation or reviewer |
| `guide` to an active task | `delegate_task` | Send guidance to its attached running worker |
| `submit_review` | `kanban_request_review` | Submit successful implementation evidence for a named reviewer |
| `accept` | `kanban_complete` | Accept the current successful reviewer run |
| `request_changes` | `kanban_request_changes`, `delegate_task`, `kanban_heartbeat` | Return to the implementation worker with retained context |
| `cancel` | `delegate_task`, `kanban_block` | Request execution cancellation and park the task after terminal evidence |

Bot guidance also requires the existing Bot Chat authorization. Room guidance
requires an explicit current room message grant. Discovery and knowing an ID
do not grant these capabilities. Dispatcher task workers and delegated children
do not receive parent team control. The service rechecks dispatcher exclusion
before any board or action access, including styled and direct calls. Bot Chat's
`message_agent` tool is session-injected rather than globally registered; the
team service requires both that real injection and the current canonical Bot
Chat authorization.

## Example: research, review, correct

A person can ask: “Have my researcher compare these proposals. Ask my checker
to review the result, correct any errors with the same researcher, then prepare
a summary once the comparison is accepted.” The parent uses operations like:

```json
{"action":"create","title":"Compare proposals","profile":"researcher","idempotency_key":"comparison-1"}
```

Use the returned reference in later calls. Here `task:comparison` illustrates
that returned value; it is not an ID to copy literally.

```json
{"action":"create","title":"Prepare summary","profile":"researcher","parent_refs":["task:comparison"]}
{"action":"start","task_ref":"task:comparison"}
{"action":"guide","targets":["task:comparison"],"message":"Check the dates as well as the totals."}
```

The summary remains dependency-gated. Discover or inspect the returned worker
and run references, and use the selected worker interface to wait for the result.
Once the implementation succeeds:

```json
{"action":"submit_review","task_ref":"task:comparison","reviewer":"checker","summary":"Comparison ready for verification."}
{"action":"start","task_ref":"task:comparison"}
```

The second `start` claims the task's review phase. After that reviewer succeeds,
the parent can request a correction or accept the result:

Before the review transition, the parent reads the successful implementation
through its own worker authority and stores a bounded, secret-redacted summary
receipt with the review intent. The reviewer receives that evidence in its
assignment. Worker and run references record provenance; they do not grant the
reviewer access to an implementation sibling.

```json
{"action":"request_changes","task_ref":"task:comparison","message":"Recheck the second proposal's date."}
```

A correction uses the original implementation worker ID and a new linked run.
After it succeeds, submit it for review again, start the reviewer, collect the
review result, and accept:

```json
{"action":"accept","task_ref":"task:comparison","summary":"Dates and totals verified."}
```

Only acceptance marks the task done and releases its dependent summary. Review
is a phase of the same Kanban task, with its own claim and run identity.

## Task, worker and run records

<img src="/img/worker-orchestration/task-run-link.svg" width="1100" alt="A claimed Kanban attempt is connected to a pending worker run before execution is scheduled; results are reviewed separately." />

Kanban owns task dependencies, claims and acceptance. WorkerStore owns the
worker conversation, execution lease, queues, checkpoints and receipts.
Parent-managed tasks use `execution_mode: parent`; existing tasks keep the
dispatcher default. The dispatcher checks that routing mode during selection
and the claim itself.

Team admission prepares an identified held worker assignment, attaches its
reference to the exact Kanban run, and then schedules it through a trusted entry.
Ordinary FIFO and exact-run scheduling cannot lease a held assignment. The team
entry rechecks the parent, current unexpired claim, immutable attachment and
native executable permissions before and after lease acquisition. A matching
retry can reuse the record; changed content must fail. An already-running
assignment is observed. An uncertain tool effect must be reconciled through the
existing worker controls before continuation; a reference is never permission
to replay.

Cancellation interrupts only the exact attached worker run. A request
acknowledgment is not proof that they stopped. `pending_terminal_worker_evidence`
means the parent should wait and inspect before treating the task as cancelled.
The current team `cancel` action targets an attached active task; it is not yet
a saved workflow's bulk cancellation operation.

## Guidance to Bots and rooms

`guide` accepts several typed targets and returns an `outcomes` entry for each.
Read each recipient's status; there is no blanket delivered guarantee. A queued
message has not necessarily been consumed, and an ambiguous result must not be
resent automatically. An idempotency key is not a universal exactly-once promise
across Bot and room transports.

To allow messaging an existing hosted room, explicitly include `message` in
the room grant when configuring the session:

```yaml
orchestration:
  discovery:
    rooms:
      - id: existing-room-id
        actions: [inspect, message]
        participants: [researcher-bot, checker-bot]
```

Use real existing room and Bot profile identifiers. The trusted gateway binds
the grant to the live session, profile, service, policy, participants, gateway
and authority epoch, and checks them again for sending. This does not create,
adopt or transfer authority over a room. Inspection-only grants remain read-only.

Saved repeatable workflow definitions are the next increment. A team task is
not yet a saved workflow template or an automatically resumed script.
