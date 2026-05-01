---
sidebar_position: 8
title: "ADR: Agent Orchestration Primitives"
description: "Why Hermes has delegate_task, agent_control, and Kanban as separate orchestration primitives"
---

# ADR: Agent Orchestration Primitives

Status: Proposed

## Context

Hermes now has three related, but intentionally separate, ways to coordinate agent work:

- `delegate_task`: spawn short-lived subagents inside the current agent process.
- `agent_control`: command persistent Hermes profiles as peer agents through ACP.
- Kanban: coordinate durable asynchronous work through a shared task board.

These should not collapse into one abstraction. They solve different lifecycle, identity, and reliability problems.

This ADR also supersedes the direction explored in [PR #14009](https://github.com/NousResearch/hermes-agent/pull/14009), which introduced an ACP session client for external orchestration. The new direction keeps ACP as the control protocol, but moves orchestration into Hermes itself as an opt-in toolset with durable handles, run records, and per-agent leases.

## Decision

Keep three primitives:

| Primitive | Shape | Identity and memory | State | Parent behavior | Best use | Not for |
|---|---|---|---|---|---|---|
| `delegate_task` | Synchronous RPC to an isolated child `AIAgent` | Anonymous child, fresh context, no shared memory writes | In process only | Parent waits for final summary | Bounded subtasks, parallel research, code review slices, context isolation | Long-lived roles, resumable work, named teammates |
| `agent_control` | Direct command channel to another Hermes profile over ACP | Real profile with its own config, skills, tools, memory, and session history | `~/.hermes/agent-control.db` plus ACP session storage | Parent sends prompt and waits for reply | Manager/team patterns, durable specialists, reusable reviewer/researcher/operator profiles | Fire-and-forget queues, multi-day task boards, broad human workflow |
| Kanban | Shared durable task board and state machine | Named profiles can pick up tasks over time | `~/.hermes/kanban.db` | Parent can create work and continue | Async workflows, retries, human-in-the-loop, multi-agent pipelines | Immediate request/response control of one profile |

## Why `agent_control` Exists

`delegate_task` is deliberately disposable. That is good for isolation, but bad for "team" behavior. A manager agent needs to address a reviewer, researcher, planner, or operator that has:

- a stable profile identity
- its own configured tool surface
- memory and skills selected for that role
- a session history that can be resumed
- durable run metadata for auditing

Kanban solves a different problem: distributed coordination. It is the right primitive when work should be visible, recoverable, and picked up asynchronously. It is heavier than a direct command channel and does not replace the need to say "ask this named profile to do this now and return the answer."

## Reliability Boundaries

`agent_control` is designed as a production-oriented direct-control primitive:

- Handles and runs are persisted in SQLite.
- A session lease keyed by `(profile, session_id)` prevents two orchestrators or duplicate handles from racing the same ACP session.
- Forking takes the same source-session lease before branching history.
- Expired leases are marked as errors instead of leaving handles stuck forever.
- Permission requests from controlled profiles fail closed by default.
- Permissive approval policies are process/admin configuration, not model-selected tool arguments.
- `delegate_task` children cannot inherit `agent_control`, preventing recursive escalation.
- Cancellation is intentionally omitted from the model-facing toolset until the control plane has an async or pollable run API with response dispatching.

It does not try to become a scheduler, queue, or workflow engine. Those remain Kanban's job.

## Relationship to ACP

ACP remains the transport boundary between the orchestrator and the controlled profile. This keeps controlled agents close to normal Hermes execution:

- profile selection still flows through `hermes -p <profile> acp`
- session creation/loading/forking uses existing ACP methods
- editor ACP behavior remains unchanged by default
- agent-control subprocesses opt into the controlled profile's CLI toolsets

This avoids introducing a second internal agent runtime while still giving the orchestrator durable control handles.

## Expected User Mental Model

Use `delegate_task` when you want "do this isolated subtask and summarize back."

Use `agent_control` when you want "ask Alice the reviewer, Bob the researcher, or Ops the operator, each as a real Hermes profile."

Use Kanban when you want "put this work on the board so agents and humans can pick it up, retry it, comment on it, and complete it over time."
