---
sidebar_position: 8
title: "Agent Control"
description: "Control persistent Hermes profiles as peer agents through ACP"
---

# Agent Control

`agent_control` lets an orchestrator profile command other Hermes profiles as persistent peer agents. These are not anonymous `delegate_task` children: each controlled agent runs through ACP with its own profile config, tools, skills, memory providers, and session history.

This toolset is intentionally opt-in. Enable it only for profiles that are meant to orchestrate other agents.

Agent Control requires the ACP extra:

```bash
pip install 'hermes-agent[acp]'
# From a source checkout:
pip install -e '.[acp]'
```

## Enable the toolset

For a one-off orchestrator run:

```bash
hermes -p orchestrator --toolsets agent_control
```

Or add `agent_control` to the orchestrator profile's configured toolsets.

The controlled profiles must already exist under `~/.hermes/profiles/<profile-name>`. When `agent_control` launches a profile through ACP, the profile uses its configured CLI toolsets instead of the editor-focused `hermes-acp` default.

## Tools

| Tool | Purpose |
|---|---|
| `agent_start` | Create or attach to a durable handle for a profile session. |
| `agent_prompt` | Send work to a controlled profile and wait for its response. |
| `agent_status` | Inspect the handle and latest run. |
| `agent_list` | List known controlled profile handles. |
| `agent_fork` | Branch a controlled agent's session for speculative work. |

State is stored in `~/.hermes/agent-control.db`. Handles survive process restarts; active subprocesses do not. A later `agent_prompt` loads the stored ACP session before sending new work.

## Basic pattern

```python
agent_start(
    profile="researcher",
    cwd="/home/user/project",
    idempotency_key="team:researcher"
)

agent_prompt(
    agent_id="agent-abc123",
    prompt="""Inspect the repository and return:
    - findings with file paths
    - tests run
    - open blockers
    """,
    timeout_seconds=900
)

agent_status(agent_id="agent-abc123")
```

Use idempotency keys for named team roles. If the orchestrator retries after a transient failure, it can reuse the same handle instead of spawning duplicate sessions.

## Permissions

`agent_prompt`, `agent_start`, and `agent_fork` deny dangerous local permission requests by default. The model cannot choose a permissive approval policy through tool arguments.

For trusted local automation, an administrator can set `HERMES_AGENT_CONTROL_APPROVAL_POLICY=allow_once` in the Hermes process environment. Keep the default `deny` policy for untrusted or shared orchestrator profiles.

## Agent Control vs. Delegation vs. Kanban

Use `delegate_task` for short, context-isolated subtasks where the parent needs an answer before continuing. Delegated children are fresh in-process agents and only their final summary returns.

Use `agent_control` when identity matters: a reviewer, researcher, operator, or planner should keep a real profile, durable session, tool configuration, and memory across prompts.

Use Kanban when the work should be asynchronous, recoverable, and visible as a task board across multiple profiles or humans. `agent_control` is a direct command channel; Kanban is a durable coordination surface.

## Reliability model

Each controlled ACP session has a SQLite lease keyed by profile and session id. Only one orchestrator run can own the same profile session at a time, even if multiple handles point at it, so concurrent prompts do not corrupt session history. Forks take the same lease before branching source history. Runs are persisted with status, response, error, and usage metadata.

`agent_control` intentionally exposes synchronous request/response commands only. Cancellation is not a model-facing tool until Hermes has an async or pollable run API with response dispatching strong enough to cancel in-flight ACP work reliably.

For production orchestration, ask controlled agents for structured handoffs: files touched, commands run, tests passed or failed, artifacts produced, and blockers. The control plane makes execution durable; the prompt contract makes team behavior predictable.
