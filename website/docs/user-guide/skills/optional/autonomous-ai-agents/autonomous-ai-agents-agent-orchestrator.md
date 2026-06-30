---
title: "Agent Orchestrator — Use Agent Orchestrator as an external supervision layer for parallel coding agents"
sidebar_label: "Agent Orchestrator"
description: "Use Agent Orchestrator as an external supervision layer for parallel coding agents"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agent Orchestrator

Use Agent Orchestrator as an external supervision layer for parallel coding agents. Best for teams that want isolated worktrees, PR/CI feedback routing, and a live dashboard without expanding Hermes core.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/agent-orchestrator` |
| Path | `optional-skills/autonomous-ai-agents/agent-orchestrator` |
| Version | `0.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Agent-Orchestrator`, `Parallel-Agents`, `Worktrees`, `CI`, `PR-Review`, `Dashboard` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), `using-git-worktrees` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Agent Orchestrator

Use [Agent Orchestrator](https://github.com/AgentWrapper/agent-orchestrator) when the user wants a dedicated external control plane for many coding agents working in parallel. Hermes stays the operator and reviewer; Agent Orchestrator provides the long-running daemon, isolated workspaces, agent adapters, and CI/review feedback routing.

This belongs as an optional skill, not a Hermes core feature: it is a separate product with its own daemon, Electron UI, SQLite state, and adapter ecosystem.

## When to Use

- The user wants dozens of parallel coding runs with isolated git worktrees.
- The team wants PR comments, CI failures, and merge conflicts routed back to the owning agent automatically.
- A long-running desktop dashboard is desirable.
- The user already works with terminal-native agents like Codex, Claude Code, Aider, or OpenCode.

Prefer Hermes `delegate_task` for small local subtasks. Prefer this skill when the orchestration itself is the product.

## What It Adds

- Long-running Go daemon on loopback
- Electron/React control plane
- Agent adapters for many terminal agents
- Worktree-per-run isolation
- Durable state in SQLite
- CI / PR / merge-conflict feedback loops

## Hermes Adoption Target

Adopt the pattern, not the product:

- keep Hermes `delegate_task` for small local reasoning jobs
- borrow the worktree-per-run and feedback-routing ideas for repo-local workflows
- treat the Agent Orchestrator daemon/UI as an external operator surface
- avoid importing its state model or dashboard concerns into Hermes core

## Prerequisites

- Go 1.25+
- Node.js 20+
- pnpm
- git
- Optional but commonly useful: `tmux`, `gh`

## Install

### From source

```bash
terminal(command="git clone https://github.com/AgentWrapper/agent-orchestrator.git", workdir="/tmp")
terminal(command="pnpm install", workdir="/tmp/agent-orchestrator/frontend", timeout=600)
terminal(command="go test -race ./...", workdir="/tmp/agent-orchestrator/backend", timeout=600)
```

### Prebuilt desktop app

Use the platform release artifact from GitHub Releases if the user wants the full desktop app quickly.

## Operating Pattern with Hermes

1. Let Hermes inspect the repo and decide whether external orchestration is warranted.
2. Use Hermes to prepare the repo, secrets, and agent CLI prerequisites.
3. Start Agent Orchestrator separately.
4. Let Agent Orchestrator own the parallel coding loop.
5. Use Hermes for repo-specific review, verification, or follow-up fixes.

## Verification

Check the daemon health endpoint after startup:

```bash
terminal(command="curl -fsS http://127.0.0.1:3001/healthz", timeout=30)
```

If using the source tree, upstream documents these checks:

```bash
terminal(command="go test -race ./...", workdir="/path/to/agent-orchestrator/backend", timeout=600)
terminal(command="pnpm test", workdir="/path/to/agent-orchestrator/frontend", timeout=600)
```

## Pitfalls

- The daemon is loopback-only by design; do not expose it directly.
- It is another orchestration layer, so avoid overlapping responsibility with Hermes cron jobs and background processes.
- Worktree-heavy operation needs enough disk and clean git hygiene.
- Treat it as a separate app lifecycle, not a pip/uv helper inside Hermes.

## Related

- Upstream: https://github.com/AgentWrapper/agent-orchestrator
- Use alongside Hermes skills for worktree discipline and verification.
