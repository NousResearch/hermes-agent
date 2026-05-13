---
sidebar_position: 11
title: "Named Agents"
description: "Define reusable agent workers with Markdown files — stored globally in HERMES_HOME or per-project — and run them by name via assign_agent"
---

# Named Agents

Named agents are reusable worker definitions stored as Markdown files. Each agent has its own identity, instructions, and optional tool constraints. Instead of copy-pasting a system prompt into every conversation, you define it once and invoke it by name.

## Agents vs Skills

Skills are **procedures** — reusable playbooks that get loaded into your *current* agent session. Agents are **workers** — independent agent instances with their own context that you delegate tasks to.

| | Agents | Skills |
|--|--------|--------|
| Execution | Spawned as isolated subagents via `assign_agent` | Inlined into current session via `/skill` or skill tools |
| State | Stateless (no memory between calls) | Stateless |
| Storage | Markdown files on disk | Markdown files on disk |
| Use case | Dedicated specialists (code reviewer, tester, writer) | Reusable workflows (refactor, release, debug) |

## Storage and Override Rules

Agents are discovered from two locations:

| Scope | Location |
|-------|----------|
| **Global** | `$HERMES_HOME/agents/*.md`, `$HERMES_HOME/agents/**/*.md`, `$HERMES_HOME/agents/AGENT.md` |
| **Project-local** | `<project>/.hermes/agents/*.md`, `<project>/.hermes/agents/**/*.md`, `<project>/.hermes/agents/AGENT.md` |

A project-local agent with the same `name` as a global one **shadows** the global definition. This lets projects override or specialize shared agents without editing global files.

Discovery is non-recursive at the top level — `agents/*.md` is scanned flat; subdirectories are traversed recursively. Project-local agents are found by walking from the current working directory toward the filesystem root.

## Agent File Format

Agent files are Markdown with YAML frontmatter. Required fields: `schema_version`, `name`, `description`, and a non-empty body prompt.

```markdown
---
schema_version: 1
name: code-reviewer
description: "Thorough code reviewer for pull requests — checks logic, style, and test coverage"
tags: [coding, review]
tools:
  mode: restrict
  allow_toolsets: [terminal, file]
delegation:
  role: leaf
---

You are a senior code reviewer. Your job:

- Read the changed files carefully
- Focus on logic errors, edge cases, and potential bugs
- Flag style violations and suggest improvements
- Check that tests cover the changes
- Be specific and constructive in your feedback

When done, summarize your findings in a concise report.
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `schema_version` | Yes | Must be `1`. Reserved for future schema changes. |
| `name` | Yes | Unique identifier. Lowercase, hyphens allowed. Used by `assign_agent`. |
| `description` | Yes | Human-readable summary shown in `agents_list` output. |
| `tags` | No | Labels used by the `agents_list(category=...)` filter. |
| `tools.mode` / `tools.allow_toolsets` | No | Optional toolset restriction for the delegated worker. |
| `delegation.role` | No | Optional child role (`leaf` or `orchestrator`) passed to the delegation layer. |

:::warning No secrets in frontmatter
Do not put API keys, tokens, or other secrets in agent frontmatter. Secrets belong in `~/.hermes/.env` or your environment, not in files that may be checked into version control.
:::

## Running Agents with `assign_agent`

Use the `assign_agent` tool to delegate a task synchronously to a named agent:

```
assign_agent(agent_name="code-reviewer", task="Review PR #42 in /home/nuuair/myproject")
```

The agent runs with its own isolated context. Only the final result is returned — intermediate tool calls and thoughts stay private to the subagent.

### Delegation Toolset

`assign_agent` lives in the `delegation` toolset alongside `delegate_task`. Enable it with:

```
enabled_toolsets: [delegation]
```

## Listing and Inspecting Agents

Two read-only tools are available in the `agents` toolset:

### `agents_list`

Returns compact metadata for all registered agents — names, descriptions, source paths, and enabled/disabled/shadowed status. Does not include prompt bodies.

```
agents_list()                          # all agents
agents_list(category="review")          # filter by tag
agents_list(include_shadowed=true)     # include shadowed globals
agents_list(workdir="/path/to/project") # include project-local agents for that project
```

### `agent_view`

Returns the full agent definition including the prompt body, for inspection or verification.

```
agent_view(name="code-reviewer")       # from default search paths
agent_view(name="code-reviewer", source="global")   # global only
agent_view(name="code-reviewer", source="project") # project-local only
agent_view(name="code-reviewer", workdir="/path/to/project")
```

## Security Notes

- **Agents toolset is read-only.** You can list and view agents, but cannot create, edit, or delete them via tools.
- **`assign_agent` is in the delegation toolset**, not the agents toolset. It requires explicit enablement.
- **Project-local agents cannot set arbitrary CLI or ACP command execution.** They are scoped to the toolsets they declare.
- **Secrets in frontmatter are rejected** at load time. Never put credentials in agent files.

## PR1 Limitations

Named agents are a new capability (PR1). Current limitations:

- **No `spawn_agent` / background execution.** Agents run synchronously via `assign_agent` and return when complete.
- **No marketplace or install wizard.** Agent files are created and managed by hand.
- **No edit or update tools.** To change an agent, edit its Markdown file directly.
- **PR1 routing only.** Explicit routing to specific models or CLI runner modes is rejected. Agents use inherited delegation routing via `delegate_task`.
- **No agent-to-agent chaining.** You cannot currently have one agent spawn another as a subagent.

These limitations will be addressed in future releases.
