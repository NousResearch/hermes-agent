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
| `routing.mode` / `routing.provider` / `routing.model` | No | Optional native Hermes provider routing. `routing.mode: hermes` runs the child with the named provider/model through Hermes' normal credential resolver. |
| `runner.mode` / `runner.name` / `runner.continue` | No | Optional execution runner. Defaults to `delegate_task`; `cli` uses a trusted runner from `agent_runners` config. |

:::warning No secrets in frontmatter
Do not put API keys, tokens, or other secrets in agent frontmatter. Secrets belong in `~/.hermes/.env` or your environment, not in files that may be checked into version control.
:::

## Running Agents with `assign_agent`

Use the `assign_agent` tool to delegate a task synchronously to a named agent:

```
assign_agent(agent_name="code-reviewer", task="Review PR #42 in /home/nuuair/myproject")
```

The agent runs with its own isolated context. Only the final result is returned — intermediate tool calls and thoughts stay private to the subagent.

### Native provider routing

By default, a named agent inherits the normal delegation model/provider. To pin a named agent to a specific Hermes-supported provider and model, use `routing.mode: hermes`:

```markdown
---
schema_version: 1
name: fast-code-explorer
description: "Repository exploration specialist backed by Kimi"
routing:
  mode: hermes
  provider: kimi-coding
  model: kimi-k2.6
tools:
  mode: restrict
  allow_toolsets: [file]
---

You explore repositories and report concise findings.
```

When `assign_agent` runs this agent, the child uses the configured provider/model while the parent session keeps its own model. Credentials, base URLs, API modes, OAuth state, and credential pools are resolved by Hermes' normal provider machinery. Do not put API keys in agent files.

`routing.mode: hermes` is intended for native Hermes providers such as OpenRouter, Anthropic, Kimi, MiniMax, Z.AI/GLM, and other configured provider integrations. External command-backed agents should use trusted CLI runners instead.

### CLI-backed agents

By default, `assign_agent` uses the in-process `delegate_task` runner. Advanced users can route a named agent to a trusted external CLI runner by adding `runner` frontmatter:

```markdown
---
schema_version: 1
name: code-architect
description: "Architecture planner backed by an external CLI"
runner:
  mode: cli
  name: gemini-cli
  continue: auto   # off | auto | require
---

You are a careful software architect.
```

The executable command is **not** read from the agent file. It must come from trusted user-owned config:

```yaml
agent_runners:
  gemini-cli:
    type: cli
    command: gemini
    args: ["--output-format", "stream-json"]
    resume_arg: "--resume"
    allowed_from_project_agents: true
```

Continuation modes:

| Mode | Behavior |
|------|----------|
| `off` | Always start a fresh external CLI session. |
| `auto` | Resume when Hermes has a stored external session id for this parent session, agent, runner, and workdir; otherwise start fresh. |
| `require` | Fail closed unless a stored external CLI session exists. |

External CLI session ids are stored under `$HERMES_HOME/agent-runner-sessions.json`. Project-local agents may request a runner by name, but they cannot define commands, args, env vars, or executable paths.

### Inspecting runtime routing

Hermes writes a small, best-effort runtime trace when `assign_agent` resolves and runs a named agent. This lets you verify which agent file won discovery, which runner mode was used, and whether CLI session continuity resumed.

The trace is stored as JSONL:

```text
$HERMES_HOME/logs/runtime-trace.jsonl
```

Use the read-only `runtime_inspect` tool from the `debugging` toolset:

```python
runtime_inspect(session_id="20260513_...")
runtime_inspect(session_id="20260513_...", agent_name="code-architect", limit=20)
```

Typical events include:

- `model.request` — ordinary LLM call metadata: provider, model, API mode, base URL host, tool count, and API call count. It does not include prompts, messages, API keys, auth headers, or the full base URL.
- `assign_agent.requested` — the parent session requested a named agent.
- `assign_agent.resolved` — registry lookup chose a global or project-local agent and resolved runner/model/provider metadata.
- `assign_agent.dispatched` — Hermes selected `delegate_task` or `cli_runner` execution.
- `assign_agent.completed` — execution finished, including success, duration, runner name, resume status, return code, and external CLI session id when available.

Runtime trace payloads redact secret-looking fields and intentionally omit full prompts and configured CLI command argv.
 
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
- **Native provider routing is trusted metadata.** Agent files may request a provider/model with `routing.mode: hermes`, but API keys and credential lookup stay outside the file. Internal provider/model overrides are not exposed on the public `delegate_task` tool schema.
- **Project-local agents cannot define arbitrary CLI or ACP commands.** CLI commands, args, resume flags, and executable paths must come from trusted `agent_runners` config. Project-local agents can only request a configured runner by name, and only when that runner sets `allowed_from_project_agents: true`.
- **Secrets in frontmatter are rejected** at load time. Never put credentials in agent files.

## Current Limitations

Named agents are a new capability. Current limitations:

- **No `spawn_agent` / background execution.** Agents run synchronously via `assign_agent` and return when complete.
- **No marketplace or install wizard.** Agent files are created and managed by hand.
- **No edit or update tools.** To change an agent, edit its Markdown file directly.
- **Limited routing.** `delegate_task` agents can inherit delegation routing or use `routing.mode: hermes` for native provider/model routing; `cli` agents can use trusted external runners from `agent_runners`. Arbitrary ACP commands from agent files are not supported.
- **No agent-to-agent chaining.** You cannot currently have one agent spawn another as a subagent.

These limitations will be addressed in future releases.
