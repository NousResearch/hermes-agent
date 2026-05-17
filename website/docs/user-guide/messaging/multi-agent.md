---
sidebar_position: 30
title: "Multi-Agent Routing"
description: "Run multiple isolated AI agents from a single gateway process — each with its own model, personality, memory, and skills"
---

# Multi-Agent Routing

Run multiple isolated AI agents from a single gateway process. Each agent can have its own **model**, **system prompt** (SOUL.md), **memory**, **skills**, and **sessions** — all routed automatically based on where the message comes from.

## Why Multi-Agent?

- **Specialized personalities** — a coding agent with Opus, a research agent with Sonnet, a creative agent with GPT-5, all in one gateway
- **Memory isolation** — each agent's MEMORY.md and USER.md are separate; they don't leak knowledge across personalities
- **Skill isolation** — skills created by one agent don't clutter another's context
- **Cost control** — route low-stakes chats to cheaper models, high-stakes threads to premium ones
- **Workspace separation** — different teams/channels get different agents with different toolsets

## Quick Start

```bash
# 1. Add agents
hermes agent add coder --model anthropic/claude-opus-4-6
hermes agent add research --model anthropic/claude-sonnet-4-6

# 2. List agents and see their IDs
hermes agent list

# 3. Edit ~/.hermes/config.yaml to add routes (see below)

# 4. Restart the gateway
hermes gateway start
```

## Configuration

Multi-agent config lives in `~/.hermes/config.yaml` under top-level keys `agents`, `routes`, and `default_agent`.

### Agents

Each entry under `agents:` defines an agent profile:

```yaml
default_agent: main

agents:
  main:
    # Inherits everything from gateway defaults — no overrides needed

  coder:
    model: "anthropic/claude-opus-4-6"
    provider: "anthropic"
    home_dir: "~/.hermes/profiles/coder"  # Optional: isolate memory/skills/SOUL.md
    enabled_toolsets: [filesystem, terminal, web, skills]

  research:
    model: "anthropic/claude-sonnet-4-6"
    home_dir: "~/.hermes/profiles/research"
    enabled_toolsets: [web, browser, vision, skills]
```

| Field | Description |
|-------|-------------|
| `model` | Model string (same syntax as `hermes model`) |
| `provider` | Inference provider (same as top-level `model.provider`) |
| `home_dir` | Profile directory for SOUL.md, memory/, skills/, sessions/. Defaults to `~/.hermes` for `main`, or `~/.hermes/profiles/<id>` for others. |
| `enabled_toolsets` | List of toolset names this agent can use |
| `disabled_toolsets` | List of toolset names to explicitly remove |

### Routes

Routes match incoming messages and assign them to an agent. Evaluated in **declaration order** — first match wins.

```yaml
routes:
  # Route a specific Telegram forum topic to coder
  - match: { platform: telegram, chat_id: "-1001234567890", thread_id: "42" }
    agent: coder

  # Route the rest of that group to research
  - match: { platform: telegram, chat_id: "-1001234567890" }
    agent: research

  # Route a Slack workspace to coder
  - match: { platform: slack, guild_id: "T0ABC123" }
    agent: coder

  # Route a Discord channel to creative
  - match: { platform: discord, chat_id: "123456789012345678" }
    agent: creative

  # Route a specific user (everywhere) to research
  - match: { platform: telegram, user_id: "123456789" }
    agent: research
```

Supported match keys (all string equality, no globs/regex in MVP):

| Key | Matches |
|-----|---------|
| `platform` | Platform name: `telegram`, `discord`, `slack`, `whatsapp`, etc. |
| `chat_id` | Chat / channel / group ID |
| `thread_id` | Thread / topic ID (Telegram forums, Slack threads, etc.) |
| `user_id` | Sender's platform user ID |
| `user_id_alt` | Alternative user identifier (platform-specific) |
| `guild_id` | Workspace / server ID (Discord, Slack, etc.) |
| `parent_chat_id` | Parent chat ID for nested contexts |

### Resolution Order

When a message arrives, the gateway resolves the agent in this order:

1. **Routes match** — first matching route in declaration order
2. **`select_agent` plugin hook** — plugins can override or supplement routing
3. **`default_agent`** — fallback from config (default: `main`)
4. **`"main"`** — hardcoded final fallback

## CLI Management

```bash
hermes agent list              # Show all agents with model, home dir, route count
hermes agent show coder        # Display full details: paths, routes, SOUL preview
hermes agent add coder         # Add a new agent (interactive or --flags)
hermes agent add coder --from-profile main   # Clone existing profile directory
hermes agent remove coder      # Remove agent (warns about orphaned routes)
hermes agent remove coder --yes              # Force removal without confirmation
```

## Profile Isolation

Each agent with a distinct `home_dir` gets fully isolated storage:

| What | Where | Shared? |
|------|-------|---------|
| SOUL.md | `<home_dir>/SOUL.md` | Per-agent |
| MEMORY.md / USER.md | `<home_dir>/memory/` | Per-agent |
| Skills | `<home_dir>/skills/` | Per-agent |
| Sessions | `~/.hermes/state.db` (SQLite, `agent_id` column) | Shared DB, per-agent rows |
| Cron jobs | `<home_dir>/cron/jobs.json` | Per-agent (directory isolation) |
| Config | `~/.hermes/config.yaml` only | Shared |

The `main` agent uses `~/.hermes` directly. Other agents default to `~/.hermes/profiles/<id>` unless you set `home_dir` explicitly.

## Plugin Hook: `select_agent`

Plugins can implement custom routing logic beyond declarative routes:

```python
# In your plugin
from hermes_cli.plugins import register_hook

@register_hook("select_agent")
def my_custom_router(event, gateway, route_match):
    # route_match is the agent_id from declarative routes (or None)
    if event.source.user_id == "my_boss":
        return "coder"  # Boss always gets the premium model
    return None  # Fall through to next hook or default
```

The first hook returning a non-None string wins. Hooks run after route matching, so `route_match` contains the declarative result (if any).

## Backward Compatibility

Existing single-agent installs require **zero changes**:

- No `agents:` / `routes:` config → everything routes to `main` with existing `~/.hermes` home
- Session keys default to `agent:main:...` — existing sessions continue uninterrupted
- SQLite databases are migrated automatically with `agent_id` column defaulting to `"main"`
- Cron jobs without `agent_id` default to `"main"`

## Limitations (MVP)

- Routes use exact string matching only — no globs, regex, or range matching
- Route specificity is manual — declare more specific routes before general ones
- Per-agent token budgets and priority queues are not yet implemented
- Agents share the same Python process — no filesystem sandboxing guards
- A2A (agent-to-agent) communication is not yet implemented

## Examples

### Team Channel Routing

Route different Slack channels to different agents:

```yaml
agents:
  dev:
    model: "anthropic/claude-opus-4-6"
    enabled_toolsets: [terminal, file, web, skills]
  ops:
    model: "anthropic/claude-sonnet-4-6"
    enabled_toolsets: [terminal, web, cronjob]

routes:
  - match: { platform: slack, chat_id: "C1234567890" }
    agent: dev
  - match: { platform: slack, chat_id: "C0987654321" }
    agent: ops
```

### Forum Topic Routing

Route Telegram forum topics to specialized agents:

```yaml
agents:
  support:
    model: "anthropic/claude-sonnet-4-6"
  sales:
    model: "openai/gpt-5"
    provider: "openrouter"

routes:
  - match: { platform: telegram, chat_id: "-1001234", thread_id: "1" }
    agent: support
  - match: { platform: telegram, chat_id: "-1001234", thread_id: "2" }
    agent: sales
```

### Model Tier Routing

Route VIP users to premium models, everyone else to standard:

```yaml
agents:
  premium:
    model: "anthropic/claude-opus-4-6"
  standard:
    model: "anthropic/claude-sonnet-4-6"

routes:
  - match: { platform: telegram, user_id: "123456789" }
    agent: premium
  - match: { platform: telegram }
    agent: standard
```
