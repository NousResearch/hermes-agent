---
sidebar_position: 20
---

# Plugins

Hermes has a plugin system for adding custom tools, hooks, slash commands, and integrations without modifying core code.

**→ [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin)** — step-by-step guide with a complete working example.

## Quick overview

Drop a directory into `~/.hermes/plugins/` with a `plugin.yaml` and Python code:

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # manifest
├── __init__.py      # register() — wires schemas to handlers
├── schemas.py       # tool schemas (what the LLM sees)
└── tools.py         # tool handlers (what runs when called)
```

Start Hermes — your tools appear alongside built-in tools. The model can call them immediately.

Project-local plugins under `./.hermes/plugins/` are disabled by default. Enable them only for trusted repositories by setting `HERMES_ENABLE_PROJECT_PLUGINS=true` before starting Hermes.

## What plugins can do

| Capability | How |
|-----------|-----|
| Add tools | `ctx.register_tool(name, schema, handler)` |
| Add hooks | `ctx.register_hook("post_tool_call", callback)` |
| Add slash commands | `ctx.register_command("mycommand", handler)` |
| Ship data files | `Path(__file__).parent / "data" / "file.yaml"` |
| Bundle skills | Copy `skill.md` to `~/.hermes/skills/` at load time |
| Gate on env vars | `requires_env: [API_KEY]` in plugin.yaml |
| Distribute via pip | `[project.entry-points."hermes_agent.plugins"]` |

## Plugin discovery

| Source | Path | Use case |
|--------|------|----------|
| User | `~/.hermes/plugins/` | Personal plugins |
| Project | `.hermes/plugins/` | Project-specific plugins (requires `HERMES_ENABLE_PROJECT_PLUGINS=true`) |
| pip | `hermes_agent.plugins` entry_points | Distributed packages |

## Available hooks

| Hook | Fires when |
|------|-----------|
| `pre_tool_call` | Before any tool executes |
| `post_tool_call` | After any tool returns |
| `pre_llm_call` | Before LLM API request |
| `post_llm_call` | After LLM API response |
| `on_session_start` | Session begins |
| `on_session_end` | Session ends |

## Slash commands

Plugins can register slash commands that work in both CLI and messaging platforms:

```python
def register(ctx):
    ctx.register_command(
        name="greet",
        handler=lambda args: f"Hello, {args or 'world'}!",
        description="Greet someone",
        args_hint="[name]",
        aliases=("hi",),
    )
```

The handler receives the argument string (everything after `/greet`) and returns a string to display. Registered commands automatically appear in `/help`, tab autocomplete, Telegram bot menu, and Slack subcommand mapping.

| Parameter | Description |
|-----------|-------------|
| `name` | Command name without slash |
| `handler` | Callable that takes `args: str` and returns `str | None` |
| `description` | Shown in `/help` |
| `args_hint` | Usage hint, e.g. `"[name]"` |
| `aliases` | Tuple of alternative names |
| `cli_only` | Only available in CLI |
| `gateway_only` | Only available in messaging platforms |

## Managing plugins

```
/plugins              # list loaded plugins in a session
hermes config set display.show_cost true  # show cost in status bar
```

See the **[full guide](/docs/guides/build-a-hermes-plugin)** for handler contracts, schema format, hook behavior, error handling, and common mistakes.
