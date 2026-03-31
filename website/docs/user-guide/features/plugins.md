---
sidebar_position: 11
sidebar_label: "Plugins"
title: "Plugins"
description: "Extend Hermes with custom tools, hooks, and integrations via the plugin system"
---

# Plugins

Hermes has a plugin system for adding custom tools, hooks, and integrations without modifying core code.

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
| Inject messages | `ctx.inject_message(content, role="user")` — see [Injecting Messages](#injecting-messages) |
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

Plugins can register callbacks for these lifecycle events. See the **[Event Hooks page](/docs/user-guide/features/hooks#plugin-hooks)** for full details, callback signatures, and examples.

| Hook | Fires when |
|------|-----------|
| `pre_tool_call` | Before any tool executes |
| `post_tool_call` | After any tool returns |
| `pre_llm_call` | Once per turn, before the LLM loop — can return `{"context": "..."}` to inject into the system prompt |
| `post_llm_call` | Once per turn, after the LLM loop completes |
| `on_session_start` | New session created (first turn only) |
| `on_session_end` | End of every `run_conversation` call |

## Managing plugins

```bash
hermes plugins                  # interactive toggle UI — enable/disable with checkboxes
hermes plugins list             # table view with enabled/disabled status
hermes plugins install user/repo  # install from Git
hermes plugins update my-plugin   # pull latest
hermes plugins remove my-plugin   # uninstall
hermes plugins enable my-plugin   # re-enable a disabled plugin
hermes plugins disable my-plugin  # disable without removing
```

Running `hermes plugins` with no arguments launches an interactive curses checklist (same UI as `hermes tools`) where you can toggle plugins on/off with arrow keys and space.

Disabled plugins remain installed but are skipped during loading. The disabled list is stored in `config.yaml` under `plugins.disabled`:

```yaml
plugins:
  disabled:
    - my-noisy-plugin
```

In a running session, `/plugins` shows which plugins are currently loaded.

## Injecting Messages

Plugins can inject messages into the active conversation using `ctx.inject_message()`:

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**Signature:** `ctx.inject_message(content: str, role: str = "user") -> bool`

How it works:

- If the agent is **idle** (waiting for user input), the message is queued as the next input and starts a new turn.
- If the agent is **mid-turn** (actively running), the message interrupts the current operation — the same as a user typing a new message and pressing Enter.
- For non-`"user"` roles, the content is prefixed with `[role]` (e.g. `[system] ...`).
- Returns `True` if the message was queued successfully, `False` if no CLI reference is available (e.g. in gateway mode).

This enables plugins like remote control viewers, messaging bridges, or webhook receivers to feed messages into the conversation from external sources.

:::note
`inject_message` is only available in CLI mode. In gateway mode, there is no CLI reference and the method returns `False`.
:::

See the **[full guide](/docs/guides/build-a-hermes-plugin)** for handler contracts, schema format, hook behavior, error handling, and common mistakes.
