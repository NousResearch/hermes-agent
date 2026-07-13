---
sidebar_position: 11
sidebar_label: "Plugins"
title: "Plugins"
description: "Extend Hermes with custom tools, hooks, and integrations via the plugin system"
---

# Plugins

Hermes has a plugin system for adding custom tools, hooks, and integrations without modifying core code.

If you want to create a custom tool for yourself, your team, or one project,
this is usually the right path. The developer guide's
[Adding Tools](/docs/developer-guide/adding-tools) page is for built-in Hermes
core tools that live in `tools/` and `toolsets.py`.

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

### Minimal working example

Here is a complete plugin that adds a `hello_world` tool and logs every tool call via a hook.

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Minimal Hermes plugin — registers a tool and a hook."""

import json


def register(ctx):
    # --- Tool: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Returns a friendly greeting for the given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
        description="Return a friendly greeting for the given name.",
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

Drop both files into `~/.hermes/plugins/hello-world/`, restart Hermes, and the model can immediately call `hello_world`. The hook prints a log line after every tool invocation.

Project-local plugins under `./.hermes/plugins/` are disabled by default. Enable them only for trusted repositories by setting `HERMES_ENABLE_PROJECT_PLUGINS=true` before starting Hermes.

## What plugins can do

Every `ctx.*` API below is available inside a plugin's `register(ctx)` function.

| Capability | How |
|-----------|-----|
| Add tools | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| Add hooks | `ctx.register_hook("post_tool_call", callback)` |
| Add slash commands | `ctx.register_command(name, handler, description)` — adds `/name` in CLI and gateway sessions |
| Dispatch tools from commands | `ctx.dispatch_tool(name, args)` — invokes a registered tool with parent-agent context auto-wired |
| Add CLI commands | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — adds `hermes <plugin> <subcommand>` |
| Inject messages | `ctx.inject_message(content, role="user")` — see [Injecting Messages](#injecting-messages) |
| Ship data files | `Path(__file__).parent / "data" / "file.yaml"` |
| Bundle skills | `ctx.register_skill(name, path)` — namespaced as `plugin:skill`, loaded via `skill_view("plugin:skill")` |
| Gate on env vars | `requires_env: [API_KEY]` in plugin.yaml — prompted during `hermes plugins install` |
| Distribute via pip | `[project.entry-points."hermes_agent.plugins"]` |
| Register a gateway platform (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — see [Adding Platform Adapters](/docs/developer-guide/adding-platform-adapters) |
| Register an image-generation backend | `ctx.register_image_gen_provider(provider)` — see [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) |
| Register a video-generation backend | `ctx.register_video_gen_provider(provider)` — see [Video Generation Provider Plugins](/docs/developer-guide/video-gen-provider-plugin) |
| Register a context-compression engine | `ctx.register_context_engine(engine)` — see [Context Engine Plugins](/docs/developer-guide/context-engine-plugin) |
| Register a memory backend | Subclass `MemoryProvider` in `plugins/memory/<name>/__init__.py` — see [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) (uses a separate discovery system) |
| Run a host-owned LLM call | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — borrow the user's active model + auth for a one-shot completion with optional JSON schema validation. See [Plugin LLM Access](/docs/developer-guide/plugin-llm-access) |
| Register an inference backend (LLM provider) | `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/__init__.py` — see [Model Provider Plugins](/docs/developer-guide/model-provider-plugin) (uses a separate discovery system) |

## Plugin discovery

| Source | Path | Use case |
|--------|------|----------|
| Bundled | `<repo>/plugins/` | Ships with Hermes — see [Built-in Plugins](/docs/user-guide/features/built-in-plugins) |
| User | `~/.hermes/plugins/` | Personal plugins |
| Project | `.hermes/plugins/` | Project-specific plugins (requires `HERMES_ENABLE_PROJECT_PLUGINS=true`) |
| pip | `hermes_agent.plugins` entry_points | Distributed packages |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS declarative installs — see [Nix Setup](/docs/getting-started/nix-setup#plugins) |

Later sources override earlier ones on name collision, so a user plugin with the same name as a bundled plugin replaces it.

### Plugin sub-categories

Within each source, Hermes also recognizes sub-category directories that route plugins to specialized discovery systems:

| Sub-directory | What it holds | Discovery system |
|---|---|---|
| `plugins/` (root) | General plugins — tools, hooks, slash commands, CLI commands, bundled skills | `PluginManager` (kind: `standalone` or `backend`) |
| `plugins/platforms/<name>/` | Gateway channel adapters (`ctx.register_platform()`) | `PluginManager` (kind: `platform`, one level deeper) |
| `plugins/image_gen/<name>/` | Image-generation backends (`ctx.register_image_gen_provider()`) | `PluginManager` (kind: `backend`, one level deeper) |
| `plugins/memory/<name>/` | Memory providers (subclass `MemoryProvider`) | **Own loader** in `plugins/memory/__init__.py` (kind: `exclusive` — one active at a time) |
| `plugins/context_engine/<name>/` | Context-compression engines (`ctx.register_context_engine()`) | **Own loader** in `plugins/context_engine/__init__.py` (one active at a time) |
| `plugins/model-providers/<name>/` | LLM provider profiles (`register_provider(ProviderProfile(...))`) | **Own loader** in `providers/__init__.py` (lazily scanned on first `get_provider_profile()` call) |

User plugins at `~/.hermes/plugins/model-providers/<name>/` and `~/.hermes/plugins/memory/<name>/` override bundled plugins of the same name — last-writer-wins in `register_provider()` / `register_memory_provider()`. Drop a directory in, and it replaces the built-in without any repo edits.

Sub-category plugins surface in `hermes plugins list` and the interactive `hermes plugins` UI under their **path-derived key** — e.g. `observability/langfuse`, `image_gen/openai`, `platforms/teams`. That key (not the bare manifest `name:`) is the value you pass to `hermes plugins enable …` / `disable …` and the string to add under `plugins.enabled` in `config.yaml`.

## Plugins are opt-in (with a few exceptions)

**General plugins and user-installed backends are disabled by default** — discovery finds them (so they show up in `hermes plugins` and `/plugins`), but nothing with hooks or tools loads until you add the plugin's name to `plugins.enabled` in `~/.hermes/config.yaml`. This stops third-party code from running without your explicit consent.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # optional deny-list — always wins if a name appears in both
    - noisy-plugin
```

Three ways to flip state:

```bash
hermes plugins                    # interactive toggle (space to check/uncheck)
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove from allow-list + add to disabled
```

After `hermes plugins install owner/repo`, you're asked `Enable 'name' now? [y/N]` — defaults to no. Skip the prompt for scripted installs with `--enable` or `--no-enable`.

### What the allow-list does NOT gate

Several categories of plugin bypass `plugins.enabled` — they're part of Hermes' built-in surface and would break basic functionality if gated off by default:

| Plugin kind | How it's activated instead |
|---|---|
| **Bundled platform plugins** (IRC, Teams, etc. under `plugins/platforms/`) | Auto-loaded so every shipped gateway channel is available. The actual channel turns on via `gateway.platforms.<name>.enabled` in `config.yaml`. |
| **Bundled backends** (image-gen providers under `plugins/image_gen/`, etc.) | Auto-loaded so the default backend "just works". Selection happens via `<category>.provider` in `config.yaml` (e.g. `image_gen.provider: openai`). |
| **Memory providers** (`plugins/memory/`) | All discovered; exactly one is active, chosen by `memory.provider` in `config.yaml`. |
| **Context engines** (`plugins/context_engine/`) | All discovered; one is active, chosen by `context.engine` in `config.yaml`. |
| **Model providers** (`plugins/model-providers/`) | All bundled providers under `plugins/model-providers/` discover and register at the first `get_provider_profile()` call. The user picks one at a time via `--provider` or `config.yaml`. |
| **Pip-installed `backend` plugins** | Opt-in via `plugins.enabled` (same as general plugins). |
| **User-installed platforms** (under `~/.hermes/plugins/platforms/`) | Opt-in via `plugins.enabled` — third-party gateway adapters need explicit consent. |

In short: **bundled "always-works" infrastructure loads automatically; third-party general plugins are opt-in.** The `plugins.enabled` allow-list is the gate specifically for arbitrary code a user drops into `~/.hermes/plugins/`.

### Migration for existing users

When you upgrade to a version of Hermes that has opt-in plugins (config schema v21+), any user plugins already installed under `~/.hermes/plugins/` that weren't already in `plugins.disabled` are **automatically grandfathered** into `plugins.enabled`. Your existing setup keeps working. Bundled standalone plugins are NOT grandfathered — even existing users have to opt in explicitly. (Bundled platform/backend plugins never needed grandfathering because they were never gated.)

## Available hooks

Plugins can register callbacks for these lifecycle events. See the **[Event Hooks page](/docs/user-guide/features/hooks#plugin-hooks)** for full details, callback signatures, and examples.

| Hook | Fires when |
|------|-----------|
| [`pre_tool_call`](/docs/user-guide/features/hooks#pre_tool_call) | Before any tool executes |
| [`post_tool_call`](/docs/user-guide/features/hooks#post_tool_call) | After any tool returns |
| [`pre_llm_call`](/docs/user-guide/features/hooks#pre_llm_call) | Once per turn, before the LLM loop — can return `{"context": "..."}` to [inject context into the user message](/docs/user-guide/features/hooks#pre_llm_call) |
| [`post_llm_call`](/docs/user-guide/features/hooks#post_llm_call) | Once per turn, after the LLM loop (successful turns only) |
| [`on_session_start`](/docs/user-guide/features/hooks#on_session_start) | New session created (first turn only) |
| [`on_session_end`](/docs/user-guide/features/hooks#on_session_end) | End of every `run_conversation` call + CLI exit handler |
| [`on_session_finalize`](/docs/user-guide/features/hooks#on_session_finalize) | CLI/gateway tears down an active session (`/new`, GC, CLI quit) |
| [`on_session_reset`](/docs/user-guide/features/hooks#on_session_reset) | Gateway swaps in a new session key (`/new`, `/reset`, `/clear`, idle rotation) |
| [`subagent_stop`](/docs/user-guide/features/hooks#subagent_stop) | Once per child after `delegate_task` finishes |
| [`pre_gateway_dispatch`](/docs/user-guide/features/hooks#pre_gateway_dispatch) | Gateway received a user message, before auth + dispatch. Return `{"action": "skip" \| "rewrite" \| "allow", ...}` to influence flow. |

## Plugin types

Hermes has four kinds of plugins:

| Type | What it does | Selection | Location |
|------|-------------|-----------|----------|
| **General plugins** | Add tools, hooks, slash commands, CLI commands | Multi-select (enable/disable) | `~/.hermes/plugins/` |
| **Memory providers** | Replace or augment built-in memory | Single-select (one active) | `plugins/memory/` |
| **Context engines** | Replace the built-in context compressor | Single-select (one active) | `plugins/context_engine/` |
| **Model providers** | Declare an inference backend (OpenRouter, Anthropic, …) | Multi-register, picked by `--provider` / `config.yaml` | `plugins/model-providers/` |

Memory providers and context engines are **provider plugins** — only one of each type can be active at a time. Model providers are also plugins, but many load simultaneously; the user picks one at a time via `--provider` or `config.yaml`. General plugins can be enabled in any combination.

## Pluggable interfaces — where to go for each

The table above shows the four plugin categories, but within "General plugins" the `PluginContext` exposes several distinct extension points — and Hermes also accepts extensions outside the Python plugin system (config-driven backends, shell-hooked commands, external servers, etc.). Use this table to find the right doc for what you want to build:

| Want to add… | How | Authoring guide |
|---|---|---|
| A **tool** the LLM can call | Python plugin — `ctx.register_tool()` | [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin) · [Adding Tools](/docs/developer-guide/adding-tools) |
| A **lifecycle hook** (pre/post LLM, session start/end, tool filter) | Python plugin — `ctx.register_hook()` | [Hooks reference](/docs/user-guide/features/hooks) · [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin) |
| A **slash command** for the CLI / gateway | Python plugin — `ctx.register_command()` | [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin) · [Extending the CLI](/docs/developer-guide/extending-the-cli) |
| A **subcommand** for `hermes <thing>` | Python plugin — `ctx.register_cli_command()` | [Extending the CLI](/docs/developer-guide/extending-the-cli) |
| A bundled **skill** that your plugin ships | Python plugin — `ctx.register_skill()` | [Creating Skills](/docs/developer-guide/creating-skills) |
| An **inference backend** (LLM provider: OpenAI-compat, Codex, Anthropic-Messages, Bedrock) | Provider plugin — `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/` | **[Model Provider Plugins](/docs/developer-guide/model-provider-plugin)** · [Adding Providers](/docs/developer-guide/adding-providers) |
| A **gateway channel** (Discord / Telegram / IRC / Teams / etc.) | Platform plugin — `ctx.register_platform()` in `plugins/platforms/<name>/` | [Adding Platform Adapters](/docs/developer-guide/adding-platform-adapters) |
| A **memory backend** (Honcho, Mem0, Supermemory, …) | Memory plugin — subclass `MemoryProvider` in `plugins/memory/<name>/` | [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) |
| A **context-compression strategy** | Context-engine plugin — `ctx.register_context_engine()` | [Context Engine Plugins](/docs/developer-guide/context-engine-plugin) |
| An **image-generation backend** (DALL·E, SDXL, …) | Backend plugin — `ctx.register_image_gen_provider()` | [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) |
| A **video-generation backend** (Veo, Kling, Pixverse, Grok-Imagine, Runway, …) | Backend plugin — `ctx.register_video_gen_provider()` | [Video Generation Provider Plugins](/docs/developer-guide/video-gen-provider-plugin) |
| A **TTS backend** (any CLI — Piper, VoxCPM, Kokoro, xtts, voice-cloning scripts, …) | Config-driven — declare under `tts.providers.<name>` with `type: command` in `config.yaml` | [TTS setup](/docs/user-guide/features/tts#custom-command-providers) |
| An **STT backend** (custom whisper binary, local ASR CLI) | Config-driven — set `HERMES_LOCAL_STT_COMMAND` env var to a shell template | [Voice Message Transcription (STT)](/docs/user-guide/features/tts#voice-message-transcription-stt) |
| **External tools via MCP** (filesystem, GitHub, Linear, Notion, any MCP server) | Config-driven — declare `mcp_servers.<name>` with `command:` / `url:` in `config.yaml`. Hermes auto-discovers the server's tools and registers them alongside built-ins. | [MCP](/docs/user-guide/features/mcp) |
| **Additional skill sources** (custom GitHub repos, private skill indexes) | CLI — `hermes skills tap add <repo>` | [Skills Hub](/docs/user-guide/features/skills#skills-hub) · [Publishing a custom tap](/docs/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **Gateway event hooks** (fire on `gateway:startup`, `session:start`, `agent:end`, `command:*`) | Drop `HOOK.yaml` + `handler.py` into `~/.hermes/hooks/<name>/` | [Event Hooks](/docs/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hooks** (run a shell command on events — notifications, audit logs, desktop alerts) | Config-driven — declare under `hooks:` in `config.yaml` | [Shell Hooks](/docs/user-guide/features/hooks#shell-hooks) |

:::note
Not everything is a Python plugin. Some extension surfaces intentionally use **config-driven shell commands** (TTS, STT, shell hooks) so any CLI you already have becomes a plugin without writing Python. Others are **external servers** (MCP) the agent connects to and auto-registers tools from. And some are **drop-in directories** (gateway hooks) with their own manifest format. Pick the right surface for the integration style that fits your use case; the authoring guides in the table above each cover placeholders, discovery, and examples.
:::

## NixOS declarative plugins

On NixOS, plugins can be installed declaratively via the module options — no `hermes plugins install` needed. See the **[Nix Setup guide](/docs/getting-started/nix-setup#plugins)** for full details.

```nix
services.hermes-agent = {
  # Directory plugin (source tree with plugin.yaml)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin (pip package)
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # Enable in config
  settings.plugins.enabled = [ "my-plugin" ];
};
```

Declarative plugins are symlinked with a `nix-managed-` prefix — they coexist with manually installed plugins and are cleaned up automatically when removed from the Nix config.

## Managing plugins

```bash
hermes plugins                                       # unified interactive UI
hermes plugins list                                  # table: enabled / disabled / not enabled
hermes plugins install user/repo                     # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable            # install AND enable (no prompt)
hermes plugins install user/repo --no-enable         # install but leave disabled (no prompt)
hermes plugins update my-plugin                      # pull latest
hermes plugins remove my-plugin                      # uninstall
hermes plugins enable my-plugin                      # add to allow-list (flat plugin)
hermes plugins enable observability/langfuse         # add to allow-list (sub-category plugin)
hermes plugins disable my-plugin                     # remove from allow-list + add to disabled
```

For plugins under a sub-category directory (e.g. `plugins/observability/langfuse/`, `plugins/image_gen/openai/`), use the full `<category>/<plugin>` key — that's exactly what `hermes plugins list` shows in the **Name** column.

### Interactive UI

Running `hermes plugins` with no arguments opens a composite interactive screen:

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]
   [ ] observability/langfuse — Trace turns / LLM calls / tools to Langfuse [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **General Plugins section** — checkboxes, toggle with SPACE. Checked = in `plugins.enabled`, unchecked = in `plugins.disabled` (explicit off).
- **Provider Plugins section** — shows current selection. Press ENTER to drill into a radio picker where you choose one active provider.
- Bundled plugins appear in the same list with a `[bundled]` tag.

Provider plugin selections are saved to `config.yaml`:

```yaml
memory:
  provider: "honcho"      # empty string = built-in only

context:
  engine: "compressor"    # default built-in compressor
```

### Enabled vs. disabled vs. neither

Plugins occupy one of three states:

| State | Meaning | In `plugins.enabled`? | In `plugins.disabled`? |
|---|---|---|---|
| `enabled` | Loaded on next session | Yes | No |
| `disabled` | Explicitly off — won't load even if also in `enabled` | (irrelevant) | Yes |
| `not enabled` | Discovered but never opted in | No | No |

The default for a newly-installed or bundled plugin is `not enabled`. `hermes plugins list` shows all three distinct states so you can tell what's been explicitly turned off vs. what's just waiting to be enabled.

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
