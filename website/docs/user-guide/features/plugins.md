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
[Adding Tools](/developer-guide/adding-tools) page is for built-in Hermes
core tools that live in `tools/` and `toolsets.py`.

**→ [Build a Hermes Plugin](/developer-guide/plugins)** — step-by-step guide with a complete working example.

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
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

Drop both files into `~/.hermes/plugins/hello-world/`, restart Hermes, and the model can immediately call `hello_world`. The hook prints a log line after every tool invocation.

The model-facing tool description belongs in `schema["description"]`. The optional `ctx.register_tool(description=...)` value is separate `ToolEntry` registry metadata: when omitted, it defaults to the schema description, but Hermes does not copy it back into a schema that lacks `description`. Prefer defining the text once in the schema. If you provide both values, keep them synchronized; the model sees the schema value.

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
| Inject messages | `ctx.inject_message(content, role="user", session_key=...)` - see [Injecting Messages](#injecting-messages) |
| Ship data files | `Path(__file__).parent / "data" / "file.yaml"` |
| Bundle skills | `ctx.register_skill(name, path)` — namespaced as `plugin:skill`, loaded via `skill_view("plugin:skill")` |
| Gate on env vars | `requires_env: [API_KEY]` in plugin.yaml — prompted during `hermes plugins install` |
| Distribute via pip | `[project.entry-points."hermes_agent.plugins"]` |
| Register a gateway platform (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — see [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| Register an image-generation backend | `ctx.register_image_gen_provider(provider)` — see [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| Register a video-generation backend | `ctx.register_video_gen_provider(provider)` — see [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| Register a context-compression engine | `ctx.register_context_engine(engine)` — see [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| Register a terminal execution backend (cloud sandbox) | `ctx.register_terminal_environment_provider(provider)` — see [Terminal Environment Plugins](/developer-guide/terminal-environment-plugin) |
| Route human approval prompts | `ctx.register_approval_transport(name, present_fn)` — see [Approval transports](#approval-transports) |
| Register a memory backend | Subclass `MemoryProvider` in `plugins/memory/<name>/__init__.py` — see [Memory Provider Plugins](/developer-guide/memory-provider-plugin) (uses a separate discovery system) |
| Run a host-owned LLM call | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — borrow the user's active model + auth for a one-shot completion with optional JSON schema validation. See [Plugin LLM Access](/developer-guide/plugin-llm-access) |
| Call an MCP tool (capability-gated) | `ctx.call_mcp(server, tool, arguments, timeout=30)` — see [Calling MCP servers from plugins](#calling-mcp-servers-from-plugins) |
| Register an inference backend (LLM provider) | `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/__init__.py` — see [Model Provider Plugins](/developer-guide/model-provider-plugin) (uses a separate discovery system) |

## Plugin discovery

| Source | Path | Use case |
|--------|------|----------|
| Bundled | `<repo>/plugins/` | Ships with Hermes — see [Built-in Plugins](/user-guide/features/built-in-plugins) |
| User | `~/.hermes/plugins/` | Personal plugins |
| Project | `.hermes/plugins/` | Project-specific plugins (requires `HERMES_ENABLE_PROJECT_PLUGINS=true`) |
| pip | `hermes_agent.plugins` entry_points | Distributed packages |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS declarative installs — see [Nix Setup](/getting-started/nix-setup#plugins) |

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

## Plugins are opt-in (with a few exceptions)

**General plugins and user-installed backends are disabled by default** — discovery finds them (so they show up in `hermes plugins` and `/plugins`), but nothing with hooks or tools loads until you add the plugin's name to `plugins.enabled` in `~/.hermes/config.yaml`. This stops third-party code from running without your explicit consent.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # optional deny-list — always wins if a name appears in both
    - noisy-plugin
  # Optional: wall-clock cap (seconds) for timeout-bounded in-process Python
  # plugin hook callbacks (hot-path observers + pre_tool_call). Default 30;
  # set 0 to disable; values above 600 are clamped. Timed-out pre_tool_call
  # callbacks fail closed (block the tool). Caller-thread hooks such as
  # subagent_stop are never moved onto a timeout worker.
  # Shell hooks keep their own per-entry timeout under the top-level hooks: key.
  hook_callback_timeout: 30
```

Three ways to flip state:

```bash
hermes plugins                    # interactive toggle (space to check/uncheck)
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove from allow-list + add to disabled
```

After `hermes plugins install owner/repo`, you're asked `Enable 'name' now? [y/N]` — defaults to no. Skip the prompt for scripted installs with `--enable` or `--no-enable`.

For a reproducible install, pin a full immutable commit (tags, branches, and
abbreviated SHAs are not accepted):

```bash
hermes plugins install owner/repo --ref 0123456789abcdef0123456789abcdef01234567
```

Hermes checks out the commit detached, verifies that `HEAD` exactly matches the
requested SHA, and records the canonical source, installed revision, and pin
status in the current profile. `hermes plugins update` refuses to move a pinned
plugin; choose a new exact commit explicitly with
`hermes plugins install <source> --force --ref <new-commit>`. The
profile-local install metadata contains no config values, environment values,
secrets, or capability grants.

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

### Approval transports

An approval transport changes **where a human sees and answers** an existing
Hermes tool-approval request. It does not decide whether a command needs
approval and it is not an authorization-policy API.

```python
def present(request):
    # Deliver request.command and request.description to your UI, wait for
    # its authenticated human response, then return a request-bound decision.
    choice = send_to_my_ui_and_wait(request)  # once/session/always/deny
    return request.respond(choice)


def register(ctx):
    ctx.register_approval_transport("my-ui", present)
```

`present` may be synchronous or async. Hermes runs it on a bounded worker and
enforces the canonical `approvals.timeout` even if the plugin does not. The
request is immutable and contains redacted display text, its host presentation
class (`cli` or `gateway`), the host timeout, allowed choices, and an opaque
request ID/digest.
Return the result of
`request.respond(choice)`; unbound dictionaries and stale or changed request
IDs/digests are rejected. A plugin cannot return a scope that the host did not
offer (for example, `always` on a once-only request).

Registration alone does nothing. Enabling the plugin and explicitly selecting
its transport are separate consent steps:

```yaml
plugins:
  enabled: [my-approval-plugin]

security:
  approval:
    transport: my-ui
    transport_fallback: deny     # default
```

Transport exceptions, timeouts, unavailable registrations, invalid choices,
and stale responses deny by default. To deliberately show the prompt on the
ordinary CLI/TUI/gateway/ACP surface when the selected transport fails, set
`transport_fallback: builtin`. Without that exact opt-in, Hermes never
materializes the prompt on another surface.

Hermes still owns hardline blocks, sudo-stdin protection, user deny rules,
request binding, allowed scopes, persistence, hooks, and final authorization.
Hardline commands are blocked before any transport callback. There is
intentionally **no plugin approval policy, auto-allow callback, or required
`pre_tool_call` policy** in this interface. A future approval-policy capability
may use the plugin capability-consent model, but transport selection does not
grant it.

### Migration for existing users

When you upgrade to a version of Hermes that has opt-in plugins (config schema v21+), any user plugins already installed under `~/.hermes/plugins/` that weren't already in `plugins.disabled` are **automatically grandfathered** into `plugins.enabled`. Your existing setup keeps working. Bundled standalone plugins are NOT grandfathered — even existing users have to opt in explicitly. (Bundled platform/backend plugins never needed grandfathering because they were never gated.)

## Available hooks

Plugins can register the 26 lifecycle events currently accepted by `hermes_cli.plugins.VALID_HOOKS`. The **[Event Hooks catalog](/user-guide/features/hooks#shipped-plugin-hook-catalog)** is canonical for exact timing, return handling, payload fields, and privacy notes.

| Descriptive category | Shipped hooks |
|---|---|
| **Directive/control** | `pre_tool_call`, `pre_llm_call`, `pre_verify`, `pre_gateway_dispatch` |
| **Transform** | `transform_tool_result`, `transform_terminal_output`, `transform_llm_output`, `pre_transcription`, `post_transcription` |
| **Observer** | `post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`, `api_request_error`, `on_stream_start`, `on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `on_skill_lifecycle`, `subagent_start`, `subagent_stop`, `pre_approval_request`, `post_approval_response`, `pre_command`, `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked` |

These categories describe current behavior rather than defining future naming rules. Plugin middleware remains a separate registry/surface.
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
| A **tool** the LLM can call | Python plugin — `ctx.register_tool()` | [Build a Hermes Plugin](/developer-guide/plugins) · [Adding Tools](/developer-guide/adding-tools) |
| A **lifecycle hook** (pre/post LLM, session start/end, tool filter) | Python plugin — `ctx.register_hook()` | [Hooks reference](/user-guide/features/hooks) · [Build a Hermes Plugin](/developer-guide/plugins) |
| A **slash command** for the CLI / gateway | Python plugin — `ctx.register_command()` | [Build a Hermes Plugin](/developer-guide/plugins) · [Extending the CLI](/developer-guide/extending-the-cli) |
| A **subcommand** for `hermes <thing>` | Python plugin — `ctx.register_cli_command()` | [Extending the CLI](/developer-guide/extending-the-cli) |
| A bundled **skill** that your plugin ships | Python plugin — `ctx.register_skill()` | [Creating Skills](/developer-guide/creating-skills) |
| An **inference backend** (LLM provider: OpenAI-compat, Codex, Anthropic-Messages, Bedrock) | Provider plugin — `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/` | **[Model Provider Plugins](/developer-guide/model-provider-plugin)** · [Adding Providers](/developer-guide/adding-providers) |
| A **gateway channel** (Discord / Telegram / IRC / Teams / etc.) | Platform plugin — `ctx.register_platform()` in `plugins/platforms/<name>/` | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| A **memory backend** (Honcho, Mem0, Supermemory, …) | Memory plugin — subclass `MemoryProvider` in `plugins/memory/<name>/` | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| A **context-compression strategy** | Context-engine plugin — `ctx.register_context_engine()` | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| An **image-generation backend** (DALL·E, SDXL, …) | Backend plugin — `ctx.register_image_gen_provider()` | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| A **video-generation backend** (Veo, Kling, Pixverse, Grok-Imagine, Runway, …) | Backend plugin — `ctx.register_video_gen_provider()` | [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| A **TTS backend** (any CLI — Piper, VoxCPM, Kokoro, xtts, voice-cloning scripts, …) | Config-driven (recommended) — declare under `tts.providers.<name>` with `type: command` in `config.yaml`. OR Python backend plugin — `ctx.register_tts_provider()` for Python-SDK / streaming engines that need more than a shell template. | [TTS Setup](/user-guide/features/tts#custom-command-providers) · [Python plugin guide](/user-guide/features/tts#python-plugin-providers) |
| An **STT backend** (any CLI — whisper.cpp, custom whisper binary, local ASR CLI) | Config-driven (recommended) — declare under `stt.providers.<name>` with `type: command` in `config.yaml`, or set `HERMES_LOCAL_STT_COMMAND` for the legacy single-command escape hatch. OR Python backend plugin — `ctx.register_transcription_provider()` for Python-SDK engines (OpenRouter, SenseAudio, Gemini-STT, etc.). | [STT Setup](/user-guide/features/tts#stt-custom-command-providers) · [Python plugin guide](/user-guide/features/tts#python-plugin-providers-stt) |
| **External tools via MCP** (filesystem, GitHub, Linear, Notion, any MCP server) | Config-driven — declare `mcp_servers.<name>` with `command:` / `url:` in `config.yaml`. Hermes auto-discovers the server's tools and registers them alongside built-ins. | [MCP](/user-guide/features/mcp) |
| **Additional skill sources** (custom GitHub repos, private skill indexes) | CLI — `hermes skills tap add <repo>` | [Skills Hub](/user-guide/features/skills#skills-hub) · [Publishing a custom tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **Gateway event hooks** (fire on `gateway:startup`, `session:start`, `agent:end`, `command:*`) | Drop `HOOK.yaml` + `handler.py` into `~/.hermes/hooks/<name>/` | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hooks** (run a shell command on events — notifications, audit logs, desktop alerts) | Config-driven — declare under `hooks:` in `config.yaml` | [Shell Hooks](/user-guide/features/hooks#shell-hooks) |

:::note
Not everything is a Python plugin. Some extension surfaces intentionally use **config-driven shell commands** (TTS, STT, shell hooks) so any CLI you already have becomes a plugin without writing Python. Others are **external servers** (MCP) the agent connects to and auto-registers tools from. And some are **drop-in directories** (gateway hooks) with their own manifest format. Pick the right surface for the integration style that fits your use case; the authoring guides in the table above each cover placeholders, discovery, and examples.
:::

## NixOS declarative plugins

On NixOS, plugins can be installed declaratively via the module options — no `hermes plugins install` needed. See the **[Nix Setup guide](/getting-started/nix-setup#plugins)** for full details.

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
hermes plugins                               # unified interactive UI
hermes plugins list                          # table: enabled / disabled / not enabled
hermes plugins search <term>                 # search the community plugin index
hermes plugins install <name>                # install by index name (resolved to repo @ pinned ref)
hermes plugins install user/repo             # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable    # install AND enable (no prompt)
hermes plugins install user/repo --no-enable # install but leave disabled (no prompt)
hermes plugins update my-plugin              # pull latest (local edits are autostashed and re-applied)
hermes plugins remove my-plugin              # uninstall
hermes plugins enable my-plugin              # add to allow-list
hermes plugins disable my-plugin             # remove from allow-list + add to disabled
hermes plugins capabilities [my-plugin]      # declared vs granted capabilities
```

### One-click install links (Desktop)

Hermes Desktop registers the `hermes://` URL scheme, so a website, README, or
chat message can link straight to a plugin install:

```
hermes://plugin/install?repo=owner/repo            # main install link
hermes://plugin/install?repo=owner/repo&enable=1   # enable the agent plugin after install
hermes://plugin/install?repo=owner/repo&force=1    # replace an existing install
```

Clicking one opens Hermes and shows a **confirmation dialog** — the repo id,
a "Before you install" note, and GitHub browse + clone links — then
shallow-clones the repo to detect what it ships (an **agent plugin** —
backend Python, a **desktop plugin** — app UI, or both). You pick the
components with checkboxes and confirm. Nothing is installed until you do;
deep links never auto-install, and agent-plugin installs go through the same
[install-time security scanning](#install-time-security-scanning) as
`hermes plugins install`.

Hybrid repos (agent + desktop halves in one repo) use one link and one
dialog. The same modal is reachable without a link via **Settings → Plugins →
Install from Git**. Legacy `hermes://plugin-agent/…` and
`hermes://plugin-desktop/…` URLs route into the same dialog. In dev builds
(`npm run dev`) the scheme is `hermes-dev://`.

Websites need no SDK — a normal anchor works:

```html
<a href="hermes://plugin/install?repo=owner/repo&enable=1">Install in Hermes</a>
```

MCP servers have the equivalent link form — see
[Add to Hermes link](/reference/mcp-config-reference#add-to-hermes-link).

### Plugin capabilities and consent

Plugins can declare the privileged host surfaces they want in their
`plugin.yaml`:

```yaml
name: my-plugin
capabilities:
  - tools.override        # replace built-in tools
  - llm.model_override    # pick the model for host-owned LLM calls
```

When a plugin declares capabilities, `hermes plugins install` (and
`hermes plugins enable`) shows the list with one-line risk descriptions and
asks once. Consenting records the grant under
`plugins.entries.<id>.granted_capabilities` together with a consent hash and
timestamp. Declining leaves the plugin enabled with those capabilities off —
a well-behaved plugin probes with `ctx.has_capability()` and degrades
gracefully.

**Update re-consent:** if a plugin update declares capabilities you haven't
granted, `hermes plugins update` surfaces the additions and asks again. New
capabilities stay off until you consent — a plugin update can never silently
widen its access.

**Non-interactive sessions fail closed:** installing or updating without a
TTY completes the install, but declared capabilities are *not* granted. Run
`hermes plugins enable <id>` interactively to grant them later.

Inspect the state at any time:

```bash
hermes plugins capabilities             # all plugins with declared/granted capabilities
hermes plugins capabilities my-plugin   # one plugin, declared vs granted
```

Capability ids map 1:1 to the older per-feature config gates, which keep
working but are **deprecated** in favor of the consent flow:

| Capability | Legacy key (`plugins.entries.<id>.…`) |
|---|---|
| `tools.override` | `allow_tool_override` |
| `llm.provider_override` | `llm.allow_provider_override` |
| `llm.model_override` | `llm.allow_model_override` |
| `llm.agent_id_override` | `llm.allow_agent_id_override` |
| `llm.profile_override` | `llm.allow_profile_override` |
| `llm.task_override` | `llm.allow_task_override` |
| `gateway.platform_actions` | `allow_platform_actions` |

A gate is open when *either* the capability is granted *or* the legacy key is
set — existing configs keep working unchanged.

:::warning Not a sandbox
Capabilities are a **consent and audit layer**, not isolation. Plugins run as
regular in-process Python: a malicious plugin can ignore every gate here.
Granting a capability is a statement of trust in the plugin author — it is
not a code audit, and Hermes has not reviewed the plugin's code. Only install
plugins from sources you trust.
:::

### Platform actions

`ctx.platform_actions` gives a plugin a minimal, capability-gated verb set for
acting on connected chat platforms through the live gateway adapter registry —
the sanctioned alternative to monkeypatching an adapter. **It is off by
default**: every call re-checks the `gateway.platform_actions` capability
(legacy key `plugins.entries.<id>.allow_platform_actions`), and an ungranted
call returns a structured error instead of acting.

v1 verbs (both `async`, both return a plain dict, and neither ever raises into
hook dispatch):

```python
result = await ctx.platform_actions.add_reaction(
    platform="telegram", chat_id="-100123", message_id="456", emoji="👍",
)
result = await ctx.platform_actions.set_thread_title(
    platform="discord", chat_id="123", thread_id="456", title="New title",
)
if not result["ok"]:
    print(result["error"], result.get("detail"))
```

Success is `{"ok": True, "action": <verb>}`. Failures are
`{"ok": False, "error": <code>, "detail": <str>}` with stable error codes:
`capability_not_granted`, `invalid_argument`, `gateway_unavailable`,
`unknown_platform`, `adapter_not_registered`, `adapter_disconnected`,
`unsupported_platform_action`, `action_failed`. Actions validate that the
target adapter exists and is connected before acting; a disconnected or
missing adapter degrades to a structured error, never an exception.

Platforms supported in v1: Telegram and Discord. Telegram's `add_reaction`
*sets* the bot's reaction (the Bot API replaces a previous bot reaction rather
than stacking). Every action — allowed or denied — is written to the log with
the plugin id, verb, platform, and outcome.

:::warning Security note
Platform actions are a **messaging-as-the-bot power**: a granted plugin can
react and rename threads in any chat the gateway bot can reach, not just the
chat that triggered the hook. Grant `gateway.platform_actions` only to plugins
you trust, and prefer plugins that document exactly which actions they take.
Raw platform SDK payload/handle access is deliberately **not** part of this
surface — per the #64176 round-2 design correction it requires its own
capability (`gateway.raw_events`) with a "no stability guarantee" label and a
separate design, and has not shipped.
:::

### Discovering community plugins

`hermes plugins search <term>` searches the **community plugin index** — a
static, machine-readable JSON catalog of community plugins. Matching is fuzzy
across name, description, and tags:

```bash
hermes plugins search telegram               # fuzzy search
hermes plugins search                        # browse the whole index
hermes plugins search --capability platform  # filter by declared capability
hermes plugins search media --json           # machine-readable output
hermes plugins search --refresh              # bypass the 24h local cache
```

Once you've found a plugin, install it by bare name — the name is resolved
through the index to its `owner/repo` plus the index-pinned commit:

```bash
hermes plugins install hermes-media-studio
```

If a name matches more than one entry, the candidates are listed and nothing
is installed. Explicit `owner/repo` or Git-URL identifiers never touch the
index and keep working exactly as before. An explicit `--ref <sha>` always
overrides the index pin.

**How the index is fetched.** The index lives at a canonical URL
(`https://raw.githubusercontent.com/NousResearch/hermes-plugin-index/main/index.json`,
overridable via `hermes config set plugins.index_url <url>`). Fetches are
cached under `~/.hermes/cache/plugin_index.json` for 24 hours; when the
remote is unreachable the stale cache is used, and when there is no cache at
all a bundled seed copy ships with Hermes — so search works fully offline.

**Index entry format.** Each entry is a JSON object:

```json
{
  "name": "hermes-media-studio",
  "description": "Generative media workspace plugin.",
  "author": "NousResearch",
  "tags": ["media", "image-gen"],
  "repo": "NousResearch/hermes-media-studio",
  "ref": "<40-char commit SHA>",
  "subdir": null,
  "homepage": "https://github.com/NousResearch/hermes-media-studio",
  "capabilities": ["tools", "dashboard"],
  "api_version": 1,
  "added_at": "2026-08-12"
}
```

`repo` is the `owner/name` GitHub identifier, `ref` pins an immutable commit
SHA, and optional `subdir` supports monorepos. The bundled seed file
(`hermes_cli/data/plugin_index.json` in the repo) is the format reference.

**Submitting a plugin.** The index is maintained as a plain JSON file —
submit a pull request to the
[hermes-plugin-index](https://github.com/NousResearch/hermes-plugin-index)
repository adding your entry (name, description, author, tags, `owner/repo`,
and a pinned commit SHA). Review covers the entry's *metadata* only.

:::warning Indexed ≠ audited
Inclusion in the community index means the entry's metadata was reviewed —
**it is not a code audit**. Installing still goes through the normal
consent/review flow (plugins install disabled by default, enabling is an
explicit step, and tool-override rights require a separate grant). Review a
plugin's source before enabling it.
:::

### Plugin packs

A **plugin pack** is a declarative, shareable YAML file (`hermes-pack.yaml`)
that pins a set of plugins — like sharing a modpack. Installing a pack fans
out to ordinary pinned installs; nothing new exists at runtime.

```yaml
name: voice-assistant-pack
description: STT + streaming TTS + approval relay
author: hyper
version: 1.0.0
plugins:
  - name: hermes-media-studio            # bare community-index name…
    ref: e8d59971d2b7901405b39dac7b03bdd616272d0d
  - repo: owner/approval-relay           # …or explicit owner/repo (or git URL)
    ref: 8f3c2d1a9b4e5f6071829304a5b6c7d8e9f00112
    subdir: plugins/relay                # optional monorepo path
config:                                  # optional, non-secret seeds only
  hermes-media-studio:
    default_model: flux-3
skills: []                               # declared list only (not auto-installed yet)
```

```bash
hermes plugins pack show ./hermes-pack.yaml     # dry-run review
hermes plugins pack install ./hermes-pack.yaml  # review → confirm → install
hermes plugins pack export > hermes-pack.yaml   # snapshot the current install
hermes plugins pack export --enabled-only       # only plugins.enabled
```

**Supply-chain posture.** Every entry's `ref` must be an exact 40-character
commit SHA — tags and branch names are rejected with an error naming the
entry, the same rule as the community index. Pack installs ride the exact
same pinned install path as `hermes plugins install --ref <sha>` and record
the same provenance in `plugins/.install-metadata.json`, so two installs of
the same pack resolve identically. Packs build on the
[manifest v2 fields](/developer-guide/plugins) (`manifest_version`,
`api_version`, `requires_plugins`) — each plugin's own manifest still
validates through the normal install path.

**Consent is never bulk-granted.** `pack install` shows a mandatory review
screen (every plugin, source, pinned ref, and the capabilities it declares),
then asks **one** confirmation for the pack contents. After that, each
plugin's declared capabilities go through the standard per-plugin
capability-consent prompt — identical to a single `hermes plugins install`.
There is no `--yes`, and non-interactive sessions cannot install packs.

**Secrets never travel in packs.** `config:` seeds are limited to
non-secret `plugins.entries.<id>` keys — secret-shaped key names
(`*token*`, `*key*`, `*password*`, …), capability grants, and the deprecated
`allow_*` trust gates are rejected on install and stripped on export.
Plugins that need secrets declare them in their own `requires_env`, which
prompts during install as usual. Existing user values in
`plugins.entries.<id>` always win over pack seeds.

**Partial failure.** Each plugin installs independently; failures are
reported per plugin, the rest continue, and the command exits non-zero if
any plugin failed.

**Export caveats.** `pack export` only includes plugins with known Git
provenance (installed via `hermes plugins install`). Local-only plugins are
listed as warning comments in the emitted YAML, not as installable entries.

The `skills:` list is parsed and displayed at install time but not yet
auto-installed — install those manually for now (`hermes skills`). Wiring
skill-hub ids into pack install is a documented follow-up seam.

### Install-time security scanning

Every `hermes plugins install` and `hermes plugins update` runs a static
security scan over the plugin tree before it is activated (inspired by
Claude Cowork's skill & plugin security scanning). The scanner reuses the
same threat-pattern engine as the [Skills Hub guard](/user-guide/features/skills)
— exfiltration of credential stores, reverse shells, destructive commands,
persistence mechanisms, obfuscated execution, and prompt injection in
documentation files — with plugin-aware exemptions: a provider plugin
reading its **own** API key from the environment (the documented
`requires_env` pattern) is not flagged.

Three verdicts, matching Cowork's pass/warn/fail:

| Verdict | Behavior |
|---|---|
| **safe** | Installs normally, no extra output |
| **caution** | Findings are shown; you confirm `Install anyway? [y/N]` (or pass `--force`) |
| **dangerous** | Blocked. `--force` does **not** override |

On `hermes plugins update`, a dangerous verdict on the updated tree
disables the plugin until you review the findings and re-enable it.

Scanning is on by default; disable it in `config.yaml`:

```yaml
plugins:
  scan_on_install: false
```

### Interactive UI

Running `hermes plugins` with no arguments opens a composite interactive screen:

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]

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

Plugins can inject messages into a CLI conversation or a known gateway session using `ctx.inject_message()`:

```python
# Active CLI conversation
ctx.inject_message("New data arrived from the webhook", role="user")

# Existing gateway conversation
ctx.inject_message(
    "New data arrived from the webhook",
    role="user",
    session_key="agent:main:telegram:dm:123456789",
)
```

**Signature:** `ctx.inject_message(content: str, role: str = "user", *, session_key: str | None = None) -> bool`

In CLI mode:

- If the agent is **idle** (waiting for user input), the message is queued as the next input and starts a new turn.
- If the agent is **mid-turn** (actively running), the message interrupts the current operation — the same as a user typing a new message and pressing Enter.
- For non-`"user"` roles, the content is prefixed with `[role]` (e.g. `[system] ...`).
- Returns `True` if the message was queued successfully.

In gateway mode:

- `session_key` is required and must identify an existing gateway session. It is the stable routing key, not the CLI session ID.
- Hermes reuses that session's stored platform, chat, thread, profile, and conversation history. Plugins cannot supply a new chat route through this API.
- Hermes rechecks the stored route against the gateway's current authorisation rules before dispatch.
- Routes that relied only on an adapter-time or upstream authorisation decision are rejected unless Hermes can revalidate them from current core allowlists, pairing, or explicit allow-all configuration.
- Injected text is always conversational input. It cannot invoke slash commands, approve tools, or resolve pending confirmation and clarification prompts.
- The route and conversation are pinned while dispatch is pending. Hermes drops the request if topic recovery changes the route or the session rotates before handling starts.
- The request enters the platform adapter's normal message path. Active sessions use the existing busy-session queue rather than starting a competing turn.
- Returns `True` when the live gateway accepts the request for asynchronous dispatch. This does not confirm that the agent turn or platform delivery has completed.
- Returns `False` when `session_key` is omitted, the permission is not granted, or no live gateway can accept the request. Unknown or unroutable session keys discovered after asynchronous acceptance are written to the gateway log.

This enables plugins like remote control viewers, messaging bridges, or webhook receivers to feed messages into the conversation from external sources.

Gateway injection can send an agent response to an external messaging platform. It is disabled by default for every plugin. Grant it per plugin in `config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:
      allow_gateway_injection: true
```

:::warning
Only grant gateway injection to plugins you trust. Hermes checks this host API permission and restricts it to existing session routes, but Python plugins run in-process and this setting is not a sandbox.
:::

:::note
This plugin API does not expose a public HTTP endpoint or CLI command for external processes. The plugin must already know the target gateway `session_key`, for example from its own trusted configuration or previously retained session state.
:::

## Calling MCP servers from plugins

`ctx.call_mcp()` lets a plugin call a tool on one of the user's configured MCP servers — synchronously, from any hook or tool handler — routing through Hermes' existing native MCP client (same connections, trust-tier gates, circuit breaker, and reconnect logic as model-invoked MCP tools; never a parallel client).

```python
result = ctx.call_mcp(
    "knowledge_rag",            # server name from mcp.servers
    "query_knowledge",          # tool on that server
    {"query": "deploy runbook"},
    timeout=30,                 # seconds; clamped to 1–600
)
if result["ok"]:
    print(result["result"])
else:
    print("MCP error:", result["error"])
```

**Signature:** `ctx.call_mcp(server: str, tool: str, arguments: dict | None = None, timeout: float = 30) -> dict`

Returns a stable envelope: `{"ok": True, "result": ...}` (plus `structuredContent` when the server provides it) or `{"ok": False, "error": "..."}`. Results over ~64 KB are truncated and flagged with `"truncated": True`.

### Security: default-off, per-server allowlist

A plugin has **no MCP access by default**. The operator must grant each server explicitly in `config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:
      mcp_allowlist: ["knowledge_rag", "github"]
```

- Calling a server not in the list raises `PermissionError` naming the exact config key to set.
- The grant is per-server and per-plugin — never ambient authority over every configured server, and `"*"` wildcards are not honored.
- Every call has an enforced timeout (default 30 s) so a hung MCP server cannot stall the hook or tool pipeline that invoked it.
- MCP servers return untrusted content. Treat `result` as data, not instructions — don't feed it into privileged decisions (approvals, command execution) without validation.

:::warning
Granting `mcp_allowlist` gives the plugin the same access to that MCP server as the model has — including any write-capable tools the server exposes (subject to the server's `trust` tier gates). Grant only servers the plugin genuinely needs.
:::

See the **[full guide](/developer-guide/plugins)** for handler contracts, schema format, hook behavior, error handling, and common mistakes.
