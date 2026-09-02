---
sidebar_label: "Build a Plugin"
slug: /developer-guide/plugins
title: "Build a Hermes Plugin"
description: "Step-by-step guide to building a complete Hermes plugin with tools, hooks, data files, and skills"
---

# Build a Hermes Plugin

This guide walks through building a complete Hermes plugin from scratch. By the end you'll have a working plugin with multiple tools, lifecycle hooks, shipped data files, and a bundled skill — everything the plugin system supports.

:::info Not sure which guide you need?
Hermes has several distinct pluggable interfaces — some use Python `register_*` APIs, others are config-driven or drop-in directories. Use this map first:

| If you want to add… | Read |
|---|---|
| Custom tools, hooks, slash commands, skills, or CLI subcommands | **This guide** (the general plugin surface) |
| A **native desktop app** extension (panes, pages, status bar, palette, themes) | [Desktop Plugin SDK](/developer-guide/desktop-plugin-sdk) |
| A **web dashboard** extension (tabs, shell slots, themes) | [Extending the Dashboard](/user-guide/features/extending-the-dashboard) |
| An **LLM / inference backend** (new provider) | [Model Provider Plugins](/developer-guide/model-provider-plugin) |
| A **gateway channel** (Discord/Telegram/IRC/Teams/etc.) | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| A **memory backend** (Honcho/Mem0/Supermemory/etc.) | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| A **context-compression engine** | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| An **image-generation backend** | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| A **video-generation backend** | [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| A **web-search / extract backend** | [Web Search Provider Plugins](/developer-guide/web-search-provider-plugin) |
| A **cloud browser backend** (Browserbase-style CDP session provider) | [Browser Provider Plugins](/developer-guide/browser-provider-plugin) |
| A **secret-manager backend** (vault / password manager / OS keystore) | [Secret Source Plugins](/developer-guide/secret-source-plugin) |
| A **dashboard OIDC/auth provider** | [Web Dashboard — custom providers](/user-guide/features/web-dashboard#custom-providers) — `ctx.register_dashboard_auth_provider()` |
| A **TTS backend** (any CLI — Piper, VoxCPM, Kokoro, voice cloning, …) | [TTS custom command providers](/user-guide/features/tts#custom-command-providers) — config-driven, no Python needed |
| An **STT backend** (custom whisper / ASR CLI) | [Voice Message Transcription](/user-guide/features/tts#voice-message-transcription-stt) — set `HERMES_LOCAL_STT_COMMAND` to an argv-tokenized template |
| A **realtime speech-to-speech provider** (Gemini Live, xAI, a self-hosted duplex model, …) | [Register a realtime voice provider](#register-a-realtime-voice-provider) — `ctx.register_realtime_voice_provider(provider)`; driven by `hermes realtime --provider <name>` |
| **External tools via MCP** (filesystem, GitHub, Linear, any MCP server) | [MCP](/user-guide/features/mcp) — declare `mcp_servers.<name>` in `config.yaml` |
| **Gateway event hooks** (fire on startup, session events, commands) | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) — drop `HOOK.yaml` + `handler.py` into `~/.hermes/hooks/<name>/` |
| **Shell hooks** (run a shell command on events) | [Shell Hooks](/user-guide/features/hooks#shell-hooks) — declare under `hooks:` in `config.yaml` |
| **Additional skill sources** (custom GitHub repos, private skill indexes) | [Skills](/user-guide/features/skills) — `hermes skills tap add <repo>` · [Publishing a tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| A first-class **core** inference provider (not a plugin) | [Adding Providers](/developer-guide/adding-providers) |

See the full [Pluggable interfaces table](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each) for a consolidated view of every extension surface including config-driven (TTS, STT, MCP, shell hooks) and drop-in directory (gateway hooks) styles.
:::

:::caution Third-party-product plugins ship standalone — not into the core tree
Plugins that integrate **someone else's product or project** — observability/metrics backends, vendor SaaS connectors, analytics dashboards, paid-service tie-ins — are built and distributed as **standalone plugin repos**, not merged into `NousResearch/hermes-agent`. Users install them into `~/.hermes/plugins/` or via a pip entry point; everything in this guide works the same way from a standalone repo. This is a coupling-and-maintenance decision (the core moves fast and we don't own your backend), not a quality bar — a plugin can be excellent and still belong in its own repo. Promote it in the Nous Research Discord `#plugins-skills-and-skins` channel. See [CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) for the policy.
:::

## Portable Agent Plugins v1 packages

Hermes can also install and load directory packages that target the Agent
Plugins v1.0.0 format. This is a compatibility adapter for the portable
components Hermes already owns. It does not replace native `plugin.yaml` plus
`register(ctx)` plugins.

```text
my-portable-plugin/
├── plugin.json
├── skills/
│   └── summarize/
│       ├── SKILL.md
│       └── references/
└── mcp.json
```

Install and activate a portable package through the normal workflow:

```bash
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
```

Portable packages are disabled after installation unless you explicitly enable
them. An enabled package may provide immediate `skills/*/SKILL.md` directories
and stdio MCP servers from root `mcp.json`. Skills are read-only, namespaced,
and loaded through `skills_list` plus `skill_view`. MCP commands are passed as
one executable token with a separate argument list, never through a shell.
Use `skills_list` to discover the full qualified skill name. Portable skill
namespaces have the deterministic form `agent-plugin-<slug>-<hash>`, derived
from the discovered plugin key so sanitized names cannot collide.

Hermes validates `plugin.json`, Agent Skills frontmatter, fixed component
locations, `mcp.json`, resolved paths, and symlink containment locally. It does
not fetch JSON schemas while loading a package. A bad skill or MCP entry is
skipped at its own boundary when valid sibling components can still load.
`PLUGIN_ROOT` points to the resolved package root. `PLUGIN_DATA` points to a
profile-scoped writable directory managed by Hermes.
Values declared in portable MCP `env` are visible package data, not a secret
storage mechanism. Do not place credentials in `mcp.json`.

The current portable subset supports stdio and Streamable HTTP MCP entries.
Portable `streamable-http` entries are routed through Hermes' existing native
remote MCP client (the same runtime that powers URL-based `mcp_servers`
config), with the v1 boundary rules enforced: the URL must be absolute
http(s) with no user information or fragment, plain HTTP is accepted only
for `localhost`/loopback hosts, and configured headers are never forwarded
across a cross-origin redirect. Legacy `sse` entries are reported and
skipped. Agent Plugins v1 does not define trust, permissions, provenance, or a
sandbox. Enabling a package grants its instructions and local executable the
same full-trust posture as other installed Hermes plugins.

The [rendered specification](https://agent-plugins.org/specification) currently
labels v1.0.0 a Working Draft, while the
[versioned specification repository](https://github.com/agentplugins/agent-plugins-spec/blob/main/spec/1.0.0.md)
records it as Published. Hermes keys behavior on the canonical v1.0.0 schema
identifiers and normative text, not either mutable status label. This is an
explicit supported subset, not a claim of full Agent Plugins conformance.

## Native plugin compatibility contract

Native `plugin.yaml` plus `register(ctx)` plugins are protected by behavior,
not by one global plugin API number. Hermes does not expose a
`PLUGIN_API_VERSION`, require a manifest-wide `api:` match, or attach an API
version to unrelated values. A plugin that uses a documented behavior should
continue to work after a normal Hermes upgrade.

The compatibility rules are:

- **Evolve additively.** Documented `PluginContext` methods are not removed or
  renamed. New parameters are optional, have defaults, and should be
  keyword-only. Existing return fields are not removed or silently retyped.
- **Hook payloads are keyword payloads.** New hook data is added as keyword
  fields, never by changing the meaning or position of an existing field.
  Hermes inspects callback signatures: a legacy callback receives the fields it
  declares, while a callback with `**kwargs` receives the complete current
  payload. New plugins should accept `**kwargs` so they can opt into additive
  data without another signature change.
- **Manifests are open to additions.** Unknown `plugin.yaml` fields are ignored.
  Older Hermes releases can therefore load a plugin whose manifest contains
  metadata introduced by a newer release, provided the plugin code itself uses
  supported runtime behavior.
- **Provider interfaces grow through defaults.** New provider methods have a
  default implementation. New callback context is optional and forwarded only
  when signature inspection shows that a provider accepts it. Adding an
  abstract method or an unconditionally forwarded argument requires a
  migration window rather than a flag-day signature change.
- **Version the contract that crosses a boundary.** A capability may carry its
  own schema version when it defines a wire payload or persisted format (for
  example, observer payloads or secret-source state). Keep fields additive
  within that local schema. Persisted plugin state and config must remain
  readable, or ship an explicit migration; resumed sessions written by the old
  format must still replay. Do not add version literals to unrelated callback
  or context values.

### Deprecation policy

A documented native plugin behavior may be deprecated only with all of the
following:

1. a replacement and migration instructions in the plugin guide and release
   notes;
2. a warning emitted at most once per process, naming the replacement and the
   earliest removal release;
3. support for the old behavior through at least two subsequent minor
   releases; and
4. behavior-based compatibility coverage for both the legacy path and the
   replacement throughout that window.

Removal after the window must include any migration needed for persisted data
or resumable sessions. In practice, additive aliases and adapters are preferred
to removal.

Hermes enforces this contract with frozen external-plugin fixtures discovered
from an isolated `HERMES_HOME`. Those tests load and invoke the plugin through
`PluginManager`; they assert real registration and callback outcomes rather
than internal symbol lists or source-code shape.

## What you're building

A **calculator** plugin with two tools:
- `calculate` — evaluate math expressions (`2**16`, `sqrt(144)`, `pi * 5**2`)
- `unit_convert` — convert between units (`100 F → 37.78 C`, `5 km → 3.11 mi`)

Plus a hook that logs every tool call, and a bundled skill file.

## Step 1: Create the plugin directory

Create a directory and continue with Step 2:

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

### Validate with Plugin Doctor

`hermes plugins doctor [path-or-id]` runs the same directory discovery,
manifest parser, namespaced import, `register(ctx)`, hook registry, and tool
registry used by Hermes itself. It reports invalid hook names, callbacks that do
not accept `**kwargs`, registration failures, and drift between declared and
registered tools/hooks. Pass `--ci` to exit non-zero on an error:

```bash
hermes plugins doctor . --ci
```

Doctor uses a temporary `HERMES_HOME`, restores plugin registration state after
the check, and blocks direct Python socket connections to catch accidental
network access while registration runs. This is not a sandbox: plugin code still
executes in-process with the current user's permissions and can spawn subprocesses,
so only run Doctor on code you trust enough to import.

## Step 2: Write the manifest

Create `plugin.yaml`:

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

This tells Hermes: "I'm a plugin called calculator, I provide tools and hooks." The `provides_tools` and `provides_hooks` fields are lists of what the plugin registers.

Optional fields you could add:
```yaml
author: Your Name
requires_env:          # gate loading on env vars; prompted during install
  - SOME_API_KEY       # simple format — plugin disabled if missing
  - name: OTHER_KEY    # rich format — shows description/url during install
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
capabilities:          # privileged host surfaces you request (consent flow)
  - tools.override     # replace built-in tools (needs user consent)
  - llm.model_override # choose the model for host-owned LLM calls
```

### Declaring capabilities

If your plugin needs a privileged host surface — overriding a built-in tool,
picking the model for `ctx.llm` calls, etc. — declare it in `capabilities:`.
At install/enable time the user sees the list and consents once; if a later
version adds a capability, the update flow asks again for just the addition.
Undeclared or unconsented capabilities are simply off (fail closed), so
**probe before using them and degrade gracefully**:

```python
def register(ctx):
    if ctx.has_capability("tools.override"):
        ctx.register_tool(..., override=True)
    else:
        ctx.register_tool(...)   # register under a non-conflicting name
```

Known capability ids: `tools.override`, `llm.provider_override`,
`llm.model_override`, `llm.agent_id_override`, `llm.profile_override`,
`llm.task_override` (see `hermes_cli/plugin_capabilities.py` for the
canonical registry). Unknown ids are ignored. The older per-capability
config keys (`plugins.entries.<id>.allow_tool_override`, …) still work but
are deprecated — declare capabilities instead so users get a single,
auditable consent screen. Capabilities are consent + audit, **not a
sandbox**: they gate host API surfaces, nothing more.

**Pip-distributed plugins** have no `plugin.yaml` directory once installed,
so declare capabilities in distribution metadata instead, via the companion
`hermes_agent.plugin_capabilities` entry-point group. Each declaration is
named `<plugin-id>.<capability-id>` and points at the same object as your
`hermes_agent.plugins` entry point:

```toml
[project.entry-points."hermes_agent.plugins"]
calculator = "my_pkg:register"

[project.entry-points."hermes_agent.plugin_capabilities"]
"calculator.tools.override" = "my_pkg:register"
```

Hermes reads these from installed metadata without importing your code, so
`hermes plugins capabilities` and the consent flow stay accurate for pip
installs.

### Manifest v2 reference

`plugin.yaml` also supports an additive **v2 schema** (#64165). Every field is
optional; a manifest without `manifest_version` is a v1 manifest and stays
fully supported forever. Unknown fields never break loading — they are ignored
with a warning (forward compatibility), and a `manifest_version` newer than
this Hermes understands still loads with a warning.

| Field | Type | Meaning |
|---|---|---|
| `manifest_version` | int | Manifest **file-format** version. Absent = `1`. Current max: `2`. Independent from `api_version`. |
| `api_version` | int | Runtime **plugin API generation** the plugin targets (ctx surface / hook signatures). Deliberately a separate axis from `manifest_version` — an `api_version: 1` plugin can use a v2 manifest. |
| `requires_plugins` | list | Inter-plugin dependencies: `- id: other-plugin` with optional `version_range: ">=1.0,<2"`. **Advisory**: a missing dependency logs a clear warning but the plugin still loads — probe at runtime with `ctx.has_plugin("other-plugin")`. Load **order** honors these edges: when A requires B, B's `register()` runs before A's (topological sort, alphabetical tiebreak; cycles warn and fall back to alphabetical order). |
| `python_dependencies` | list of str | Declared pip requirements (e.g. `"requests>=2.0,<3"`). **Declaration seam only** — Hermes validates them, and `hermes plugins install` / `hermes plugins doctor` surface missing ones with a `pip install` hint, but Hermes **never auto-installs** them. Pin upper bounds. |
| `config_schema` | mapping | JSON-schema-ish description of keys under `plugins.entries.<id>.settings`: `api_url: {type: str, default: "", description: "...", required: false}`. Validated at load; mismatches log actionable warnings naming the key and expected type — never load failures. Types: `str`, `int`, `float`, `bool`, `list`, `dict` (plus JSON-schema aliases). |
| `license` | str | SPDX-style license id (e.g. `MIT`). |
| `homepage` | str | Project URL. |
| `tags` | list of str | Free-form discovery tags (e.g. `[gateway, telegram]`). |

```yaml
# plugin.yaml — manifest v2 example
name: my-plugin
version: 1.2.0
manifest_version: 2
api_version: 1
license: MIT
homepage: https://github.com/owner/my-plugin
tags: [gateway, demo]
requires_plugins:
  - id: other-plugin
    version_range: ">=1.0,<2"
python_dependencies:
  - "somepkg>=1.0,<2"     # surfaced, never auto-installed
config_schema:
  api_url: {type: str, default: "", description: "Service endpoint"}
```

:::note pip-dependency isolation is deferred
`python_dependencies` is intentionally declare-and-surface only. Installing
arbitrary packages into Hermes' shared venv is a conflict and supply-chain
surface, so the install seam's isolation design (constraints-file installs
against the host lock vs. per-plugin vendored dirs vs. conflict detection
with refusal) is an explicitly deferred follow-up — see the round-2 review on
[#64165](https://github.com/NousResearch/hermes-agent/issues/64165) and
[#15220](https://github.com/NousResearch/hermes-agent/issues/15220). Plugin
packs (#64166) build on these v2 fields.
:::

## Step 3: Write the tool schemas

Create `schemas.py` — this is what the LLM reads to decide when to call your tools:

```python
"""Tool schemas — what the LLM sees."""

CALCULATE = {
    "name": "calculate",
    "description": (
        "Evaluate a mathematical expression and return the result. "
        "Supports arithmetic (+, -, *, /, **), functions (sqrt, sin, cos, "
        "log, abs, round, floor, ceil), and constants (pi, e). "
        "Use this for any math the user asks about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2**10', 'sqrt(144)')",
            },
        },
        "required": ["expression"],
    },
}

UNIT_CONVERT = {
    "name": "unit_convert",
    "description": (
        "Convert a value between units. Supports length (m, km, mi, ft, in), "
        "weight (kg, lb, oz, g), temperature (C, F, K), data (B, KB, MB, GB, TB), "
        "and time (s, min, hr, day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'km', 'lb', 'F', 'GB')",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit (e.g., 'mi', 'kg', 'C', 'MB')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
}
```

**Why schemas matter:** The `description` field is how the LLM decides when to use your tool. Be specific about what it does and when to use it. The `parameters` define what arguments the LLM passes.

## Step 4: Write the tool handlers

Create `tools.py` — this is the code that actually executes when the LLM calls your tools:

```python
"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import math

# Safe globals for expression evaluation — no file/network access
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}


def calculate(args: dict, **kwargs) -> str:
    """Evaluate a math expression safely.

    Rules for handlers:
    1. Receive args (dict) — the parameters the LLM passed
    2. Do the work
    3. Return a JSON string — ALWAYS, even on error
    4. Accept **kwargs for forward compatibility
    """
    expression = args.get("expression", "").strip()
    if not expression:
        return json.dumps({"error": "No expression provided"})

    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero"})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Invalid: {e}"})


# Conversion tables — values are in base units
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}


def _convert_temp(value, from_u, to_u):
    # Normalize to Celsius
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # Convert to target
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)


def unit_convert(args: dict, **kwargs) -> str:
    """Convert between units."""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # Temperature
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # Ratio-based conversions
        for table in (_LENGTH, _WEIGHT, _DATA, _TIME):
            lc = {k.lower(): v for k, v in table.items()}
            if from_unit.lower() in lc and to_unit.lower() in lc:
                result = float(value) * lc[from_unit.lower()] / lc[to_unit.lower()]
                return json.dumps({"input": f"{value} {from_unit}",
                                 "result": round(result, 6),
                                 "output": f"{round(result, 6)} {to_unit}"})

        return json.dumps({"error": f"Cannot convert {from_unit} → {to_unit}"})
    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}"})
```

**Key rules for handlers:**
1. **Signature:** `def my_handler(args: dict, **kwargs) -> str`
2. **Return:** Always a JSON string. Success and errors alike.
3. **Never raise:** Catch all exceptions, return error JSON instead.
4. **Accept `**kwargs`:** Hermes may pass additional context in the future.

## Step 5: Write the registration

Create `__init__.py` — this wires schemas to handlers:

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# Track tool usage via hooks
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """Hook: runs after every tool call (not just ours)."""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def register(ctx):
    """Wire schemas to handlers and register hooks."""
    ctx.register_tool(name="calculate",    toolset="calculator",
                      schema=schemas.CALCULATE,    handler=tools.calculate)
    ctx.register_tool(name="unit_convert", toolset="calculator",
                      schema=schemas.UNIT_CONVERT, handler=tools.unit_convert)

    # This hook fires for ALL tool calls, not just ours
    ctx.register_hook("post_tool_call", _on_post_tool_call)
```

**What `register()` does:**
- Called exactly once at startup
- `ctx.register_tool()` puts your tool in the registry — the model sees it immediately
- `ctx.register_hook()` subscribes to lifecycle events
- `ctx.register_cli_command()` registers a CLI subcommand (e.g. `hermes my-plugin <subcommand>`)
- `ctx.register_command()` registers an in-session slash command (e.g. `/myplugin <args>` inside CLI / gateway chat) — see [Register slash commands](#register-slash-commands) below
- `ctx.register_realtime_voice_provider(provider)` registers a realtime speech-to-speech backend — see [Register a realtime voice provider](#register-a-realtime-voice-provider) below
- `ctx.dispatch_tool(name, arguments)` — call any other tool (built-in or from another plugin) with the parent agent's context (approvals, credentials, task_id) wired up automatically. Useful from slash-command handlers that need to invoke `terminal`, `read_file`, or any other tool as if the model had called it directly.
- `ctx.get_config()` / `ctx.set_config()` access only this plugin's settings namespace; `ctx.state` stores plugin-owned runtime data under the active profile.
- If this function crashes, the plugin is disabled but Hermes continues fine

**`dispatch_tool` example — a slash command that runs a tool:**

```python
def handle_scan(ctx, raw_args: str):
    """Implement /scan by invoking the terminal tool through the registry."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{raw_args}'"})
    return result  # returned to the caller's chat UI

def register(ctx):
    # Handlers receive a single raw_args string; close over ctx via a lambda.
    ctx.register_command(
        "scan",
        lambda raw: handle_scan(ctx, raw),
        description="Find files matching a glob",
    )
```

The dispatched tool goes through the normal approval, redaction, and budget pipelines — it's a real tool invocation, not a shortcut around them.

### Store settings and runtime state

Use plugin-relative config keys for user-visible behavior. Hermes resolves them
under `plugins.entries.<plugin-id>.settings` and rejects global, cross-plugin,
and traversal paths:

```python
def register(ctx):
    endpoint = ctx.get_config("endpoint", default="https://example.invalid")
    retries = ctx.get_config("retry.attempts", default=3)

    ctx.set_config("endpoint", endpoint)
    ctx.set_config("retry.attempts", retries)
```

Use `ctx.state` for plugin-owned cursors, caches, and deduplication data rather
than placing runtime bookkeeping in `config.yaml`:

```python
def register(ctx):
    cursor = ctx.state.get("cursor", default={"page": 0})
    ctx.state.set("cursor", {"page": cursor["page"] + 1})
```

State is profile-scoped, atomically replaced, safe across concurrent writers,
and limited to 10 MiB per plugin. Portable packages share the same directory as
their `PLUGIN_DATA`; native plugins receive a collision-resistant,
Windows-safe namespace. Malformed existing state is reported and preserved.

Config and state have different owners: settings are user-visible behavior in
`config.yaml`, while state is plugin-owned runtime data under
`<HERMES_HOME>/plugin-data/`. Neither API exposes another plugin's namespace.

## Step 6: Test it

Start Hermes:

```bash
hermes
```

You should see `calculator: calculate, unit_convert` in the banner's tool list.

Try these prompts:
```
What's 2 to the power of 16?
Convert 100 fahrenheit to celsius
What's the square root of 2 times pi?
How many gigabytes is 1.5 terabytes?
```

Check plugin status:
```
/plugins
```

Output:
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

### Debugging plugin discovery

If your plugin doesn't show up — or shows up but isn't loading — set `HERMES_PLUGINS_DEBUG=1` to get verbose discovery logs on stderr:

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

You'll see, for every plugin source (bundled, user, project, entry-points):

- which directories were scanned and how many manifests each yielded
- per manifest: resolved key, name, kind, source, on-disk path
- skip reasons: `disabled via config`, `not enabled in config`, `exclusive plugin`, `no plugin.yaml, depth cap reached`
- on load: the plugin being imported, plus a one-line summary of what `register(ctx)` registered (tools, hooks, slash commands, CLI commands)
- on parse failure: a full traceback for the exception (YAML scanner errors, etc.)
- on `register()` failure: a full traceback pointing at the line in your `__init__.py` that raised

The same logs are always written to `~/.hermes/logs/agent.log` at WARNING level (failures only) and DEBUG level (everything) when the env var is set. So if you can't run with the env var (e.g. from inside the gateway), tail the log file instead:

```bash
hermes logs --level WARNING | grep -i plugin
```

Common reasons a plugin doesn't appear:

- **Not enabled in config** — plugins are opt-in. Run `hermes plugins enable <name>` (the name comes from the `plugins list` output, which can be `<category>/<plugin>` for nested layouts).
- **Wrong directory layout:** Native packages use `~/.hermes/plugins/<plugin-name>/plugin.yaml` (flat) or one category level. Portable packages use root `plugin.json` in the same locations. Anything deeper is ignored.
- **Missing `__init__.py`:** Native packages need both `plugin.yaml` and `__init__.py` with a `register(ctx)` function. Portable packages do not import Python and do not require `__init__.py`.
- **Wrong `kind`** — gateway adapters need `kind: platform` in their manifest. Memory providers are auto-detected as `kind: exclusive` and routed through the `memory.provider` config instead of `plugins.enabled`.

## Your plugin's final structure

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "I'm calculator, I provide tools and hooks"
├── __init__.py      # Wiring: schemas → handlers, register hooks
├── schemas.py       # What the LLM reads (descriptions + parameter specs)
└── tools.py         # What runs (calculate, unit_convert functions)
```

Four files, clear separation:
- **Manifest** declares what the plugin is
- **Schemas** describe tools for the LLM
- **Handlers** implement the actual logic
- **Registration** connects everything

## What else can plugins do?

### Register a realtime voice provider

A realtime voice provider is a long-lived, bidirectional speech-to-speech session:
audio in, audio out, interruptions, and tool calls on one channel. Hermes ships one
(the bundled OpenAI Realtime backend in `plugins/realtime_voice/openai/`) and drives
any registered provider from `hermes realtime --provider <name>` — see
[Realtime Voice](/user-guide/features/realtime-voice).

The split is fixed by the contract in `agent/realtime_voice_provider.py`:

- **The provider owns** its SDK, credentials, wire protocol, audio formats, turn
  detection, and transport lifecycle. It translates native events into the frozen
  typed events (`SessionReady`, `InputSpeechStarted`, `InputTranscript`,
  `ResponseStarted`, `OutputAudio`, `OutputTranscript`, `ToolCall`,
  `ToolCallCancelled`, `ResponseCompleted`, `SessionFailure`, `SessionClosed`, …).
- **Hermes owns** tool execution, approvals, history, and memory. The orchestrator
  (`agent/realtime_voice_orchestrator.py`) runs every `ToolCall` on its own cancellable
  task under a per-call timeout, answers each `call_id` exactly once through
  `submit_tool_results()`, fences the audio tail of a cancelled response, and turns a
  barge-in into cancel + truncate (or a local drop when the provider cannot truncate).

Capabilities are declared, never guessed. Advertise only what the wire can do; the
orchestrator checks `session.supports(...)` before calling an optional operation, and
the base class refuses a session that advertises a capability without overriding its
hook.

```python
# ~/.hermes/plugins/my-realtime/__init__.py
import os

from agent.realtime_voice_provider import (
    PCM16_24K,
    RealtimeAudioFormat,
    RealtimeCapability,
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    RealtimeVoiceSetup,
    SessionClosed,
    SessionReady,
)


class MySession(RealtimeVoiceSession):
    def __init__(self, socket):
        super().__init__(
            {RealtimeCapability.TOOL_CALLING, RealtimeCapability.RESPONSE_CANCELLATION},
            input_audio=RealtimeAudioFormat(sample_rate_hz=16_000),  # what send_audio expects
            output_audio=PCM16_24K,                                   # what OutputAudio carries
        )
        self._socket = socket

    async def send_audio(self, audio: bytes) -> None:
        await self._socket.send_pcm(audio)

    async def _submit_tool_results(self, results, continue_response) -> None:
        await self._socket.send_tool_outputs([(r.call_id, r.output) for r in results])

    async def _cancel_response(self, response_id) -> None:
        await self._socket.cancel()

    def _events(self):
        async def stream():
            yield SessionReady(session_id=self._socket.id)
            async for frame in self._socket:
                event = translate(frame)   # your wire -> contract events
                if event is not None:
                    yield event
            yield SessionClosed(reason="end of stream")
        return stream()

    async def _close(self) -> None:
        await self._socket.close()


class MyProvider(RealtimeVoiceProvider):
    capabilities = frozenset({RealtimeCapability.TOOL_CALLING, RealtimeCapability.RESPONSE_CANCELLATION})

    @property
    def name(self) -> str:
        return "my-realtime"          # what --provider matches

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_REALTIME_KEY"))

    async def open_session(self, setup: RealtimeVoiceSetup) -> RealtimeVoiceSession:
        socket = await connect(model=setup.model, voice=setup.voice,
                               instructions=setup.instructions, tools=setup.tools)
        return MySession(socket)


def register(ctx):
    ctx.register_realtime_voice_provider(MyProvider())
```

Rules the orchestrator relies on:

- `ToolCall.arguments` is the raw JSON text from the wire; Hermes parses it and answers
  a malformed payload with an error result so the turn completes.
- A `SessionFailure` with `terminal=False` is a warning (one bad frame must never end a
  call); `terminal=True` or `SessionClosed` ends the stream.
- Emit `ToolCallCancelled` when your provider retracts calls (Gemini Live's
  `toolCallCancellation`); Hermes will never submit results for those ids.
- Providers may redeliver events after a reconnect; Hermes deduplicates tool work by
  `call_id`.
- The API key stays inside the provider. Never put it in `provider_data` or logs.

Registration is profile-scoped and reversible: unloading the plugin restores whatever
was registered under that name before, and a provider targeting a different
`api_version` than the host is ignored with a warning.

**How to test a provider without audio hardware:** drive it with the orchestrator and a
scripted fake socket, the way `tests/plugins/realtime_voice/test_openai_provider.py` does —
feed frames in, assert the commands your session writes out.

### Ship data files

Put any files in your plugin directory and read them at import time:

```python
# In tools.py or __init__.py
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

That's for files you *ship*. State you *write* is different — see the next
section.

### Store durable state

Never write runtime state into your plugin directory: that's the install
tree, and `hermes plugins update` / `remove` git-pull or delete it — your
users' data dies with it. The sanctioned home is the per-plugin data root,
which survives both and follows the active profile:

```python
from plugins.plugin_storage import plugin_data_dir, plugin_db

# <hermes home>/plugin-data/<name>/ — created on first use
state_file = plugin_data_dir("my-plugin") / "state.json"

# Or a SQLite database at <data dir>/data.db (WAL mode, thread-friendly)
conn = plugin_db("my-plugin")
conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY)")
```

One directory per plugin means every plugin's data is inspectable in one
predictable place. Secrets don't belong here — credential reads go through
the standard `.env` / secret-scope path like everywhere else.

### Bundle skills

Plugins can ship skill files that the agent loads via `skill_view("plugin:skill")`. Register them in your `__init__.py`:

```
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
from pathlib import Path

def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

The agent can now load your skills with their namespaced name:

```python
skill_view("my-plugin:my-workflow")   # → plugin's version
skill_view("my-workflow")              # → built-in version (unchanged)
```

**Key properties:**
- Plugin skills are **read-only** — they don't enter `~/.hermes/skills/` and can't be edited via `skill_manage`.
- Plugin skills are **not** listed in the system prompt's `<available_skills>` index — they're opt-in explicit loads.
- Bare skill names are unaffected — the namespace prevents collisions with built-in skills.
- When the agent loads a plugin skill, a bundle context banner is prepended listing sibling skills from the same plugin.

:::tip Legacy pattern
The old `shutil.copy2` pattern (copying a skill into `~/.hermes/skills/`) still works but creates name collision risk with built-in skills. Prefer `ctx.register_skill()` for new plugins.
:::

### Gate on environment variables

If your plugin needs an API key:

```yaml
# plugin.yaml — simple format (backwards-compatible)
requires_env:
  - WEATHER_API_KEY
```

If `WEATHER_API_KEY` isn't set, the plugin is disabled with a clear message. No crash, no error in the agent — just "Plugin weather disabled (missing: WEATHER_API_KEY)".

When users run `hermes plugins install`, they're **prompted interactively** for any missing `requires_env` variables. Values are saved to `.env` automatically.

For a better install experience, use the rich format with descriptions and signup URLs:

```yaml
# plugin.yaml — rich format
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Environment variable name |
| `description` | No | Shown to user during install prompt |
| `url` | No | Where to get the credential |
| `secret` | No | If `true`, input is hidden (like a password field) |

Both formats can be mixed in the same list. Already-set variables are skipped silently.

### Lazy-install optional Python dependencies

If your plugin wraps an SDK that not every user will have installed (a vendor SDK, a heavy ML lib, a platform-specific package), don't `import` it at the top of the module. Use the `tools.lazy_deps.ensure(...)` helper inside the tool handler — Hermes will install the package on first use, gated by the user's `security.allow_lazy_installs` config.

```python
# tools.py
from tools.lazy_deps import ensure, FeatureUnavailable

def my_tool_handler(args, **kwargs):
    try:
        ensure("my-plugin.my-backend")   # key must be in LAZY_DEPS
    except FeatureUnavailable as exc:
        return {"error": str(exc)}

    import my_backend_sdk   # safe now
    ...
```

Two rules from the security model in `tools/lazy_deps.py`:

| Rule | Why |
|---|---|
| Your feature key must appear in the in-tree `LAZY_DEPS` allowlist | Prevents a malicious config from coaxing Hermes into installing arbitrary packages — only specs Hermes itself ships are eligible |
| Specs are PyPI-by-name only | No `--index-url`, `git+https://`, or file: paths. Pin versions with PEP 440 (`"my-sdk>=1.2,<2"`) inside the allowlist entry |

For third-party plugins distributed via pip, declare the optional deps as `[project.optional-dependencies]` extras in your own `pyproject.toml` and tell users to `pip install your-plugin[backend]` — that path doesn't go through `lazy_deps`. The lazy-install dance is most useful for **bundled** plugins where shipping a hard dependency on every install would bloat the base Hermes footprint.

When `security.allow_lazy_installs: false` is set globally, `ensure()` raises `FeatureUnavailable` immediately with a remediation hint — your plugin should catch it and degrade gracefully (return an error result, not crash the tool loop).



### Thread-safe lazy singletons

Plugins often cache an expensive object — an SDK client, an HTTP session, a connection pool — in a module-level variable built on first use:

```python
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    _client = ExpensiveClient(...)   # ← TOCTOU race
    return _client
```

This is a footgun. Hermes runs multiple threads in one process (delegated tool calls, background workers, the self-improvement fork), so two threads can hit `get_client()` before `_client` is set, **both** pass the `is not None` check, **both** run the expensive build, and the second write clobbers the first — leaking whatever resource the loser opened (connection, file handle, background thread).

Don't hand-roll the lock. Use the helpers in `plugins/plugin_utils.py`:

```python
from plugins.plugin_utils import lazy_singleton, SingletonSlot

# Zero-arg accessor → decorate it:
@lazy_singleton
def get_client():
    return ExpensiveClient(load_config())   # runs exactly once

client = get_client()    # safe across threads
get_client.reset()       # drop the instance (tests / teardown)


# Accessor that takes a build argument → use a slot:
_slot: SingletonSlot = SingletonSlot()

def get_client(config=None):
    return _slot.get(lambda: ExpensiveClient(resolve(config)))

def reset_client():
    _slot.reset()
```

Both serialize concurrent first calls with double-checked locking and run the factory at most once. If the factory raises, nothing is cached and the next call retries. The honcho memory plugin (`plugins/memory/honcho/client.py`) is the reference consumer.

> Rule of thumb: any time you write `global _something` followed by a `is None` check and a build, reach for one of these instead.



### Conditional tool availability

For tools that depend on optional libraries:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = tool hidden from model
)
```

### Overriding a built-in tool

To replace a built-in tool with your own implementation (e.g. swap the
default browser tool for a headed-Chrome CDP backend, or replace
`web_search` with a custom corporate index), pass `override=True`:

```python
def register(ctx):
    ctx.register_tool(
        name="browser_navigate",             # same name as the built-in
        toolset="plugin_my_browser",         # your own toolset namespace
        schema={...},
        handler=my_custom_navigate,
        override=True,                       # explicit opt-in
    )
```

Without `override=True`, the registry rejects any registration that would
shadow an existing tool from a different toolset — this prevents
accidental overwrites. Overriding a **built-in** tool additionally
requires the operator to opt in via
`plugins.entries.<plugin_id>.allow_tool_override: true` in `config.yaml`;
without that gate, `register_tool(override=True)` raises
`PluginToolOverrideError`. The override is logged so it's
auditable in `~/.hermes/logs/agent.log`. Plugins load after built-in
tools, so the registration order is correct: your handler replaces the
built-in one.

**Non-bundled plugins also need an operator grant.** For any plugin that
does not ship with Hermes core (user, project, or pip source),
`override=True` against an existing built-in tool additionally requires a
per-plugin opt-in in `config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:                    # the plugin's registry key from `hermes plugins list`
      allow_tool_override: true
```

Without the grant, `ctx.register_tool(..., override=True)` raises
`PluginToolOverrideError`; since `register()` exceptions are caught by the
loader, the plugin is disabled and Hermes continues. The gate exists
because an enabled plugin that silently replaces a privileged built-in
like `shell_exec` or `write_file` could intercept everything the model
routes through it. Bundled plugins are exempt: an override there is a
maintainer decision. If config cannot be loaded, the gate fails closed.

You normally never edit this key by hand. `hermes plugins enable <name>`
asks whether to grant the capability when enabling a non-bundled plugin
(defaulting to no), and the `--allow-tool-override` /
`--no-allow-tool-override` flags skip the prompt for scripted installs.
The same grant also gates `deregister()`: without it, a plugin cannot
remove a tool it does not own (which would otherwise be a way around the
override check).

### Register multiple hooks

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### Hook reference

Each hook is documented in full on the **[Event Hooks reference](/user-guide/features/hooks#plugin-hooks)** — callback signatures, parameter tables, exactly when each fires, and examples. Here's the summary:

| Hook | Fires when | Callback signature | Returns |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | Before any tool executes | `tool_name: str, args: dict, task_id: str` | optional directive: `{"action": "block", "message": ...}` vetoes the call; `{"action": "approve", "message": ...}` escalates to the human-approval gate |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | After any tool returns | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | ignored |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | Once per turn, before the tool-calling loop | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [context injection](#pre_llm_call-context-injection) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | Once per turn, after the tool-calling loop (successful turns only) | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | ignored |
| `pre_api_request` | Before each raw provider API request (several per turn when the model calls tools) | `session_id: str, model: str, provider: str, base_url: str, api_mode: str, api_call_count: int, message_count: int, tool_count: int, approx_input_tokens: int, max_tokens: int, request: dict` | ignored |
| `post_api_request` | After each raw provider API request returns | `pre_api_request` fields plus `api_duration: float, finish_reason: str, response_model: str \| None, usage: dict, response: dict, assistant_content_chars: int, assistant_tool_call_count: int` | ignored |
| `api_request_error` | A provider API call raised | correlation fields plus `status_code: int \| None, retry_count: int \| None, max_retries: int \| None, retryable: bool \| None, reason: str \| None, error: dict, request: dict` | ignored |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | New session created (first turn only) | `session_id: str, model: str, platform: str` | ignored |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | End of every `run_conversation` call + CLI exit | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | ignored |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/gateway tears down an active session | `session_id: str \| None, platform: str` | ignored |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | Gateway swaps in a new session key (`/new`, `/reset`) | `session_id: str, platform: str` | ignored |
| [`gateway_platform_event`](/user-guide/features/hooks#gateway_platform_event) | An authorized platform-native event is normalized at the gateway boundary (Telegram reactions currently) | `platform: str, event_type: str, payload: dict` | ignored |
| `kanban_task_claimed` | A kanban task is claimed (dispatcher process, before the worker spawns) | `task_id: str, board: str \| None, assignee: str \| None, run_id: int \| None, profile_name: str` | ignored |
| `kanban_task_completed` | A kanban task completes (worker process) | `task_id, board, assignee, run_id, profile_name, summary: str \| None` | ignored |
| `kanban_task_blocked` | A kanban task is blocked (worker process) | `task_id, board, assignee, run_id, profile_name, reason: str \| None` | ignored |

Most hooks are fire-and-forget observers — their return values are ignored. The exceptions are `pre_llm_call`, which can inject context into the conversation, and `pre_tool_call`, which can return a block/approve directive.

All callbacks should accept `**kwargs` for forward compatibility. If a hook callback crashes, it's logged and skipped. Other hooks and the agent continue normally.

The kanban lifecycle hooks fire **after** the board DB change commits, so a callback always sees durable state and can never hold the SQLite write lock. Because kanban workers run as separate `hermes -p <profile> chat -q` subprocesses, `kanban_task_claimed` fires in the **dispatcher** process while `kanban_task_completed` / `kanban_task_blocked` fire in the **worker** process — hook in the dispatcher to observe every transition centrally, or in the worker for per-task in-session context.

The **API request hooks** are observers for the raw provider request, one level below the per-turn `pre_llm_call` / `post_llm_call` pair: a single turn that calls tools makes several API requests, and these hooks fire around each one. They exist for observability plugins (tracing, cost accounting, latency dashboards). The `request` and `response` kwargs are sanitized, size-capped JSON views of the provider payload (sensitive keys redacted, long strings truncated, SDK objects normalized), and `usage` is a plain token-summary dict. Every payload carries the correlation fields `turn_id`, `api_request_id`, `task_id`, `session_id`, and `api_call_count`, so a plugin can stitch requests, tool calls, and turns together. `api_request_error` fires when a provider call raises and adds `status_code`, `retry_count` / `max_retries`, `retryable`, `reason`, and an `error` dict with `type` and `message`.

### `pre_llm_call` context injection

This is the only hook whose return value matters. When a `pre_llm_call` callback returns a dict with a `"context"` key (or a plain string), Hermes injects that text into the **current turn's user message**. This is the mechanism for memory plugins, RAG integrations, guardrails, and any plugin that needs to provide the model with additional context.

#### Return format

```python
# Dict with context key
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# Plain string (equivalent to the dict form above)
return "Recalled memories:\n- User prefers dark mode"

# Return None or don't return → no injection (observer-only)
return None
```

Any non-None, non-empty return with a `"context"` key (or a plain non-empty string) is collected and appended to the user message for the current turn.

#### Oversized-context spill

Per-hook context is capped at `10,000` characters by default. Anything above the cap is written to `$HERMES_HOME/hook_outputs/<session_id>/<uuid>.txt` and replaced with a head/tail preview plus the saved path. The model can read the full content via `read_file` or `terminal` if it genuinely needs it. This keeps a runaway plugin from inflating every subsequent turn's prompt and blowing out the prompt cache prefix. Tune in `config.yaml`:

```yaml
hooks:
  output_spill:
    enabled: true          # default: true
    max_chars: 10000       # default; set higher to opt out of spilling
    preview_head: 500      # chars shown at the top of the preview
    preview_tail: 500      # chars shown at the bottom of the preview
    # directory: null      # default: $HERMES_HOME/hook_outputs
```

#### How injection works

Injected context is appended to the **user message**, not the system prompt. This is a deliberate design choice:

- **Prompt cache preservation** — the system prompt stays identical across turns. Anthropic and OpenRouter cache the system prompt prefix, so keeping it stable saves 75%+ on input tokens in multi-turn conversations. If plugins modified the system prompt, every turn would be a cache miss.
- **Ephemeral** — the injection happens at API call time only. The original user message in the conversation history is never mutated, and nothing is persisted to the session database.
- **The system prompt is Hermes's territory** — it contains model-specific guidance, tool enforcement rules, personality instructions, and cached skill content. Plugins contribute context alongside the user's input, not by altering the agent's core instructions.

#### Example: Memory recall plugin

```python
"""Memory plugin — recalls relevant context from a vector store."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """Called before each LLM turn. Returns recalled memories."""
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None  # nothing to inject

        text = "Recalled context from previous sessions:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # fail silently, don't break the agent

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

#### Example: Guardrails plugin

```python
"""Guardrails plugin — enforces content policies."""

POLICY = """You MUST follow these content policies for this session:
- Never generate code that accesses the filesystem outside the working directory
- Always warn before executing destructive operations
- Refuse requests involving personal data extraction"""

def inject_guardrails(**kwargs):
    """Injects policy text into every turn."""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### Example: Observer-only hook (no injection)

```python
"""Analytics plugin — tracks turn metadata without injecting context."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """Fires before each LLM call. Returns None — no context injected."""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # No return → no injection

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### Multiple plugins returning context

When multiple plugins return context from `pre_llm_call`, their outputs are joined with double newlines and appended to the user message together. The order follows plugin discovery order (alphabetical by plugin directory name).

### Middleware: change what happens

Hooks observe the agent loop (with the few documented steering shapes above). **Middleware changes what happens**: request middleware rewrites the effective payload before anything downstream sees it, and execution middleware wraps the actual call. Register it from the same `register(ctx)` entry point:

```python
def cap_find_output(tool_name, args, **kwargs):
    """Rewrite terminal find commands to cap their output."""
    command = args.get("command", "")
    if tool_name == "terminal" and command.startswith("find "):
        return {
            "args": {**args, "command": command + " | head -100"},
            "source": "my-plugin",
            "reason": "cap find output",
        }
    return None  # leave the call unchanged

def register(ctx):
    ctx.register_middleware("tool_request", cap_find_output)
```

The canonical list of kinds is `VALID_MIDDLEWARE` in `hermes_cli/middleware.py`:

| Kind | Receives | Return contract |
|------|----------|-----------------|
| `tool_request` | `tool_name`, `args`, `original_args`, context kwargs | Return `{"args": {...}}` to replace the effective tool arguments before hooks, guardrails, approvals, and execution see them. Return `None` to leave the call unchanged. |
| `llm_request` | `request`, `original_request`, context kwargs | Return `{"request": {...}}` to replace the effective provider kwargs before Hermes sends them. |
| `tool_execution` | the payload plus `next_call` | Wraps tool execution. Call `next_call(payload)` exactly once to run the downstream chain (or skip it to short-circuit) and return the result. |
| `llm_execution` | the payload plus `next_call` | Same shape, wrapping the provider call. |

**Rules that matter in practice:**

- Request middleware chains: each callback sees the payload as rewritten by earlier callbacks, while `original_args` / `original_request` always carries the pre-middleware copy. Payloads are copied between callbacks, so mutate freely.
- You can include `source`, `reason`, and `name` strings in the returned dict. They land in the middleware trace, which downstream observer hooks receive as the `middleware_trace` kwarg.
- `next_call` in execution middleware is **single-use**. Calling it twice raises, because it would re-run the provider or tool.
- A middleware callback that raises is logged and skipped; the chain continues. A downstream failure raised after your `next_call` propagates as itself. Middleware can never break the base runtime path.
- Middleware payloads carry `middleware_schema_version` (`hermes.middleware.v1`) alongside the observer telemetry fields.
- Unknown kinds register with a warning instead of failing, so a plugin written against a newer Hermes still loads on an older one.

### Register CLI commands

Plugins can add their own `hermes <plugin>` subcommand tree:

```python
def _my_command(args):
    """Handler for hermes my-plugin <subcommand>."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """Build the argparse tree for hermes my-plugin."""
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subs.add_parser("config", help="Show plugin config")
    subparser.set_defaults(func=_my_command)

def register(ctx):
    ctx.register_tool(...)
    ctx.register_cli_command(
        name="my-plugin",
        help="Manage my plugin",
        setup_fn=_setup_argparse,
        handler_fn=_my_command,
    )
```

After registration, users can run `hermes my-plugin status`, `hermes my-plugin config`, etc.

**Memory provider plugins** use a convention-based approach instead: add a `register_cli(subparser)` function to your plugin's `cli.py` file. The memory plugin discovery system finds it automatically — no `ctx.register_cli_command()` call needed. See the [Memory Provider Plugin guide](/developer-guide/memory-provider-plugin#adding-cli-commands) for details.

**Active-provider gating:** Memory plugin CLI commands only appear when their provider is the active `memory.provider` in config. If a user hasn't set up your provider, your CLI commands won't clutter the help output.

### Register slash commands

Plugins can register in-session slash commands — commands users type during a conversation (like `/lcm status` or `/ping`). These work in both CLI and gateway (Telegram, Discord, etc.).

```python
def _handle_status(raw_args: str) -> str:
    """Handler for /mystatus — called with everything after the command name."""
    if raw_args.strip() == "help":
        return "Usage: /mystatus [help|check]"
    return "Plugin status: all systems nominal"

def register(ctx):
    ctx.register_command(
        "mystatus",
        handler=_handle_status,
        description="Show plugin status",
    )
```

After registration, users can type `/mystatus` in any session. The command appears in autocomplete, `/help` output, and the Telegram bot menu.

**Signature:** `ctx.register_command(name: str, handler: Callable, description: str = "", args_hint: str = "")`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Command name without the leading slash (e.g. `"lcm"`, `"mystatus"`) |
| `handler` | `Callable[[str], str \| None]` | Called with the raw argument string. May also be `async`. |
| `description` | `str` | Shown in `/help`, autocomplete, and Telegram bot menu |

**Key differences from `register_cli_command()`:**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| Invoked as | `/name` in a session | `hermes name` in a terminal |
| Where it works | CLI sessions, Telegram, Discord, etc. | Terminal only |
| Handler receives | Raw args string | argparse `Namespace` |
| Use case | Diagnostics, status, quick actions | Complex subcommand trees, setup wizards |

**Conflict protection:** If a plugin tries to register a name that conflicts with a built-in command (`help`, `model`, `new`, etc.), the registration is silently rejected with a log warning. Built-in commands always take precedence.

**Async handlers:** The gateway dispatch automatically detects and awaits async handlers, so you can use either sync or async functions:

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### Dispatch tools from slash commands

Slash command handlers that need to orchestrate tools (spawn a subagent via `delegate_task`, call `file_edit`, etc.) should use `ctx.dispatch_tool()` instead of reaching into framework internals. The parent-agent context (workspace hints, spinner, model inheritance) is wired up automatically.

```python
def register(ctx):
    def _handle_deliver(raw_args: str):
        result = ctx.dispatch_tool(
            "delegate_task",
            {
                "goal": raw_args,
                "toolsets": ["terminal", "file", "web"],
            },
        )
        return result

    ctx.register_command(
        "deliver",
        handler=_handle_deliver,
        description="Delegate a goal to a subagent",
    )
```

**Signature:** `ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tool name as registered in the tool registry (e.g. `"delegate_task"`, `"file_edit"`) |
| `args` | `dict` | Tool arguments, same shape the model would send |
| `parent_agent` | `Agent \| None` | Optional override. When omitted, resolves from the current CLI agent (or degrades gracefully in gateway mode) |

**Runtime behavior:**

- **CLI mode:** `parent_agent` is resolved from the active CLI agent so workspace hints, spinner, and model selection inherit as expected.
- **Gateway mode:** There is no CLI agent, so tools degrade gracefully — workspace is read from the configured terminal working directory and no spinner is shown.
- **Explicit override:** If the caller passes `parent_agent=` explicitly, it is respected and not overwritten.

This is the public, stable interface for tool dispatch from plugin commands. Plugins should not reach into `ctx._cli_ref.agent` or similar private state.

### Act from inside a hook (profile + tools)

`ctx._cli_ref` is only populated in an **interactive CLI** session. It is `None` in the gateway, in non-interactive `hermes chat -q` runs, and in **kanban-spawned worker sessions** — so any plugin logic that reaches through `_cli_ref` silently no-ops in exactly those contexts. Two stable, session-agnostic APIs cover what hooks actually need:

- **`ctx.profile_name`** — the active profile name (e.g. `"default"`, or the assignee profile in a kanban worker). Derived from `HERMES_HOME`, so it works everywhere with no `_cli_ref` dependency.
- **`ctx.dispatch_tool(name, args)`** — invoke any registered tool (built-in or plugin), including the `kanban_*` tools, `delegate_task`, `terminal`, `read_file`, etc. Works from hook callbacks regardless of which process the hook fires in.

Together these let a kanban lifecycle hook observe a transition and act on the board without touching framework internals:

```python
def register(ctx):
    def on_blocked(*, task_id, reason=None, **kw):
        # Runs in the worker process; ctx._cli_ref is None here.
        ctx.dispatch_tool("kanban_comment", {
            "task_id": task_id,
            "comment": f"[{ctx.profile_name}] auto-noted block: {reason}",
        })
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

For running a full `hermes <subcommand>` (e.g. `hermes kanban show`), shell out with the `terminal` tool via `ctx.dispatch_tool("terminal", {"command": "hermes kanban show ..."})` — there is no in-process slash-command bridge for headless worker sessions, and tools are the supported way to drive Hermes from a hook.

### Handle Slack Block Kit button clicks

Plugins that post Block Kit messages with interactive elements (buttons, overflow menus, datepickers, etc.) can register the click handlers directly with the Slack adapter — no monkey-patching of `slack_bolt.AsyncApp` required.

```python
def register(ctx):
    async def _on_approve(ack, body, action):
        # ack within 3 seconds — slack_bolt requirement.
        await ack()
        # body["channel"]["id"], body["user"]["id"], body["message"]["ts"]
        # action["action_id"], action["value"]
        sweep_id = (action.get("value") or "").split("|", 1)[-1]
        # ...do the deterministic work, then post a follow-up.

    ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
```

**Signature:** `ctx.register_slack_action_handler(action_id, callback) -> None`

| Parameter | Type | Description |
|-----------|------|-------------|
| `action_id` | `str \| re.Pattern \| dict` | Whatever `slack_bolt.App.action()` accepts: a literal `action_id`, a compiled regex matching multiple ids, or a constraint dict like `{"action_id": "...", "block_id": "..."}` |
| `callback` | async callable | Receives `(ack, body, action)` per the slack_bolt convention |

**Runtime behavior:**

- The handler is queued at plugin-load time and wired into the adapter's `slack_bolt.AsyncApp` when the Slack platform connects.
- Each callback is wrapped defensively: if your handler raises, the gateway logs the error and best-effort-acks the click so Slack stops retrying.
- Standard slack_bolt rules apply — `await ack()` within 3 seconds, then do longer work.
- For multi-workspace deployments the handler fires for clicks from any connected workspace; use `body["team"]["id"]` if you need to scope behaviour.

This is the public way for plugins to participate in Slack interactivity. Older plugins may patch `SlackAdapter.connect`; prefer this API instead. For the full slack_bolt surface (events, shortcuts, commands — not just Block Kit actions), use the generic `register_platform_handler("slack", ...)` below.

### Register native platform handlers (any platform)

Plugins that need to receive platform events the core adapter doesn't route — extra update types, native button callbacks, reaction/member events, webhook routes — can register a handler factory that the platform's adapter invokes at connect time. This works on **every** gateway platform.

```python
def register(ctx):
    def _wire(native, adapter):
        # native: the platform's client/app object (see table below)
        # adapter: the platform adapter instance (treat as read-only)
        # Import platform SDKs HERE so register() works without them.
        ...

    ctx.register_platform_handler("discord", _wire)
```

**Signature:** `ctx.register_platform_handler(platform, factory) -> None`

| Parameter | Type | Description |
|-----------|------|-------------|
| `platform` | `str` | Gateway platform name, lowercase (`"telegram"`, `"discord"`, `"slack"`, `"matrix"`, ...) |
| `factory` | callable | Receives `(native, adapter)` at connect time |

**What `native` is, per platform:**

| Platform | `native` object | Typical hooks |
|----------|-----------------|---------------|
| `telegram` | PTB `Application` | `add_handler` — any update type, pattern-scoped callbacks |
| `discord` | `discord.ext.commands.Bot` | `add_listener` — reactions, member events, threads, voice |
| `slack` | `slack_bolt.AsyncApp` | `app.event()` / `app.action()` / `app.command()` |
| `matrix` | Matrix client | event callbacks |
| `teams` | Teams `App` | `on_message` / `on_card_action` decorators |
| `dingtalk` | `DingTalkStreamClient` | `register_callback_handler` for other stream topics |
| `feishu` | lark_oapi client | API calls; event routing |
| `line`, `api_server`, `msgraph_webhook` | aiohttp `web.Application` | `router.add_get/post` — custom routes (wired before the router freezes) |
| everything else (whatsapp, signal, irc, email, sms, ntfy, wecom, weixin, bluebubbles, yuanbao, ...) | `None` | connect-time hook; work through the `adapter` handle |

**Runtime behavior:**

- Factories are queued at plugin-load time and invoked when the platform connects — for platforms where dispatch order matters (Telegram, Slack, Teams, aiohttp routers) they run **before** the core handlers register, so scoped plugin handlers take precedence and everything else falls through.
- **Always scope handlers you add to first-match dispatch tables.** On Telegram, use `CallbackQueryHandler(..., pattern=r"^myplugin:")` — an unscoped handler would swallow the core button flows (exec approvals, model picker, clarify prompts).
- Each factory is isolated: if it raises, the error is logged and the platform still connects.
- Import platform SDKs inside the factory body, not at module level — `register()` must work when the SDK isn't installed.
- One plugin can register factories for several platforms; each fires only when its platform connects.

**Telegram alias:** `ctx.register_telegram_handler(factory)` is a back-compat alias for `ctx.register_platform_handler("telegram", factory)`.

Example — Telegram, pattern-scoped inline buttons:

```python
def register(ctx):
    def _wire(application, adapter):
        from telegram.ext import CallbackQueryHandler

        async def _on_button(update, context):
            query = update.callback_query
            await query.answer()
            # ...handle "myplugin:*" callbacks

        application.add_handler(
            CallbackQueryHandler(_on_button, pattern=r"^myplugin:")
        )

    ctx.register_platform_handler("telegram", _wire)
```

Example — Discord, reaction events:

```python
def register(ctx):
    def _wire(bot, adapter):
        async def on_raw_reaction_add(payload):
            ...  # e.g. reaction-based voting / moderation

        bot.add_listener(on_raw_reaction_add, "on_raw_reaction_add")

    ctx.register_platform_handler("discord", _wire)
```

:::tip
This guide covers **general plugins** (tools, hooks, slash commands, CLI commands). The sections below sketch the authoring pattern for each specialized plugin type; each links to its full guide for field reference and examples.
:::

## Specialized plugin types

Hermes has five specialized plugin types beyond the general surface. Each ships as a directory under `plugins/<category>/<name>/` (bundled) or `~/.hermes/plugins/<category>/<name>/` (user). The contract differs by category — pick the one you need, then read its full guide.

### Model provider plugins — add an LLM backend

Drop a profile into `plugins/model-providers/<name>/`:

```python
# plugins/model-providers/acme/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="acme",
    aliases=("acme-inference",),
    display_name="Acme Inference",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=("acme-large-v3", "acme-medium-v3"),
))
```

```yaml
# plugins/model-providers/acme/plugin.yaml
name: acme-provider
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
```

Lazy-discovered the first time anything calls `get_provider_profile()` or `list_providers()` — `auth.py`, `config.py`, `doctor.py`, `models.py`, `runtime_provider.py`, and the chat_completions transport auto-wire to it. User plugins override bundled ones by name.

**Full guide:** [Model Provider Plugins](/developer-guide/model-provider-plugin) — field reference, overridable hooks (`prepare_messages`, `build_extra_body`, `build_api_kwargs_extras`, `fetch_models`), api_mode selection, auth types, testing.

### Platform plugins — add a gateway channel

Drop an adapter into `plugins/platforms/<name>/`:

```python
# plugins/platforms/myplatform/adapter.py
from gateway.platforms.base import BasePlatformAdapter

class MyPlatformAdapter(BasePlatformAdapter):
    async def connect(self): ...
    async def send(self, chat_id, text): ...
    async def disconnect(self): ...

def check_requirements():
    import os
    return bool(os.environ.get("MYPLATFORM_TOKEN"))

def _env_enablement():
    import os
    tok = os.getenv("MYPLATFORM_TOKEN", "").strip()
    if not tok:
        return None
    return {"token": tok}

def register(ctx):
    ctx.register_platform(
        name="myplatform",
        label="MyPlatform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        required_env=["MYPLATFORM_TOKEN"],
        # Auto-populate PlatformConfig.extra from env so env-only setups
        # show up in `hermes gateway status` without SDK instantiation.
        env_enablement_fn=_env_enablement,
        # Opt in to cron delivery: `deliver=myplatform` routes to this var.
        cron_deliver_env_var="MYPLATFORM_HOME_CHANNEL",
        emoji="💬",
        platform_hint="You are chatting via MyPlatform. Keep responses concise.",
    )
```

```yaml
# plugins/platforms/myplatform/plugin.yaml
name: myplatform-platform
label: MyPlatform
kind: platform
version: 1.0.0
description: MyPlatform gateway adapter
requires_env:
  - name: MYPLATFORM_TOKEN
    description: "Bot token from the MyPlatform console"
    password: true
optional_env:
  - name: MYPLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

**Full guide:** [Adding Platform Adapters](/developer-guide/adding-platform-adapters) — complete `BasePlatformAdapter` contract, message routing, auth gating, setup wizard integration. Look at `plugins/platforms/irc/` for a stdlib-only working example.

### Memory provider plugins — add a cross-session knowledge backend

Drop an implementation of `MemoryProvider` into `plugins/memory/<name>/`:

```python
# plugins/memory/my-memory/__init__.py
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_content, assistant_content, *,
                  session_id="", messages=None) -> None:
        ...

    def prefetch(self, query, *, session_id="") -> str:
        ...

    def get_tool_schemas(self) -> list[dict]:
        return []   # required @abstractmethod — see full guide

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

Memory providers are single-select — only one is active at a time, chosen via `memory.provider` in `config.yaml`.

**Full guide:** [Memory Provider Plugins](/developer-guide/memory-provider-plugin) — full `MemoryProvider` ABC, threading contract, profile isolation, CLI command registration via `cli.py`.

### Context engine plugins — replace the context compressor

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def update_from_response(self, usage) -> None: ...
    def should_compress(self, prompt_tokens: int = None) -> bool: ...
    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context="") -> list: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

Context engines are single-select — chosen via `context.engine` in `config.yaml`.

**Full guide:** [Context Engine Plugins](/developer-guide/context-engine-plugin).

### Image-generation backends

Drop a provider into `plugins/image_gen/<name>/`:

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, aspect_ratio="landscape", **kwargs) -> dict:
        # returns success_response(...) / error_response(...)
        ...

def register(ctx):
    ctx.register_image_gen_provider(MyImageGenProvider())
```

```yaml
# plugins/image_gen/my-imggen/plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
description: Custom image generation backend
```

**Full guide:** [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) — full `ImageGenProvider` ABC, `list_models()` / `get_setup_schema()` metadata, `success_response()`/`error_response()` helpers, base64 vs URL output, user overrides, pip distribution.

**Reference examples:** `plugins/image_gen/openai/` (DALL-E / GPT-Image via OpenAI SDK), `plugins/image_gen/openai-codex/`, `plugins/image_gen/xai/` (Grok image gen).

## Non-Python extension surfaces

Hermes also accepts extensions that aren't Python plugins at all. These are shown in the [Pluggable interfaces table](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each); the sections below sketch each authoring style briefly.

### MCP servers — register external tools

Model Context Protocol (MCP) servers register their own tools into Hermes without any Python plugin. Declare them in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    timeout: 120

  linear:
    url: "https://mcp.linear.app/sse"
    auth:
      type: "oauth"
```

Hermes connects to each server at startup, lists its tools, and registers them alongside built-ins. The LLM sees them exactly like any other tool. **Full guide:** [MCP](/user-guide/features/mcp).

### Gateway event hooks — fire on lifecycle events

Drop a manifest + handler into `~/.hermes/hooks/<name>/`:

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Send a push notification when a long task finishes
events:
  - agent:end
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
async def handle(event_type: str, context: dict) -> None:
    if context.get("duration_seconds", 0) > 120:
        # send notification …
        pass
```

Events include `gateway:startup`, `session:start`, `session:end`, `session:reset`, `agent:start`, `agent:step`, `agent:end`, and wildcard `command:*`. Errors in hooks are caught and logged — they never block the main pipeline.

**Full guide:** [Gateway Event Hooks](/user-guide/features/hooks#gateway-event-hooks).

### Shell hooks — run a shell command on tool calls

If you just want to run a script when a tool fires (notifications, audit logs, desktop alerts, auto-formatters), use shell hooks in `config.yaml` — no Python required:

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

Supports all the same events as Python plugin hooks (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `pre_gateway_dispatch`) plus structured JSON output for `pre_tool_call` blocking decisions.

**Full guide:** [Shell Hooks](/user-guide/features/hooks#shell-hooks).

### Skill sources — add a custom skill registry

If you maintain a GitHub repo of skills (or want to pull from a community index beyond the built-in sources), add it as a **tap**:

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

Publishing your own tap is just a GitHub repo with `skills/<skill-name>/SKILL.md` directories — no server or registry signup needed.

**Full guides:** [Skills Hub](/user-guide/features/skills#skills-hub) · [Publishing a custom tap](/user-guide/features/skills#publishing-a-custom-skill-tap) (repo layout, minimal example, non-default paths, trust levels).

### TTS / STT via command templates

Any CLI that reads/writes audio or text can be plugged in through `config.yaml` — no Python code:

```yaml
tts:
  provider: voxcpm
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      voice_compatible: true
```

For STT, point `HERMES_LOCAL_STT_COMMAND` at an argv-tokenized template. It runs without implicit shell interpretation; wrap it in `sh -c`, `cmd /c`, or PowerShell explicitly if the trusted local command requires shell syntax. Supported placeholders: `{input_path}`, `{output_path}`, `{format}`, `{voice}`, `{model}`, `{speed}` (TTS); `{input_path}`, `{output_dir}`, `{language}`, `{model}` (STT). Any path-interacting CLI is automatically a plugin.

**Full guides:** [TTS custom command providers](/user-guide/features/tts#custom-command-providers) · [STT](/user-guide/features/tts#voice-message-transcription-stt).

## Distribute via pip

For sharing plugins publicly, add an entry point to your Python package:

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# Plugin auto-discovered on next hermes startup
```

## Distribute for NixOS

:::warning Nix is no longer explicitly supported
Nix/NixOS is no longer an explicitly supported install path (best-effort only) — see [Nix Setup](/getting-started/nix-setup). This section is kept for users already deploying on NixOS.
:::

NixOS users can install your plugin declaratively if you provide a `pyproject.toml` with entry points:

**Entry-point plugins** (recommended for distribution):
```nix
# User's configuration.nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

**Directory plugins** (no `pyproject.toml` needed):
```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "you";
    repo = "hermes-my-plugin";
    rev = "v1.0.0";
    hash = "sha256-...";
  })
];
```

See the [Nix Setup guide](/getting-started/nix-setup#plugins) for complete documentation including overlay usage and collision checking.

## Common mistakes

**Handler doesn't return JSON string:**
```python
# Wrong — returns a dict
def handler(args, **kwargs):
    return {"result": 42}

# Right — returns a JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**Missing `**kwargs` in handler signature:**
```python
# Wrong — will break if Hermes passes extra context
def handler(args):
    ...

# Right
def handler(args, **kwargs):
    ...
```

**Handler raises exceptions:**
```python
# Wrong — exception propagates, tool call fails
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# Right — catch and return error JSON
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**Schema description too vague:**
```python
# Bad — model doesn't know when to use it
"description": "Does stuff"

# Good — model knows exactly when and how
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
