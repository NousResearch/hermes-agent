---
sidebar_position: 9
title: "Tools Runtime"
description: "Runtime behavior of the tool registry, toolsets, dispatch, and terminal environments"
---

# Tools Runtime

Hermes tools are self-registering functions grouped into toolsets and executed through a central registry/dispatch system.

Primary files:

- `tools/registry.py`
- `model_tools.py`
- `toolsets.py`
- `agent/tool_executor.py`
- `tools/approval.py`
- `tools/terminal_tool.py`
- `tools/environments/*`

## Tool registration model

Each tool module calls `registry.register(...)` at import time.

`model_tools.py` is responsible for importing/discovering tool modules and building the schema list used by the model.

### How `registry.register()` works

Each self-registering tool module in `tools/` calls `registry.register()` at module level to declare itself. The registration shape is:

```python
registry.register(
    name="terminal",               # Unique tool name (used in API schemas)
    toolset="terminal",            # Toolset this tool belongs to
    schema={...},                  # OpenAI function-calling schema (description, parameters)
    handler=handle_terminal,       # The function that executes when the tool is called
    check_fn=check_terminal,       # Optional: returns True/False for availability
    requires_env=["SOME_VAR"],     # Optional: env vars needed (for UI display)
    is_async=False,                # Whether the handler is an async coroutine
    description="Run commands",    # Human-readable description
    emoji="💻",                    # Emoji for spinner/progress display
    max_result_size_chars=None,    # Optional per-tool output budget
    dynamic_schema_overrides=None, # Optional callable for runtime schema fields
    override=False,                # Explicitly replace a tool from another toolset
)
```

Each call creates a `ToolEntry` stored in the singleton `ToolRegistry._tools` dict keyed by tool name. If that name is already registered:

- Re-registering it in the same toolset replaces the existing entry silently. This supports reconnect and refresh flows.
- If both the existing and new toolsets start with `mcp-`, the new MCP entry replaces the old one and the collision is logged at `DEBUG` level. This supports MCP server refreshes and overlapping names from different MCP servers.
- For any other cross-toolset replacement, `override=True` explicitly allows the new entry to replace the old one and logs the override at `INFO` level. Plugin overrides also require the operator to enable `plugins.entries.<plugin_id>.allow_tool_override` in `config.yaml`.
- Otherwise, the registry logs an error, rejects the new registration, and keeps the existing entry. Accidental cross-toolset shadowing is never allowed by default.

### Discovery: `discover_builtin_tools()`

When `model_tools.py` is imported, it calls `discover_builtin_tools()` from `tools/registry.py`. This function scans every `tools/*.py` file using AST parsing to find modules that contain top-level `registry.register()` calls, then imports them:

```python
# tools/registry.py (simplified)
def discover_builtin_tools(tools_dir=None):
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).parent
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        if _module_registers_tools(path):  # AST check for top-level registry.register()
            importlib.import_module(f"tools.{path.stem}")
```

This auto-discovery means new tool files are picked up automatically — no manual list to maintain. The AST check only matches top-level `registry.register()` calls (not calls inside functions), so helper modules in `tools/` are not imported.

Each import triggers the module's `registry.register()` calls. Import errors in optional tool modules are caught and logged — they don't prevent other tools from loading.

After core tool discovery, MCP tools and plugin tools are also discovered:

1. **MCP tools** — `tools.mcp_tool.discover_mcp_tools()` reads MCP server config and registers tools from external servers.
2. **Plugin tools** — `hermes_cli.plugins.discover_plugins()` loads user/project/pip plugins that may register additional tools.

## Tool availability checking (`check_fn`)

Each tool can optionally provide a `check_fn` — a callable that returns `True` when the tool is available and `False` otherwise. Typical checks include:

- **API key present** — e.g., `lambda: bool(os.environ.get("SERP_API_KEY"))` for web search
- **Service running** — e.g., checking if the Honcho server is configured
- **Binary installed** — e.g., verifying `playwright` is available for browser tools

When `registry.get_definitions()` builds the schema list for the model, it evaluates availability through `_check_fn_cached()`:

```python
# Simplified from registry.py
if entry.check_fn and not _check_fn_cached(entry.check_fn):
    continue  # Skip this tool for this schema build
```

Key behaviors:

- A per-call cache ensures that multiple tools sharing the same `check_fn` evaluate it only once during one `get_definitions()` pass.
- Results are also cached across calls for about 30 seconds. Call `invalidate_check_fn_cache()` after relevant configuration changes to clear this cache immediately.
- A `False` result or exception within about 60 seconds of that function's last successful check is treated as transient: the tool remains available, the failure is not cached, and the next call probes again. Without a recent success, the failure is cached and the tool is omitted.
- `is_toolset_available()` evaluates the registered tools individually and returns `True` when at least one tool in the toolset is exposable. A single unavailable tool does not hide a mixed toolset.

## Toolset resolution

Toolsets are named bundles of tools. Hermes resolves them through:

- explicit enabled/disabled toolset lists
- platform presets (`hermes-cli`, `hermes-telegram`, etc.)
- dynamic MCP toolsets
- curated special-purpose sets like `hermes-acp`

### How `get_tool_definitions()` filters tools

The main entry point is `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`:

1. **If `enabled_toolsets` is provided** — only tools from those toolsets are included. Each toolset name is resolved via `resolve_toolset()` which expands composite toolsets into individual tool names.

2. **If `disabled_toolsets` is provided** — subtract those toolsets from the selected set at the end, whether the starting set came from `enabled_toolsets` or from the default. For platform bundles and posture toolsets, Hermes preserves shared core tools and subtracts only the bundle's non-core delta.

3. **If `enabled_toolsets` is omitted** — start with all known toolsets before applying any disabled-toolset subtraction.

4. **Registry filtering** — the resolved tool name set is passed to `registry.get_definitions()`, which applies `check_fn` filtering and returns OpenAI-format schemas.

5. **Dynamic schema patching** — `registry.get_definitions()` first applies each entry's optional `dynamic_schema_overrides`. `model_tools` then rebuilds `execute_code` from the available sandbox tools, rebuilds `discord` / `discord_admin` from detected capabilities and configuration, and removes unavailable web-tool references from `browser_navigate`.

### Legacy toolset names

Old toolset names with `_tools` suffixes (e.g., `web_tools`, `terminal_tools`) are mapped to their modern tool names via `_LEGACY_TOOLSET_MAP` for backward compatibility.

## Dispatch

At runtime, registry-backed tools are dispatched through the central registry. Tools that require live agent state, callbacks, or provider-owned state are routed by the agent runtime instead.

### Dispatch flow: model tool_call → handler execution

When the model returns a `tool_call`, the flow is:

```
Model response with tool_call
    ↓
agent/tool_executor.py
    ↓
[Tool request middleware]
    ↓
[Plugin pre-hook + tool-loop guardrails]
    ↓
[Agent-owned tool?] → execute with agent/context/provider state
[Registry-backed tool?] → model_tools.handle_function_call(...)
    ↓
[Tool execution middleware]
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
Look up ToolEntry by name
[Async handler?] → bridge via _run_async()
[Sync handler?]  → call directly
    ↓
Normalize to a string or supported multimodal envelope
    ↓
[Plugin post-hook]
    ↓
[Registry-backed only: optional transform_tool_result hook]
```

### Error wrapping

Registry-backed tool execution is wrapped in error handling at two levels:

1. **`registry.dispatch()`** — catches handler exceptions, sanitizes the error, and returns a JSON error string. It accepts normal string results and the supported multimodal envelope; unsupported result types become a `tool_result_contract` JSON error.

2. **`handle_function_call()`** — wraps orchestration around registry dispatch in a secondary try/except and also returns a sanitized JSON error string on failure.

Agent-owned paths perform their own exception handling in the tool executor. Successful tool results do not have to be JSON; ordinary handlers return strings, while the registry and orchestration error wrappers above use the JSON error shape.

### Agent-owned tools

Several built-in tools are intercepted before registry dispatch because they need agent-level state or runtime callbacks:

- `todo` — planning/task tracking
- `memory` — persistent memory writes
- `session_search` — cross-session recall
- `delegate_task` — spawns subagent sessions
- `clarify` — uses the active clarification callback
- `read_terminal` — reads from the active terminal UI callback

Context-engine tools and memory-provider tools are also routed through their owning runtime components rather than the central registry. The built-in agent-owned tools above still register schemas for `get_tool_definitions`; the agent runtime supplies the state and callbacks required for their normal execution path.

### Async bridging

When a tool handler is async, `_run_async()` bridges it to the sync dispatch path:

- **CLI path (no running loop)** — uses a persistent event loop to keep cached async clients alive
- **Gateway path (running loop)** — spins up a disposable thread with `asyncio.run()`
- **Worker threads (parallel tools)** — uses per-thread persistent loops stored in thread-local storage

## The DANGEROUS_PATTERNS approval flow

The terminal tool integrates a dangerous-command approval system defined in `tools/approval.py`:

1. **Pattern detection** — `DANGEROUS_PATTERNS` is a list of `(regex, description)` tuples covering destructive operations:
   - Recursive deletes (`rm -rf`)
   - Filesystem formatting (`mkfs`, `dd`)
   - SQL destructive operations (`DROP TABLE`, `DELETE FROM` without `WHERE`)
   - System config overwrites (`> /etc/`)
   - Service manipulation (`systemctl stop`)
   - Remote code execution (`curl | sh`)
   - Fork bombs, process kills, etc.

2. **Detection** — before executing a command in a guarded terminal environment, `check_all_command_guards()` combines hardline blocks, user deny rules, Tirith findings, and `detect_dangerous_command(command)`. Isolated container backends skip these guards; Docker stops skipping them when host paths are mounted.

3. **Approval prompt** — if a match is found:
   - **CLI mode** — an interactive prompt asks the user to approve, deny, or allow permanently
   - **Gateway mode** — an async approval callback sends the request to the messaging platform
   - **Smart approval** — optionally, an auxiliary LLM can auto-approve low-risk commands that match patterns (e.g., `rm -rf node_modules/` is safe but matches "recursive delete")

4. **Session state** — approvals are tracked per-session. Once you approve "recursive delete" for the session, subsequent commands matching that pattern don't re-prompt. Catastrophic hardline commands and user deny rules remain blocked before the approval bypasses.

5. **Permanent allowlist** — the "allow permanently" option writes the pattern to `config.yaml`'s `command_allowlist`, persisting across sessions.

## Terminal/runtime environments

The terminal system supports multiple backends:

- local
- docker
- ssh
- singularity
- modal
- daytona

It also supports:

- per-task cwd overrides
- background process management
- PTY mode
- approval callbacks for dangerous commands

## Concurrency

Tool calls may execute sequentially or concurrently depending on the tool mix and interaction requirements.

## Related docs

- [Toolsets Reference](../reference/toolsets-reference.md)
- [Built-in Tools Reference](../reference/tools-reference.md)
- [Agent Loop Internals](./agent-loop.md)
- [ACP Internals](./acp-internals.md)
