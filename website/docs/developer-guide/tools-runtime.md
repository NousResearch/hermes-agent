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
- `tools/terminal_tool.py`
- `tools/environments/*`

## Tool registration model

Each tool module calls `registry.register(...)` at import time.

`model_tools.py` is responsible for importing/discovering tool modules and building the schema list used by the model.

### How `registry.register()` works

Every tool file in `tools/` calls `registry.register()` at module level to declare itself. The function signature is:

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
)
```

Each call creates a `ToolEntry` stored in the singleton `ToolRegistry._tools` dict keyed by tool name. If a name collision occurs across toolsets, a warning is logged and the later registration wins.

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

Each import triggers the module's `registry.register()` calls. Errors in optional tools (e.g., missing `fal_client` for image generation) are caught and logged — they don't prevent other tools from loading.

After core tool discovery, MCP tools and plugin tools are also discovered:

1. **MCP tools** — `tools.mcp_tool.discover_mcp_tools()` reads MCP server config and registers tools from external servers.
2. **Plugin tools** — `hermes_cli.plugins.discover_plugins()` loads user/project/pip plugins that may register additional tools.

## Tool availability checking (`check_fn`)

Each tool can optionally provide a `check_fn` — a callable that returns `True` when the tool is available and `False` otherwise. Typical checks include:

- **API key present** — e.g., `lambda: bool(os.environ.get("SERP_API_KEY"))` for web search
- **Service running** — e.g., checking if the Honcho server is configured
- **Binary installed** — e.g., verifying `playwright` is available for browser tools

When `registry.get_definitions()` builds the schema list for the model, it runs each tool's `check_fn()`:

```python
# Simplified from registry.py
if entry.check_fn:
    try:
        available = bool(entry.check_fn())
    except Exception:
        available = False   # Exceptions = unavailable
    if not available:
        continue            # Skip this tool entirely
```

Key behaviors:
- Check results are **cached per-call** — if multiple tools share the same `check_fn`, it only runs once.
- Exceptions in `check_fn()` are treated as "unavailable" (fail-safe).
- The `is_toolset_available()` method checks whether a toolset's `check_fn` passes, used for UI display and toolset resolution.

## Toolset resolution

Toolsets are named bundles of tools. Hermes resolves them through:

- explicit enabled/disabled toolset lists
- platform presets (`hermes-cli`, `hermes-telegram`, etc.)
- dynamic MCP toolsets
- curated special-purpose sets like `hermes-acp`

### How `get_tool_definitions()` filters tools

The main entry point is `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`:

1. **If `enabled_toolsets` is provided** — only tools from those toolsets are included. Each toolset name is resolved via `resolve_toolset()` which expands composite toolsets into individual tool names.

2. **If `disabled_toolsets` is provided** — start with ALL toolsets, then subtract the disabled ones.

3. **If neither** — include all known toolsets.

4. **Registry filtering** — the resolved tool name set is passed to `registry.get_definitions()`, which applies `check_fn` filtering and returns OpenAI-format schemas.

5. **Dynamic schema patching** — after filtering, `execute_code` and `browser_navigate` schemas are dynamically adjusted to only reference tools that actually passed filtering (prevents model hallucination of unavailable tools).

### Legacy toolset names

Old toolset names with `_tools` suffixes (e.g., `web_tools`, `terminal_tools`) are mapped to their modern tool names via `_LEGACY_TOOLSET_MAP` for backward compatibility.

## Dispatch

At runtime, tools are dispatched through the central registry, with agent-loop exceptions for some agent-level tools such as memory/todo/session-search handling.

### Dispatch flow: model tool_call → handler execution

When the model returns a `tool_call`, the flow is:

```
Model response with tool_call
    ↓
run_agent.py agent loop
    ↓
model_tools.handle_function_call(name, args, task_id, user_task)
    ↓
[Agent-loop tools?] → handled directly by agent loop (todo, memory, session_search, delegate_task)
    ↓
[Plugin pre-hook] → invoke_hook("pre_tool_call", ...)
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
Look up ToolEntry by name
    ↓
[Async handler?] → bridge via _run_async()
[Sync handler?]  → call directly
    ↓
Return result string (or JSON error)
    ↓
[Plugin post-hook] → invoke_hook("post_tool_call", ...)
```

### Error wrapping

All tool execution is wrapped in error handling at two levels:

1. **`registry.dispatch()`** — catches any exception from the handler and returns `{"error": "Tool execution failed: ExceptionType: message"}` as JSON.

2. **`handle_function_call()`** — wraps the entire dispatch in a secondary try/except that returns `{"error": "Error executing tool_name: message"}`.

This ensures the model always receives a well-formed JSON string, never an unhandled exception.

### Agent-loop tools

Four tools are intercepted before registry dispatch because they need agent-level state (TodoStore, MemoryStore, etc.):

- `todo` — planning/task tracking
- `memory` — persistent memory writes
- `session_search` — cross-session recall
- `delegate_task` — spawns subagent sessions

These tools' schemas are still registered in the registry (for `get_tool_definitions`), but their handlers return a stub error if dispatch somehow reaches them directly.

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

2. **Detection** — before executing any terminal command, `detect_dangerous_command(command)` checks against all patterns.

3. **Approval prompt** — if a match is found:
   - **CLI mode** — an interactive prompt asks the user to approve, deny, or allow permanently
   - **Gateway mode** — an async approval callback sends the request to the messaging platform
   - **Smart approval** — optionally, an auxiliary LLM can auto-approve low-risk commands that match patterns (e.g., `rm -rf node_modules/` is safe but matches "recursive delete")

4. **Session state** — approvals are tracked per-session. Once you approve "recursive delete" for a session, subsequent `rm -rf` commands don't re-prompt.

5. **Permanent allowlist** — the "allow permanently" option writes the pattern to `config.yaml`'s `command_allowlist`, persisting across sessions.

## Terminal/runtime environments

The terminal system supports multiple backends:

- local
- docker
- ssh
- singularity
- modal
- daytona
- vercel_sandbox

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
