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
    schema={...},                  # Model-facing schema (description, parameters)
    handler=handle_terminal,       # The function that executes when the tool is called
    check_fn=check_terminal,       # Optional: returns True/False for availability
    requires_env=["SOME_VAR"],     # Optional: env vars needed (for UI display)
    is_async=False,                # Whether the handler is an async coroutine
    description="Run commands",    # Optional ToolEntry registry metadata
    emoji="💻",                    # Emoji for spinner/progress display
)
```

Each call creates a `ToolEntry` stored in the singleton `ToolRegistry._tools` dict keyed by tool name. A registration that would shadow an existing tool from a **different** toolset is rejected (with an error log) unless the caller passes `override=True`; plugin overrides of built-in tools additionally require the operator opt-in `plugins.entries.<plugin_id>.allow_tool_override: true` in `config.yaml`.

`schema["description"]` is the authoritative model-facing description. The separate `description=` argument populates `ToolEntry.description`; when it is omitted, the registry metadata falls back to the schema description. `get_definitions()` builds the OpenAI function definition from `entry.schema` and does not copy `entry.description` into a schema that lacks `description`. Therefore, `description=` alone does not describe the tool to the model, and when both values differ the model sees the schema value. Prefer defining the description once in the schema unless a registry consumer intentionally needs different metadata.

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

1. **MCP tools** — `tools.mcp_tool_discovery.discover_mcp_tools()` (re-exported by the `tools.mcp_tool` facade) reads MCP server config and registers tools from external servers.
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
agent loop (`agent/conversation_loop.py`, via `run_agent.py`'s `AIAgent` facade)
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

### Executor abandonment and nested Clarify

A plugin invoking `clarify` still has its outer tool's finite executor deadline;
waiting for an answer does not exempt arbitrary plugin work from that deadline or
borrow approval-only human-wait accounting. The platform callback is framework
context, never a value supplied in model arguments.

Sequential and concurrent workers bind a call-local Clarify wait scope. Prepared
terminal workers create it before middleware/consent preparation; the sequential
executor reuses that same scope with their future, and batch closure cancels all
of its slots. When an executor abandons that call, it removes and wakes only that scope's gateway
registrations. Registration and cancellation share the queue lock, so a late
callback cannot leave a new actionable prompt. A gateway waiter receives the
existing cancellation sentinel so the platform can retire its card and disarm
late-delivery watchers. Before returning to plugin code, `clarify_tool` raises the
internal `ClarifyWaitAbandoned` control-flow exception; the executor catches it.
It deliberately bypasses ordinary `Exception`-to-tool-error wrapping: an abandoned
answer (or callback error) must not resume the plugin as a normal result. This is
not the session `/stop` signal and does not clear unrelated prompts. An
already-abandoned call shows no prompt, so `on_human_input_request` does not fire;
a prompt that was shown still fires `on_human_input_resolved` exactly once, with
the callback's outcome (`cancelled` from a gateway wait the executor abandoned).

Concurrent workers and their executor share a call-local terminal-event claim:
the first `post_tool_call` event wins. Hooks are not deferred or reordered:
`post_tool_call` still precedes `transform_tool_result` in the worker. If timeout
wins before dispatch finishes, later worker events are suppressed. If dispatch
already emitted its terminal event before outer middleware stalls, that event
cannot be retracted; the model-facing outer result may still time out, without a
second terminal event. Sequential registry calls retain their existing
outer-owned terminal-event suppression.

Delivery failure releases only its exact prompt, never every prompt in the
session. Initial and fallback sends recheck ownership under the queue lock when
scheduled, then again when their coroutine starts. The text fallback keeps its
human-decision `is_approval_prompt` marker, so it is still pushed on Telegram.
A boundary/flush wait cannot cause an already-abandoned worker to start a new prompt.
A send already in flight when abandonment occurs may still reach the platform;
this mechanism does not claim network rollback.

This is cooperative settlement, not general thread termination. Non-cooperating
plugin code can still outlive its deadline and its effects remain unknown; the
executor must not wait indefinitely for it or claim rollback. Normal submitted
answers and platform cancellation keep their existing results.

### Error wrapping

All tool execution is wrapped in error handling at two levels:

1. **`registry.dispatch()`** — catches any exception from the handler and returns `{"error": "Tool execution failed: ExceptionType: message"}` as JSON.

2. **`handle_function_call()`** — wraps the entire dispatch in a secondary try/except that returns `{"error": "Error executing tool_name: message"}`.

This ensures the model always receives a well-formed JSON string, never an unhandled exception.

### Agent-loop tools

Four tools are intercepted before registry dispatch because they need agent-level state (TodoStore, MemoryStore, etc.):

- `todo_list` — planning/task tracking
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

`tools/process_registry_checkpoint.py` owns running-process checkpoints and
PID-safe adoption. Completed output is separate: `tools/process_registry_results.py`
writes one atomic, redacted receipt per process under the profile's
`logs/process-results/`. Producers cannot overwrite another parent's results by
rewriting the shared PID checkpoint. The registry persists the receipt before
releasing its completion event; one-shot linger waits on that event. The existing
process query methods load retained snapshots without adopting PIDs or enqueuing
notifications. Reads require the commissioning durable session or its compression
continuation; knowing a handle alone does not authorize a retained result read.
The registry captures that owner before starting any output reader, including on
CLI and non-notifying processes, and preserves the producer's profile context in
reader threads. Receipt redaction is forced independently of live-output opt-out;
retention is bounded by age and count.

## Concurrency

Tool calls may execute sequentially or concurrently depending on the tool mix and interaction requirements.

## Related docs

- [Toolsets Reference](../reference/toolsets-reference.md)
- [Built-in Tools Reference](../reference/tools-reference.md)
- [Agent Loop Internals](./agent-loop.md)
- [ACP Internals](./acp-internals.md)
