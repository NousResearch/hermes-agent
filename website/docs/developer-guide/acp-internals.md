---
sidebar_position: 2
title: "ACP Internals"
description: "How the ACP adapter works: lifecycle, sessions, event bridge, approvals, and tool rendering"
---

# ACP Internals

The ACP adapter wraps Hermes' synchronous `AIAgent` in an async JSON-RPC stdio server.

Key implementation files:

- `acp_adapter/entry.py`
- `acp_adapter/server.py`
- `acp_adapter/session.py`
- `acp_adapter/events.py`
- `acp_adapter/permissions.py`
- `acp_adapter/tools.py`
- `acp_adapter/auth.py`

## Boot flow

```text
hermes acp / hermes-acp / python -m acp_adapter
  -> acp_adapter.entry.main()
  -> parse --version / --check / --setup before server startup
  -> load ~/.hermes/.env
  -> configure stderr logging
  -> construct HermesACPAgent
  -> acp.run_agent(agent, use_unstable_protocol=True)
```

Stdout is reserved for ACP JSON-RPC transport. Human-readable logs go to stderr.

## Major components

### `HermesACPAgent`

`acp_adapter/server.py` implements the ACP agent protocol.

Responsibilities:

- initialize / authenticate
- new/load/resume/fork/list/cancel/close session methods
- prompt execution
- session model switching
- wiring sync AIAgent callbacks into ACP async notifications

### `SessionManager`

`acp_adapter/session.py` tracks live ACP sessions.

Each session stores:

- `session_id`
- `agent`
- `cwd`
- `model`
- `history`
- `cancel_event`
- runtime queue/lock/interrupted-prompt state
- `mode` and `config_options`

The manager is thread-safe and supports:

- create
- get
- remove
- unload (preserving persisted history)
- fork
- list
- cleanup
- cwd updates

### Event bridge

`acp_adapter/events.py` converts AIAgent callbacks into ACP `session_update` events.

Bridged callbacks:

- `tool_progress_callback`
- `reasoning_callback` (provider/model reasoning; `thinking_callback` is set to
  `None` so local waiting-status text does not become an ACP thought)
- `step_callback`

Because `AIAgent` runs in a worker thread while ACP I/O lives on the main event loop, the bridge uses:

```python
asyncio.run_coroutine_threadsafe(...)
```

### Permission bridge

`acp_adapter/permissions.py` adapts dangerous terminal approval prompts into ACP permission requests.

Mapping:

- `allow_once` -> Hermes `once`
- `allow_session` -> Hermes `session`
- `allow_always` -> Hermes `always`
- reject options -> Hermes `deny`

Timeouts and bridge failures deny by default.

### Tool rendering helpers

`acp_adapter/tools.py` maps Hermes tools to ACP tool kinds and builds editor-facing content.

Examples:

- `patch` / `write_file` -> file diffs
- `terminal` -> shell command text
- `read_file` / `search_files` -> text previews
- large results -> truncated text blocks for UI safety

## Session lifecycle

```text
new_session(cwd)
  -> create SessionState
  -> create AIAgent(platform="acp", enabled_toolsets=["hermes-acp"])
  -> bind task_id/session_id to cwd override

prompt(..., session_id)
  -> convert text/image/resource blocks into model content
  -> reset cancel event
  -> install callbacks + approval bridge
  -> run AIAgent in ThreadPoolExecutor
  -> update session history
  -> emit final agent message chunk
```

### Cancelation

`cancel(session_id)`:

- sets the session cancel event
- requests a hard agent interrupt
- causes the prompt response to return `stop_reason="cancelled"`

### Load, resume, and close

- `load_session()` restores persisted state and replays history before its
  response.
- `resume_session()` restores state without replaying history.
- `close_session()` cancels active work, clears session approvals, and unloads
  the live session while preserving its `state.db` transcript for a later load.
  ACP-provided MCP servers are reference-counted and selectively released after
  their last owning session closes.

### Forking

`fork_session()` deep-copies message history into a new live session, preserving conversation state while giving the fork its own session ID and cwd.

## Provider/auth behavior

ACP does not implement its own auth store.

Instead it reuses Hermes' runtime resolver:

- `acp_adapter/auth.py`
- `hermes_cli/runtime_provider.py`

So ACP advertises and uses the currently configured Hermes provider/credentials. It also always advertises a terminal setup auth method (`hermes-setup`, args `--setup`) so first-run ACP clients can open Hermes' interactive model/provider configuration before starting a normal ACP session.

## Working directory binding

ACP sessions carry an editor cwd.

The session manager binds that cwd to the ACP session ID via task-scoped terminal/file overrides, so file and terminal tools operate relative to the editor workspace.

## Duplicate same-name tool calls

The event bridge tracks tool IDs FIFO per tool name, not just one ID per name. This is important for:

- parallel same-name calls
- repeated same-name calls in one step

Without FIFO queues, completion events would attach to the wrong tool invocation.

## Approval callback restoration

ACP temporarily installs an approval callback on the terminal tool during prompt execution, then restores the previous callback afterward. This avoids leaving ACP session-specific approval handlers installed globally forever.

## Current limitations

- Audio prompt blocks are not converted into model input.
- Linked resources must be local files to be read directly; embedded and local
  resources are bounded to 512 KiB when inlined.
- History replay reconstructs text, reasoning, and tool activity but does not
  reconstruct prior image/audio attachments.
- Editor-specific UX varies by ACP client implementation.

## Related files

- `tests/acp/` — ACP test suite
- `toolsets.py` — `hermes-acp` toolset definition
- `hermes_cli/main.py` — `hermes acp` CLI subcommand
- `pyproject.toml` — `[acp]` optional dependency + `hermes-acp` script
