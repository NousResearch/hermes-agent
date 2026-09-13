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
- new/load/resume/fork/list/cancel session methods
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

The manager is thread-safe and supports:

- create
- get
- remove
- fork
- list
- cleanup
- cwd updates

### Event bridge

`acp_adapter/events.py` converts AIAgent callbacks into ACP `session_update` events.

Bridged callbacks:

- `tool_progress_callback`
- `reasoning_callback` (provider reasoning; `thinking_callback` stays `None`)
- `step_callback`
- `stream_delta_callback`
- `interim_assistant_callback` and `background_review_callback` (negotiated extension below)

Because `AIAgent` runs in a worker thread while ACP I/O lives on the main event loop, the bridge uses:

```python
asyncio.run_coroutine_threadsafe(...)
```

### Optional commentary extension

Clients opt in with `clientCapabilities._meta.hermes.messagePhases: 1` in
`initialize`. Hermes acknowledges the exact integer version in
`agentCapabilities._meta.hermes.messagePhases: 1`. Other versions retain ordinary
ACP behavior. For each `session/prompt`, the client supplies a UUID at
`_meta.hermes.turnId`; it is an opaque correlation token, never owner identity or
prompt content. Missing or invalid tokens disable additional notifications for
that prompt.

Hermes emits an additional `agent_message_chunk` for each interim callback and
background review summary, with a unique `messageId` and:

```json
{"hermes":{"messagePhases":1,"phase":"commentary","source":"assistant","turnId":"<client prompt UUID>"}}
```

Background summaries use `source: "background_review"` and retain their
Self-improvement review label. These copies leave raw response chunks and the
ordinary final response unchanged, including when `already_streamed` is true.
They contain no provider reasoning or raw tool output. For Codex-backed Hermes,
installing the commentary consumer replaces the existing fallback that displayed
commentary in thought events; real analysis/reasoning and final text are preserved.

Callbacks capture the connection, session, and turn token. The background worker
snapshots its callback before it starts, including a disabled `None`, so later
turns cannot change its destination. Prompt callbacks restore their previous
values before draining queued prompts. Queued prompts without a client token
produce only ordinary ACP output.

A background summary can arrive after its initiating prompt finishes or during
another prompt. Clients must correlate using the captured token, exclude these
additional copies from ordinary output, and avoid attributing them to the active
turn. A relay should use bounded, connection-scoped token mappings, reject
unknown/replayed tokens, honor its notification toggle at delivery time, and
reauthorize the original recipient. Hermes does not persist or replay these
notifications or supply recipient identities. Reconnect requires negotiation
again; a late worker retains its old connection.

### Permission bridge

`acp_adapter/permissions.py` adapts dangerous terminal approval prompts into ACP permission requests.

Mapping:

- `allow_once` -> Hermes `once`
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
  -> extract text from ACP content blocks
  -> reset cancel event
  -> install callbacks + approval bridge
  -> run AIAgent in ThreadPoolExecutor
  -> update session history
  -> emit final agent message chunk
```

### Cancelation

`cancel(session_id)`:

- sets the session cancel event
- calls `agent.interrupt()` when available
- causes the prompt response to return `stop_reason="cancelled"`

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

- ACP sessions are persisted to the shared `~/.hermes/state.db` (SessionDB) and transparently restored across process restarts; they appear in `session_search`
- non-text prompt blocks are currently ignored for request text extraction
- editor-specific UX varies by ACP client implementation

## Related files

- `tests/acp/` — ACP test suite
- `toolsets.py` — `hermes-acp` toolset definition
- `hermes_cli/main.py` — `hermes acp` CLI subcommand
- `pyproject.toml` — `[acp]` optional dependency + `hermes-acp` script
