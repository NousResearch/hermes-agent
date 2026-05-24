# Tools Guide

This directory owns built-in tool implementations, the tool registry, terminal
backends, file operations, patch handling, browser/computer tools, skills tools,
and tool-specific state.

## Registry and Exposure

Every built-in tool module registers itself with `tools.registry.registry`.
Handlers must return JSON strings.

Creating a core tool requires two steps:

1. Add `tools/<tool>.py` with a top-level `registry.register(...)`.
2. Add the tool name to `toolsets.py` so the agent can see it.

Auto-discovery imports tool modules, but toolsets are the deliberate exposure
boundary. `_HERMES_CORE_TOOLS` is the default bundle most platforms inherit.

For local or custom capabilities, prefer a plugin with `ctx.register_tool(...)`
instead of editing core.

## Tool Schema Rules

Do not hardcode cross-tool references in schema descriptions. A tool may be
available while the referenced tool is disabled, missing credentials, or removed
from the active toolset. If a cross-reference is needed, add it dynamically in
`model_tools.py` after the active toolset is known.

If schema descriptions mention Hermes paths, use `display_hermes_home()` so
profile-specific homes display correctly.

## State Files

Persistent tool state must live under `get_hermes_home()`, never
`Path.home() / ".hermes"`. This includes caches, logs, checkpoints, memory,
plans, auth sidecars, and tool-specific state.

## Agent-loop Tools

Some tools are intercepted by the agent loop before normal dispatch:

- `todo`
- `memory`
- `session_search`
- `delegate_task`

Follow existing patterns such as `tools/todo_tool.py` and
`tools/memory_tool.py` when adding loop-level behavior.

## File and Patch Tools

`tools/file_operations.py`, `tools/file_tools.py`, and `tools/patch_parser.py`
coordinate reads, writes, patching, validation, and LSP diagnostics.

Important invariants:

- Preserve sibling-subagent file-state checks.
- Keep LSP diagnostics best-effort and non-fatal.
- Surface introduced diagnostics from writes/patches without flooding the agent
  with pre-existing shifted diagnostics.
- Do not bypass centralized write/patch helpers for new file mutation paths.

## Terminal Environments

Terminal backends live under `tools/environments/`. The local backend is not the
only runtime; Docker, SSH, Modal, Daytona, Singularity, and other backends may
have different filesystem visibility.

Host-side services such as LSP cannot always see remote container files. Gate
host-only features accordingly.

## Delegation Tool

`tools/delegate_tool.py` spawns synchronous child agents.

Key rules:

- Leaf subagents cannot call `delegate_task`, `clarify`, `memory`,
  `send_message`, or `execute_code`.
- Orchestrator subagents may spawn workers only when enabled by config and
  within `delegation.max_spawn_depth`.
- Subagents inherit only toolsets allowed by the parent and config.
- Dangerous-command approvals in subagent worker threads default to deny unless
  `delegation.subagent_auto_approve` is set.
- Use `cronjob`, kanban, or background terminal work for durable tasks.

## Skills Tools

`tools/skills_tool.py`, `tools/skill_manager_tool.py`, `tools/skills_hub.py`,
and skill guard/audit helpers are the runtime surface for skills. Follow
`skills/AGENTS.md` for skill authoring standards.

## Kanban Tools

`tools/kanban_tools.py` exposes the worker/orchestrator toolset:
`kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`,
`kanban_comment`, `kanban_create`, and `kanban_link`. Profiles that explicitly
enable the `kanban` toolset outside dispatcher-spawned tasks can also get
routing helpers such as `kanban_list` and `kanban_unblock`.

The board is the hard isolation boundary. Workers are spawned with
`HERMES_KANBAN_BOARD` pinned; do not let tools cross boards implicitly.
