# tools/ + toolsets.py + model_tools.py — model tools

Applies on top of the root `AGENTS.md`: settle the **Footprint Ladder** before adding anything here.
Most capabilities should NOT be core tools. Long-form: `website/docs/developer-guide/adding-tools.md`,
`tools-runtime.md`.

## Registry and discovery

`tools/registry.py` has no deps and is imported by every tool file; each `tools/*.py` calls
`registry.register()` at import time; `model_tools.py` imports the registry and triggers discovery
(`discover_builtin_tools()`), then `run_agent.py`, `cli.py`, `batch_runner.py`, `environments/`
consume it. Any `tools/*.py` with a top-level `registry.register()` is imported automatically — no
manual import list. The registry handles schema collection, dispatch (`handle_function_call()`),
availability (`check_fn`, TTL-cached process-wide), and error wrapping. **All handlers return a JSON
string.**

## Adding a core tool (2 files) — only when the user is explicitly contributing a core tool

For custom/local-only tools do NOT edit core: create `~/.hermes/plugins/<name>/plugin.yaml` +
`__init__.py` and call `ctx.register_tool(...)`; plugin toolsets are discovered automatically and
toggled without touching `tools/` or `toolsets.py` (`plugins/AGENTS.md`).

1. `tools/your_tool.py`:
   ```python
   from tools.registry import registry
   def check_requirements() -> bool: return bool(os.getenv("EXAMPLE_API_KEY"))
   def example_tool(param: str, task_id: str = None) -> str: return json.dumps({"success": True, ...})
   registry.register(name="example_tool", toolset="example",
       schema={"name": "example_tool", "description": "...", "parameters": {...}},
       handler=lambda args, **kw: example_tool(param=args.get("param", ""), task_id=kw.get("task_id")),
       check_fn=check_requirements, requires_env=["EXAMPLE_API_KEY"])
   ```
2. `toolsets.py`: add the name to `_HERMES_CORE_TOOLS` (all platforms) or a new toolset. **Required**
   — discovery registers the schema, but a tool is only exposed if a toolset names it.
   `_HERMES_CORE_TOOLS` is the default bundle every platform's base toolset inherits, not dead code.

Rules for tool code:
- **Schema descriptions must not name tools from other toolsets** (`browser_navigate` saying "prefer
  web_search"). Those tools may be unavailable (missing key, disabled toolset) and the model
  hallucinates calls to them. Cross-references are added dynamically in `get_tool_definitions()` in
  `model_tools.py` — see the `browser_navigate` / `execute_code` post-processing blocks.
- **Paths in schema descriptions use `display_hermes_home()`** (schema is built at import, after
  `_apply_profile_override()` set `HERMES_HOME`). **State files use `get_hermes_home()`**, never
  `Path.home()/.hermes`, so each profile gets its own state.
- **No `offset`/`limit` on instructional tools** (skills, prompts, playbooks) — models read page 1
  and skip the rest (root rubric).
- **`check_fn` answers reachability/opt-in, never surface.** It is TTL-cached process-wide, and one
  process serves many sessions; GUI-only tools go in a named toolset (`desktop_ui`, `project`)
  folded in by `_load_enabled_toolsets(platform)` (root: capability is a property of the SESSION).
- **Agent-level tools** (`todo`, `memory`) are intercepted before `handle_function_call()` via the
  `INLINE_TOOL_EXECUTORS` table (`agent/inline_tool_executors.py`; `agent/AGENTS.md`).
- **`_last_resolved_tool_names`** is a process-global in `model_tools.py`; `_run_single_child()` in
  `delegate_tool.py` saves/restores it around child runs — readers may see it stale mid-delegation.
- New tools integrate with existing setup UX (`hermes tools`, `hermes setup`, auto-install) rather
  than a raw env var; secrets go in `OPTIONAL_ENV_VARS` (`hermes_cli/AGENTS.md`).

## Toolsets (`toolsets.py`)

Single `TOOLSETS` dict. Keys today: `browser, clarify, code_execution, cronjob, debugging,
delegation, discord, discord_admin, feishu_doc, feishu_drive, file, homeassistant, image_gen,
kanban, memory, messaging, moa, rl, safe, search, session_search, skills, spotify, terminal, todo,
tts, video, vision, web, yuanbao` (don't assert the list in tests). Per-platform enable/disable via
`hermes tools` (curses) or `tools.<platform>.enabled/disabled` in config.yaml. `browser_exec`
replaces the other browser tools when `browser.backend` is `browser-use`.

## Backends and providers inside tools/

Several tools front pluggable backends: terminal environments in `tools/environments/` (local,
docker, ssh, modal, daytona, singularity; `terminal_tool_backends.py`, `tool_backend_helpers.py`),
browser (`browser_tool_*.py`: cdp, cloud, install, lifecycle, session, real_profile, vision), MCP
client (`mcp_tool_*.py`: config, discovery, transport, registration, content, errors), TTS
(`tts_tool_providers.py`, `tts_command_provider.py`), skills hub sources (`skills_hub_official.py`
`OptionalSkillSource`). Adding a backend = a new sibling or provider entry in the existing table,
never an `elif` on a backend name (root shape rules). Remote-backend file visibility problems are
fixed at the mount, not by adding a tool.

## Delegation (`tools/delegate_tool.py`)

Spawns a subagent with isolated context + terminal session; the parent waits for the summary unless
`background=true`, which returns a delegation id and re-enters the result via the async-delegation
completion queue. Shapes: single (`goal` + optional `context`, `toolsets`) or batch (`tasks: [...]`,
concurrency capped by `delegation.max_concurrent_children`, default 3). Roles: `leaf` (default;
no `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`; keeps `execute_code`) and
`orchestrator` (keeps `delegate_task`; gated by `delegation.orchestrator_enabled`, bounded by
`delegation.max_spawn_depth`, default 2). Config knobs under `delegation:`:
`max_concurrent_children, max_spawn_depth, child_timeout_seconds, orchestrator_enabled,
subagent_auto_approve, inherit_mcp_toolsets, max_iterations`. **Durability:** background
delegation is process-local; work that must survive restart uses `cronjob` or
`terminal(background=True, notify_on_complete=True)`. API: `website/docs/developer-guide/subagent-lifecycle-api.md`.

## Tests

`tests/tools/`. Test the handler through the registry (real dispatch), not the bare function only;
assert contracts ("every registered tool has a toolset", "no schema description names a tool from
another toolset") rather than tool counts. Approval/security-boundary tools are E2E'd with real
imports against a temp `HERMES_HOME` (see `tests/tools/test_approval_config_readonly.py`).
