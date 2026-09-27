# tools/ + toolsets.py + model_tools.py — model tools

Applies on top of the root `AGENTS.md`: settle the **Footprint Ladder** before adding anything here.
Most capabilities should NOT be core tools. Long-form: `website/docs/developer-guide/adding-tools.md`,
`tools-runtime.md`.

## Registry and discovery

`tools/registry.py` has no deps and is imported by every tool file; each `tools/*.py` calls
`registry.register()` at import time; `model_tools.py` imports the registry and triggers discovery
(`discover_builtin_tools()`), then `run_agent.py`, `cli.py`, `batch_runner.py`, `environments/`
consume it. Any `tools/*.py` with a top-level `registry.register()` is imported automatically — no
manual import list. A tool that is a whole package (`tools/connectors/`) registers from
`tools/<pkg>/tool.py`, the only file discovery scans inside a package; every sibling in the package
is a library by construction, and the package needs an `__init__.py` or discovery skips it with a
warning (setuptools would drop it from the wheel). The registry handles schema collection, dispatch (`handle_function_call()`),
availability (`check_fn`, TTL-cached process-wide), and error wrapping. **All handlers return a JSON
string.**

## Adding a core tool (2 files) — only when the user is explicitly contributing a core tool

For custom/local-only tools do NOT edit core: create `~/.hermes/plugins/<name>/plugin.yaml` +
`__init__.py` and call `ctx.register_tool(...)`; plugin toolsets are discovered automatically and
toggled without touching `tools/` or `toolsets.py` (`plugins/AGENTS.md`).

1. `tools/your_tool.py`:
   ```python
   from agent.secret_scope import get_secret
   from tools.registry import registry
   def check_requirements() -> bool: return bool(get_secret("EXAMPLE_API_KEY"))
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
- **Paths in schema descriptions use `display_hermes_home()`** (schema is built at import; under
  multiplex it shows the launch home, which is display-only). **State files use `get_hermes_home()`**
  at call time, never `Path.home()/.hermes` and never a module constant, so each served profile gets
  its own state.
- **No `offset`/`limit` on instructional tools** (skills, prompts, playbooks) — models read page 1
  and skip the rest (root rubric).
- **`check_fn` answers reachability/opt-in for the profile it runs under, never surface.** Results
  are TTL-cached in `registry.py::_check_fn_cache` keyed by `hermes_home_key()`, and one process
  serves many sessions AND many profiles: a probe reads credentials through
  `agent.secret_scope.get_secret`, never bare `os.getenv` (that answers with the launch profile's
  `.env` for everyone). The registry classifies an `UnscopedSecretError` from
  `current_secret_scope()` at the catch site — a boot-time probe with no scope is DEBUG, not a
  traceback. GUI-only tools go in a named toolset (`desktop_ui`, `project`) folded in by
  `_load_enabled_toolsets(platform)` (root: capability is a property of the SESSION).
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

**Native vision embeds are history, not one-shot payloads.** `vision_tools.py::_vision_analyze_native`
(and the browser screenshot twins in `browser_tool_vision.py` / `browser_use_cli.py`) bake the image
into a tool result that is re-sent on every later API call. Size and repeat policy live in
`vision_tools_history_budget.py` (config section `vision`: `embed_target_bytes`, `max_calls_per_image`);
the repeat counter is keyed on (session id, resolved source) so region crops share their file's count,
and its default cap applies only inside `agent.delegation_context.is_delegated_child_process_context()`.
Put new embed-cost rules there, never a second counter in a tool.

**Every spawn goes through one env builder.** `environments/local.py::build_subprocess_env` (+
`hermes_constants.apply_subprocess_home_env`, `env_passthrough.py::resolve_passthrough_value`) is
how a terminal, `execute_code`, background process, delegation child, ACP or MCP stdio child gets
its environment; a child that acts FOR the served profile (`hermes -p X` workers, `key_cmd`
helpers, browser drivers, Bot Chat relay turns) uses `environments/local.py::
served_profile_child_env(target_home=, inherit_credentials=)`: launch-profile `.env` /
`TERMINAL_*` residue dropped (`strip_launch_profile_env`), the target home pinned, only the
target's own secrets overlaid. `os.environ.copy()` / `dict(os.environ)` pins the launch profile;
contextvars do not cross process boundaries, so resolve before `Popen`. A child's
`UnscopedSecretError` is a spawn-site bug, never a reason to add environ fallthrough. New threads
from scoped code use `agent.memory_provider.spawn_context_thread` (a bare `threading.Thread`
drops the scope). **MCP trust is a per-profile record:** `mcp_tool_registration.py::
_record_scope_trust` keys trust on the home; a secondary never adopts the launch profile's trust
for a same-named server, and `mcp_tool_handlers.py::_trust_gate_check` consults the calling
session's profile.

**Background-process teardown signals the parent first.** `process_registry.py::ProcessRegistry.
_terminate_host_pid` snapshots the descendants, SIGTERMs only the recorded parent, waits
`terminal.daemon_term_grace_seconds` for it to exit and reap its own children, then SIGTERMs the
snapshot survivors and SIGKILLs whatever ignored both (so a supervisor that reaps its tree — a
Chromium/Electron browser reaping its zygotes, a shell trap — exits cleanly, while a shell whose
children ignore SIGHUP still leaves no orphan). Never SIGTERM descendants before the parent: killing
a browser's zygote mid-shutdown turns exit 0 into a SIGTRAP core dump. `_stop_systemd_unit` (scope
teardown, kills the worker cgroup) runs only after that PID kill in `kill()`, or on a parent already
proven dead/recycled (`session.exited`, `_signal_kill` recycled-PID path, checkpoint recovery) —
it is never the first signal a live parent receives.

## Delegation (`tools/delegate_tool.py`)

Spawns a subagent with isolated context + terminal session; the parent waits for the summary unless
`background=true`, which returns a delegation id and re-enters the result via the async-delegation
completion queue. Shapes: single (`goal` + optional `context`, `toolsets`) or batch (`tasks: [...]`,
concurrency capped by `delegation.max_concurrent_children`, default 3). A background batch returns as ONE
completion by default; with `delegation.independent_completions` it is split into completion **units**
(`delegate_tool_dispatch._units_of`): tasks sharing a `group` join and report together; each ungrouped
task reports alone as it finishes. Units of one call share ONE pool slot (`slot_key` in
`async_delegation._dispatch`) — never count units against capacity; the executor is sized by live UNITS
and the stall clock arms when the runner starts, so a queued unit is never judged stalled. Roles: `leaf` (default;
no `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`; keeps `execute_code`) and
`orchestrator` (keeps `delegate_task`; gated by `delegation.orchestrator_enabled`, bounded by
`delegation.max_spawn_depth`, default 2). Config knobs under `delegation:`:
`max_concurrent_children, independent_completions, max_spawn_depth, child_timeout_seconds, orchestrator_enabled,
subagent_auto_approve, inherit_mcp_toolsets, max_iterations`. **Child processes:** a child's background
processes are killed at its teardown and their notices are suppressed in the parent; `process_manage(action="handoff")`
(children only) flips `ProcessSession.owner_task_id` to the parent under the registry lock
(`process_registry.transfer_ownership`) so the completion routes and reaps by the new owner; un-handed leftovers land on
the result as `orphaned_processes`, exited-but-never-read notify processes as `unread_completions` (`_ChildRun.account_background_processes`, before `cleanup` kills them). **Child kernels:** a child's `execute_code` kernels are keyed `<parent-owner>::child::<child-session-id>`, pinned against the `max_session_kernels` LRU cap while the child runs and disposed by `cleanup` (`code_kernel.shutdown_kernels_for_delegated_child`) — never let a finished child's kernel squat the cap. **output_schema:** a miss after the one retry keeps `status: completed` with the raw text in `summary` plus `schema_valid: false` / `schema_errors` / `schema_note` — never discard a child's result. **Durability:** background
delegation is process-local; work that must survive restart uses `cronjob` or
`terminal(background=True, notify_on_complete=True)`. `heartbeat=N` on the same spawn emits a `heartbeat` event (type on the completion queue; delta output only, `HEARTBEAT_MIN_SECONDS` floor, one daemon timer thread for all sessions in `process_registry.py::ProcessRegistry._heartbeat_loop`) — every surface that renders `watch_match` must render `heartbeat` (`process_registry_notifications.py`, `gateway/run.py::_drain_gateway_watch_events`, `tui_gateway/session_notifications.py::_DEDUP_EXTRA_FIELDS`). **Stop fan-out:** a turn's hard interrupt reaches only
`_active_children`; background units are detached at dispatch, so every stop surface (gateway `/stop` busy AND
idle paths, TUI `session.interrupt`, ACP `cancel`, CLI `/stop` via `interrupt_all`) calls
`async_delegation.interrupt_for_session` too. Depth>0 delegations are always synchronous
(`_model_background_value`), so the stop recurses through the child's own `_active_children` fan-out; an
interrupted entry's `summary` is the child's last real assistant text (`_build_result_entry`). API: `website/docs/developer-guide/subagent-lifecycle-api.md`.

## Tests

`tests/tools/`. Test the handler through the registry (real dispatch), not the bare function only;
assert contracts ("every registered tool has a toolset", "no schema description names a tool from
another toolset") rather than tool counts. Approval/security-boundary tools are E2E'd with real
imports against a temp `HERMES_HOME` (see `tests/tools/test_approval_config_readonly.py`).
