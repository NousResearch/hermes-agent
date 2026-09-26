# Model tools

The root footprint ladder applies before any change here. Read `website/docs/developer-guide/adding-tools.md`, `website/docs/developer-guide/tools-runtime.md`, and `website/docs/developer-guide/subagent-lifecycle-api.md`.

## Registry and exposure

`tools/registry.py` owns registration, schema collection, dispatch, availability, and error wrapping. Tool modules register at import. Package tools register from `tools/<package>/tool.py` and require an `__init__.py`. Handlers return JSON strings.

Registration does not expose a tool. `toolsets.py` must place it in the core set or a named toolset. A custom or local tool belongs in a plugin with `ctx.register_tool`, not in core.

- Tool schema text never names a tool from another toolset. Add conditional cross-tool guidance in `model_tools.py::get_tool_definitions`, where availability is known.
- Instructional tools load their full document and do not offer pagination.
- Schema display paths use `display_hermes_home()`. Tool state resolves `get_hermes_home()` at call time.
- `check_fn` answers configured reachability or opt-in, not session/client identity. Its cache is keyed by `hermes_home_key()`, and credential probes use `agent.secret_scope.get_secret`.
- A GUI-only capability lives in a session-enabled named toolset. It is not a process-wide environment or `check_fn` decision.
- Agent-level tools use `agent/inline_tool_executors.py::INLINE_TOOL_EXECUTORS` before registry dispatch.
- New tools use existing setup UX and register secrets through the CLI configuration system.

Extend backend tables and topical siblings for terminal, browser, MCP, TTS, and skills-hub providers. Do not add name ladders. Fix remote file visibility in the backend mount rather than adding another file tool.

Native image results remain conversation history and are resent on later calls. Keep image size and delegated-repeat policy centralized in `vision_tools_history_budget.py`. Do not add a second counter in a tool.

## Spawn and trust boundaries

Every process environment comes from `tools/environments/local.py::build_subprocess_env`. A child acting for a served profile uses `served_profile_child_env`, which strips launch-profile residue, pins the target home, and overlays only target-profile secrets. Context variables do not cross `Popen`. Resolve scope first. A child `UnscopedSecretError` means the spawn site is incomplete, not that environment fallback is allowed.

MCP trust is per profile. Record it under the current home and consult the calling session's profile. A secondary profile never inherits trust for a same-named launch-profile server.

Scoped threads use `agent.memory_provider.spawn_context_thread`.

Background teardown signals the recorded parent first and gives it the configured grace period to reap children. Only then signal surviving descendants and finally force survivors. Systemd scope teardown follows the parent attempt unless the parent is already proven dead or recycled. Do not kill browser or supervisor descendants before their live parent.

## Delegation

`tools/delegate_tool.py` creates an isolated child context and terminal session. Leaf children cannot delegate or use parent-only coordination tools. Orchestrator children require explicit enablement and depth bounds.

- Concurrency is counted by live execution units. Grouped completion units from one call share the call's pool slot, and stall timing begins when work starts rather than while queued.
- Background child processes die at child teardown unless the child explicitly hands ownership to the parent through `process_manage`. Report unhanded live work and unread completion notifications in the result.
- Child `execute_code` kernels are namespaced, pinned while running, and disposed during child cleanup.
- If structured-output validation still fails after its retry, retain the raw child result with `schema_valid: false`. Do not discard completed work.
- Background delegation is process-local. Work that must survive a restart uses cron or a supervised background process with completion notification.
- Heartbeats are completion-queue events. Every surface that renders watcher events handles them and includes the required deduplication fields.
- Hard stop reaches synchronous children and detached background units. Every stop surface calls both active-child interruption and `async_delegation.interrupt_for_session`.
- Interrupted results preserve the child's last real assistant text.

`_last_resolved_tool_names` in `model_tools.py` is process-global. Delegation saves and restores it, so observers tolerate a transient stale value.

## Tests

Run `tests/tools/` through `scripts/run_tests.sh`. Dispatch tools through the registry, assert that every exposed tool belongs to a toolset, and test approval/security boundaries with real imports and isolated homes. Test child cleanup, handoff, profile-specific MCP trust, and stop fan-out at their real boundaries. Do not assert toolset sizes or current inventories.