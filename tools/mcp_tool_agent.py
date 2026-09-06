"""Live-agent tool-list maintenance after MCP (re)discovery: refreshing an
AIAgent's tools/tool names, preserving the cached tools[] prefix across rebuilds,
and re-injecting post-build tools."""

import logging
import json
import threading
from typing import Optional
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")

# Serializes in-place swaps of ``agent.tools`` / ``agent.valid_tool_names`` by
# the reload RPC, gateway reload and late-binding refresh thread; the run loop
# reads them during tool iteration and must never see a half-updated pair.
_agent_tools_lock = threading.Lock()


def _def_name(tool_def: dict) -> str:
    return (tool_def.get("function") or {}).get("name", "")


def _agent_tool_defs(agent) -> list:
    return list(getattr(agent, "tools", None) or [])


def _resolve_refresh_toolsets(agent, enabled_override, disabled_override):
    """Explicit reloads pass freshly-resolved toolsets (so a server just ENABLED in config is
    picked up) and the agent's selection is updated to match; automatic paths pass nothing
    and reuse the build-time selection."""
    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    if enabled_override is not None or disabled_override is not None:
        enabled = enabled_override if enabled_override is not None else enabled
        disabled = disabled_override if disabled_override is not None else disabled
        agent.enabled_toolsets, agent.disabled_toolsets = enabled, disabled
    return enabled, disabled


def _tool_defs_content_changed(agent, new_defs: list) -> bool:
    """Byte-level diff of the serialized tool arrays (dynamic schemas change CONTENT under
    stable names); False if either side fails to serialize."""
    try:
        dump = lambda defs: json.dumps(defs, sort_keys=True, separators=(",", ":"), default=str)  # noqa: E731
        return dump(_agent_tool_defs(agent)) != dump(new_defs)
    except Exception:  # noqa: BLE001
        return False


def _publish_tool_snapshot(
    agent, new_defs: list, new_names: set, *, snapshot_generation: int,
    staged_engine_names: set, content_aware: bool, prefix_registered: Optional[set]) -> Optional[set]:
    """Single atomic read-diff-publish under ``_agent_tools_lock`` so ``added`` matches what
    was published and a stale (older-generation) rebuild can't overwrite a newer one. Returns
    the added names, or None when nothing was published (unchanged, or a newer snapshot won)."""
    with _agent_tools_lock:
        # Tolerate an agent that never set the generation (or a non-int mock).
        published_gen = getattr(agent, "_tool_snapshot_generation", -1)
        published_gen = published_gen if isinstance(published_gen, int) else -1
        if snapshot_generation < published_gen:
            return None  # a newer snapshot already won
        current_defs = _agent_tool_defs(agent)
        current = {_def_name(t) for t in current_defs}
        if prefix_registered is not None:
            new_defs, new_names = _merge_preserving_prefix(current_defs, new_defs, prefix_registered)
        # Record the generation even when unchanged so an in-flight older caller can't clobber.
        agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
        # Same NAME set: no change for MCP-reload callers. Content-aware callers
        # (compaction boundary) also diff serialized bytes.
        if new_names == current and not (content_aware and _tool_defs_content_changed(agent, new_defs)):
            return None
        agent.tools = new_defs
        agent.valid_tool_names = new_names
        # Publish context-engine routing names atomically with the snapshot.
        engine_names = getattr(agent, "_context_engine_tool_names", None)
        if isinstance(engine_names, set):
            engine_names.clear()
            engine_names.update(staged_engine_names)
        return new_names - current


def refresh_agent_mcp_tools(
    agent, *, enabled_override=None, disabled_override=None, quiet_mode: bool = True,
    content_aware: bool = False, preserve_prefix: bool = False) -> set:
    """Re-derive an already-built agent's tool snapshot from the live registry; returns the
    newly-added tool names (empty when unchanged). The agent snapshots ``agent.tools`` at build
    time, so servers that connect later (slow OAuth, ``/reload-mcp``) are invisible until
    rebuilt. Shared by the TUI RPC, gateway reload, late-binding thread and between-turns
    refresh: respects the toolset filter, diffs by tool NAME (a count compare misses an
    equal-size swap), re-injects the memory-provider / context-engine tools ``agent_init``
    appends after ``get_tool_definitions``, publishes ``(tools, valid_tool_names)`` together.

    ``preserve_prefix``: for rebuilds inside a live conversation the tool array is a cached
    request prefix and any moved byte re-prefills the whole history — existing tools keep their
    slot (schemas still refresh), a still-registered tool whose ``check_fn`` merely flapped is
    carried forward (``check_fn`` gates exposure, never invocation), a deregistered tool is
    dropped, new tools append at the tail. The caller owns the prompt-cache contract."""
    from model_tools import get_tool_definitions
    from tools.registry import registry
    enabled, disabled = _resolve_refresh_toolsets(agent, enabled_override, disabled_override)
    # Generation captured BEFORE the slow get_tool_definitions call (a slower caller holding an
    # OLDER set must not clobber a newer one); definitions computed OUTSIDE the lock.
    snapshot_generation = registry._generation
    new_defs = list(get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=quiet_mode) or [])
    new_names = {_def_name(t) for t in new_defs}
    # Post-build families re-appended on LOCALS only; live attributes untouched until publish.
    staged_engine_names = _reinject_post_build_tools(agent, new_defs, new_names)
    # Registry membership is read OUTSIDE ``_agent_tools_lock``: taking ``registry._lock``
    # under the tools lock would be the first nesting of the two.
    prefix_registered: Optional[set] = None
    if preserve_prefix:
        try:
            prefix_registered = {entry.name for entry in registry.get_all_entries()}
        except Exception:  # noqa: BLE001
            pass  # fail open to the plain rebuild
    added = _publish_tool_snapshot(
        agent, new_defs, new_names, snapshot_generation=snapshot_generation,
        staged_engine_names=staged_engine_names, content_aware=content_aware, prefix_registered=prefix_registered)
    if added is None:
        return set()
    persist_agent_tool_names(agent)  # re-pin so a rebuild after agent-cache eviction restores this order
    return added


def reprobe_tool_availability() -> None:
    """Explicit ``/reload-mcp`` hatch out of the tools[] freeze: drop the ``check_fn`` verdict
    cache AND the ``get_tool_definitions`` memo (keyed on registry generation, so it would
    otherwise replay the stale verdicts)."""
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def persist_agent_tool_names(agent) -> None:
    """Best-effort: write ``agent.tools`` names to the session row (freeze pin)."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not db or not session_id:
        return
    try:
        db.update_session_tool_names(session_id, [_def_name(t) for t in _agent_tool_defs(agent)])
    except Exception:  # noqa: BLE001
        logger.debug("tool_names persist skipped", exc_info=True)


def restore_agent_tool_prefix(agent, saved_names: list) -> bool:
    """Fold a freshly built agent's ``tools`` onto the session's saved order; True if changed.
    After agent-cache eviction the gateway rebuilds a NEW AIAgent with no predecessor to
    preserve, so the saved name list stands in (``_merge_preserving_prefix`` rule; a saved
    tool still registered but failing its probe is carried forward from the registry schema)."""
    if not saved_names:
        return False
    from tools.registry import registry
    fresh_defs = _agent_tool_defs(agent)
    fresh = {_def_name(t): t for t in fresh_defs}

    def _saved_def(name):
        if name in fresh:
            return fresh[name]
        entry = registry.get_entry(name)
        return None if entry is None else {"type": "function", "function": {**entry.schema, "name": entry.name}}

    saved_defs = [d for d in map(_saved_def, saved_names) if d is not None]
    registered_names = {entry.name for entry in registry.get_all_entries()}
    merged, merged_names = _merge_preserving_prefix(saved_defs, fresh_defs, registered_names)
    with _agent_tools_lock:
        if merged == fresh_defs:
            return False
        agent.tools = merged
        agent.valid_tool_names = merged_names
    if [_def_name(t) for t in merged] != list(saved_names):
        persist_agent_tool_names(agent)
    return True


def _merge_preserving_prefix(current_defs: list, new_defs: list, registered_names: set) -> tuple[list, set]:
    """Fold a fresh tool snapshot into a live one without moving existing bytes. Ordered by
    ``current_defs`` (the cached request prefix): a name in both keeps its slot but takes the
    fresh schema; a name only in the live list is kept if still registered (``check_fn``
    flapped), else dropped; a name only in the fresh list is appended at the tail."""
    fresh = {_def_name(entry): entry for entry in new_defs if _def_name(entry)}
    merged = []
    for entry in current_defs:
        name = _def_name(entry)
        replacement = fresh.pop(name, None)
        if replacement is not None:
            merged.append(replacement)
        elif name and name in registered_names:
            merged.append(entry)
    merged.extend(fresh.values())
    return merged, {_def_name(t) for t in merged}


def _reinject_post_build_tools(agent, tools_list: list, name_set: set) -> set:
    """Append memory-provider and context-engine tools onto the caller's staged ``tools_list``
    / ``name_set`` (never the live agent attributes), mirroring ``agent_init``'s post-build
    injection. Idempotent and fail-soft. Returns the context-engine routing names THIS rebuild
    appended: a name already owned by a registry/plugin tool is not claimed, matching agent_init."""
    def _add(schema) -> bool:
        name = schema.get("name", "") if isinstance(schema, dict) else ""
        if not name or name in name_set:
            return False
        tools_list.append({"type": "function", "function": schema})
        name_set.add(name)
        return True

    def _schema_getter(attr: str, method: str):
        getter = getattr(getattr(agent, attr, None) or None, method, None)
        return getter if callable(getter) else None

    enabled = getattr(agent, "enabled_toolsets", None)
    try:
        get_mem_schemas = _schema_getter("_memory_manager", "get_all_tool_schemas")
        if get_mem_schemas is not None:
            from agent.memory_manager import memory_provider_tools_enabled  # same gate inject_memory_provider_tools uses
            if memory_provider_tools_enabled(
                    enabled, getattr(agent, "disabled_toolsets", None), memory_tool_present="memory" in name_set):
                for schema in get_mem_schemas():
                    _add(schema)
    except Exception:
        logger.debug("Memory-provider tool re-injection skipped", exc_info=True)

    # The `context_engine` toolset is intentionally empty, so lcm_* tools exist only via this
    # append. Honor the enabled_toolsets gate agent_init uses, or a restricted-toolset platform
    # would re-leak tools the build excluded.
    # See #5544.
    staged_engine_names: set = set()
    try:
        get_schemas = _schema_getter("context_compressor", "get_tool_schemas")
        if (enabled is None or "context_engine" in enabled) and get_schemas is not None:
            # Claim the routing name only when WE appended the schema.
            staged_engine_names.update(s["name"] for s in get_schemas() if _add(s))
    except Exception:
        logger.debug("Context-engine tool re-injection skipped", exc_info=True)
    return staged_engine_names
