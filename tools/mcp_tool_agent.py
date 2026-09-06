"""Live-agent tool-list maintenance after MCP (re)discovery: refreshing an
AIAgent's tools/tool names, preserving the cached tools[] prefix across rebuilds,
and re-injecting post-build tools."""

import logging
import json
import threading
from typing import Optional

logger = logging.getLogger("tools.mcp_tool")

# Serializes in-place swaps of ``agent.tools`` / ``agent.valid_tool_names`` by
# the reload RPC, gateway reload and late-binding refresh thread; the run loop
# reads them during tool iteration and must never see a half-updated pair.
_agent_tools_lock = threading.Lock()
_MAX_TOOL_REFRESH_GENERATION_ATTEMPTS = 3


def _def_name(tool_def: dict) -> str:
    return (tool_def.get("function") or {}).get("name", "")


def _agent_tool_defs(agent) -> list:
    return list(getattr(agent, "tools", None) or [])


def agent_tool_names(agent) -> list:
    """Names of ``agent.tools`` in wire order (unnamed entries skipped)."""
    return [name for name in map(_def_name, _agent_tool_defs(agent)) if name]


def _tool_defs_content_changed(agent, new_defs: list) -> bool:
    """Byte-level diff of the serialized tool arrays (dynamic schemas change CONTENT under
    stable names); False if either side fails to serialize."""
    try:
        dump = lambda defs: json.dumps(defs, sort_keys=True, separators=(",", ":"), default=str)  # noqa: E731
        return dump(_agent_tool_defs(agent)) != dump(new_defs)
    except Exception:  # noqa: BLE001
        return False




def _refresh_tool_policy(agent, enabled_override, disabled_override):
    # Explicit reloads (/reload-mcp) pass freshly-resolved toolsets so a server
    # the user just ENABLED in config is picked up; the agent's stored selection
    # is then updated to match. The automatic paths (between-turns, late-binding)
    # pass nothing and reuse the agent's build-time selection unchanged.
    explicit_policy = enabled_override is not None or disabled_override is not None
    # Policy epochs order explicit reloads against automatic rebuilds even when
    # both derive from the same registry generation.  Policy is still staged:
    # only a successful winning refresh publishes enabled/disabled toolsets.
    with _agent_tools_lock:
        published_policy_epoch = getattr(agent, "_tool_published_policy_epoch", 0)
        if not isinstance(published_policy_epoch, int):
            published_policy_epoch = 0
        latest_policy_epoch = getattr(
            agent, "_tool_policy_epoch", published_policy_epoch
        )
        if not isinstance(latest_policy_epoch, int):
            latest_policy_epoch = published_policy_epoch
        pending_policy = getattr(agent, "_tool_pending_policy", None)
        if not (
            isinstance(pending_policy, tuple)
            and len(pending_policy) == 3
            and isinstance(pending_policy[0], int)
            and pending_policy[0] == latest_policy_epoch
            and pending_policy[0] > published_policy_epoch
        ):
            pending_policy = None
        if explicit_policy:
            base_enabled = (
                pending_policy[1]
                if pending_policy is not None
                else getattr(agent, "enabled_toolsets", None)
            )
            base_disabled = (
                pending_policy[2]
                if pending_policy is not None
                else getattr(agent, "disabled_toolsets", None)
            )
            enabled = (
                enabled_override if enabled_override is not None else base_enabled
            )
            disabled = (
                disabled_override if disabled_override is not None else base_disabled
            )
            refresh_policy_epoch = latest_policy_epoch + 1
            agent._tool_policy_epoch = refresh_policy_epoch
            agent._tool_pending_policy = (
                refresh_policy_epoch,
                enabled,
                disabled,
            )
            policy_refresh = True
        elif pending_policy is not None:
            refresh_policy_epoch, enabled, disabled = pending_policy
            policy_refresh = True
        else:
            refresh_policy_epoch = published_policy_epoch
            enabled = getattr(agent, "enabled_toolsets", None)
            disabled = getattr(agent, "disabled_toolsets", None)
            policy_refresh = False

    return enabled, disabled, refresh_policy_epoch, policy_refresh


def _publish_staged_policy(agent, enabled, disabled, refresh_policy_epoch) -> None:
    agent.enabled_toolsets = enabled
    agent.disabled_toolsets = disabled
    agent._tool_published_policy_epoch = refresh_policy_epoch
    pending = getattr(agent, "_tool_pending_policy", None)
    if (
        isinstance(pending, tuple)
        and pending
        and pending[0] == refresh_policy_epoch
    ):
        agent._tool_pending_policy = None



def _published_memory_provider_names(agent, tools) -> set:
    """Explicit ownership wins; old direct callers retain manager-based inference."""
    names = getattr(agent, "_memory_provider_tool_names", None)
    if isinstance(names, set):
        return set(names)
    has_tool = getattr(getattr(agent, "_memory_manager", None), "has_tool", None)
    return {
        tool["function"]["name"] for tool in tools
        if callable(has_tool) and has_tool(tool["function"]["name"])
    }


def refresh_agent_mcp_tools(
    agent,
    *,
    enabled_override=None,
    disabled_override=None,
    quiet_mode: bool = True,
    content_aware: bool = False,
    preserve_prefix: bool = False,
) -> set:
    """Re-derive an already-built agent's tool snapshot from the live registry.

    The agent snapshots ``agent.tools`` once at build time and never re-reads
    the registry (see ``run_agent`` / ``agent_init``).  When MCP servers connect
    *after* that snapshot — a slow HTTP/OAuth server that misses the bounded
    startup wait, or a ``/reload-mcp`` — their tools are invisible until the
    snapshot is rebuilt.  This is the single shared rebuild used by every such
    caller (the TUI ``reload.mcp`` RPC, the gateway reload, the late-binding
    refresh thread, and the per-turn between-turns refresh) so they can't drift
    apart again.

    The rebuild respects the agent's own ``enabled_toolsets`` /
    ``disabled_toolsets`` (the same filtering it was built with) and diffs by
    tool **name** (not count — a count compare misses an equal-size add/remove
    swap).

    Crucially it is **additive-preserving**: ``get_tool_definitions`` returns
    only the registry-derived tools, but ``agent_init`` appends two further
    families directly onto ``agent.tools`` *after* that — external
    memory-provider tools (mem0/honcho/…) and context-engine tools
    (``lcm_*``).  A naive ``agent.tools = get_tool_definitions(...)`` would
    silently DELETE those.  So after rebuilding the registry set we re-run the
    same post-build injectors ``agent_init`` used, reconstructing the full
    surface.  The new ``(tools, valid_tool_names)`` pair is published together
    under ``_agent_tools_lock`` so a concurrent reader never sees a
    cross-attribute half-swap.

    Returns the set of newly-added tool names (empty when nothing changed), so
    callers can decide whether to notify the user / re-emit session info.  The
    caller owns the prompt-cache contract: this helper does NOT check turn state,
    because each caller has a different policy (``/reload-mcp`` rebuilds after
    explicit user consent; the late-binding and between-turns paths only rebuild
    at a turn boundary, before that turn's ``tools=`` prefix is assembled).
    """
    from model_tools import get_tool_definitions
    from tools.registry import registry

    enabled, disabled, refresh_policy_epoch, policy_refresh = _refresh_tool_policy(
        agent, enabled_override, disabled_override
    )

    for attempt in range(_MAX_TOOL_REFRESH_GENERATION_ATTEMPTS):
        # Strict provider fallback is derived from the published ownership and
        # schemas as one snapshot. Processing stays outside the lock, but a
        # concurrent publisher cannot make us combine names from one tool
        # surface with schemas from another.
        with _agent_tools_lock:
            fallback_provider_tools = list(getattr(agent, "tools", None) or [])
            fallback_provider_names = _published_memory_provider_names(agent, fallback_provider_tools)
            fallback_engine_names = set(getattr(agent, "_context_engine_tool_names", None) or set())
            snapshot_epoch_raw = getattr(agent, "_tool_snapshot_epoch", 0)
            snapshot_epoch = (
                snapshot_epoch_raw if isinstance(snapshot_epoch_raw, int) else 0
            )

        # Capture immutable routes with the generation the rebuild derives
        # from. A later mutation forces a retry before publication.
        (
            snapshot_generation,
            snapshot_registry_entries,
        ) = registry.snapshot_entries_with_generation()

        # Registry-derived tools (built-ins + MCP), filtered to the agent's
        # toolsets. Assembly remains outside the lock; only coherent snapshots
        # and the final publication use the critical section.
        try:
            new_defs = list(
                get_tool_definitions(
                    enabled_toolsets=enabled,
                    disabled_toolsets=disabled,
                    quiet_mode=quiet_mode,
                )
                or []
            )
            new_names = {t["function"]["name"] for t in new_defs}
            registry_scope_defs = new_defs
            if {
                "tool_search",
                "tool_describe",
                "tool_call",
            } <= new_names:
                registry_scope_defs = list(
                    get_tool_definitions(
                        enabled_toolsets=enabled,
                        disabled_toolsets=disabled,
                        quiet_mode=True,
                        skip_tool_search_assembly=True,
                    )
                    or []
                )
            scoped_registry_names = {
                tool.get("function", {}).get("name")
                for tool in registry_scope_defs
                if isinstance(tool, dict)
            }
            scoped_registry_names.intersection_update(
                snapshot_registry_entries
            )

            # Re-append post-build injected families on staged locals.
            (
                staged_engine_names,
                staged_provider_names,
            ) = _reinject_post_build_tools(
                agent,
                new_defs,
                new_names,
                enabled_toolsets=enabled,
                disabled_toolsets=disabled,
                strict_memory_schemas=True,
                fallback_provider_names=fallback_provider_names,
                fallback_provider_tools=fallback_provider_tools,
                reserved_registry_names=scoped_registry_names,
            )
            _reinject_authorized_dynamic_tools(agent, new_defs, new_names)
            # A transient availability probe may omit a still-registered tool.
            # Retain its existing schema and position at a live prefix boundary.
            if preserve_prefix:
                new_defs, new_names = _merge_preserving_prefix(
                    fallback_provider_tools, new_defs,
                    set(snapshot_registry_entries) - fallback_provider_names - fallback_engine_names,
                )
                scoped_registry_names.update(new_names & snapshot_registry_entries.keys())
            staged_registry_routes = {
                name: snapshot_registry_entries[name]
                for name in scoped_registry_names
            }
        except Exception:
            # A failed policy rebuild remains pending at its unpublished epoch.
            # The next automatic refresh retries that exact policy; a newer
            # explicit policy replaces it under the lock and keeps ordering.
            raise

        # Single atomic read-diff-publish so the returned ``added`` is
        # consistent with what was actually published.
        with _agent_tools_lock:
            if not registry.generation_is_current(snapshot_generation):
                if attempt + 1 >= _MAX_TOOL_REFRESH_GENERATION_ATTEMPTS:
                    raise RuntimeError(
                        "tool registry generation did not stabilize after "
                        f"{_MAX_TOOL_REFRESH_GENERATION_ATTEMPTS} attempts"
                    )
                continue
            # Defensive: the published generation should be an int, but tolerate an
            # agent that never set it (or set a non-int, e.g. a test mock) rather
            # than throwing TypeError on the comparison and silently failing the
            # whole refresh.
            published_gen_raw = getattr(agent, "_tool_snapshot_generation", -1)
            published_gen = published_gen_raw if isinstance(published_gen_raw, int) else -1
            if snapshot_generation < published_gen:
                # A newer snapshot already won. An automatic refresh can be
                # dropped, but an explicit policy must be rebuilt against the
                # winning generation before its epoch can be published.
                if (
                    policy_refresh
                    and getattr(agent, "_tool_policy_epoch", None)
                    == refresh_policy_epoch
                ):
                    if attempt + 1 >= _MAX_TOOL_REFRESH_GENERATION_ATTEMPTS:
                        raise RuntimeError(
                            "tool registry generation did not stabilize after "
                            f"{_MAX_TOOL_REFRESH_GENERATION_ATTEMPTS} attempts"
                        )
                    continue
                return set()
            if refresh_policy_epoch < getattr(agent, "_tool_policy_epoch", 0):
                # A newer explicit policy reload started after this snapshot.
                return set()
            published_snapshot_epoch_raw = getattr(
                agent, "_tool_snapshot_epoch", 0
            )
            published_snapshot_epoch = (
                published_snapshot_epoch_raw
                if isinstance(published_snapshot_epoch_raw, int)
                else 0
            )
            if snapshot_epoch != published_snapshot_epoch:
                # Another caller published a same-generation snapshot after
                # this attempt captured its provider fallback. Automatic
                # refreshes are stale and stop; a current explicit policy
                # retries against the winning snapshot.
                if (
                    policy_refresh
                    and getattr(agent, "_tool_policy_epoch", None)
                    == refresh_policy_epoch
                ):
                    if attempt + 1 >= _MAX_TOOL_REFRESH_GENERATION_ATTEMPTS:
                        raise RuntimeError(
                            "agent tool snapshot did not stabilize after "
                            f"{_MAX_TOOL_REFRESH_GENERATION_ATTEMPTS} attempts"
                        )
                    continue
                return set()
            current_tools = list(getattr(agent, "tools", None) or [])
            current = {
                t["function"]["name"]
                for t in current_tools
            }
            current_provider_names = _published_memory_provider_names(agent, current_tools)
            current_engine_names = set(
                getattr(agent, "_context_engine_tool_names", set()) or set()
            )
            current_registry_routes = getattr(agent, "_tool_registry_routes", {})
            if not isinstance(current_registry_routes, dict):
                current_registry_routes = {}
            registry_routes_changed = (
                current_registry_routes.keys() != staged_registry_routes.keys()
                or any(
                    current_registry_routes[name] is not entry
                    for name, entry in staged_registry_routes.items()
                )
            )
            provider_availability_changed = (
                current_provider_names != staged_provider_names
            )
            snapshot_changed = (
                current != new_names
                or (content_aware and _tool_defs_content_changed(agent, new_defs))
                or provider_availability_changed
                or current_engine_names != staged_engine_names
                or registry_routes_changed
            )
            if not snapshot_changed:
                agent._memory_provider_tool_names = set(staged_provider_names)
                # The complete schema and routing snapshot is unchanged, so
                # preserve the live list object and avoid churn.
                # Record the generation so an in-flight older caller can't clobber.
                agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
                if policy_refresh:
                    _publish_staged_policy(agent, enabled, disabled, refresh_policy_epoch)
                return set()
            agent.tools = new_defs
            agent.valid_tool_names = new_names
            if policy_refresh:
                _publish_staged_policy(agent, enabled, disabled, refresh_policy_epoch)
            # Publish context-engine routing names atomically with the snapshot.
            engine_names = getattr(agent, "_context_engine_tool_names", None)
            if isinstance(engine_names, set):
                engine_names.clear()
                engine_names.update(staged_engine_names)
            else:
                agent._context_engine_tool_names = set(staged_engine_names)
            agent._memory_provider_tool_names = set(staged_provider_names)
            agent._tool_registry_routes = dict(staged_registry_routes)
            agent._tool_snapshot_generation = max(published_gen, snapshot_generation)
            agent._tool_snapshot_epoch = published_snapshot_epoch + 1
            if provider_availability_changed:
                agent._cached_system_prompt = None
            added = new_names - current
        persist_agent_tool_names(agent)
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
    generation, entries = registry.snapshot_entries_with_generation()
    with _agent_tools_lock:
        epoch = getattr(agent, "_tool_snapshot_epoch", 0)
        fresh_defs = _agent_tool_defs(agent)
        current_routes = dict(getattr(agent, "_tool_registry_routes", {}) or {})
    fresh = {_def_name(t): t for t in fresh_defs}

    def _saved_def(name):
        if name in fresh:
            return fresh[name]
        entry = entries.get(name)
        return None if entry is None else {"type": "function", "function": {**entry.schema, "name": entry.name}}

    saved_defs = [d for d in map(_saved_def, saved_names) if d is not None]
    registered_names = set(entries)
    merged, merged_names = _merge_preserving_prefix(saved_defs, fresh_defs, registered_names)
    _reinject_authorized_dynamic_tools(agent, merged, merged_names)
    with _agent_tools_lock:
        if (not registry.generation_is_current(generation)
                or getattr(agent, "_tool_snapshot_epoch", 0) != epoch):
            return False
        if merged == fresh_defs:
            return False
        agent.tools = merged
        agent.valid_tool_names = merged_names
        agent._tool_registry_routes = {
            name: entries[name] for name in merged_names | current_routes.keys()
            if name in entries
        }
        agent._tool_snapshot_generation = generation
        agent._tool_snapshot_epoch = epoch + 1
    if [_def_name(t) for t in merged] != list(saved_names):
        persist_agent_tool_names(agent)
    return True


def _merge_preserving_prefix(current_defs: list, new_defs: list, registered_names: set) -> tuple[list, set]:
    """Fold a fresh tool snapshot into a live one without moving existing bytes. Ordered by
    ``current_defs`` (the cached request prefix): a name in both keeps its slot but takes the
    fresh schema; a name only in the live list is kept if still registered (``check_fn``
    flapped), else dropped; a name only in the fresh list is appended at the tail.

    The bridge tools keep their BUILT entry, not the fresh one: ``tool_search``'s description
    is derived from the session (deferred count, listing, whether ``manage_connections`` was
    present), so a late MCP server or a ``check_fn`` flap would rewrite it every turn. Search
    reads the live catalog at dispatch, so the stale count costs nothing."""
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    fresh = {_def_name(entry): entry for entry in new_defs if _def_name(entry)}
    merged = []
    for entry in current_defs:
        name = _def_name(entry)
        replacement = fresh.pop(name, None)
        if name in BRIDGE_TOOL_NAMES:
            merged.append(entry)
        elif replacement is not None:
            merged.append(replacement)
        elif name and name in registered_names:
            merged.append(entry)
    merged.extend(fresh.values())
    return merged, {_def_name(t) for t in merged}


def _reinject_authorized_dynamic_tools(agent, tools_list: list, name_set: set) -> None:
    """``message_agent`` is injected by an auth gate, never registered, so a registry-derived
    rebuild drops it. Scrub any stale copy from the STAGED pair and re-add it only when the live
    gate re-authorizes, so the publisher exposes a coherent ``(tools, valid_tool_names)``."""
    from tools.bot_mode_dm import MESSAGE_AGENT_TOOL_NAME, message_agent_authorized, message_agent_tool_schema

    tools_list[:] = [entry for entry in tools_list if _def_name(entry) != MESSAGE_AGENT_TOOL_NAME]
    name_set.discard(MESSAGE_AGENT_TOOL_NAME)
    if message_agent_authorized(agent):
        tools_list.append(message_agent_tool_schema())
        name_set.add(MESSAGE_AGENT_TOOL_NAME)


def _reinject_post_build_tools(
    agent,
    tools_list: list,
    name_set: set,
    *,
    enabled_toolsets=None,
    disabled_toolsets=None,
    strict_memory_schemas: bool = False,
    fallback_provider_names=None,
    fallback_provider_tools=None,
    reserved_registry_names=None,
) -> tuple[set, set]:
    """Append memory-provider and context-engine tools onto staged locals.

    Mirrors the post-``get_tool_definitions`` injection in ``agent_init`` so a
    snapshot rebuild reconstructs the FULL tool surface, not just the
    registry-derived subset. Operates ONLY on the caller's staged ``tools_list``
    / ``name_set`` (never the live agent attributes) so the rebuild stays atomic.
    Idempotent (skips names already present) and fail-soft.

    Returns the context-engine and memory-provider names actually appended by
    THIS rebuild.  The sets match ``agent_init``'s dedup behavior (a name
    already provided by a registry/plugin tool is NOT claimed by an injected
    family).
    """
    reserved_names = set(reserved_registry_names or set())

    def _add(schema: dict) -> bool:
        name = schema.get("name", "")
        if not name or name in name_set or name in reserved_names:
            return False
        tools_list.append({"type": "function", "function": schema})
        name_set.add(name)
        return True

    # Memory-provider tools (mem0/honcho/byterover/supermemory/…).
    staged_provider_names: set = set()
    try:
        memory_manager = getattr(agent, "_memory_manager", None)
        get_mem_schemas = getattr(memory_manager, "get_all_tool_schemas", None) if memory_manager else None
        if callable(get_mem_schemas):
            # Honor the same final toolset policy inject_memory_provider_tools
            # uses, while remaining independently mergeable from the startup
            # policy helpers.
            from agent.memory_manager import normalize_tool_schema
            strict_get_schemas = getattr(
                memory_manager, "get_all_tool_schemas_strict", None
            )
            schema_callback = (
                strict_get_schemas
                if strict_memory_schemas and callable(strict_get_schemas)
                else get_mem_schemas
            )
            memory_selected = "memory" in name_set
            if _memory_provider_family_disabled_for_refresh(disabled_toolsets):
                # Complete family revocation is independent of provider code;
                # do not let a broken plugin veto its own removal.
                effective_schemas = []
            else:
                try:
                    raw_schemas = list(schema_callback())
                    normalized_schemas = [
                        schema
                        for raw_schema in raw_schemas
                        if (schema := normalize_tool_schema(raw_schema)) is not None
                    ]
                    effective_schemas = _effective_memory_provider_schemas_for_refresh(
                        normalized_schemas,
                        enabled_toolsets=enabled_toolsets,
                        disabled_toolsets=disabled_toolsets,
                        memory_selected=memory_selected,
                    )
                except Exception:
                    if not strict_memory_schemas:
                        raise
                    # Tightening can safely filter the already-published
                    # provider contracts when fresh enumeration is unavailable.
                    # It may remove names, never invent or add them.
                    current_provider_names = set(fallback_provider_names or set())
                    current_provider_schemas = [
                        schema
                        for tool in (fallback_provider_tools or [])
                        if isinstance(tool, dict)
                        and (
                            schema := normalize_tool_schema(tool)
                        ) is not None
                        and schema["name"] in current_provider_names
                    ]
                    effective_schemas = _effective_memory_provider_schemas_for_refresh(
                        current_provider_schemas,
                        enabled_toolsets=enabled_toolsets,
                        disabled_toolsets=disabled_toolsets,
                        memory_selected=memory_selected,
                    )
                    effective_names = {
                        schema["name"] for schema in effective_schemas
                    }
                    if not effective_names <= current_provider_names:
                        raise
            for schema in effective_schemas:
                name = schema.get("name", "")
                if _add(schema) and name:
                    staged_provider_names.add(name)
    except Exception:
        if strict_memory_schemas:
            raise
        logger.debug("Memory-provider tool re-injection skipped", exc_info=True)
        staged_provider_names = set()

    # Context-engine tools (lcm_grep/lcm_describe/…) — the `context_engine`
    # toolset is intentionally empty, so these only exist via this append.
    # Honor the same enabled_toolsets gate agent_init uses (#5544): without it a
    # restricted-toolset platform (e.g. platform_toolsets: telegram: []) would
    # re-leak lcm_* tools the build deliberately excluded, and pay the local-
    # model latency penalty.
    staged_engine_names: set = set()
    try:
        compressor = getattr(agent, "context_compressor", None)
        get_schemas = getattr(compressor, "get_tool_schemas", None) if compressor else None
        if callable(get_schemas) and _context_engine_family_enabled_for_refresh(
            enabled_toolsets,
            disabled_toolsets,
        ):
            schemas = _effective_context_engine_schemas_for_refresh(
                get_schemas(),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )
            for schema in schemas:
                name = schema.get("name", "")
                # Only claim the routing name when WE appended the schema, so a
                # name already owned by a registry/plugin tool keeps its own
                # dispatch (matches agent_init.py's `continue`-before-claim).
                if _add(schema) and name:
                    staged_engine_names.add(name)
    except Exception:
        logger.debug("Context-engine tool re-injection skipped", exc_info=True)

    return staged_engine_names, staged_provider_names


def _dynamic_denied_tool_names(
    disabled_toolsets,
    *,
    family_name: str,
    family_marker: Optional[str] = None,
) -> Optional[set[str]]:
    """Resolve dynamic-tool subtraction without requiring startup helpers.

    The snapshot-consistency change must remain independently mergeable from
    the startup-policy change.  When the centralized helpers are present this
    module uses them; this local equivalent preserves the same fail-closed
    refresh behavior when this PR lands first.
    """
    if not disabled_toolsets:
        return set()
    if isinstance(disabled_toolsets, str):
        disabled_names = [disabled_toolsets]
    else:
        disabled_names = [str(name) for name in disabled_toolsets]
    if any(name in {"all", "*", family_name} for name in disabled_names):
        return None

    try:
        from toolsets import bundle_non_core_tools, get_toolset, resolve_toolset, validate_toolset

        denied: set[str] = set()
        for name in disabled_names:
            if not validate_toolset(name):
                continue
            resolved = set(
                bundle_non_core_tools(name)
                if name.startswith("hermes-") or (get_toolset(name) or {}).get("posture")
                else resolve_toolset(name)
            )
            if family_marker is not None and family_marker in resolved:
                return None
            denied.update(resolved)
    except Exception:
        logger.debug(
            "Failed to resolve disabled toolsets for %s tools",
            family_name,
            exc_info=True,
        )
        return None
    return denied



def _effective_memory_provider_schemas_for_refresh(
    raw_schemas,
    *,
    enabled_toolsets,
    disabled_toolsets,
    memory_selected: bool,
) -> list[dict]:
    """Apply provider policy during refresh on either merge order."""
    from agent import memory_manager as memory_module

    centralized = getattr(
        memory_module,
        "effective_memory_provider_tool_schemas",
        None,
    )
    if callable(centralized):
        return centralized(
            raw_schemas,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            memory_selected=memory_selected,
        )

    if not memory_selected and not memory_module.memory_provider_tools_enabled(
        enabled_toolsets,
    ):
        return []
    denied_names = _dynamic_denied_tool_names(
        disabled_toolsets,
        family_name="memory",
        family_marker="memory",
    )
    if denied_names is None:
        return []

    effective = []
    for raw_schema in raw_schemas:
        schema = memory_module.normalize_tool_schema(raw_schema)
        if schema is not None and schema["name"] not in denied_names:
            effective.append(schema)
    return effective



def _memory_provider_family_disabled_for_refresh(disabled_toolsets) -> bool:
    """Return whether refresh policy removes every provider-owned schema."""
    from agent import memory_manager as memory_module

    centralized = getattr(memory_module, "memory_provider_tools_disabled", None)
    if callable(centralized):
        return centralized(disabled_toolsets)
    return (
        _dynamic_denied_tool_names(
            disabled_toolsets,
            family_name="memory",
            family_marker="memory",
        )
        is None
    )



def _effective_context_engine_schemas_for_refresh(
    raw_schemas,
    *,
    enabled_toolsets,
    disabled_toolsets,
) -> list[dict]:
    """Apply context-engine policy during refresh on either merge order."""
    from agent import context_engine as context_module
    from agent.memory_manager import normalize_tool_schema

    centralized = getattr(
        context_module,
        "effective_context_engine_tool_schemas",
        None,
    )
    if callable(centralized):
        return centralized(
            raw_schemas,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )

    if enabled_toolsets is not None and "context_engine" not in enabled_toolsets:
        return []
    denied_names = _dynamic_denied_tool_names(
        disabled_toolsets,
        family_name="context_engine",
    )
    if denied_names is None:
        return []

    effective = []
    for raw_schema in raw_schemas:
        schema = normalize_tool_schema(raw_schema)
        if schema is not None and schema["name"] not in denied_names:
            effective.append(schema)
    return effective



def _context_engine_family_enabled_for_refresh(
    enabled_toolsets,
    disabled_toolsets,
) -> bool:
    """Check family policy before invoking provider-controlled enumeration."""
    from agent import context_engine as context_module

    centralized = getattr(
        context_module,
        "context_engine_tool_family_enabled",
        None,
    )
    if callable(centralized):
        return centralized(enabled_toolsets, disabled_toolsets)
    if enabled_toolsets is not None and "context_engine" not in enabled_toolsets:
        return False
    return (
        _dynamic_denied_tool_names(
            disabled_toolsets,
            family_name="context_engine",
        )
        is not None
    )
