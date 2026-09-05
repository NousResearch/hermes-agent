"""Registering a connected (or schema-cached) MCP server's tools into the tool registry:
include/exclude filtering, trust-tier metadata capture, utility-tool selection, name-collision
resolution and the schema-cache write-through. Both entry points (``_register_server_tools``
live, ``_register_from_cache_sync`` lazy) build ``_Candidate`` records for ``_register_candidates``."""

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional
from tools.mcp_tool_common import _parse_boolish, _core, _resolve_tool_timeout, mcp_field
from tools import mcp_tool_handlers as _handlers
from tools import mcp_tool_schema as _schema
from tools.mcp_tool_handlers import (
    _make_check_fn, _make_get_prompt_handler, _make_list_prompts_handler,
    _make_list_resources_handler, _make_read_resource_handler)
from tools.mcp_tool_schema import (
    _UTILITY_CAPABILITY_ATTRS, _build_utility_schemas, _normalize_name_filter, matches_name_filter)

if TYPE_CHECKING:  # pragma: no cover
    from tools.mcp_tool import MCPServerTask

logger = logging.getLogger("tools.mcp_tool")

_UTILITY_ORIGIN_PREFIX = "generated utility "
# Utility tool key -> handler factory; each takes (server_name, tool_timeout).
_UTILITY_HANDLER_FACTORIES = {
    "list_resources": _make_list_resources_handler, "read_resource": _make_read_resource_handler,
    "list_prompts": _make_list_prompts_handler, "get_prompt": _make_get_prompt_handler}


def _normalize_server_trust(value: Any) -> str:
    """Config ``trust`` -> tier. None -> ``full`` (compat default); unrecognized -> ``untrusted`` (fail closed)."""
    if value is None:
        return _core._TRUST_FULL
    text = str(value).strip().lower()
    if text in (_core._TRUST_FULL, _core._TRUST_UNTRUSTED):
        return text
    logger.warning("MCP trust: unrecognized trust value %r — treating as 'untrusted' (valid values: full, untrusted)",
                   value)
    return _core._TRUST_UNTRUSTED


def _annotation_read_only_hint(mcp_tool: Any) -> bool:
    """True only when annotations (SDK object or cache dict) carry ``readOnlyHint is True``; unknown = write-capable."""
    annotations = getattr(mcp_tool, "annotations", None)
    hint = annotations.get("readOnlyHint") if isinstance(annotations, dict) else getattr(annotations, "readOnlyHint", None)
    return hint is True


def _record_tool_trust_metadata(server_name: str, config: dict, tools: List[Any]) -> None:
    """Capture per-server trust and per-tool readOnlyHint at discovery — the security boundary: the call-time gate
    classifies from data we control, never re-read server-supplied state."""
    with _core._lock:
        _core._server_trust_levels[server_name] = _normalize_server_trust((config or {}).get("trust"))
        hints = _core._tool_read_only_hints.setdefault(server_name, {})
        hints.update({t.name: _annotation_read_only_hint(t) for t in tools if getattr(t, "name", None)})


def _track_mcp_tool_server(tool_name: str, server_name: str) -> None:
    """Remember the exact raw MCP server that registered *tool_name*."""
    with _core._lock:
        _core._mcp_tool_server_names[tool_name] = server_name


def _forget_mcp_tool_server(tool_name: str) -> None:
    """Forget MCP server provenance for a deregistered tool."""
    with _core._lock:
        _core._mcp_tool_server_names.pop(tool_name, None)


def _select_utility_schemas(server_name: str, server: "MCPServerTask", config: dict) -> List[dict]:
    """Utility schemas allowed by config (``tools.resources``/``tools.prompts``) and advertised
    capabilities. ``initialize_result.capabilities`` is the truth (sub-object non-None iff the
    family is served); without it fall back to the legacy session-method check, which never
    filters anything since ClientSession defines all four methods."""
    tools_filter = config.get("tools") or {}
    enabled = {f: _parse_boolish(tools_filter.get(f), default=True) for f in ("resources", "prompts")}
    advertised = getattr(getattr(server, "initialize_result", None), "capabilities", None)

    def _skip_reason(handler_key: str) -> Optional[str]:
        family = _UTILITY_CAPABILITY_ATTRS[handler_key]
        if not enabled[family]:
            return f"{family} disabled"
        if advertised is not None:
            if getattr(advertised, family, None) is None:
                return f"server does not advertise '{family}' capability"
            return None
        # Legacy gate (no initialize_result): the ClientSession method shares the handler key.
        return None if hasattr(server.session, handler_key) else f"session lacks {handler_key}"
    selected: List[dict] = []
    for entry in _build_utility_schemas(server_name):
        reason = _skip_reason(entry["handler_key"])
        if reason:
            logger.debug("MCP server '%s': skipping utility '%s' (%s)", server_name, entry["handler_key"], reason)
        else:
            selected.append(entry)
    return selected


def _existing_tool_names() -> List[str]:
    """Tool names for all connected servers plus lazy (cache-registered) servers, whose tools live only in the registry."""
    names: List[str] = []
    for server in _core._servers.values():
        names.extend(server._registered_tool_names if hasattr(server, "_registered_tool_names")
                     else (_schema._convert_mcp_schema(server.name, t)["name"] for t in server._tools))
    with _core._lock:
        names.extend(n for sname, tool_names in _core._lazy_server_tool_names.items()
                     if sname not in _core._servers for n in tool_names)
    return names


def _make_tool_filter(name: str, config: dict) -> Callable[[str], bool]:
    """Include/exclude predicate for a server's tool names: ``tools.include`` is a whitelist (``[]`` = register
    nothing), ``tools.exclude`` a blacklist; entries are exact names or fnmatch globs; include wins over exclude."""
    tools_filter = config.get("tools") or {}
    # Selective tool loading: honour include/exclude lists from config. Rules (matching issue #690 spec,
    # extended with glob support): tools.include — whitelist: only matching tool names are registered
    # tools.exclude — blacklist: all tools EXCEPT matching ones are registered entries may be exact names or
    # fnmatch globs (e.g. "*_radar_*") include takes precedence over exclude include: [] → register nothing
    # (an explicit empty whitelist, as written by the install checklist's "uncheck everything" path) Neither
    # set → register all tools (backward-compatible default)
    include_raw = tools_filter.get("include")
    include_set = _normalize_name_filter(include_raw, f"mcp_servers.{name}.tools.include")
    exclude_set = _normalize_name_filter(tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude")
    if isinstance(include_raw, (str, list, tuple, set)):
        return lambda tool_name: matches_name_filter(tool_name, include_set)
    return lambda tool_name: not (exclude_set and matches_name_filter(tool_name, exclude_set))


def _cached_tools(raws: Iterable[Any]) -> List[SimpleNamespace]:
    """Schema-cache rows -> stand-ins for MCP Tool objects; rows that are not dicts or lack a name
    are dropped. Missing or non-dict ``annotations`` (older cache files) fail closed to write-capable."""
    return [SimpleNamespace(name=raw["name"], description=raw.get("description") or "",
                            inputSchema=raw["inputSchema"] if isinstance(raw.get("inputSchema"), dict) else {},
                            annotations=raw["annotations"] if isinstance(raw.get("annotations"), dict) else None)
            for raw in raws if isinstance(raw, dict) and raw.get("name")]


@dataclass
class _Candidate:
    """One registration attempt (native tool or generated utility); ``origin`` is the provenance text in diagnostics."""

    registry_name: str
    origin: str
    schema: dict
    handler: Callable

    @property
    def is_utility(self) -> bool:
        return self.origin.startswith(_UTILITY_ORIGIN_PREFIX)


def _tool_candidates(name: str, tools: Iterable[Any], should_register: Callable[[str], bool],
                     tool_timeout) -> List[_Candidate]:
    """Native tools (live SDK objects or cache stand-ins) -> candidates. The injection scan runs on
    BOTH paths: the cache file is user-writable JSON."""
    out: List[_Candidate] = []
    for t in tools:
        if not should_register(t.name):
            logger.debug("MCP server '%s': skipping tool '%s' (filtered by config)", name, t.name)
            continue
        _schema._scan_mcp_description(name, t.name, t.description or "")
        schema = _schema._convert_mcp_schema(name, t)
        handler = _handlers._make_tool_handler(name, t.name, tool_timeout)
        out.append(_Candidate(schema["name"], f"tool {t.name!r}", schema, handler))
    return out


def _utility_candidates(name: str, entries: Iterable[Any], tool_timeout) -> List[_Candidate]:
    """``{schema, handler_key}`` rows (live selection or cache) -> candidates; malformed rows dropped."""
    out: List[_Candidate] = []
    for raw in entries:
        schema, key = (raw.get("schema"), raw.get("handler_key")) if isinstance(raw, dict) else (None, None)
        if isinstance(schema, dict) and key in _UTILITY_HANDLER_FACTORIES and schema.get("name"):
            out.append(_Candidate(schema["name"], f"{_UTILITY_ORIGIN_PREFIX}{key!r}", schema,
                                  _UTILITY_HANDLER_FACTORIES[key](name, tool_timeout)))
    return out


def _resolve_name_collisions(name: str, candidates: List[_Candidate]) -> List[_Candidate]:
    """Preflight name collisions: exact duplicates dropped silently; a utility normalizing onto
    a native tool's name is shadowed (native wins); any other multi-origin collision skips every
    colliding entry (fail closed). Returns survivors in order."""
    unique: List[_Candidate] = []
    origins_by_name: Dict[str, set[str]] = {}
    for c in candidates:
        origins = origins_by_name.setdefault(c.registry_name, set())
        if c.origin in origins:
            logger.debug("MCP server '%s': duplicate registration candidate %s for '%s'; keeping one",
                         name, c.origin, c.registry_name)
            continue
        origins.add(c.origin)
        unique.append(c)
    ambiguous: Dict[str, List[str]] = {}
    shadowed: set[tuple[str, str]] = set()
    # A generated resource/prompt utility that normalizes onto a server-native tool's name must not knock
    # that native tool out of the registry: the native tool is the capability the user connected the server
    # for, while the generated utility (read_resource/list_resources/list_prompts/get_prompt) is optional
    # sugar that only matters when the server exposes no such tool of its own (#87112). Resolve that
    # specific collision in favour of the native tool — keep it, drop the shadowed utility — and fall back
    # to the conservative skip-everything only for genuinely ambiguous collisions (two or more native tools
    # normalizing to one name, which we cannot disambiguate). The four utility keys are distinct, so a
    # colliding set holds at most one utility origin.
    for registry_name, origins in origins_by_name.items():
        if len(origins) <= 1:
            continue
        utility_origins = sorted(o for o in origins if o.startswith(_UTILITY_ORIGIN_PREFIX))
        native_origins = sorted(origins - set(utility_origins))
        if len(native_origins) == 1 and utility_origins:
            shadowed.update((registry_name, o) for o in utility_origins)
            logger.info(
                "MCP server '%s': generated utility %s normalizes onto server-native %s — keeping the native tool "
                "and dropping the utility (the utility only applies when the server has no such tool of its own)",
                name, ", ".join(utility_origins), native_origins[0])
        else:
            ambiguous[registry_name] = sorted(origins)
    for registry_name, origins in sorted(ambiguous.items()):
        logger.error("MCP server '%s': name normalization collision for '%s' from %s; skipping every colliding "
                     "entry instead of choosing an arbitrary handler", name, registry_name, ", ".join(origins))
    return [c for c in unique if c.registry_name not in ambiguous and (c.registry_name, c.origin) not in shadowed]


def _register_candidates(name: str, candidates: List[_Candidate], *, check_fn: Callable,
                         scope: Callable[[], Optional[str]], lazy: bool) -> List[str]:
    """Register candidates under toolset ``mcp-{name}``; returns the names that landed. The
    ownership pre-check is advisory (servers connect in parallel): ``registry.register()`` is
    the atomic gate and its verdict is re-read after every call."""
    from tools.registry import registry
    toolset_name = f"mcp-{name}"
    registered: List[str] = []
    for c in candidates:
        existing_toolset = registry.get_toolset_for_tool(c.registry_name)
        if existing_toolset and existing_toolset != toolset_name:  # foreign owner: skip, preserve it
            if lazy:
                if not c.is_utility:
                    logger.warning("MCP server '%s' (lazy): cached tool '%s' collides with toolset '%s' — skipping",
                                   name, c.registry_name, existing_toolset)
            elif existing_toolset.startswith("mcp-"):
                logger.error("MCP server '%s': %s normalizes to '%s', already owned by MCP toolset '%s' — skipping to "
                             "preserve the existing owner", name, c.origin, c.registry_name, existing_toolset)
            else:
                logger.warning("MCP server '%s': %s (→ '%s') collides with built-in tool in toolset '%s' — skipping to "
                               "preserve built-in", name, c.origin, c.registry_name, existing_toolset)
            continue
        registry.register(
            name=c.registry_name, toolset=toolset_name, schema=c.schema, handler=c.handler, check_fn=check_fn,
            is_async=False, description=c.schema.get("description") or "", scope=scope())
        if registry.get_toolset_for_tool(c.registry_name) == toolset_name:
            _track_mcp_tool_server(c.registry_name, name)
            registered.append(c.registry_name)
        elif not lazy:
            logger.error("MCP server '%s': registration of %s as '%s' was rejected by the registry; "
                         "skipping provenance/count updates", name, c.origin, c.registry_name)
    if registered:
        registry.register_toolset_alias(name, toolset_name)
    return registered


def _write_schema_cache(name: str, server: "MCPServerTask", config: dict, should_register) -> None:
    """Write-through: persist the manifest so the next startup registers this server lazily (no spawn). Never raises."""
    try:
        # Write-through (#56832): refresh the on-disk schema cache after a live connect so the next startup
        # can lazily register this server without spawning it. Cache failures never break registration.
        from tools.mcp_schema_cache import config_fingerprint, write_cache_entry
        tools_payload = []
        for t in server._tools:
            if not should_register(t.name):
                continue
            # mcp 2.0 renamed every Tool model field to snake_case and left camelCase as a
            # *serialization* alias only, which pydantic does not apply to attribute access: a bare
            # camelCase getattr returns None on 2.x instead of raising. That silently wrote an empty
            # ``inputSchema`` into the schema cache on every write-through, so a ``lazy: true`` server
            # registered from cache with every parameter stripped. ``mcp_field`` reads both spellings.
            schema_obj = mcp_field(t, "input_schema", "inputSchema")
            tools_payload.append({
                "name": t.name, "description": t.description or "",
                "inputSchema": schema_obj if isinstance(schema_obj, dict) else {},
                "annotations": {"readOnlyHint": _annotation_read_only_hint(t)},  # lazy path trust-gates identically
            })
        utility_payload = [{"schema": e["schema"], "handler_key": e["handler_key"]}
                           for e in _select_utility_schemas(name, server, config)]
        cache_meta = getattr(server, "_list_cache_meta", None) or {}
        write_cache_entry(name, config_fingerprint(config), tools=tools_payload, utility_tools=utility_payload,
                          ttl_ms=cache_meta.get("ttl_ms"), cache_scope=cache_meta.get("cache_scope"))
    except Exception as exc:
        logger.debug("MCP schema cache write failed for '%s': %s", name, exc)


def _register_server_tools(name: str, server: "MCPServerTask", config: dict) -> List[str]:
    """Register a connected server's tools plus utilities (initial discovery and list_changed
    refresh); returns the names. Toolset aliases derive from the live registry, not
    ``toolsets.TOOLSETS``; lossy normalization collisions (``read-file``/``read_file``) fail closed."""
    should_register = _make_tool_filter(name, config)
    _record_tool_trust_metadata(name, config, server._tools)
    candidates = _tool_candidates(name, server._tools, should_register, server.tool_timeout)
    candidates += _utility_candidates(name, _select_utility_schemas(name, server, config), server.tool_timeout)
    registered = _register_candidates(
        name, _resolve_name_collisions(name, candidates),
        check_fn=_make_check_fn(name), scope=lambda: _core._server_registry_scope(name), lazy=False)
    if registered:
        _write_schema_cache(name, server, config, should_register)
    return registered


def _register_from_cache_sync(name: str, config: dict, entry: dict) -> List[str]:
    """Lazy startup: register from a cached manifest with no child process (first real call goes
    through ``_ensure_lazy_server_connected``). Trust metadata is recorded first so the
    call-time gate is identical for live and cached registrations.

    Lazy startup (#56832, design by Vansh5632): tools appear in the registry immediately; the first real
    call routes through ``_get_connected_server_for_call`` → ``_ensure_lazy_server_connected``.
    """
    from tools.mcp_schema_cache import config_fingerprint, tools_from_cache_entry, utility_tools_from_cache_entry
    tool_timeout = _resolve_tool_timeout(config)
    cached_tools = _cached_tools(tools_from_cache_entry(entry))
    _record_tool_trust_metadata(name, config, cached_tools)
    candidates = _tool_candidates(name, cached_tools, _make_tool_filter(name, config), tool_timeout)
    candidates += _utility_candidates(name, utility_tools_from_cache_entry(entry), tool_timeout)
    registered = _register_candidates(
        name, candidates, check_fn=_make_check_fn(name), scope=_core._mcp_registry_scope, lazy=True)
    if registered:
        with _core._lock:
            _core._lazy_server_configs[name] = dict(config)
            _core._lazy_server_fingerprints[name] = config_fingerprint(config)
            _core._lazy_server_tool_names[name] = list(registered)
        logger.info("MCP server '%s' (lazy): registered %d tool(s) from schema cache", name, len(registered))
    return registered
