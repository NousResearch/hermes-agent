"""Progressive tool disclosure ("tool search"): MCP/plugin tools and a curated set of
event-triggered core tools are replaced in the model-visible array by three bridge tools —
tool_search / tool_describe / tool_call. Invariants: core tools (``toolsets._HERMES_CORE_TOOLS``)
and session-gated GUI toolsets never defer unless named in ``defer``; ANY deferrable tool
activates the bridge (the listing scales with budget, not activation); the catalog is
stateless — rebuilt from the live tool-defs every assembly (a session-keyed one drifts and
silently drops tools); bridge calls route through ``model_tools.handle_function_call``."""

from __future__ import annotations

import functools
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import tool_error
from tools.tool_search_catalog import (
    BRIDGE_TOOL_NAMES, CHARS_PER_TOKEN, TOOL_CALL_NAME, TOOL_DESCRIBE_NAME, TOOL_SEARCH_NAME,
    CatalogEntry, _corpus_stats, _fn, _listing_group_label, _registry_entry, _registry_toolset,
    build_catalog, build_catalog_listing_with_form, search_catalog)
from tools.tool_search_validation import validate_deferred_call_args

logger = logging.getLogger("tools.tool_search")
_MAX_QUERIES_PER_CALL = _MAX_DESCRIBE_NAMES_PER_CALL = 10  # bound the work one bridge call requests


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""
    enabled: str  # "auto" | "on" | "off" — "auto" is an alias of "on" today
    # Listing budget as % of context; does NOT gate activation, only bounds how much
    # the embedded manifest may consume before it degrades (full -> names -> bare).
    threshold_pct: float  # 0..100
    search_default_limit: int
    max_search_limit: int
    listing: str = "auto"  # "auto"/"on" = embed the manifest when it fits; "off" = bare bridge
    listing_max_tokens: int = 4000  # budget = min(this, threshold_pct% of context)
    # None = curated default; an explicit list replaces it wholesale ([] = defer no core tools).
    defer_tools: Optional[frozenset] = None

    @property
    def effective_defer_tools(self) -> frozenset:
        return _DEFAULT_DEFERRED_TOOLS if self.defer_tools is None else self.defer_tools

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build from a raw dict / legacy bool / None; every field is clamped and unknown
        values fall back to safe defaults — a config typo must not break the agent."""
        if not isinstance(raw, dict):  # legacy bool / None
            raw = {"enabled": "off" if raw is False else "auto"}
        max_search_limit = _clamped_int(raw.get("max_search_limit"), 25, 1, 50)
        defer_raw = raw.get("defer")
        return cls(
            enabled=_tri_state(raw.get("enabled", "auto")),
            threshold_pct=max(0.0, min(100.0, _safe_float(raw.get("threshold_pct"), 5.0))),
            search_default_limit=_clamped_int(
                raw.get("search_default_limit"), 5, 1, max_search_limit),
            max_search_limit=max_search_limit,
            listing=_tri_state(raw.get("listing", "auto")),
            listing_max_tokens=_clamped_int(raw.get("listing_max_tokens"), 4000, 200, 60000),
            defer_tools=(frozenset(str(n).strip() for n in defer_raw if str(n).strip())
                         if isinstance(defer_raw, (list, tuple, set)) else None))


_TRI_STATE_ALIASES = {"true": "on", "1": "on", "yes": "on", "false": "off", "0": "off", "no": "off"}


def _tri_state(value: Any) -> str:
    """Normalize an ``auto``/``on``/``off`` setting (bool-ish aliases accepted)."""
    text = str(value).strip().lower()
    return _TRI_STATE_ALIASES.get(text, text if text in ("auto", "on", "off") else "auto")


def _clamped_int(value: Any, fallback: int, lo: int, hi: int) -> int:
    """``int(value)`` (or ``fallback`` when unparseable) clamped to ``[lo, hi]``."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = fallback
    return max(lo, min(hi, value))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _config_from_loader(loader_name: str) -> ToolSearchConfig:
    """Tool-search config via ``hermes_cli.config.<loader_name>`` (defaults on any failure)."""
    try:
        import hermes_cli.config as _cfg_mod
        tools_cfg = (getattr(_cfg_mod, loader_name)() or {}).get("tools")
        tools_cfg = tools_cfg if isinstance(tools_cfg, dict) else {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))
    except Exception as e:
        logger.debug("Failed to load tool-search config: %s", e)
        return ToolSearchConfig.from_raw(None)


load_config = functools.partial(_config_from_loader, "load_config")
load_config_readonly = functools.partial(_config_from_loader, "load_config_readonly")  # no copy


def _core_tool_names() -> frozenset[str]:
    """Names that never defer by default (lazy: ``toolsets`` imports ``tools.registry``)."""
    try:
        from toolsets import _HERMES_CORE_TOOLS
        return frozenset(_HERMES_CORE_TOOLS)
    except Exception:
        return frozenset()


# Session-gated GUI toolsets: off ``_HERMES_CORE_TOOLS`` so non-GUI clients never pay
# their schema; once enabled they stay direct unless the deferral list names them.
_DIRECT_SURFACE_TOOLSETS = frozenset({"desktop_ui", "project"})

# Event-triggered core tools deferred BY DEFAULT (a catalog stub suffices); the ``defer``
# config replaces this wholesale ([] = everything eager). POST-rename names. ``clarify``
# is deliberately absent: A/B showed deferring it collapsed structured-clarify usage
# (18/18 -> 7/18) — the ask-the-user affordance must be ambient, a stub is not enough.
_DEFAULT_DEFERRED_TOOLS = frozenset({
    "computer_use", "session_search", "image_generate",
    "todo_list", "process_manage", "cronjob_manage",
    # Desktop GUI surface (desktop_ui + project toolsets)
    "drive_preview", "gui_tour", "desktop_preview", "annotate_preview",
    "show_tip", "setup_mcp", "desktop_project", "close_terminal",
    "apply_layout", "read_terminal", "read_window_below", "focus_pane"})


def is_deferrable_tool_name(name: str, defer_tools: Optional[frozenset] = None) -> bool:
    """True if a tool is *eligible* for deferral: named in ``defer_tools`` (curated set or
    user override), OR an MCP tool, OR neither core nor a session-gated GUI surface (i.e. a
    plugin tool). Bridge names never defer."""
    if name in BRIDGE_TOOL_NAMES:
        return False
    if defer_tools is not None and name in defer_tools:
        return True
    if name in _core_tool_names():
        return False
    toolset = _registry_toolset(name)  # None (unregistered/malformed) never defers
    return toolset is not None and (
        toolset.startswith("mcp-") or toolset not in _DIRECT_SURFACE_TOOLSETS)


def _tool_def_names(tool_defs: Iterable[Dict[str, Any]]) -> Iterable[str]:
    """Function names of a tool-defs list (``""`` for a nameless def)."""
    return (_fn(td).get("name", "") for td in tool_defs)


def classify_tools(tool_defs: List[Dict[str, Any]], defer_tools: Optional[frozenset] = None,
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable); bridge tools are dropped (re-added
    after classification)."""
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    for td, name in zip(tool_defs, _tool_def_names(tool_defs)):
        if name not in BRIDGE_TOOL_NAMES:
            (deferrable if is_deferrable_tool_name(name, defer_tools) else visible).append(td)
    return visible, deferrable


def _deferrable_in(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deferrable subset of pre-assembly ``tool_defs`` under the read-only user config."""
    return classify_tools(tool_defs, load_config_readonly().effective_defer_tools)[1]


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Token cost via the chars/4 rule (order-of-magnitude precision suffices)."""
    def _chars(td: Dict[str, Any]) -> int:
        try:
            return len(json.dumps(td, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            return len(str(td))
    return int(math.ceil(sum(map(_chars, tool_defs)) / CHARS_PER_TOKEN))


def should_activate(config: ToolSearchConfig, deferrable_tokens: int,
                    context_length: Optional[int]) -> bool:
    """``"off"`` never activates; ``"on"``/``"auto"`` activate whenever any deferrable tool
    exists ("auto" is reserved for a future budget-gated mode — do not distinguish them
    without that design). ``context_length`` is kept for caller compatibility."""
    return config.enabled != "off" and deferrable_tokens > 0


def listing_token_budget(config: ToolSearchConfig, context_length: Optional[int]) -> int:
    """``min(listing_max_tokens, threshold_pct% of context)``; unknown context uses a 10K
    percentage leg (5% of a typical 200K window)."""
    pct_leg = (int(context_length * (config.threshold_pct / 100.0))
               if context_length and context_length > 0 else 10_000)
    return max(0, min(config.listing_max_tokens, pct_leg))


def _bridge_schema(name: str, description: str, properties: Dict[str, Any],
                   required: List[str]) -> Dict[str, Any]:
    """One OpenAI-style function schema (key order is part of the frozen bytes)."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


def _search_description(deferred_count: int, listing: Optional[str], listing_form: str) -> str:
    """tool_search bridge description with the listing embedded (framing per ``listing_form``)."""
    desc = (
        f"Search {deferred_count} additional tools that are loaded on demand. "
        "Takes a list of queries searched in parallel against the same "
        "catalog; send one query per distinct capability you need. Returns "
        "matching tool names grouped per query plus a shared map with each "
        "tool's description. Follow with "
        f"`{TOOL_DESCRIBE_NAME}` to load full parameter schemas, "
        f"then `{TOOL_CALL_NAME}` to invoke. Tools listed at the top of this "
        "system prompt are already available and do not need to be searched.")
    if not listing:
        return desc
    if listing_form == "groups":
        return desc + (
            "\n\nThe servers below are connected and their tools ARE available "
            "through this bridge. For any request in these domains, search "
            "here FIRST — do not claim the capability is unavailable and do "
            "not substitute a generic tool (terminal/browser) without "
            "searching.\n\n" + listing)
    desc += (
        "\n\nEvery deferred capability is listed below. If a tool name "
        "appears here, do NOT claim it is unavailable — load it with "
        f"`{TOOL_DESCRIBE_NAME}` (skip `{TOOL_SEARCH_NAME}` when you "
        "already see the exact name).")
    if listing_form == "mixed":
        desc += (
            " For servers marked 'names not listed', the tools exist "
            f"too — find them with `{TOOL_SEARCH_NAME}` before "
            "concluding anything is missing.")
    return desc + "\n\n" + listing


def bridge_tool_schemas(deferred_count: int, listing: Optional[str] = None,
                        listing_form: str = "") -> List[Dict[str, Any]]:
    """Bridge tool schemas injected in place of deferred tools; kept short — every byte is paid
    every turn. ``listing`` is embedded in the tool_search description; per-tool forms say
    "skip search when you see the exact name", "groups" says search is mandatory."""
    return [
        _bridge_schema(
            TOOL_SEARCH_NAME,
            _search_description(deferred_count, listing, listing_form),
            {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search queries, each a few keywords describing one capability (e.g. ['create github issue', 'send slack message']). Searched in parallel; results come back grouped per query. A single string is accepted and treated as one query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches per query. Defaults to 5 and is clamped to the configured maximum (25 by default).",
                },
            },
            ["queries"],
        ),
        _bridge_schema(
            TOOL_DESCRIBE_NAME,
            f"Load the full JSON schemas for tools returned by `{TOOL_SEARCH_NAME}`. "
            f"Required before `{TOOL_CALL_NAME}` if a tool's parameters are unknown. "
            "Batch every schema you need into one call.",
            {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact tool names (as returned by tool_search). A single string is accepted and treated as one name.",
                },
            },
            ["names"],
        ),
        _bridge_schema(
            TOOL_CALL_NAME,
            "Invoke a deferred tool by name with the given arguments. Argument shape "
            f"matches the tool's schema (see `{TOOL_DESCRIBE_NAME}`). Policy, hooks, "
            "and approvals run exactly as for any directly-listed tool.",
            {
                "name": {"type": "string", "description": "Exact tool name to invoke."},
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool, matching its schema.",
                },
            },
            ["name", "arguments"],
        ),
    ]


@dataclass
class AssemblyResult:
    """Outcome of one assembly (tests and observability)."""
    tool_defs: List[Dict[str, Any]]
    activated: bool
    deferred_count: int = 0
    deferred_tokens: int = 0
    threshold_tokens: int = 0
    # 0 = passthrough; 1 = bridge + per-tool listing; 2 = bare bridge / server summary only.
    tier: int = 0
    listing_form: str = "none"  # "full" | "names" | "mixed" | "groups" | "none"


def assemble_tool_defs(tool_defs: List[Dict[str, Any]], *, context_length: Optional[int] = None,
                       config: Optional[ToolSearchConfig] = None) -> AssemblyResult:
    """Tool-defs the model should see: passthrough when inactive, else deferrable tools
    replaced by the three bridge tools. Idempotent — existing bridge tools are stripped first."""
    config = config or load_config()
    incoming = [td for td, name in zip(tool_defs, _tool_def_names(tool_defs))
                if name not in BRIDGE_TOOL_NAMES]
    visible, deferrable = classify_tools(incoming, config.effective_defer_tools)
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)
    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming, activated=False, deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=int((context_length or 0) * (config.threshold_pct / 100.0)), tier=0)
    listing, listing_form = None, "none"
    listing_budget = listing_token_budget(config, context_length)
    if config.listing != "off":
        listing, listing_form = build_catalog_listing_with_form(
            deferrable, max_tokens=listing_budget)
    bridge = bridge_tool_schemas(len(deferrable), listing=listing, listing_form=listing_form)
    tier = 1 if listing_form in ("full", "names", "mixed") else 2
    logger.info(
        "tool_search activated (tier %d): %d core/visible tools kept, %d deferred "
        "(~%d tokens), listing %s (budget ~%d tokens)",
        tier, len(visible), len(deferrable), deferrable_tokens, listing_form, listing_budget)
    return AssemblyResult(
        tool_defs=visible + bridge, activated=True, deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens, threshold_tokens=listing_budget,
        tier=tier, listing_form=listing_form)


def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def _shared_tool_record(entry: CatalogEntry) -> Dict[str, Any]:
    """One record for the shared ``tools`` map (per-query groups carry names only);
    ``required`` lets the model attempt a trivial call without a ``tool_describe`` round-trip."""
    try:
        required = entry.schema["function"]["parameters"]["required"]
    except (TypeError, KeyError, AttributeError):
        required = []
    return {"source": entry.source, "source_name": entry.source_name,
            "description": (entry.description or "")[:400],  # cap chatty MCP descriptions
            "required": [r[:64] for r in (required if isinstance(required, list) else [])
                         if isinstance(r, str)][:32]}


def _available_source_summary(catalog: List[CatalogEntry]) -> List[Dict[str, Any]]:
    """Deterministic ``[{name, tool_count}]`` of connected sources (attached to empty query
    groups so a lexical miss is not read as a missing capability)."""
    counts = Counter(_listing_group_label(entry.source_name) for entry in catalog)
    return [{"name": name, "tool_count": counts[name]} for name in sorted(counts)]


def _string_list_arg(args: Dict[str, Any], key: str, *, dedupe: bool, max_items: int,
                     retry_hint: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Read a list-of-strings bridge argument -> ``(items, error_json)``. A bare string (a
    common model slip) is a one-item list; rejects non-lists, all-blank lists, > ``max_items``."""
    raw = args.get(key)
    raw = [raw] if isinstance(raw, str) else raw
    if not isinstance(raw, list):
        return None, tool_error(f"{key} is required and must be an array of strings")
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and (not dedupe or text not in out):
            out.append(text)
    if not out:
        return None, tool_error(f"{key} is required and must contain at least one non-empty string")
    if len(out) > max_items:
        return None, tool_error(f"too many {key}: {len(out)} > max {max_items}. {retry_hint}")
    return out, None


def dispatch_tool_search(args: Dict[str, Any], *, current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_search`` bridge tool -> JSON ``{queries, total_available,
    results: [{query, matches: [names]}], tools: {name: {source, source_name, description,
    required}}}``. ``limit`` applies PER QUERY; empty groups get ``available_sources`` +
    ``hint`` so a lexical miss is not mistaken for a missing capability."""
    config = config or load_config()
    queries, err = _string_list_arg(args, "queries", dedupe=False, max_items=_MAX_QUERIES_PER_CALL,
                                    retry_hint="Retry with fewer, more targeted queries.")
    if err:
        return err
    raw_limit = args.get("limit")
    limit = (config.search_default_limit if raw_limit is None
             else _clamped_int(raw_limit, config.search_default_limit, 1, config.max_search_limit))
    catalog = build_catalog(_deferrable_in(current_tool_defs))
    results: List[Dict[str, Any]] = []
    tools_map: Dict[str, Dict[str, Any]] = {}
    corpus_stats = _corpus_stats(catalog)
    available_sources = _available_source_summary(catalog) if catalog else []
    for query in queries:
        hits = search_catalog(catalog, query, limit=limit, corpus_stats=corpus_stats)
        for h in hits:
            tools_map.setdefault(h.name, _shared_tool_record(h))
        group: Dict[str, Any] = {"query": query, "matches": [h.name for h in hits]}
        if not hits and catalog:
            group["available_sources"] = available_sources
            group["hint"] = (
                "This query returned no lexical matches, but the sources above "
                "are connected and their tools remain available. Retry "
                "tool_search with the service name plus a concrete action or "
                "object before concluding the capability is unavailable.")
        results.append(group)
    return json.dumps({"queries": queries, "total_available": len(catalog), "results": results,
                       "tools": tools_map}, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any], *, current_tool_defs: List[Dict[str, Any]],
                           config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_describe`` bridge tool -> JSON ``{tools: {name: {description,
    parameters}}, not_found: [...]  (unknown / not in this assembly; never fails the call),
    errors: {name: msg}  (registered but non-deferrable)}``. Duplicates dedupe silently."""
    config = config or load_config_readonly()
    names, err = _string_list_arg(
        args, "names", dedupe=True, max_items=_MAX_DESCRIBE_NAMES_PER_CALL,
        retry_hint="Retry with fewer names per call.")
    if err:
        return err
    deferrable = _deferrable_in(current_tool_defs)
    by_name = {name: _fn(td) for td, name in zip(deferrable, _tool_def_names(deferrable)) if name}
    tools: Dict[str, Dict[str, Any]] = {}
    not_found: List[str] = []
    errors: Dict[str, str] = {}
    for name in names:
        fn = by_name.get(name)
        if fn is not None:
            tools[name] = {"description": fn.get("description", ""),
                           "parameters": fn.get("parameters", {})}
        elif _registry_entry(name) is not None and not is_deferrable_tool_name(
            name, load_config_readonly().effective_defer_tools):
            # Registered but bridge/core/GUI-surface: a real name, wrong door.
            errors[name] = (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search.")
        else:
            not_found.append(name)
    result: Dict[str, Any] = {"tools": tools}
    if not_found:
        result["not_found"] = not_found
        result["hint"] = "Names in not_found are not currently available. Re-run tool_search to refresh."
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


def scoped_deferrable_names(tool_defs: List[Dict[str, Any]]) -> frozenset[str]:
    """Deferrable names in the *pre-assembly* ``tool_defs`` of the session scope — the
    universe ``tool_call`` may reach. Gates bridge dispatch AND the executor unwrap so a
    restricted session cannot invoke an out-of-scope tool via the bridge."""
    defer_tools = load_config_readonly().effective_defer_tools
    return frozenset(n for n in _tool_def_names(tool_defs)
                     if n and is_deferrable_tool_name(n, defer_tools))


def _infer_flat_tool_call_target(flat_args: Dict[str, Any]) -> Optional[str]:
    """Infer the intended direct tool when a model omits 'name' and flattens
    arguments to the top level (e.g. Qwen2.5 via Ollama sends
    {"query": "..."} as the tool_call payload instead of
    {"name": "web_search", "arguments": {"query": "..."}}). Returns a tool
    name if the flat args uniquely match a registered tool's required
    parameters, else None. Cheap heuristic — only runs on the error path."""
    if not isinstance(flat_args, dict) or not flat_args:
        return None
    # Ignore bridge-level keys if somehow present
    candidate_keys = {k for k in flat_args.keys() if k not in ("name", "arguments")}
    if not candidate_keys:
        return None
    try:
        from tools.registry import registry as _reg
        best: list[str] = []
        for tname in _reg.get_all_tool_names():
            schema = _reg.get_schema(tname)
            if not isinstance(schema, dict):
                continue
            fn = schema.get("function") if schema.get("type") == "function" else schema
            params = fn.get("parameters") if isinstance(fn, dict) else None
            if not isinstance(params, dict):
                continue
            required = params.get("required")
            properties = params.get("properties")
            if not isinstance(properties, dict):
                continue
            if isinstance(required, list) and required:
                # All required keys must be present in the flat payload
                if not all(isinstance(r, str) and r in flat_args for r in required):
                    continue
            else:
                # No required list — require at least one known property overlap
                if not candidate_keys.intersection(properties.keys()):
                    continue
            # Every supplied key should be a known property (avoid false positives)
            if not candidate_keys.issubset(properties.keys()):
                continue
            best.append(tname)
        if len(best) == 1:
            return best[0]
        # Prefer obvious web_search/web_extract disambiguation when multiple
        # candidates remain (e.g. both accept 'query' due to loose schemas)
        if "web_search" in best and candidate_keys == {"query"}:
            return "web_search"
        if "web_search" in best and candidate_keys == {"query", "limit"}:
            return "web_search"
        return None
    except Exception:
        return None


def resolve_underlying_call(args: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse a ``tool_call`` invocation -> (underlying_name, args, error_msg); ``(None, {}, msg)``
    on error. Shared by dispatch, display and the trajectory recorder so all three agree."""
    name = str(args.get("name") or "").strip()
    if not name:
        # Qwen/Ollama and similar weak tool-call models flatten the intended
        # tool's arguments to the top level and omit the required 'name'
        # field, yielding {"query": "..."} instead of
        # {"name": "web_search", "arguments": {"query": "..."}}. Detect that
        # shape and return an actionable hint (or auto-infer when unambiguous)
        # so the model does not loop on the generic error 13+ times (#103752).
        inferred = _infer_flat_tool_call_target(args)
        if inferred:
            # Tell the model to call the tool directly — tool_call is only for
            # deferred (bridge) tools, and most inferred targets (web_search,
            # etc.) are core direct tools.
            return None, {}, (
                f"tool_call requires a 'name' argument — it looks like you meant to call "
                f"'{inferred}' with arguments {json.dumps(args, ensure_ascii=False)}. "
                f"Call '{inferred}' directly instead of via tool_call (tool_call is only for "
                f"deferred tools discovered via tool_search/tool_describe). "
                f"Example: {{\"name\": \"{inferred}\", \"arguments\": {json.dumps(args, ensure_ascii=False)}}}")
        hint_keys = ", ".join(sorted(str(k) for k in args.keys())[:5]) if isinstance(args, dict) and args else ""
        hint = f" Received keys: [{hint_keys}]." if hint_keys else ""
        return None, {}, (
            "tool_call requires a 'name' argument and an 'arguments' object. "
            "Use tool_call as {\"name\": \"<tool>\", \"arguments\": {...}} for deferred tools only; "
            "for directly-listed tools (e.g. web_search) call the tool by its own name."
            + hint)
    if name in BRIDGE_TOOL_NAMES:
        return None, {}, f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
    raw_args = args.get("arguments")
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, {}, f"tool_call 'arguments' is not valid JSON: {e}"
    raw_args = {} if raw_args is None else raw_args
    if not isinstance(raw_args, dict):
        return None, {}, "tool_call 'arguments' must be an object"
    if not is_deferrable_tool_name(name, load_config_readonly().effective_defer_tools):
        return None, {}, (
            f"'{name}' is not a deferrable tool. If it appears in the model-facing tools "
            "list already, call it directly instead of via tool_call.")
    return name, raw_args, None


__all__ = [
    "TOOL_SEARCH_NAME", "TOOL_DESCRIBE_NAME", "TOOL_CALL_NAME", "BRIDGE_TOOL_NAMES",
    "ToolSearchConfig", "CatalogEntry", "AssemblyResult", "load_config", "is_deferrable_tool_name",
    "classify_tools", "estimate_tokens_from_schemas", "should_activate", "build_catalog",
    "build_catalog_listing_with_form", "listing_token_budget", "search_catalog",
    "bridge_tool_schemas", "assemble_tool_defs", "is_bridge_tool", "dispatch_tool_search",
    "dispatch_tool_describe", "resolve_underlying_call", "scoped_deferrable_names",
    "validate_deferred_call_args"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Literal  # noqa: F401,E402
import copy  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import re  # noqa: F401,E402
import snowballstemmer  # noqa: F401,E402
import threading  # noqa: F401,E402

def build_catalog_listing(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Optional[str]:
    """Render a skills-style manifest of the deferred catalog.

    One line per tool — ``name: short description`` — grouped under a
    heading per source (MCP server / plugin toolset), exactly like the
    bundled-skills listing in the system prompt:

        github tools: (44)
        - create_issue: Open a new issue in a GitHub repository.
        - merge_pull_request: Merge an open pull request.
        ...

    Ordering is deterministic (groups and tools sorted by name) so the
    rendered block is byte-stable across assemblies of the same catalog —
    this keeps the request prefix cacheable across turns.

    Token-budget fallbacks (cheap chars/4 estimate, same rule as the
    activation gate):
      1. full listing (names + short descriptions)
      2. names-only listing, still grouped
      3. server-level summary — one line per MCP server / plugin toolset
         (name + tool count), so the model always knows WHICH domains are
         reachable through the bridge even when per-tool names don't fit
      4. ``None`` — only when the summary itself exceeds the budget
    """
    text, _form = build_catalog_listing_with_form(deferrable, max_tokens=max_tokens)
    return text
# ---- END PLUGIN-COMPAT ----
