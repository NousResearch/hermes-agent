"""Resolve a weak model's MCP ``tool_call`` onto a registered tool and run it.

Exact names pass through. A name that is the unique suffix of one registered MCP
tool is that tool. A guessed ``mcp__<server>__...`` name is remapped only when one
read tool on that server shares a specific word with the guess. Write tools run
only when the guess names that action. Unknown argument keys are dropped so the
schema check does not reject the call before the server sees it.

Several MCP calls in one ``tool_call`` run in order. Identical calls run once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from tools.tool_search_catalog import CatalogEntry, _tokenize, build_catalog

MCP_BATCH_SENTINEL = "__mcp_batch__"

_GENERIC = frozenset(_tokenize(
    "get list fetch tool mcp data info show find query all the please call"
))
_WRITE = frozenset(_tokenize(
    "send delete remove create update publish unpublish archive grant revoke "
    "cancel pause resume execute trigger import upsert destroy"
))
# A guessed key fills the schema key when the tool does not use the guessed name.
_ARG_ALIASES = {
    "broadcast_id": ("campaign_id", "broadcastId", "campaignId"),
    "workspace_id": ("workspaceId",),
    "account_id": ("accountId",),
}


class McpPrepared(NamedTuple):
    name: Optional[str]
    arguments: Dict[str, Any]
    resolved_from: Optional[str]
    error: Optional[str]


def _tool_names() -> List[str]:
    from tools.registry import registry
    return registry.get_all_tool_names()


def _schema_fn(name: str) -> Dict[str, Any]:
    from tools.registry import registry
    schema = registry.get_schema(name) or {}
    if not isinstance(schema, dict):
        return {}
    fn = schema.get("function") if schema.get("type") == "function" else schema
    return fn if isinstance(fn, dict) else {}


def _is_mcp_tool(name: str) -> bool:
    from tools.tool_search_catalog import _registry_toolset
    toolset = _registry_toolset(name)
    return isinstance(toolset, str) and toolset.startswith("mcp-")


def unique_mcp_suffix(name: str) -> Optional[str]:
    """The one registered MCP tool whose name ends with ``__<name>``, if there is exactly one."""
    if not name or name.startswith("mcp__") or "__" in name:
        return None
    suffix = f"__{name}"
    matches = [n for n in _tool_names() if n.startswith("mcp__") and n.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def is_mcp_call_name(name: str) -> bool:
    """True when this tool_call entry is an MCP tool, or a guess at one."""
    if name.startswith("mcp__") and name.count("__") >= 2:
        return True
    return unique_mcp_suffix(name) is not None


def _is_write(name: str) -> bool:
    return bool(set(_tokenize(name.split("__")[-1])) & _WRITE)


def _properties(name: str) -> Dict[str, Any]:
    params = _schema_fn(name).get("parameters") or {}
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def _fit_args(name: str, arguments: Dict[str, Any], *, query: str, remap: bool) -> Dict[str, Any]:
    """Keep arguments the tool declares. A remapped click/open guess sets that flag."""
    args = dict(arguments or {})
    props = _properties(name)
    if props:
        for dest, sources in _ARG_ALIASES.items():
            if dest in props and dest not in args:
                for source in sources:
                    if source in args:
                        args[dest] = args.pop(source)
                        break
        if _schema_fn(name).get("parameters", {}).get("additionalProperties") is not True:
            args = {key: value for key, value in args.items() if key in props}
        if remap:
            stems = set(_tokenize(query))
            if "click" in stems and "has_clicked" in props and "has_clicked" not in args:
                args["has_clicked"] = True
            if "open" in stems and "has_opened" in props and "has_opened" not in args:
                args["has_opened"] = True
    return args


def _catalog(server: str) -> List[CatalogEntry]:
    prefix = f"mcp__{server}__"
    defs = []
    for name in _tool_names():
        if not name.startswith(prefix) or not _is_mcp_tool(name):
            continue
        fn = _schema_fn(name)
        defs.append({"type": "function", "function": {
            "name": name,
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {},
        }})
    return build_catalog(defs)


def _specific(query: str, server: str) -> frozenset:
    skip = set(_GENERIC)
    skip.update(_tokenize(server))
    return frozenset(_tokenize(query)) - skip


def _score(entry: CatalogEntry, specific: frozenset) -> Tuple[int, int]:
    name_hits = len(specific & set(_tokenize(entry.name)))
    all_hits = len(specific & set(entry._tokens))
    return name_hits, all_hits


def _pick(catalog: Sequence[CatalogEntry], query: str, server: str) -> Tuple[Optional[CatalogEntry], List[str]]:
    """Return ``(winner, [])`` or ``(None, closest names)`` when the guess is ambiguous or empty."""
    specific = _specific(query, server)
    ranked = sorted(
        ((score, entry) for entry in catalog if (score := _score(entry, specific))[1] > 0),
        key=lambda pair: (-pair[0][0], -pair[0][1], pair[1].name),
    )
    if not ranked:
        return None, [entry.name for entry in catalog[:4]]
    request_write = bool(specific & _WRITE)
    if request_write:
        pool = [pair for pair in ranked if _is_write(pair[1].name)]
    else:
        pool = [pair for pair in ranked if not _is_write(pair[1].name)]
    if not pool:
        return None, [pair[1].name for pair in ranked[:4]]
    if len(pool) > 1 and pool[0][0] == pool[1][0]:
        return None, [pool[0][1].name, pool[1][1].name]
    return pool[0][1], []


def _unresolved(name: str, closest: Sequence[str]) -> str:
    hint = f" Closest: {', '.join(closest)}." if closest else " Use tool_search to find the exact name."
    return f"'{name}' is not a known tool name.{hint}"


def prepare_mcp_call(name: str, arguments: Dict[str, Any]) -> Optional[McpPrepared]:
    """Resolve one MCP entry. ``None`` means this entry is not an MCP call."""
    if name.startswith("mcp__") and name.count("__") >= 2:
        if _is_mcp_tool(name):
            return McpPrepared(name, _fit_args(name, arguments, query="", remap=False), None, None)
        server = name.split("__")[1]
        rest = name.split("__", 2)[2]
        rest = re.sub(rf"^{re.escape(server)}_", "", rest)
        catalog = _catalog(server)
        winner, closest = _pick(catalog, rest.replace("_", " "), server) if catalog else (None, [])
        if winner is None:
            return McpPrepared(None, {}, None, _unresolved(name, closest))
        return McpPrepared(
            winner.name, _fit_args(winner.name, arguments, query=rest, remap=True), name, None)
    hit = unique_mcp_suffix(name)
    if hit is None:
        return None
    return McpPrepared(hit, _fit_args(hit, arguments, query="", remap=False), None, None)


def dispatch_mcp_batch(
    calls: List[Dict[str, Any]], ids, *, user_task, enabled_tools, middleware_trace,
    enabled_toolsets, disabled_toolsets, scoped_names: frozenset,
) -> str:
    """Run each resolved MCP call in order. Identical name+arguments run once."""
    from model_tools import handle_function_call
    from tools.interrupt import is_interrupted

    unique: List[Dict[str, Any]] = []
    seen = set()
    for call in calls:
        tool_name = call.get("name")
        if not tool_name:
            unique.append(call)
            continue
        key = (tool_name, json.dumps(call.get("arguments") or {}, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)

    results: List[Dict[str, Any]] = []
    for call in unique:
        if is_interrupted():
            results.append({"name": call.get("name"), "error": "Stopped by the user before this call was made."})
            break
        if call.get("error") or not call.get("name"):
            results.append({
                "name": call.get("resolved_from") or call.get("name"),
                "error": call.get("error") or "unresolved",
            })
            continue
        if call["name"] not in scoped_names:
            results.append({
                "name": call["name"],
                "error": f"'{call['name']}' is not available in this session.",
            })
            continue
        payload = handle_function_call(
            call["name"], call.get("arguments") or {}, **asdict(ids), user_task=user_task,
            enabled_tools=enabled_tools, tool_request_middleware_trace=list(middleware_trace),
            skip_pre_tool_call_hook=False, skip_tool_request_middleware=False,
            skip_tool_execution_middleware=False,
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        )
        results.append({
            "name": call["name"],
            "resolved_from": call.get("resolved_from"),
            "response": payload,
        })

    if len(results) == 1 and "response" in results[0]:
        body = results[0]["response"]
        if results[0].get("resolved_from"):
            parsed: Any = body
            if isinstance(body, str):
                try:
                    parsed = json.loads(body)
                except ValueError:
                    parsed = body
            return json.dumps({
                "tool": results[0]["name"],
                "resolved_from": results[0]["resolved_from"],
                "result": parsed,
            }, ensure_ascii=False)
        return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return json.dumps({"results": results}, ensure_ascii=False)
