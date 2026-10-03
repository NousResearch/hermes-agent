"""``manage_catalog``: the setup agent's catalog search and install-through-the-card.

``search`` reads the plugin catalog (the Plugins tab's resolver) and the skills hub and says what is
already installed in ``default``. ``install`` opens the same connection operation
``manage_connections`` opens, with rows of kind ``plugin`` / ``skill``; the host installs each row
the user approves (``tools/connectors/catalog.py``). The model sends catalog ids and an action,
nothing else: the pin, scan, target profile and activation are the host's.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from tools.registry import tool_error

ACTIONS = ("search", "install")
ITEM_KINDS = ("plugin", "skill")
_TOP_KEYS = frozenset({"action", "kind", "query", "items", "reason"})
_ITEM_KEYS = frozenset({"kind", "id"})
SEARCH_LIMIT = 10

NOT_HERE = (
    "manage_catalog is available only in the Hermes desktop app, where the approval card can be drawn. "
    "Tell the user to install from a terminal instead: `hermes plugins install <id>` for a plugin, "
    "`hermes skills install <id>` for a skill."
)

NOTE = (
    "Settled once; do not re-offer a row the user skipped. Each installed row names target_profile: "
    "its tools and skills are live now in that profile's open chats and in every new chat of it; "
    "they are callable in this chat only when this chat runs in that profile. A failed row carries "
    "the reason in detail."
)

MANAGE_CATALOG_SCHEMA = {
    "name": "manage_catalog",
    "description": (
        "Find and install Hermes catalog plugins and hub skills for the user. 'search' lists matches "
        "(id, kind, display, tier, platforms, installed) and changes nothing. 'install' shows the user "
        "one approval card with a row per item and blocks until every row is installed, skipped, or "
        "the card is closed; the host installs each approved row into the user's default profile at "
        "the catalog's reviewed version. Pass only catalog ids exactly as 'search' returns them; the "
        "host decides the source, version, profile and settings, and the user can change them on the "
        "card. Offer an install only for something the user asked for or agreed to."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "kind": {"type": "string", "enum": list(ITEM_KINDS),
                     "description": "search only: limit results to plugins or skills."},
            "query": {"type": "string", "description": "search only: words to match; empty lists plugins."},
            "items": {
                "type": "array",
                "description": "install only: the catalog items to offer, one card row each.",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(ITEM_KINDS)},
                        "id": {"type": "string", "description": "Catalog id as 'search' returns it."},
                    },
                    "required": ["kind", "id"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string", "description": "install only: one line on why these are offered."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _check_args(args: Dict[str, Any]) -> Optional[str]:
    unknown = sorted(set(args) - _TOP_KEYS)
    if unknown:
        return (f"unknown parameter(s) {', '.join(unknown)}: manage_catalog takes action, kind, query, "
                "items and reason. The host decides source, version, profile and settings.")
    if args.get("action") not in ACTIONS:
        return f"action must be one of {', '.join(ACTIONS)}."
    if args.get("kind") not in (None, *ITEM_KINDS):
        return f"kind must be one of {', '.join(ITEM_KINDS)}."
    if args["action"] == "search":
        return None
    items = args.get("items")
    if not isinstance(items, list) or not items:
        return "install needs 'items': [{\"kind\": \"plugin\", \"id\": \"<catalog id>\"}, ...]."
    for item in items:
        if not isinstance(item, dict):
            return "every item is an object {kind, id}."
        extra = sorted(set(item) - _ITEM_KEYS)
        if extra:
            return (f"unknown item field(s) {', '.join(extra)}: an item is {{kind, id}}. The host decides "
                    "source, version, profile and settings.")
        if item.get("kind") not in ITEM_KINDS or not str(item.get("id") or "").strip():
            return "every item needs kind (plugin or skill) and a non-empty id."
    return None


def manage_catalog(
    args: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    card_surface: bool = False,
    installer: Any = None,
) -> str:
    """``card_surface``: the session is a desktop chat, the one surface that draws catalog rows."""
    error = _check_args(args)
    if error:
        return tool_error(error)
    if connection_callback is None or not card_surface:
        return tool_error(NOT_HERE)
    if args["action"] == "search":
        return json.dumps(search(str(args.get("query") or ""), args.get("kind"), installer=installer),
                          ensure_ascii=False)
    return install(args["items"], session_id=session_id, tool_call_id=tool_call_id,
                   connection_callback=connection_callback, installer=installer)


def install(items: List[Dict[str, str]], *, session_id: Optional[str], tool_call_id: Optional[str],
            connection_callback: Callable[[Dict[str, Any]], Optional[str]], installer: Any = None) -> str:
    from tools.connectors.catalog import open_runner
    from tools.connectors.gateway.config import operation_session_key
    from tools.connectors.operation import Target
    from tools.connectors.run import Kind, run_operation

    seen, keys = set(), []
    for item in items:
        key = (item["kind"], str(item["id"]).strip())
        if key not in seen:
            seen.add(key)
            keys.append(key)
    counts = Counter(identifier for _, identifier in keys)
    reserved = {identifier for _, identifier in keys}
    identifiers, targets = {}, []
    for kind, identifier in keys:
        name = identifier
        if counts[identifier] > 1:
            name = f"{kind}:{identifier}"
            while name in reserved:
                name = f"{kind}:{name}"
            reserved.add(name)
        identifiers[name] = identifier
        targets.append(Target(name=name, kind=kind, action="install"))
    runner = open_runner(installer, identifiers=identifiers)
    try:
        return run_operation(
            targets, Kind(prepare=runner.prepare, observe=runner.observe, note=NOTE),
            session_key=operation_session_key(session_id), tool_call_id=tool_call_id,
            connection_callback=connection_callback, with_urls_in_result=False,
        )
    finally:
        runner.close()


def search(query: str, kind: Optional[str], *, installer: Any = None) -> Dict[str, Any]:
    from tools.connectors.catalog import DEFAULT_PROFILE, target_scope

    rows: List[Dict[str, Any]] = []
    with target_scope(DEFAULT_PROFILE):
        if kind in (None, "plugin"):
            rows += _plugin_rows(query)
        if kind in (None, "skill") and query.strip():
            rows += _skill_rows(query)
    return {"results": rows, "installed_in": DEFAULT_PROFILE}


def _plugin_rows(query: str) -> List[Dict[str, Any]]:
    from hermes_cli.plugin_catalog import filter_entries, load_catalog_live
    from hermes_cli.plugins_cmd import PluginOperationError, _read_install_metadata
    from tools.connectors.catalog import _display

    try:
        installed = {str(rec["catalog"]["name"]) for rec in _read_install_metadata().values()
                     if isinstance(rec, dict) and isinstance(rec.get("catalog"), dict) and rec["catalog"].get("name")}
    except PluginOperationError:
        installed = set()
    return [{"id": e.name, "kind": "plugin", "display": _display(e.name), "tier": e.tier,
             "platforms": list(e.platforms), "installed": e.name in installed}
            for e in filter_entries(load_catalog_live(), query)[:SEARCH_LIMIT]]


def _skill_rows(query: str) -> List[Dict[str, Any]]:
    from tools.skills_hub import HubLockFile
    from tools.skills_hub_github import GitHubAuth
    from tools.skills_hub_search import create_source_router, unified_search

    installed = {str(e.get("identifier")) for e in HubLockFile().list_installed()}
    return [{"id": r.identifier, "kind": "skill", "display": r.name,
             "tier": "official" if r.source == "official" else "community", "platforms": [],
             "installed": r.identifier in installed}
            for r in unified_search(query, create_source_router(GitHubAuth()), limit=SEARCH_LIMIT)]
