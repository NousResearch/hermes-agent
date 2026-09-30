from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.connectors.gateway.bridge import SIGN_IN_EXPIRED, UNREACHABLE
from tools.connectors.gateway.names import format_connector_name, is_connector_name, vendor_slug_candidates
from tools.tool_search_catalog import CatalogEntry, _fn, _tokenize

logger = logging.getLogger(__name__)


def connections_in_scope(tool_defs: Iterable[Dict[str, Any]]) -> bool:
    return any(_fn(td).get("name") == "manage_connections" for td in tool_defs)


def connectors_unavailable(failure: str, *, verb: str,
                           names: Optional[List[str]] = None) -> Dict[str, Any]:
    hint = (f"Hosted connector tools could not be {verb} right now. "
            "Do not conclude the app is missing.")
    if failure == SIGN_IN_EXPIRED:
        hint += " The user must sign in to Nous again."
    field: Dict[str, Any] = {"status": "unavailable", "reason": failure, "hint": hint}
    if names:
        field["names"] = names
    return field


def _connector_entry(name: str, connector: str, slug: str, schema: Dict[str, Any]) -> CatalogEntry:
    description = str(schema.get("description") or "")
    input_schema = schema.get("input_schema")
    parameters = input_schema if isinstance(input_schema, dict) else {}
    tool_def = {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters}}
    text = f"{connector} {slug.replace('_', ' ')} {description}"
    return CatalogEntry(name=name, description=description, schema=tool_def,
                        source="connectors", source_name=connector, _tokens=_tokenize(text))


def connector_entries_by_group(
    queries: List[str],
    connector_search: Optional[Any] = None,
) -> Tuple[List[List[CatalogEntry]], Optional[str]]:
    per_query: List[List[CatalogEntry]] = [[] for _ in queries]
    try:
        if connector_search is None:
            from tools.connectors.gateway.bridge import connector_search_hits as connector_search
        leg = connector_search([{"use_case": q} for q in queries])
        if leg.failure:
            return per_query, leg.failure
        hits = leg.payload or {}
        schemas = hits.get("schemas")
        groups = hits.get("results")
        if not isinstance(schemas, dict) or not isinstance(groups, list):
            return per_query, (UNREACHABLE if hits else None)
        for position, group in enumerate(groups[: len(queries)]):
            if not isinstance(group, dict):
                continue
            echoed = group.get("use_case")
            if isinstance(echoed, str) and echoed and echoed != queries[position]:
                continue
            slugs = group.get("tools") if isinstance(group.get("tools"), list) else []
            picked: Dict[str, tuple[str, CatalogEntry]] = {}
            for slug in slugs:
                schema = schemas.get(slug)
                if not isinstance(schema, dict) or not schema.get("connector"):
                    continue
                # Gateway policy matches normalized lowercase connector slugs; tool slugs stay verbatim.
                slug = str(slug)
                connector = str(schema["connector"]).lower()
                name = format_connector_name(connector, slug)
                prior = picked.get(name)
                if prior is not None and prior[0] != slug:
                    # Keep the slug this composed name resolves to; its twin would execute differently.
                    reaches = vendor_slug_candidates(connector, name.split("__", 2)[2])[0]
                    logger.warning("connector %s: vendor slugs %s and %s both compose to %s, which reaches %s",
                                   connector, prior[0], slug, name, reaches)
                    if slug != reaches:
                        continue
                elif prior is not None:
                    continue
                picked[name] = (slug, _connector_entry(name, str(schema["connector"]), slug, schema))
            per_query[position] = [entry for _, entry in picked.values()]
    except Exception:
        logger.debug("connector search merge failed (D32)", exc_info=True)
        return [[] for _ in queries], UNREACHABLE
    return per_query, None


def remote_schemas_for(
    names: List[str],
    current_tool_defs: List[Dict[str, Any]],
    connector_describe: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    connector_names = [n for n in names if is_connector_name(n)]
    if not connector_names or not connections_in_scope(current_tool_defs):
        return {}, None
    try:
        if connector_describe is None:
            from tools.connectors.gateway.bridge import connector_describe
        leg = connector_describe(connector_names)
        if leg.failure:
            return {}, leg.failure
        tools = leg.payload.get("tools")
        if isinstance(tools, dict):
            return tools, None
        return {}, (UNREACHABLE if leg.payload else None)
    except Exception:
        logger.debug("connector describe merge failed (D32)", exc_info=True)
        return {}, UNREACHABLE
