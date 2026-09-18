"""Remote connector adapter for tool search and descriptions.

Failures return no connector results so local search behavior is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from tools.connectors.gateway.names import format_connector_name, is_connector_name, vendor_slug_candidates
from tools.tool_search_catalog import CatalogEntry, _fn, _tokenize

logger = logging.getLogger(__name__)


def connections_in_scope(tool_defs: Iterable[Dict[str, Any]]) -> bool:
    return any(_fn(td).get("name") == "manage_connections" for td in tool_defs)


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
) -> List[List[CatalogEntry]]:
    """Correlate remote response groups by position, never their wire index."""
    per_query: List[List[CatalogEntry]] = [[] for _ in queries]
    try:
        if connector_search is None:
            from tools.connectors.gateway.bridge import connector_search_hits as connector_search
        hits = connector_search([{"use_case": q} for q in queries]) or {}
        schemas = hits.get("schemas")
        groups = hits.get("results")
        if not isinstance(schemas, dict) or not isinstance(groups, list):
            return per_query
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
        logger.debug("connector search merge failed silently (D32)", exc_info=True)
        return [[] for _ in queries]
    return per_query


def remote_schemas_for(
    names: List[str],
    current_tool_defs: List[Dict[str, Any]],
    connector_describe: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    connector_names = [n for n in names if is_connector_name(n)]
    if not connector_names or not connections_in_scope(current_tool_defs):
        return {}
    try:
        if connector_describe is None:
            from tools.connectors.gateway.bridge import connector_describe
        remote = connector_describe(connector_names)
        if isinstance(remote, dict) and isinstance(remote.get("tools"), dict):
            return remote["tools"]
    except Exception:
        logger.debug("connector describe merge failed silently (D32)", exc_info=True)
    return {}
