"""Connector (remote) leg of the tool-search bridge.

Connector tools live on the managed tool gateway, not in the local registry.
``tool_search`` asks the gateway for hits per query and ranks them in the same
BM25 pass as local tools; ``tool_describe`` fetches their schemas by name.
Every failure path (signed out, config off, gateway dark, bad shapes) yields
empty results so local search behaves exactly as without connectors (D32).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from tools.tool_gateway.names import format_connector_name, is_connector_name, vendor_slug_candidates
from tools.tool_search_catalog import CatalogEntry, _fn, _tokenize

logger = logging.getLogger(__name__)


def connections_in_scope(tool_defs: Iterable[Dict[str, Any]]) -> bool:
    """The session granted connections and its availability check passed."""
    return any(_fn(td).get("name") == "manage_connections" for td in tool_defs)


def _connector_entry(name: str, connector: str, slug: str, schema: Dict[str, Any]) -> CatalogEntry:
    """A gateway hit as a catalog document, so it ranks in the same BM25 pass as local
    tools. The search text is the connector name, the slug's words and the description:
    the same fields a local entry indexes, so the rarest-token gate treats both alike."""
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
    """Remote connector hits for ``dispatch_tool_search`` as catalog entries, one list per
    query, in the gateway's order.

    Correlation with the remote response is by ARRAY POSITION only: the wire ``index`` field
    is 1-based vendor passthrough on the search route and is never read.
    """
    per_query: List[List[CatalogEntry]] = [[] for _ in queries]
    try:
        if connector_search is None:
            from tools.tool_gateway.bridge import connector_search_hits as connector_search
        hits = connector_search([{"use_case": q} for q in queries]) or {}
        schemas = hits.get("schemas")
        groups = hits.get("results")
        if not isinstance(schemas, dict) or not isinstance(groups, list):
            return per_query
        for position, group in enumerate(groups[: len(queries)]):
            if not isinstance(group, dict):
                continue
            # Correlate by position, then verify the echoed use_case when the
            # gateway provides one, never the wire index (NS-734). A
            # mismatched echo means the response groups don't line up with
            # our queries; drop the group rather than mis-attribute hits.
            echoed = group.get("use_case")
            if isinstance(echoed, str) and echoed and echoed != queries[position]:
                continue
            slugs = group.get("tools") if isinstance(group.get("tools"), list) else []
            picked: Dict[str, tuple[str, CatalogEntry]] = {}  # name -> (slug, entry), gateway order
            for slug in slugs:
                schema = schemas.get(slug)
                if not isinstance(schema, dict) or not schema.get("connector"):
                    continue  # cannot compose a callable name without its connector
                # Lowercase the connector half at composition: the search
                # route leaks vendor-cased connector slugs for custom
                # toolkits (live-verified 2026-08-25: connections say
                # custom_nous_lab_deepwiki while schemas say
                # CUSTOM_NOUS_LAB_DEEPWIKI in the SAME response), and the
                # gateway's policy gates compare case-sensitively against
                # the lowercase catalog form. Tool slugs stay verbatim.
                # No-op once the gateway normalizes its own surface.
                slug = str(slug)
                connector = str(schema["connector"]).lower()
                name = format_connector_name(connector, slug)
                prior = picked.get(name)
                if prior is not None and prior[0] != slug:
                    # Composition is not injective: GMAIL_X and a literal X on gmail
                    # both compose to connectors__gmail__X, and describe and execute
                    # decode that name to GMAIL_X first. Keep the twin the name
                    # reaches; describing the other under this name would run a
                    # different tool.
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
    """Schemas for the ``connectors__*`` names in ``names``, keyed by name, for
    ``dispatch_tool_describe``. Empty when no connector names were asked for, when
    connections are out of scope, or on any gateway failure; the caller then reports
    those names as ``not_found``."""
    connector_names = [n for n in names if is_connector_name(n)]
    if not connector_names or not connections_in_scope(current_tool_defs):
        return {}
    try:
        if connector_describe is None:
            from tools.tool_gateway.bridge import connector_describe
        remote = connector_describe(connector_names)
        if isinstance(remote, dict) and isinstance(remote.get("tools"), dict):
            return remote["tools"]
    except Exception:
        logger.debug("connector describe merge failed silently (D32)", exc_info=True)
    return {}
