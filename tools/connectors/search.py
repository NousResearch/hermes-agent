"""Connector search merge: hosted-connector hits for tool search.

The hosted tool gateway is not part of this build, so hosted connector search
always returns no hits. Local MCP servers surface through their own catalog
and are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.tool_search_catalog import CatalogEntry, _fn

logger = logging.getLogger(__name__)

UNREACHABLE = "unreachable"
SIGN_IN_EXPIRED = "sign_in_expired"


def connections_in_scope(tool_defs: Iterable[Dict[str, Any]]) -> bool:
    return any(_fn(td).get("name") == "manage_connections" for td in tool_defs)


def connectors_unavailable(failure: str, *, verb: str,
                           names: Optional[List[str]] = None) -> Dict[str, Any]:
    hint = (f"Hosted connector tools could not be {verb} right now. "
            "Do not conclude the app is missing.")
    field: Dict[str, Any] = {"status": "unavailable", "reason": failure, "hint": hint}
    if names:
        field["names"] = names
    return field


def connector_entries_by_group(
    queries: List[str],
    connector_search: Optional[Any] = None,
) -> Tuple[List[List[CatalogEntry]], Optional[str]]:
    """No hosted gateway means no hosted hits; local MCP servers are searched elsewhere."""
    return [[] for _ in queries], None


def remote_schemas_for(
    names: List[str],
    current_tool_defs: List[Dict[str, Any]],
    connector_describe: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """No hosted gateway means no remote schemas to merge."""
    return {}, None
