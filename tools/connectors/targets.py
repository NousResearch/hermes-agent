from __future__ import annotations

import logging
from typing import Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

CONNECTOR_ACTIONS = ("status", "connect", "reconnect")
MCP_ACTIONS = ("install", "enable", "authorize")
ALL_ACTIONS = CONNECTOR_ACTIONS + MCP_ACTIONS

_TARGET_FIELDS = frozenset({"name", "mcp"})


def normalize_targets(raw: Any) -> Tuple[List[str], List[str], Optional[str]]:
    if raw is None:
        return [], [], None
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return [], [], "'connectors' must be a list of names or {name, mcp} objects."
    managed: List[str] = []
    mcp: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            unknown = sorted(set(item) - _TARGET_FIELDS)
            if unknown:
                return [], [], (
                    f"unknown target field(s) {', '.join(unknown)}: a target is "
                    "{\"name\": \"<slug>\"} or {\"name\": \"<server>\", \"mcp\": true}. Transport, "
                    "URLs and credentials come from the catalog manifest, never from the call."
                )
            name = str(item.get("name") or "").strip().lower()
            is_mcp = bool(item.get("mcp", False))
        else:
            name, is_mcp = str(item or "").strip().lower(), False
        if not name:
            return [], [], "every target needs a non-empty 'name'."
        bucket = mcp if is_mcp else managed
        if name not in bucket:
            bucket.append(name)
    return managed, mcp, None


def validate_action(action: str, managed: List[str], mcp: List[str]) -> Optional[str]:
    if action not in ALL_ACTIONS:
        return (
            f"action must be one of {', '.join(ALL_ACTIONS)}. "
            f"{', '.join(MCP_ACTIONS)} apply to local MCP servers "
            "(targets {\"name\": ..., \"mcp\": true}); the rest apply to managed connectors. "
            "Disconnecting an account is done by the user in the Nous Portal dashboard, not "
            "through this tool."
        )
    if action in MCP_ACTIONS:
        if managed:
            return (
                f"'{action}' is an MCP action: every target must carry \"mcp\": true "
                f"(got managed connector(s) {', '.join(managed)}). Managed connectors use "
                "connect / reconnect / status."
            )
        if not mcp:
            return (
                f"'{action}' requires 'connectors': the MCP server name(s), e.g. "
                "[{\"name\": \"linear\", \"mcp\": true}]."
            )
    elif mcp:
        return (
            f"'{action}' is a managed-connector action; MCP targets ({', '.join(mcp)}) use "
            f"{', '.join(MCP_ACTIONS)}."
        )
    return None


def catalog_names() -> Set[str]:
    try:
        from hermes_cli.mcp_catalog import list_catalog

        return {e.name for e in list_catalog()}
    except Exception as exc:
        logger.debug("MCP catalog for the routing check failed: %s", exc)
        return set()


def hosted_names() -> Optional[Set[str]]:
    try:
        from tools.connectors.gateway.client import ConnectorClient
        from tools.connectors.gateway.config import connectors_available

        if not connectors_available():
            return None
        return {str(item.get("connector", "")).lower()
                for item in ConnectorClient().list_connectors() if isinstance(item, dict)}
    except Exception as exc:
        logger.debug("connector list for the routing check failed: %s", exc)
        return None


def misrouted_to_hosted_error(name: str) -> str:
    return (
        f"{name} is a local MCP server, not a hosted connector account. "
        f"Call manage_connections with action install and connectors "
        f"[{{\"name\": \"{name}\", \"mcp\": true}}]."
    )


def misrouted_to_mcp_error(action: str, name: str) -> str:
    return (
        f"{name} is a hosted connector account, not a local MCP server, so '{action}' "
        f"does not apply. Call manage_connections with action connect and connectors [\"{name}\"]."
    )
