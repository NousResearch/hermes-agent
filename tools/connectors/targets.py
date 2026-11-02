from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

CONNECTOR_ACTIONS = ("display", "status", "connect", "reconnect", "rename")
MCP_ACTIONS = ("enable", "authorize")
ALL_ACTIONS = CONNECTOR_ACTIONS + MCP_ACTIONS

_TARGET_FIELDS = frozenset({"name", "mcp", "alias", "to"})

ALIAS_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"
_ALIAS_RE = re.compile(ALIAS_PATTERN)
_ALIAS_FORMAT = "1-32 lowercase letters, digits or '-', starting with a letter or digit"


@dataclass(frozen=True)
class HostedTarget:
    """A hosted connector target. ``alias`` names one account of that connector; ``to`` is the
    new alias of a rename."""

    name: str
    alias: Optional[str] = None
    to: Optional[str] = None


def _bad_alias(key: str, value: Optional[str]) -> Optional[str]:
    if value is None or _ALIAS_RE.fullmatch(value):
        return None
    return f"'{key}' {value!r} is not a valid account name: use {_ALIAS_FORMAT}."


def normalize_targets(raw: Any) -> tuple[list[HostedTarget], list[str], Optional[str]]:
    if raw is None:
        return [], [], None
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return [], [], "'connectors' must be a list of names or {name, mcp} objects."
    managed: list[HostedTarget] = []
    mcp: list[str] = []
    for item in raw:
        alias = to = None
        if isinstance(item, dict):
            unknown = sorted(set(item) - _TARGET_FIELDS)
            if unknown:
                return [], [], (
                    f"unknown target field(s) {', '.join(unknown)}: a target is "
                    "{\"name\": \"<slug>\"} or {\"name\": \"<server>\", \"mcp\": true}. Transport, "
                    "URLs and credentials come from the server's mcp_servers entry, never from the call."
                )
            name = str(item.get("name") or "").strip().lower()
            is_mcp = bool(item.get("mcp", False))
            if is_mcp and ("alias" in item or "to" in item):
                return [], [], "'alias' and 'to' name hosted connector accounts; an MCP target takes neither."
            alias = str(item["alias"]).strip() if item.get("alias") is not None else None
            to = str(item["to"]).strip() if item.get("to") is not None else None
        else:
            name, is_mcp = str(item or "").strip().lower(), False
        if not name:
            return [], [], "every target needs a non-empty 'name'."
        if is_mcp:
            if name not in mcp:
                mcp.append(name)
            continue
        target = HostedTarget(name, alias, to)
        if all((t.name, t.alias) != (name, alias) for t in managed):
            managed.append(target)
    return managed, mcp, None


# A session whose cached schema predates the MCP catalog's removal can still send install.
INSTALL_RETIRED = (
    "'install' is no longer an action: there is no bundled MCP catalog. A server already in mcp_servers "
    "uses enable or authorize (targets {\"name\": ..., \"mcp\": true}); a new one is added by the user "
    "with `hermes mcp add <name> --url <url>` (or --command for a local server)."
)


def validate_action(action: str, managed: list[HostedTarget], mcp: list[str]) -> Optional[str]:
    if action == "install":
        return INSTALL_RETIRED
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
                f"(got managed connector(s) {', '.join(t.name for t in managed)}). Managed connectors use "
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
    return _validate_accounts(action, managed)


def _validate_accounts(action: str, managed: list[HostedTarget]) -> Optional[str]:
    if action != "rename" and any(t.to for t in managed):
        return "'to' is the new name of an account and applies to action rename only."
    if action == "display" and any(t.alias for t in managed):
        return "display takes connector slugs to suggest (e.g. [\"gmail\"]); accounts are named on connect."
    if action == "rename":
        if len(managed) != 1 or not managed[0].alias or not managed[0].to:
            return ("rename takes exactly one target {\"name\": \"<slug>\", \"alias\": \"<current name>\", "
                    "\"to\": \"<new name>\"}. Use action status to list each connector's accounts.")
        # The current name may be the vendor's label of an unnamed account, so only the new one is checked.
        return _bad_alias("to", managed[0].to)
    if any(t.alias for t in managed) and len(managed) > 1:
        return "a call with an 'alias' takes that one target only; name other accounts in separate calls."
    if action == "reconnect":
        # Like rename, a repair may address an unnamed account by its label; the account read checks it.
        return None
    return next((e for t in managed if (e := _bad_alias("alias", t.alias))), None)


def hosted_names() -> Optional[set[str]]:
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


def misrouted_to_mcp_error(action: str, name: str) -> str:
    return (
        f"{name} is a hosted connector account, not a local MCP server, so '{action}' "
        f"does not apply. Call manage_connections with action connect and connectors [\"{name}\"]."
    )
