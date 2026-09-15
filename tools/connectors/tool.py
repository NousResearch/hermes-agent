#!/usr/bin/env python3
"""Connection lifecycle tool for managed gateway accounts and local MCP servers.

Disconnecting accounts remains a portal-only user decision.
"""

from typing import Any, Callable, Dict, Optional

from tools.connectors.gateway import config as gateway_config
from tools.connectors.managed import run_managed_action
from tools.connectors.mcp import run_mcp_operation
from tools.connectors.targets import ALL_ACTIONS, MCP_ACTIONS, normalize_targets, validate_action
from tools.registry import registry, tool_error


def manage_connections(
    args: Dict[str, Any],
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    mcp_backend: Optional[Any] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    connectors_available: Optional[Callable[[], bool]] = None,
) -> str:
    action = str(args.get("action") or "status").strip().lower()
    managed, mcp_targets, target_error = normalize_targets(args.get("connectors"))
    if target_error:
        return tool_error(target_error)
    action_error = validate_action(action, managed, mcp_targets)
    if action_error:
        return tool_error(action_error)

    if action in MCP_ACTIONS:
        return run_mcp_operation(
            mcp_targets, action, backend=mcp_backend,
            connection_callback=connection_callback, session_id=session_id, tool_call_id=tool_call_id,
        )

    return run_managed_action(
        action, managed, args,
        client_factory=client_factory, session_id=session_id, tool_call_id=tool_call_id,
        connection_callback=connection_callback, connectors_available=connectors_available,
    )


MANAGE_CONNECTIONS_SCHEMA = {
    "name": "manage_connections",
    "description": (
        "Connect the user to apps: managed connector accounts (Gmail, Notion, ...) served "
        "through the tool gateway, and local MCP servers from the catalog. Targets go in "
        "'connectors': a bare slug or {\"name\": \"gmail\"} is a managed connector; "
        "{\"name\": \"linear\", \"mcp\": true} is a local MCP server. "
        "Managed actions: 'status' lists connectors and whether each is connected; 'connect' "
        "starts an authorization for the given connectors; 'reconnect' checks each one and "
        "repairs only what is not connected ('force': true restarts even a working one, for an "
        "account switch). Pass SEVERAL slugs in one call. In the desktop app the call shows the "
        "user a card and blocks until every app is connected, skipped, or the deadline passes; "
        "the result lists each target as connected / skipped / not_connected and never carries "
        "a link. Elsewhere the result carries a connect_url per app for the USER to open in a "
        "browser (never open it yourself); ask them to say when they are done, then use 'status'. "
        "When a connector tool call returns CONNECTION_REQUIRED, use 'connect'. "
        "MCP actions (targets must carry \"mcp\": true): 'install' adds a catalog entry, "
        "'enable' re-enables a disabled configured server, 'authorize' runs its OAuth. "
        "They show the user an approval card and block until it settles. Never hand-edit "
        "mcp_servers config — always use this tool. Never re-ask after a skip or timeout: continue "
        "without the app or ask in chat. A newly installed or authorized server's tools arrive on "
        "your next turn. Off the desktop app an MCP target runs at once and the result says what "
        "happened, with a link for the user to open when one is needed. This tool can NOT "
        "disconnect, delete, or revoke an account — that is deliberately user-only. When asked, say so and direct the user to the "
        "Nous Portal (their org's Connectors page) or the desktop app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ALL_ACTIONS),
                "description": "Defaults to status. install/enable/authorize need mcp:true targets.",
            },
            "connectors": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mcp": {"type": "boolean", "description": "true = local MCP server."},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "description": (
                    "Targets. REQUIRED for every action but status "
                    "(e.g. [\"gmail\", {\"name\": \"linear\", \"mcp\": true}]); optional filter for status."
                ),
            },
            "force": {
                "type": "boolean",
                "description": "reconnect only: restart the authorization even if the app is connected (account switch).",
            },
        },
        "required": [],
    },
}


registry.register(
    name="manage_connections",
    toolset="connections",
    schema=MANAGE_CONNECTIONS_SCHEMA,
    # The portal gate decides schema presence: an account the portal has not enabled for
    # connectors never sees the tool, so the model cannot call it and read the gateway's
    # 404 back to them. The handler runs the same gate so the RPC path (methods_connectors) and
    # a cached schema agree. Read as a module attribute so tests patch
    # ``gateway.config.connectors_available`` at one seam.
    handler=lambda args, **kw: manage_connections(
        args, session_id=kw.get("session_id"), connectors_available=gateway_config.connectors_available,
    ),
    check_fn=lambda: gateway_config.connectors_available(),
    emoji="🔗",
)
