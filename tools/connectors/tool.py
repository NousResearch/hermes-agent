#!/usr/bin/env python3
"""Connection lifecycle tool for managed gateway accounts and local MCP servers.

Disconnecting accounts remains a portal-only user decision.
"""

from typing import Any, Callable, Dict, Optional

from tools.connectors.catalog_tool import MANAGE_CATALOG_SCHEMA, manage_catalog
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
        "Connect the user to apps. Two kinds: a hosted connector account (Gmail, Notion, ...) "
        "served through the tool gateway, and a local MCP server from the bundled catalog. "
        "Targets go in 'connectors': a bare slug or {\"name\": \"gmail\"} is a hosted connector "
        "account; {\"name\": \"linear\", \"mcp\": true} is a local MCP server. Many names exist on "
        "both sides, so the user's own words decide. Pass \"mcp\": true only when the user asks "
        "for an MCP server, a local server or an install, or when the name exists only as a "
        "catalog entry; otherwise the target is hosted. 'connect' and 'reconnect' are hosted "
        "actions; 'install', 'enable' and 'authorize' are MCP actions. A target the other side "
        "owns is refused with the call that does work. "
        "Hosted actions: 'status' lists connectors and whether each is connected; 'connect' "
        "starts an authorization for the given connectors; 'reconnect' checks each one and "
        "repairs only what is not connected ('force': true restarts even a working one, for an "
        "account switch). Pass SEVERAL slugs in one call. In the desktop app, the terminal UI and "
        "the interactive CLI the call shows the user a card and blocks until every app is "
        "connected, skipped, or the deadline passes; the result lists each target as connected / "
        "skipped / not_connected and never carries a link. Where no card exists (a one-shot run, "
        "a scheduled job, a messaging platform) the result carries a connect_url per app for the "
        "USER to open in a browser (never open it yourself); ask them to say when they are done, "
        "then use 'status'. "
        "When a hosted connector tool call returns CONNECTION_REQUIRED, use 'connect'. "
        "MCP actions (every target must carry \"mcp\": true): 'install' adds a catalog entry, "
        "'enable' re-enables a disabled configured server, 'authorize' runs its OAuth. "
        "They show the user an approval card and block until it settles. Never hand-edit "
        "mcp_servers config — always use this tool. After a skip or a timeout, do not re-ask on "
        "your own: continue without the app or ask in chat. A later request from the USER for that "
        "same app is not a re-ask — run it. A connected server's tools are named in the result and are "
        "callable at once through tool_describe/tool_call. Where no card exists an MCP target runs at once and the result says what "
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
                "description": (
                    "Defaults to status. connect and reconnect take hosted connector slugs only. "
                    "install, enable and authorize take mcp:true targets only."
                ),
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
                                "mcp": {
                                    "type": "boolean",
                                    "description": (
                                        "true = a local MCP server from the catalog; absent or "
                                        "false = a hosted connector account."
                                    ),
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "description": (
                    "Targets. REQUIRED for every action but status "
                    "(e.g. [\"gmail\", {\"name\": \"linear\", \"mcp\": true}]); optional filter for "
                    "status. A bare slug is a hosted connector, so an MCP server needs the object "
                    "form with \"mcp\": true."
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

# The setup profile's catalog install. Reachable only through the ``setup`` toolset, which the
# profile's role grants; registry dispatch has no card callback, so it answers with the CLI pointer.
registry.register(
    name="manage_catalog",
    toolset="setup",
    schema=MANAGE_CATALOG_SCHEMA,
    handler=lambda args, **kw: manage_catalog(args, session_id=kw.get("session_id")),
    emoji="🧩",
)
