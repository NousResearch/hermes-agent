#!/usr/bin/env python3
"""Connection lifecycle tool for local MCP servers.

This build has no hosted connector accounts: the managed tool gateway was removed.
Hosted actions ('status', 'connect', 'reconnect') still parse so older frontends and
cached schemas fail with a clear answer instead of an import error.
"""

import json
from typing import Any, Callable, Dict, Optional

from tools.connectors.catalog_tool import MANAGE_CATALOG_SCHEMA, manage_catalog
from tools.connectors.gateway import config as gateway_config
from tools.connectors.mcp import run_mcp_operation
from tools.connectors.targets import ALL_ACTIONS, MCP_ACTIONS, normalize_targets, validate_action
from tools.registry import registry, tool_error

_HOSTED_REMOVED = (
    "Hosted connector accounts are not available in this build. "
    "Use a local MCP server instead: action 'install' with "
    'connectors [{"name": "<server>", "mcp": true}].'
)


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

    if action == "status":
        # No hosted connectors exist in this build; report an empty list so the
        # connectors panel renders "nothing connected" rather than an error.
        return json.dumps({"connectors": []}, ensure_ascii=False)

    return tool_error(_HOSTED_REMOVED)


MANAGE_CONNECTIONS_SCHEMA = {
    "name": "manage_connections",
    "description": (
        "Install, enable and authorize local MCP servers from the bundled catalog. "
        "Targets go in 'connectors' as objects carrying \"mcp\": true, e.g. "
        '{"name": "linear", "mcp": true}. '
        "MCP actions (every target must carry \"mcp\": true): 'install' adds a catalog entry, "
        "'enable' re-enables a disabled configured server, 'authorize' runs its OAuth. "
        "They show the user an approval card and block until it settles. Never hand-edit "
        "mcp_servers config — always use this tool. After a skip or a timeout, do not re-ask on "
        "your own: continue without the app or ask in chat. A later request from the USER for that "
        "same app is not a re-ask — run it. A connected server's tools are named in the result and are "
        "callable at once through tool_describe/tool_call. Where no card exists an MCP target runs at once and the result says what "
        "happened, with a link for the user to open when one is needed. This tool can NOT "
        "disconnect, delete, or revoke a server — that is deliberately user-only. When asked, say so and direct the user to the "
        "desktop app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ALL_ACTIONS),
                "description": (
                    "install, enable and authorize take mcp:true targets only. "
                    "status lists hosted connectors (always empty in this build); "
                    "connect and reconnect are hosted-only and unavailable."
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
                                        "true = a local MCP server from the catalog. "
                                        "Hosted connector accounts are not available in this build."
                                    ),
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "description": (
                    "Targets. REQUIRED for install, enable and authorize "
                    "(e.g. [{\"name\": \"linear\", \"mcp\": true}]); optional filter for status."
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
