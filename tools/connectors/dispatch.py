"""Hosted connector dispatch is not available in this build.

Hosted connector accounts were served through the managed tool gateway, which has
been removed. Local MCP servers are called through the normal tool pipeline
(``tool_describe`` / ``tool_call``), never through ``connectors__`` names, so any
call reaching here can only reference a hosted tool that no longer exists.
"""

import json

from tools.connectors.gateway.config import MAX_CALLS_PER_DISPATCH  # noqa: F401  (re-exported surface)

_UNAVAILABLE = {
    "error": {
        "code": "CONNECTORS_UNAVAILABLE",
        "message": (
            "Hosted connector tools are not available in this build. "
            "Install a local MCP server instead (manage_connections, action 'install')."
        ),
    }
}


def dispatch_connector_call(name, arguments, tool_call_id):
    return json.dumps(_UNAVAILABLE, ensure_ascii=False)


def dispatch_connector_batch(calls, ids, *, user_task, enabled_tools,
                             middleware_trace, enabled_toolsets, disabled_toolsets):
    return json.dumps(_UNAVAILABLE, ensure_ascii=False)
