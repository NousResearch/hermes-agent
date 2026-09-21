"""Connector integration boundary for managed gateway accounts and local MCP servers.

Only the names below are cross-package surface; imports beyond it need a design decision.
Siblings: ``contract`` (states, actors, transition table), ``operation`` (the record),
``live`` (open operation per session), ``run`` (the lifecycle loop), ``managed`` / ``mcp``
(per-kind hooks), ``targets``, ``search``, ``dispatch``, ``gateway/`` (HTTP wire + client).
"""

from tools.connectors.dispatch import dispatch_connector_batch, dispatch_connector_call
from tools.connectors.gateway.bridge import connector_describe, connector_search_hits
from tools.connectors.gateway.config import connectors_available
from tools.connectors.gateway.names import CONNECTOR_BATCH_SENTINEL, is_connector_name
from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA, manage_connections

__all__ = [
    "CONNECTOR_BATCH_SENTINEL",
    "MANAGE_CONNECTIONS_SCHEMA",
    "connector_describe",
    "connector_search_hits",
    "connectors_available",
    "dispatch_connector_batch",
    "dispatch_connector_call",
    "is_connector_name",
    "manage_connections",
]
