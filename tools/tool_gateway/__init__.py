"""Connector tool-gateway package: typed client-side plumbing for remote tools.

This package owns everything hermes-agent needs to talk to the managed tool
gateway's connector routes (search / schemas / execute / connections) and to
merge remote execute results back into ``tool_call`` result arrays.

Layering rules (enforced by review, not imports — keep them true):

- ``wire.py`` is a leaf: pydantic v2 models for the four gateway routes plus
  route path constants. No pydantic model may escape the wire/client layer;
  everything above sees plain dicts and frozen dataclasses.
- ``errors.py`` is stdlib-only: the ``ToolGatewayError`` family, the ONE
  gateway error-envelope parser, and the single producer of the
  connection-required payload shown to the model.
- ``names.py`` is stdlib-only: the ``connectors__<connector>__<tool>`` name
  codec. Parsing never raises — a malformed name is a per-entry error and
  sibling calls still run.
- ``config.py``: the ``tools.connectors`` config gate. Availability fails
  closed: config flag AND the managed Nous tools entitlement.
- ``merge.py`` is PURE: partition / splice / render with no I/O and no
  exceptions. Position in the original ``calls[]`` array is the only
  correlation key — the wire ``index`` field is never trusted.
- ``client.py`` / ``bridge.py``: the HTTP client and the only module core
  imports. Every bridge entry point is TOTAL — it catches its own
  exceptions, because the bridge branch bypasses the registry's catch-wrap.
  Approval is settled by the core BEFORE the bridge is called; denied
  entries never reach it.

Core reaches this package through ``model_tools_connectors.py``, which
dispatches one gateway request per connector entry via ``bridge.run_remote``
and re-enters core dispatch for each entry so per-tool policy fires against
the composed ``connectors__`` name.
"""

from tools.tool_gateway.config import (
    MAX_CALLS_PER_DISPATCH,
    ConnectorConfig,
    connectors_available,
)
from tools.tool_gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    IdempotencyConflict,
    ToolGatewayError,
    parse_gateway_error,
    render_connection_required,
)
from tools.tool_gateway.names import (
    CONNECTOR_BATCH_SENTINEL,
    CONNECTOR_NAME_PREFIX,
    ConnectorName,
    format_connector_name,
    is_connector_name,
    parse_connector_name,
)

__all__ = [
    "CONNECTOR_BATCH_SENTINEL",
    "CONNECTOR_NAME_PREFIX",
    "ConnectorConfig",
    "ConnectorName",
    "GatewayAuthError",
    "GatewayUnavailable",
    "IdempotencyConflict",
    "MAX_CALLS_PER_DISPATCH",
    "ToolGatewayError",
    "connectors_available",
    "format_connector_name",
    "is_connector_name",
    "parse_connector_name",
    "parse_gateway_error",
    "render_connection_required",
]
