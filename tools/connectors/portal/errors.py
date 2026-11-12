from __future__ import annotations

from tools.connectors.gateway.errors import ToolGatewayError


class InvalidConnectorSlug(ValueError):
    pass


class PortalConnectorUnavailable(ToolGatewayError):
    pass


class PortalToolsUnavailable(PortalConnectorUnavailable):
    pass
