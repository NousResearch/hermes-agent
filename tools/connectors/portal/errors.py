"""Errors specific to connector tool listings from the portal."""

from __future__ import annotations

from tools.connectors.gateway.errors import ToolGatewayError


class PortalConnectorUnavailable(ToolGatewayError):
    """The portal cannot provide connector metadata right now."""


class PortalToolsUnavailable(PortalConnectorUnavailable):
    """The portal cannot provide a connector's tool list right now."""
