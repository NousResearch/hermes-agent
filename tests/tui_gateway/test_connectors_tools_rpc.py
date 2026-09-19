"""Connectors tool-list RPC behavior through the JSON-RPC dispatch rail."""

from __future__ import annotations

import pytest

from tests.tui_gateway.conftest import ReplyTransport, reply
from tools.connectors.gateway.errors import GatewayAuthError, GatewayUnavailable
from tools.connectors.portal.tools_cache import ToolsRead
from tools.connectors.portal.wire import ConnectorToolsListing


_DESCRIPTION = "D" * 200


def _listing():
    return ConnectorToolsListing.model_validate(
        {
            "connector": "mail-service",
            "toolkitVersion": "2026.1",
            "etag": '"v1"',
            "tools": [
                {
                    "slug": "MAIL_SERVICE_SEND",
                    "name": "Send message",
                    "description": _DESCRIPTION,
                    "facet": "write",
                    "hints": ["createHint"],
                    "categories": ["messaging"],
                    "noAuth": False,
                    "deprecated": False,
                }
            ],
        }
    )


def _tools_reply(slug="mail-service"):
    return reply(ReplyTransport(), "connectors.tools", {"slug": slug})


def test_connector_tools_rpc_returns_the_typed_result(monkeypatch):
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient.require_authentication", lambda self: None)
    listing = _listing()
    result = ToolsRead(
        connector="mail-service",
        toolkit_version="2026.1",
        etag='"v1"',
        fetched_at=123.0,
        source="network",
        stale=False,
        tools=listing.tools,
    )
    monkeypatch.setattr("tools.connectors.portal.tools_cache.read_tools", lambda *args, **kwargs: result)

    answer = _tools_reply()

    from tui_gateway import server

    assert "connectors.tools" in server._LONG_HANDLERS
    assert answer["result"] == {
        "connector": "mail-service",
        "toolkit_version": "2026.1",
        "etag": '"v1"',
        "fetched_at": 123.0,
        "source": "network",
        "stale": False,
        "tools": [
            {
                "slug": "MAIL_SERVICE_SEND",
                "name": "Send message",
                "description": _DESCRIPTION,
                "facet": "write",
                "hints": ["createHint"],
                "categories": ["messaging"],
                "deprecated": False,
            }
        ],
    }


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            OSError("cache write detail"),
            {"code": 5034, "message": "Connector tools are unavailable.", "data": {"reason": "TOOLS_UNAVAILABLE"}},
        ),
        (
            GatewayAuthError("upstream detail"),
            {"code": 4032, "message": "Sign in to use connectors.", "data": {"reason": "NEEDS_NOUS_AUTH"}},
        ),
        (
            GatewayUnavailable("upstream detail", code="CONNECTOR_NOT_FOUND", status=404),
            {"code": 4041, "message": "Connector not found.", "data": {"reason": "CONNECTOR_NOT_FOUND"}},
        ),
    ],
)
def test_connector_tools_rpc_answers_a_failure_with_a_typed_error_and_no_detail(monkeypatch, raised, expected):
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient.require_authentication", lambda self: None)

    def read_tools(*args, **kwargs):
        raise raised

    monkeypatch.setattr("tools.connectors.portal.tools_cache.read_tools", read_tools)

    assert _tools_reply()["error"] == expected
