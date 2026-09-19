"""Account-level connector RPC behavior through the JSON-RPC dispatch rail."""

from __future__ import annotations

import pytest

from tests.tui_gateway.conftest import ReplyTransport, reply


@pytest.mark.parametrize("name", [
    "connectors.tools",
    "connectors.catalog",
    "connectors.accounts",
    "connectors.accounts.remove",
    "connectors.policy.get",
    "connectors.policy.set",
])
def test_every_account_connector_rpc_runs_off_the_server_loop(name):
    from tui_gateway import server

    assert name in server._LONG_HANDLERS


def test_policy_set_surfaces_conflict_and_preserves_the_seen_revision(monkeypatch):
    from tools.connectors.gateway.errors import ToolGatewayError
    from tools.connectors.portal.wire import ConnectorPolicyResponse

    class Client:
        sent = None

        def require_authentication(self):
            return None

        def policy(self):
            return ConnectorPolicyResponse.model_validate({"layers": []})

        def set_policy(self, body):
            type(self).sent = body
            raise ToolGatewayError("hidden", status=409)

    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient", Client)

    answer = reply(ReplyTransport(), "connectors.policy.set", {
        "change": {"type": "tools", "connector": "linear", "disabled_tools": ["create"]},
        "expected_revision": "shown-revision",
    })

    assert Client.sent == {
        "scope": "member",
        "disabledConnectors": [],
        "tools": {"linear": ["create"]},
        "expectedRevision": "shown-revision",
    }
    assert answer["error"] == {
        "code": 4090,
        "message": "Connector policy changed. Refresh and try again.",
        "data": {"reason": "POLICY_CONFLICT"},
    }


def test_account_remove_maps_not_found_and_returns_the_removed_account(monkeypatch):
    from tools.connectors.gateway.errors import GatewayUnavailable

    class Client:
        outcome = None

        def delete_account(self, connection_id):
            outcome = type(self).outcome
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Client)

    Client.outcome = GatewayUnavailable("hidden", code="connection_not_found", status=404)
    missing = reply(ReplyTransport(), "connectors.accounts.remove", {"connection_id": "ca_missing"})
    Client.outcome = {"connectionId": "ca_1", "status": "removed"}
    removed = reply(ReplyTransport(), "connectors.accounts.remove", {"connection_id": "ca_1"})

    assert missing["error"] == {
        "code": 4041,
        "message": "Connector account not found.",
        "data": {"reason": "CONNECTION_NOT_FOUND"},
    }
    assert removed["result"] == {"connection_id": "ca_1", "status": "removed"}
