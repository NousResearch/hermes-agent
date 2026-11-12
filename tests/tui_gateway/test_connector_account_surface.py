"""Account-owned connector RPCs come only from the desktop Connectors page, so their gateway requests
name the ``desktop`` surface even on a backend started without ``HERMES_DESKTOP`` (a URL, SSH-less or
cloud backend the app attached to)."""

import pytest

from hermes_agent.tools.connectors import live
from hermes_agent.tools.managed_gateway_identity import user_agent
from hermes_agent.tui_gateway import server


@pytest.fixture
def remote_backend(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr("hermes_agent.tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("hermes_agent.tools.connectors.ensure_guest_identity", lambda: None)
    monkeypatch.setattr("hermes_agent.tools.connectors.managed.ensure_guest_identity", lambda: None)
    live.reset_for_tests()
    yield
    live.reset_for_tests()


def _dispatch(method, params):
    return server.handle_request({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def test_account_reads_and_connects_name_the_desktop(remote_backend, monkeypatch):
    from hermes_agent.tools.connectors.gateway.errors import GatewayUnavailable

    agents = []

    def read_tools(slug, *, client, refresh):
        agents.append(user_agent())
        raise GatewayUnavailable("gone", code="connector_not_found")

    class Client:
        def __init__(self, **_):
            pass

        def connections(self, names, **_):
            agents.append(user_agent())
            return {"results": [{"connector": n, "status": "active"} for n in names]}

    monkeypatch.setattr("hermes_agent.tools.connectors.portal.client.PortalConnectorClient.require_authentication", lambda self: None)
    monkeypatch.setattr("hermes_agent.tools.connectors.portal.tools_cache.read_tools", read_tools)
    monkeypatch.setattr("hermes_agent.tools.connectors.gateway.client.ConnectorClient", Client)

    assert _dispatch("connectors.tools", {"slug": "gmail", "refresh": False})["error"]["code"] == 4041
    connected = _dispatch("connectors.connect", {"owner": {"type": "account"}, "connectors": ["gmail"]})
    assert "result" in connected, connected
    assert user_agent().split(" (")[1].startswith("cli;")  # nothing leaks out of the request
    assert [agent.split(" (")[1].split(";")[0] for agent in agents] == ["desktop", "desktop"]
