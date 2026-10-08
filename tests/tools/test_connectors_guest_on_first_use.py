"""A guest identity for connectors is created on the first hosted connector action.

Contracts, for a user with no Nous identity on a surface without the free-tier launch gate (plain
CLI, TUI, messaging gateway):
- ``manage_connections`` is offered, and its first hosted action creates exactly one guest whose
  token is connectors-only; later actions create none, and the free model is never selected
- tool_search sends nothing to the portal or gateway and points the model at ``manage_connections``
- ``nous.guest: false`` keeps the old answer: no tool, no account
Driven through the fake NAS (``anon_portal``) with Hermes' real client code.
"""

import json

import pytest

from hermes_cli import anon_auth
from tests.hermes_cli.anon_portal import install_portal


@pytest.fixture
def nas(monkeypatch, tmp_path):
    fake = install_portal(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_GUEST_ONBOARDING")
    return fake


def test_first_hosted_action_creates_one_connectors_only_guest(nas, monkeypatch):
    from tools.connectors.gateway import client
    from tools.connectors.gateway.config import connectors_available
    from tools.connectors.tool import manage_connections
    from tools.managed_tool_gateway import read_nous_access_token

    bearers = []

    class Gateway:
        """Reads the bearer the way the real gateway client does on every request."""

        def list_connectors(self, *, timeout=None):
            bearers.append(read_nous_access_token())
            return [{"connector": "gmail", "enabled": True, "connected": False}]

    monkeypatch.setattr(client, "ConnectorClient", Gateway)
    assert connectors_available()
    assert nas.creates() == 0

    for _ in range(2):
        result = json.loads(manage_connections({"action": "status"}, connectors_available=connectors_available))
        assert result["connectors"][0]["connector"] == "gmail"

    assert nas.creates() == 1
    assert [r["body"].get("purpose") for r in nas.token_requests] == ["connectors"]
    assert bearers and all(anon_auth.is_connectors_only(b) for b in bearers)

    # The free model and its surfaces stay off; the identity survives for connectors.
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.auth_nous import get_nous_auth_status_local, resolve_nous_runtime_credentials

    with pytest.raises(AuthError) as refused:
        resolve_nous_runtime_credentials()
    assert refused.value.code == "nous_auth_missing" and not refused.value.relogin_required
    assert not get_nous_auth_status_local().get("logged_in")
    assert not anon_auth.guest_notice_pending() and not anon_auth.has_free_tier_account()
    assert anon_auth.has_guest()
    assert [r["body"].get("purpose") for r in nas.token_requests] == ["connectors"]


def test_tool_search_without_an_identity_sends_nothing(nas, monkeypatch):
    from tools.connectors.tool import MANAGE_CONNECTIONS_SCHEMA
    from tools.tool_search import dispatch_tool_search

    def no_client():
        raise AssertionError("no gateway client may be built before an identity exists")

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", no_client)
    tool_defs = [{"type": "function", "function": MANAGE_CONNECTIONS_SCHEMA}]

    result = json.loads(dispatch_tool_search({"queries": ["gmail send email"]}, current_tool_defs=tool_defs))

    assert result["connectors"]["status"] == "not_set_up"
    assert "manage_connections" in result["connectors"]["hint"]
    assert nas.calls == []


def test_guest_opt_out_keeps_connectors_closed(nas):
    from hermes_constants import get_hermes_home
    from tools.connectors.gateway.config import connectors_available, ensure_guest_identity

    (get_hermes_home() / "config.yaml").write_text("nous:\n  guest: false\n", encoding="utf-8")

    assert not connectors_available()
    assert ensure_guest_identity() is None
    assert nas.calls == []
