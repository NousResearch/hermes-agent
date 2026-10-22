"""The desktop Connectors page and the guest identity behind hosted connectors.

Contracts, for a user with no Nous identity (the fake account service in
``tests.hermes_cli.anon_portal``, Hermes' real RPC handlers):
- opening the page (list, catalog, accounts, rules) creates no identity and still lists the hosted apps
- the first connect creates exactly one, and arms the connector sign-in offer, not the trial's
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from hermes_cli import free_tier_offer
from tests.hermes_cli.anon_portal import install_portal
from tests.waiting import wait_until
from tools.connectors import live

ACCOUNT = {"owner": {"type": "account"}}


@pytest.fixture
def portal(monkeypatch, tmp_path):
    fake = install_portal(monkeypatch, tmp_path)
    live.reset_for_tests()
    yield fake
    live.reset_for_tests()


def _call(method: str, params: dict | None = None) -> dict:
    response = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    assert "result" in response, response
    return response["result"]


def test_opening_the_page_creates_no_identity_and_lists_the_bundled_catalog(portal, monkeypatch):
    from tools.connectors.portal.catalog_snapshot import CATALOG

    def no_client(**_):
        raise AssertionError("no gateway client may be built before an identity exists")

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", no_client)

    assert _call("connectors.list", ACCOUNT) == {"available": True, "connectors": []}
    catalog = _call("connectors.catalog")["connectors"]
    assert [row["slug"] for row in catalog] == [row["slug"] for row in CATALOG] and catalog
    assert _call("connectors.accounts") == {"accounts": []}
    assert _call("connectors.policy.get")["layers"] == []

    assert portal.creates() == 0 and portal.calls == []


def test_first_connect_creates_one_identity_and_arms_the_connector_offer(portal, monkeypatch):
    class Gateway:
        def __init__(self, **_):
            pass

        def connections(self, names, **_):
            return {"results": [{"connector": name, "status": "active", "connection_id": "c1"} for name in names]}

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Gateway)
    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 1_000_000.0)

    _call("connectors.connect", {**ACCOUNT, "connectors": ["gmail"]})

    assert wait_until(lambda: "connector_nudge_due_in" in _call("free_tier.status"), timeout=5)
    status = _call("free_tier.status")
    assert status["connector_nudge_due_in"] == free_tier_offer.OFFER_DELAY_S
    assert "nudge_due_in" not in status and status["has_guest"] and not status["available"]
    assert portal.creates() == 1
