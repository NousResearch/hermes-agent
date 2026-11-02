"""The desktop Connectors page, the CLI and the agent tool, and the guest identity behind hosted connectors.

Contracts, for a user with no Nous identity (the fake account service in
``tests.hermes_cli.anon_portal``, Hermes' real RPC handlers and CLI):
- opening the page (list, catalog, accounts) creates no identity; the catalog comes from the Portal's
  public route, asked once and without a bearer, and rules are refused rather than invented
- ``hermes connectors list`` and the agent's ``manage_connections`` status each create exactly one
  identity and make the gateway list call, as before
- the first connect creates one identity, arms the connector sign-in offer, and the page then reads
  the signed-in routes; a connect by link that a later status read sees connected arms it too, and
  so does a page list that shows a guest's app connected on an older Hermes
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

import tui_gateway.server as srv
from hermes_cli import free_tier_offer
from tests.hermes_cli.anon_portal import install_portal
from tests.waiting import wait_until
from tools.connectors import live

ACCOUNT = {"owner": {"type": "account"}}
CATALOG = {"connectors": [{"slug": "notion", "name": "Notion", "description": "Docs", "category": "Productivity",
                           "logoUrl": None, "state": "live"}]}


class _Response:
    def __init__(self, body):
        self.status_code, self._body, self.headers = 200, body, {}

    def json(self):
        return self._body


@pytest.fixture
def portal(monkeypatch, tmp_path):
    fake = install_portal(monkeypatch, tmp_path)
    live.reset_for_tests()
    yield fake
    live.reset_for_tests()


@pytest.fixture
def connector_portal(monkeypatch):
    """The Portal's connector routes: every request as ``(path, sent a bearer)``."""
    requests = []

    class Transport:
        def request(self, method, url, *, headers=None, json=None, timeout=None):
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            requests.append((path, "Authorization" in (headers or {})))
            return _Response({"accounts": []} if path.endswith("/accounts") else CATALOG)

    monkeypatch.setattr("tools.connectors.portal.client._default_transport", Transport)
    return requests


class _Lists(list):
    connected = False


@pytest.fixture
def gateway(monkeypatch):
    """The connector gateway: every list call, and an instant connect. ``connected`` is what it lists."""
    lists = _Lists()

    class Gateway:
        def __init__(self, **_):
            pass

        def list_connectors(self, *, timeout=None):
            lists.append(1)
            return [{"connector": "notion", "enabled": True, "connected": lists.connected}]

        def connections(self, names, **_):
            return {"results": [{"connector": name, "status": "active", "connection_id": "c1"} for name in names]}

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Gateway)
    return lists


def _call(method: str, params: dict | None = None) -> dict:
    return srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


def _result(method: str, params: dict | None = None) -> dict:
    response = _call(method, params)
    assert "result" in response, response
    return response["result"]


def test_opening_the_page_creates_no_identity_and_reads_the_public_catalog(portal, connector_portal, gateway):
    assert _result("connectors.list", ACCOUNT) == {"available": True, "connectors": []}
    assert [row["slug"] for row in _result("connectors.catalog")["connectors"]] == ["notion"]
    assert _result("connectors.accounts") == {"accounts": []}
    assert _call("connectors.policy.get")["error"]["code"] == 4032

    assert connector_portal == [("/api/v1/connectors/catalog/public", False)]
    assert portal.creates() == 0 and gateway == []


@pytest.mark.parametrize("entry", ["cli", "agent"])
def test_explicit_lists_still_create_one_identity_and_list_from_the_gateway(portal, gateway, entry):
    from hermes_cli.connectors_cmd import cmd_connectors
    from tools.connectors.gateway.config import connectors_available
    from tools.connectors.tool import manage_connections

    if entry == "cli":
        assert cmd_connectors(Namespace(connectors_action="list")) == 0
    else:
        result = json.loads(manage_connections({"action": "status"}, connectors_available=connectors_available))
        assert result["connectors"][0]["connector"] == "notion"

    assert portal.creates() == 1 and gateway == [1]


def test_first_connect_creates_one_identity_then_the_page_reads_signed_in(
        portal, connector_portal, gateway, monkeypatch):
    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 1_000_000.0)

    _result("connectors.connect", {**ACCOUNT, "connectors": ["notion"]})

    assert wait_until(lambda: "connector_nudge_due_in" in _result("free_tier.status"), timeout=5)
    status = _result("free_tier.status")
    assert status["connector_nudge_due_in"] == free_tier_offer.OFFER_DELAY_S
    assert "nudge_due_in" not in status and status["has_guest"] and not status["available"]

    assert _result("connectors.list", ACCOUNT)["connectors"][0]["connector"] == "notion"
    _result("connectors.catalog")
    assert connector_portal == [("/api/v1/connectors/catalog", True)]
    assert portal.creates() == 1


def test_a_connect_by_link_seen_by_a_status_read_arms_the_connector_offer(portal, gateway, monkeypatch):
    """Messaging and one-shot runs connect by link and check with status; no operation watcher runs."""
    from tools.connectors.gateway.config import connectors_available
    from tools.connectors.tool import manage_connections

    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 1_000_000.0)
    gateway.connected = True

    for _ in range(2):
        manage_connections({"action": "status"}, connectors_available=connectors_available)

    assert free_tier_offer.offer_due_in(free_tier_offer.CONNECTOR) == free_tier_offer.OFFER_DELAY_S
    assert portal.creates() == 1


def test_the_page_list_arms_the_offer_for_apps_a_guest_connected_on_an_older_hermes(portal, gateway, monkeypatch):
    """The guest exists without the connected-app flag (an older Hermes connected its apps); the
    first page read that lists one connected must still arm the connector offer."""
    from tools.connectors import ensure_guest_identity

    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 1_000_000.0)
    assert ensure_guest_identity() is None
    gateway.connected = True

    assert _result("connectors.list", ACCOUNT)["connectors"][0]["connected"] is True

    assert free_tier_offer.offer_due_in(free_tier_offer.CONNECTOR) == free_tier_offer.OFFER_DELAY_S
    assert portal.creates() == 1


def test_a_failed_offer_write_is_retried_on_the_next_sighting(portal, gateway, monkeypatch):
    """The flag that ends the sightings is written only once the offer's schedule is, so a failed
    schedule write leaves the next list read to try again."""
    from tools.connectors import ensure_guest_identity

    monkeypatch.setattr(free_tier_offer, "_clock", lambda: 1_000_000.0)
    assert ensure_guest_identity() is None
    gateway.connected = True
    real_record = free_tier_offer.record_task_done

    def fail_once(kind=free_tier_offer.TRIAL):
        monkeypatch.setattr(free_tier_offer, "record_task_done", real_record)
        raise TimeoutError("offer record locked")

    monkeypatch.setattr(free_tier_offer, "record_task_done", fail_once)
    _result("connectors.list", ACCOUNT)
    assert free_tier_offer.offer_due_in(free_tier_offer.CONNECTOR) is None

    _result("connectors.list", ACCOUNT)
    assert free_tier_offer.offer_due_in(free_tier_offer.CONNECTOR) == free_tier_offer.OFFER_DELAY_S
