"""Portal-backed connector tool-list cache behavior."""

from __future__ import annotations

import pytest

from tests.fakes.connectors_http import FakeResponse, FakeTransport
from tools.connectors.gateway.errors import GatewayAuthError, GatewayUnavailable
from tools.connectors.portal.client import DEFAULT_TIMEOUT_SECONDS, NotModified, PortalConnectorClient
from tools.connectors.portal.errors import PortalToolsUnavailable
from tools.connectors.portal.tools_cache import TTL_SECONDS, read_tools
from tools.connectors.portal.wire import ConnectorToolsListing


def _listing(*, etag: str = '"v1"') -> ConnectorToolsListing:
    return ConnectorToolsListing.model_validate(
        {
            "connector": "mail-service",
            "toolkitVersion": "2026.1",
            "etag": etag,
            "tools": [
                {
                    "slug": "MAIL_SERVICE_SEND",
                    "name": "Send message",
                    "description": "D",
                    "facet": "write",
                    "hints": ["createHint"],
                    "categories": ["MESSAGING"],
                    "noAuth": False,
                    "deprecated": False,
                }
            ],
        }
    )


class _CacheClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def origin(self):
        return "https://portal.example.test"

    def tools(self, slug, *, if_none_match=None):
        self.calls.append({"slug": slug, "if_none_match": if_none_match})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_portal_client_parses_a_listing_and_sends_one_bounded_authorized_request():
    transport = FakeTransport(
        FakeResponse(
            200,
            {
                **_listing().model_dump(by_alias=True),
                "tools": [{
                    **_listing().tools[0].model_dump(by_alias=True),
                    "facet": "future-facet",
                }],
            },
            headers={"etag": '"header-v1"'},
        )
    )
    portal_client = PortalConnectorClient(
        transport=transport,
        endpoint_resolver=lambda: "https://portal.example.test",
        header_provider=lambda _url: {"Authorization": "Bearer fresh-token"},
    )

    parsed = portal_client.tools("mail-service")

    assert isinstance(parsed, ConnectorToolsListing)
    assert parsed.etag == '"header-v1"' and parsed.tools[0].facet == "unclassified"
    assert parsed.tools[0].description == "D"
    assert parsed.tools[0].categories == ["messaging"]
    assert transport.requests[0]["method"] == "GET"
    assert transport.requests[0]["url"] == "https://portal.example.test/api/v1/connectors/mail-service/tools"
    assert transport.requests[0]["headers"] == {"Accept": "application/json", "Authorization": "Bearer fresh-token"}
    assert transport.requests[0]["timeout"] == DEFAULT_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    ("body", "raised"),
    [({"error": "connector_not_found"}, GatewayUnavailable), (None, PortalToolsUnavailable)],
)
def test_only_the_portals_own_not_found_answer_means_the_connector_is_gone(body, raised):
    portal_client = PortalConnectorClient(
        transport=FakeTransport(FakeResponse(404, body)),
        endpoint_resolver=lambda: "https://portal.example.test",
        header_provider=lambda _url: {"Authorization": "Bearer fresh-token"},
    )

    with pytest.raises(raised):
        portal_client.tools("mail-service")


def test_cache_serves_a_fresh_entry_and_revalidates_a_stale_one(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    client = _CacheClient(_listing(), NotModified())

    initial = read_tools("mail-service", client=client, now=lambda: 100.0)
    client.calls.clear()
    fresh = read_tools("mail-service", client=client, now=lambda: 101.0)
    revalidated = read_tools(
        "mail-service", client=client, now=lambda: 100.0 + TTL_SECONDS
    )

    assert initial.source == "network"
    assert fresh.source == "cache" and client.calls == [{"slug": "mail-service", "if_none_match": '"v1"'}]
    assert revalidated.source == "revalidated"
    assert revalidated.fetched_at == 100.0 + TTL_SECONDS
    assert revalidated.tools[0].slug == "MAIL_SERVICE_SEND"


def test_future_cache_entry_revalidates_and_does_not_mask_auth_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    client = _CacheClient(_listing(), NotModified(), GatewayAuthError("expired", status=401))

    initial = read_tools("mail-service", client=client, now=lambda: 100.0)
    revalidated = read_tools("mail-service", client=client, now=lambda: 99.0)
    with pytest.raises(GatewayAuthError):
        read_tools("mail-service", client=client, refresh=True, now=lambda: 100.0)

    assert initial.source == "network"
    assert revalidated.source == "revalidated"
    assert client.calls[1] == {"slug": "mail-service", "if_none_match": '"v1"'}
    assert client.calls[2] == {"slug": "mail-service", "if_none_match": '"v1"'}


def test_cache_deletes_not_found_entries_and_serves_stale_on_tool_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    client = _CacheClient(
        _listing(),
        GatewayUnavailable("gone", code="CONNECTOR_NOT_FOUND", status=404),
        _listing(etag='"v2"'),
        PortalToolsUnavailable("unavailable", code="TOOLS_UNAVAILABLE", status=502),
    )
    read_tools("mail-service", client=client, now=lambda: 1.0)

    with pytest.raises(GatewayUnavailable):
        read_tools("mail-service", client=client, refresh=True, now=lambda: 2.0)
    replaced = read_tools("mail-service", client=client, now=lambda: 3.0)
    stale = read_tools("mail-service", client=client, refresh=True, now=lambda: 4.0)

    assert client.calls[2]["if_none_match"] is None
    assert replaced.etag == '"v2"'
    assert stale.source == "cache" and stale.stale is True
    assert stale.tools == replaced.tools
