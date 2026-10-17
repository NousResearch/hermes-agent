"""Behavior tests for ConnectorClient and the bridge entry points.

DI-callable idiom (test_managed_tool_gateway.py precedent): fakes are
injected through the constructor seams — no module mocks, no patching of
transports. FakeTransport records requests and replays queued responses.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit
from dataclasses import replace as dataclass_replace

import pytest

from tools.connectors.gateway.bridge import ConnectorLeg, connector_search_hits
from tools.connectors.gateway.client import ConnectorClient
from tools.connectors.gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    IdempotencyConflict,
    ToolGatewayError,
)
from tools.connectors.gateway.names import vendor_slug_candidates


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeTransport:
    """Records requests; replays queued responses (exceptions raise)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append(
            {"method": method, "url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout}
        )
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(transport):
    return ConnectorClient(
        transport=transport,
        endpoint_resolver=lambda: "https://tool-gateway.test",
        header_provider=lambda url: {"Authorization": "Bearer nous-token"},
    )


def execute_envelope(results):
    errors = sum(1 for r in results if r.get("error"))
    return {
        "results": results,
        "successCount": len(results) - errors,
        "errorCount": errors,
        "totalCount": len(results),
    }


PLAN_CALLS = [
    {"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}},
    {"name": "connectors__slack__POST_MESSAGE", "arguments": {}},
]


def planned(calls=PLAN_CALLS):
    from tools.connectors.gateway.merge import partition_calls

    return tuple(
        dataclass_replace(
            plan,
            tool=vendor_slug_candidates(plan.connector, plan.tool)[0],
        )
        for plan in partition_calls(calls).remote
    )


# ---------------------------------------------------------------------------
# execute: request shape + idempotency
# ---------------------------------------------------------------------------


def test_list_connectors_percent_encodes_the_opaque_cursor():
    transport = FakeTransport(
        FakeResponse(200, {"items": [{"connector": "gmail"}], "nextCursor": "page+2/&="}),
        FakeResponse(200, {"items": [{"connector": "notion"}]}),
    )

    items = make_client(transport).list_connectors(timeout=2.0)

    assert [item["connector"] for item in items] == ["gmail", "notion"]
    assert transport.requests[1]["url"].endswith(
        "/v1/connectors?limit=50&cursor=page%2B2%2F%26%3D"
    )
    assert all(request["timeout"] == 2.0 for request in transport.requests)


def test_list_connectors_preserves_cursor_over_http():
    cursor = "page+2/&= 空"
    queries = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlsplit(self.path).query)
            queries.append(query)
            connector = "notion" if "cursor" in query else "gmail"
            body = {"items": [{"connector": connector}]}
            if "cursor" not in query:
                body["nextCursor"] = cursor
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = ConnectorClient(
            endpoint_resolver=lambda: f"http://127.0.0.1:{server.server_port}",
            header_provider=lambda url: {"Authorization": "Bearer test-token"},
        )
        try:
            items = client.list_connectors(timeout=2.0)
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert [item["connector"] for item in items] == ["gmail", "notion"]
    assert queries == [{"limit": ["50"]}, {"limit": ["50"], "cursor": [cursor]}]
    assert not worker.is_alive()


def single_result(connector, tool, data):
    return {"result": {"connector": connector, "tool": tool, "data": data}}


def test_execute_sends_one_single_call_request_per_call_with_its_own_key():
    transport = FakeTransport(
        FakeResponse(200, single_result("gmail", "GMAIL_SEND_EMAIL", {"id": "m1"})),
        FakeResponse(200, single_result("slack", "SLACK_POST_MESSAGE", "ok")),
    )
    results = make_client(transport).execute(planned())

    assert [r["url"] for r in transport.requests] == ["https://tool-gateway.test/v1/connectors/execute"] * 2
    assert [r["json"] for r in transport.requests] == [
        {"call": {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "arguments": {"to": "x"}}},
        {"call": {"connector": "slack", "tool": "SLACK_POST_MESSAGE", "arguments": {}}},
    ]
    keys = [r["headers"]["x-idempotency-key"] for r in transport.requests]
    assert all(keys) and keys[0] != keys[1]
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer nous-token"
    assert results == [
        {"data": {"id": "m1"}, "error": None},
        {"data": "ok", "error": None},
    ]


def test_retry_on_5xx_reuses_the_same_idempotency_key():
    transport = FakeTransport(
        FakeResponse(502, {"error": {"code": "BAD_GATEWAY", "message": "upstream"}}),
        FakeResponse(200, single_result("gmail", "GMAIL_SEND_EMAIL", "sent")),
    )
    results = make_client(transport).execute(planned(PLAN_CALLS[:1]))

    assert len(transport.requests) == 2
    first_key = transport.requests[0]["headers"]["x-idempotency-key"]
    second_key = transport.requests[1]["headers"]["x-idempotency-key"]
    assert first_key == second_key
    assert results[0]["data"] == "sent"


def test_retry_on_transport_failure_reuses_key_then_gives_up():
    transport = FakeTransport(
        ConnectionError("reset"), ConnectionError("reset again")
    )
    with pytest.raises(ToolGatewayError) as exc_info:
        make_client(transport).execute(planned(PLAN_CALLS[:1]))
    assert exc_info.value.code == "TRANSPORT_ERROR"
    assert len(transport.requests) == 2
    assert (
        transport.requests[0]["headers"]["x-idempotency-key"]
        == transport.requests[1]["headers"]["x-idempotency-key"]
    )


def test_4xx_never_retries():
    transport = FakeTransport(
        FakeResponse(400, {"error": {"code": "BAD_REQUEST", "message": "nope"}})
    )
    with pytest.raises(ToolGatewayError):
        make_client(transport).execute(planned(PLAN_CALLS[:1]))
    assert len(transport.requests) == 1


def test_409_raises_idempotency_conflict_and_never_retries():
    transport = FakeTransport(
        FakeResponse(
            409,
            {"error": {"code": "IDEMPOTENCY_CONFLICT", "message": "key reused"}},
        )
    )
    with pytest.raises(IdempotencyConflict):
        make_client(transport).execute(planned(PLAN_CALLS[:1]))
    assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# status mapping + auth
# ---------------------------------------------------------------------------


def test_404_raises_gateway_unavailable_the_dark_signal():
    transport = FakeTransport(FakeResponse(404, {"error": {"code": "NOT_FOUND", "message": "no route"}}))
    with pytest.raises(GatewayUnavailable):
        make_client(transport).execute(planned(PLAN_CALLS[:1]))


def test_401_raises_auth_error_and_missing_token_fails_fast():
    transport = FakeTransport(
        FakeResponse(401, {"error": {"code": "UNAUTHORIZED", "message": "expired"}})
    )
    with pytest.raises(GatewayAuthError):
        make_client(transport).execute(planned(PLAN_CALLS[:1]))

    # No token -> no request at all.
    no_token = FakeTransport()
    client = ConnectorClient(
        transport=no_token,
        endpoint_resolver=lambda: "https://tool-gateway.test",
        header_provider=lambda url: {},
    )
    with pytest.raises(GatewayAuthError):
        client.execute(planned(PLAN_CALLS[:1]))
    assert no_token.requests == []


def test_connection_required_409_is_a_call_entry_with_link_and_account():
    transport = FakeTransport(
        FakeResponse(
            409,
            {
                "code": "CONNECTION_REQUIRED",
                "family": "product",
                "message": "connect gmail",
                "connector": "gmail",
                "connectUrl": "https://example.test/connect/1",
                "connectionId": "ca_1",
                "requestId": "req_1",
            },
        )
    )
    (result,) = make_client(transport).execute(planned(PLAN_CALLS[:1]))
    assert len(transport.requests) == 1
    assert result["error"]["code"] == "CONNECTION_REQUIRED"
    assert result["error"]["connect_url"] == "https://example.test/connect/1"
    assert result["error"]["connection_id"] == "ca_1"
    assert result["error"]["request_id"] == "req_1"


# ---------------------------------------------------------------------------
# bridge: connector_search_hits silent degradation (D32)
# ---------------------------------------------------------------------------


def test_search_hits_empty_on_unavailable_dark_gateway_and_names_a_real_failure():
    assert connector_search_hits(
        [{"use_case": "send mail"}], availability=lambda: False
    ) == ConnectorLeg()

    def dark_factory():
        raise GatewayUnavailable("dark", code="NOT_FOUND", status=404)

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=dark_factory,
        )
        == ConnectorLeg()
    )

    def boom_factory():
        raise RuntimeError("boom")

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=boom_factory,
        )
        == ConnectorLeg(failure="unreachable")
    )

    def rejected_factory():
        raise GatewayAuthError("token rejected", code="UNAUTHORIZED", status=401)

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=rejected_factory,
        )
        == ConnectorLeg(failure="sign_in_expired")
    )

    def forbidden_factory():
        raise GatewayAuthError("no entitlement", code="FORBIDDEN", status=403)

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=forbidden_factory,
        )
        == ConnectorLeg()
    )


def test_search_hits_pass_through_on_success():
    class FakeClient:
        def search(self, queries):
            assert queries == [{"use_case": "send mail"}]
            return {"results": [{"index": 1, "use_case": "send mail"}]}

    leg = connector_search_hits(
        [{"use_case": "send mail"}],
        availability=lambda: True,
        client_factory=lambda: FakeClient(),
    )
    assert leg.failure is None
    assert leg.payload["results"][0]["use_case"] == "send mail"


# ---------------------------------------------------------------------------
# default endpoint resolver: the SHARED origin, not a fabricated vendor
# ---------------------------------------------------------------------------


_GATEWAY_ENV_KEYS = (
    "TOOL_GATEWAY_URL",
    "CONNECTOR_GATEWAY_URL",
    "TOOL_GATEWAY_DOMAIN",
    "TOOL_GATEWAY_SCHEME",
)


def _resolve_with_env(**overrides):
    """Run the default resolver with ONLY the given gateway env keys set."""
    import os
    from unittest.mock import patch

    from tools.connectors.gateway.client import _default_endpoint_resolver

    env = {k: v for k, v in os.environ.items() if k not in _GATEWAY_ENV_KEYS}
    env.update(overrides)
    with patch.dict("os.environ", env, clear=True):
        return _default_endpoint_resolver()


def test_default_resolver_uses_the_connector_gateway_origin():
    # Connector routes live on the connectors deployment's own host, so the
    # resolver wants that origin — never a fabricated "connectors" vendor
    # passthrough host, and never the media/on-origin-vendor host.
    assert _resolve_with_env(CONNECTOR_GATEWAY_URL="http://127.0.0.1:3009") == (
        "http://127.0.0.1:3009"
    )
    assert _resolve_with_env(TOOL_GATEWAY_DOMAIN="gw.example.com") == (
        "https://connector-gateway.gw.example.com"
    )


def test_default_resolver_ignores_the_media_host_override():
    # TOOL_GATEWAY_URL moves the media/on-origin-vendor host only. Letting it
    # drag the connectors client along would silently point connector calls at
    # a host that does not serve them.
    assert _resolve_with_env(
        TOOL_GATEWAY_URL="http://127.0.0.1:3009",
        TOOL_GATEWAY_DOMAIN="gw.example.com",
    ) == "https://connector-gateway.gw.example.com"


def test_default_resolver_is_none_on_a_misconfigured_scheme():
    assert _resolve_with_env(TOOL_GATEWAY_SCHEME="ftp") is None
