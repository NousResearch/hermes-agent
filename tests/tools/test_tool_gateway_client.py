"""Behavior tests for ConnectorClient and the bridge entry points.

DI-callable idiom (test_managed_tool_gateway.py precedent): fakes are
injected through the constructor seams — no module mocks, no patching of
transports. FakeTransport records requests and replays queued responses.
"""

import json
from dataclasses import replace as dataclass_replace

import pytest

from tools.tool_gateway.bridge import connector_search_hits
from tools.tool_gateway.client import ConnectorClient
from tools.tool_gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    IdempotencyConflict,
    ToolGatewayError,
)
from tools.tool_gateway.names import vendor_slug_candidates


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
            {"method": method, "url": url, "headers": dict(headers or {}), "json": json}
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
    from tools.tool_gateway.merge import partition_calls

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


def test_execute_sends_one_request_with_camelcase_body_and_key():
    transport = FakeTransport(
        FakeResponse(
            200,
            execute_envelope(
                [
                    {"index": 0, "connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "data": {"id": "m1"}},
                    {"index": 1, "connector": "slack", "tool": "SLACK_POST_MESSAGE", "data": "ok"},
                ]
            ),
        )
    )
    results = make_client(transport).execute(planned())

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["url"].endswith("/v1/connectors/execute")
    assert request["json"] == {
        "tools": [
            {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "arguments": {"to": "x"}},
            {"connector": "slack", "tool": "SLACK_POST_MESSAGE", "arguments": {}},
        ]
    }
    assert request["headers"]["x-idempotency-key"]  # present, non-empty
    assert request["headers"]["Authorization"] == "Bearer nous-token"
    assert results == [
        {"data": {"id": "m1"}, "error": None},
        {"data": "ok", "error": None},
    ]


def test_retry_on_5xx_reuses_the_same_idempotency_key():
    transport = FakeTransport(
        FakeResponse(502, {"error": {"code": "BAD_GATEWAY", "message": "upstream"}}),
        FakeResponse(
            200,
            execute_envelope(
                [{"index": 0, "connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "data": "sent"}]
            ),
        ),
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


def test_connection_required_stays_inside_the_200_envelope():
    transport = FakeTransport(
        FakeResponse(
            200,
            execute_envelope(
                [
                    {
                        "index": 0,
                        "connector": "gmail",
                        "tool": "GMAIL_SEND_EMAIL",
                        "error": {
                            "code": "CONNECTION_REQUIRED",
                            "message": "connect gmail",
                            "connector": "gmail",
                            "connectUrl": "https://example.test/connect/1",
                        },
                    }
                ]
            ),
        )
    )
    (result,) = make_client(transport).execute(planned(PLAN_CALLS[:1]))
    assert result["error"]["code"] == "CONNECTION_REQUIRED"
    assert result["error"]["connect_url"] == "https://example.test/connect/1"


# ---------------------------------------------------------------------------
# bridge: connector_search_hits silent degradation (D32)
# ---------------------------------------------------------------------------


def test_search_hits_empty_on_unavailable_dark_gateway_and_exploding_client():
    assert connector_search_hits(
        [{"use_case": "send mail"}], availability=lambda: False
    ) == {}

    def dark_factory():
        raise GatewayUnavailable("dark", code="NOT_FOUND", status=404)

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=dark_factory,
        )
        == {}
    )

    def boom_factory():
        raise RuntimeError("boom")

    assert (
        connector_search_hits(
            [{"use_case": "send mail"}],
            availability=lambda: True,
            client_factory=boom_factory,
        )
        == {}
    )


def test_search_hits_pass_through_on_success():
    class FakeClient:
        def search(self, queries):
            assert queries == [{"use_case": "send mail"}]
            return {"results": [{"index": 1, "use_case": "send mail"}]}

    hits = connector_search_hits(
        [{"use_case": "send mail"}],
        availability=lambda: True,
        client_factory=lambda: FakeClient(),
    )
    assert hits["results"][0]["use_case"] == "send mail"


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

    from tools.tool_gateway.client import _default_endpoint_resolver

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
