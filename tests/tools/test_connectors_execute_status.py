"""Single-call execute: every failure status becomes the call's own error entry, the model-facing
shape stays what the array envelope produced, and an older gateway still works through the array form.
"""

import json
import logging

import pytest

from tools.connectors.gateway import client as client_module
from tools.connectors.gateway.client import ConnectorClient
from tools.connectors.gateway.errors import GatewayAuthError
from tools.connectors.gateway.merge import partition_calls, splice_remote_results
from tools.connectors.gateway.wire import EXECUTE_ERROR_CODES
from tools.connectors.turn import CARD, LINK, SIDE, scoped_connection_surface


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json})
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _fresh_legacy_memory(monkeypatch):
    monkeypatch.setattr(client_module, "_LEGACY_EXECUTE_ORIGINS", set())


def make_client(transport, sleeps=None):
    return ConnectorClient(
        transport=transport,
        endpoint_resolver=lambda: "https://tool-gateway.test",
        header_provider=lambda url: {"Authorization": "Bearer t"},
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


CALL = [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}}]


def planned(calls=CALL):
    return partition_calls(calls).remote


def failure(code, status, **extra):
    family = EXECUTE_ERROR_CODES[code][0]
    body = {"code": code, "family": family, "message": f"{code} message", "connector": "gmail",
            "requestId": "req_9", **extra}
    return FakeResponse(status, body)


def model_entry(transport, surface=LINK):
    plans = planned()
    with scoped_connection_surface(surface):
        return splice_remote_results(plans, make_client(transport).execute(plans))[0]


# Codes whose status is retried once: the second answer is the one the model sees.
RETRIED = {502, 503, 504}


@pytest.mark.parametrize("code", sorted(EXECUTE_ERROR_CODES))
def test_every_code_is_the_calls_error_entry_with_code_message_and_family(code):
    family, status = EXECUTE_ERROR_CODES[code]
    answers = [failure(code, status, hint="check the input")]
    if status in RETRIED:
        answers.append(failure(code, status, hint="check the input"))
    transport = FakeTransport(*answers)

    raw = make_client(transport).execute(planned())[0]
    entry = model_entry(FakeTransport(*answers))

    assert raw["data"] is None
    assert raw["error"]["status"] == status and raw["error"]["request_id"] == "req_9"
    error = entry["error"]
    assert error["code"] == code
    assert error["family"] == family
    assert error["connector"] == "gmail"
    # Transport details stay in logs; they never reach the model.
    assert "status" not in error and "request_id" not in error
    if code not in ("CONNECTION_REQUIRED", "CONNECTION_EXPIRED"):
        assert error["message"] == f"{code} message"
        assert error["hint"] == "check the input"
    assert len(transport.requests) == (2 if status in RETRIED else 1)


def test_alias_and_invalid_argument_failures_keep_the_gateways_hint():
    for code in ("ALIAS_REQUIRED", "ALIAS_UNKNOWN", "INVALID_ARGUMENTS"):
        entry = model_entry(FakeTransport(failure(code, 422, hint="name one of: work, personal")))
        assert entry["error"]["hint"] == "name one of: work, personal"


def test_an_unknown_code_with_a_family_still_renders():
    body = {"code": "SOMETHING_NEW", "family": "backend", "message": "new failure"}
    entry = model_entry(FakeTransport(FakeResponse(500, body)))
    assert entry["error"] == {"code": "SOMETHING_NEW", "message": "new failure", "family": "backend"}


def test_a_failed_call_logs_its_code_and_the_gateways_request_id(caplog):
    caplog.set_level(logging.DEBUG)
    make_client(FakeTransport(failure("TOOL_NOT_ALLOWED", 403))).execute(planned())
    lines = [r for r in caplog.records if r.getMessage().startswith("connector connectors.execute")]
    assert [(r.levelno, r.getMessage()) for r in lines] == [
        (logging.WARNING, "connector connectors.execute 403 TOOL_NOT_ALLOWED request_id=req_9 dispatch=-")]


def test_success_returns_the_result_data():
    body = {"result": {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "data": {"id": "m1"}, "account": "work"}}
    entry = model_entry(FakeTransport(FakeResponse(200, body)))
    assert entry == {"index": 0, "name": "connectors__gmail__SEND_EMAIL", "response": {"id": "m1"}, "account": "work"}


# ---------------------------------------------------------------------------
# connection codes keep the card / link behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["CONNECTION_REQUIRED", "CONNECTION_EXPIRED"])
def test_connection_codes_show_a_card_where_one_exists_and_a_link_elsewhere(code):
    link_body = dict(connector="gmail", connectUrl="https://example.test/c/1", connectionId="ca_1")

    card = model_entry(FakeTransport(failure(code, 409, **link_body)), CARD)["error"]
    assert card["code"] == code and card["connect_card_available"] is True
    assert "connect_url" not in card and "manage_connections" in card["hint"]

    side = model_entry(FakeTransport(failure(code, 409, **link_body)), SIDE)["error"]
    assert "connect_url" not in side and "Only the main agent" in side["hint"]

    link = model_entry(FakeTransport(failure(code, 409, **link_body)), LINK)["error"]
    assert link["connect_url"] == "https://example.test/c/1"


def test_expired_wording_says_reconnect_and_required_wording_is_unchanged():
    expired = model_entry(FakeTransport(failure("CONNECTION_EXPIRED", 409)), CARD)["error"]
    assert "expired" in expired["hint"] and "reconnect" in expired["hint"]

    required = model_entry(FakeTransport(failure("CONNECTION_REQUIRED", 409)), CARD)["error"]
    assert required["hint"] == (
        "This app is not connected. Call manage_connections to open a connection card for it; "
        "there is no link to show the user.")


# ---------------------------------------------------------------------------
# retry policy
# ---------------------------------------------------------------------------


def test_429_with_a_short_retry_after_waits_and_retries_under_the_same_key():
    sleeps = []
    ok = FakeResponse(200, {"result": {"data": "sent"}})
    limited = FakeResponse(429, {"code": "PROVIDER_RATE_LIMITED", "family": "backend", "message": "slow"},
                           {"Retry-After": "2"})
    transport = FakeTransport(limited, ok)
    (result,) = make_client(transport, sleeps).execute(planned())
    assert result == {"data": "sent", "error": None}
    assert sleeps == [2.0]
    keys = {r["headers"]["x-idempotency-key"] for r in transport.requests}
    assert len(keys) == 1


@pytest.mark.parametrize("headers", [{}, {"Retry-After": "120"}])
def test_429_without_a_usable_retry_after_is_returned_not_retried(headers):
    limited = FakeResponse(429, {"code": "PROVIDER_RATE_LIMITED", "family": "backend", "message": "slow"}, headers)
    transport = FakeTransport(limited)
    (result,) = make_client(transport).execute(planned())
    assert result["error"]["code"] == "PROVIDER_RATE_LIMITED"
    assert len(transport.requests) == 1


def test_product_failures_are_never_retried():
    for code, (family, status) in EXECUTE_ERROR_CODES.items():
        if family != "product":
            continue
        transport = FakeTransport(failure(code, status))
        make_client(transport).execute(planned())
        assert len(transport.requests) == 1, code


def test_a_failure_status_without_a_call_error_body_is_still_a_route_failure():
    transport = FakeTransport(FakeResponse(401, {"error": {"code": "UNAUTHORIZED", "message": "expired"}}))
    with pytest.raises(GatewayAuthError):
        make_client(transport).execute(planned())


# ---------------------------------------------------------------------------
# older gateway: array form once, then remembered
# ---------------------------------------------------------------------------


OLD_GATEWAY_400 = FakeResponse(400, {
    "error": "Invalid request body",
    "issues": [{"code": "invalid_type", "path": ["tools"], "message": "Required"}],
})


def envelope(*results):
    return FakeResponse(200, {"results": list(results), "successCount": 0, "errorCount": 0, "totalCount": 0})


def test_an_older_gateway_gets_the_array_form_once_and_is_remembered():
    two = planned(CALL + [{"name": "connectors__slack__POST_MESSAGE", "arguments": {}}])
    transport = FakeTransport(
        OLD_GATEWAY_400,
        envelope({"data": "a"}, {"error": {"code": "TOOL_NOT_ALLOWED", "message": "blocked"}}),
        envelope({"data": "c"}),
    )
    client = make_client(transport)

    first = client.execute(two)
    assert [r["json"] for r in transport.requests[:2]] == [
        {"call": {"connector": "gmail", "tool": "SEND_EMAIL", "arguments": {"to": "x"}}},
        {"tools": [{"connector": "gmail", "tool": "SEND_EMAIL", "arguments": {"to": "x"}},
                   {"connector": "slack", "tool": "POST_MESSAGE", "arguments": {}}]},
    ]
    assert first[0] == {"data": "a", "error": None}
    assert first[1]["error"]["code"] == "TOOL_NOT_ALLOWED"

    # The next execute goes straight to the array form: no second rejected single call.
    second = client.execute(planned())
    assert "tools" in transport.requests[2]["json"] and len(transport.requests) == 3
    assert second == [{"data": "c", "error": None}]


def test_a_new_gateways_own_validation_error_is_not_mistaken_for_an_old_gateway():
    body = {"code": "VALIDATION_ERROR", "message": "call.arguments must be an object"}
    transport = FakeTransport(FakeResponse(400, body))
    with pytest.raises(Exception):
        make_client(transport).execute(planned())
    assert len(transport.requests) == 1
    assert client_module._LEGACY_EXECUTE_ORIGINS == set()
