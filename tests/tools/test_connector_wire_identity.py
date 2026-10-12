"""Caller identity on connector requests: the User-Agent always, correlation ids only with consent,
the gateway's request id logged on every outcome, and ``initiator`` on each connect path."""

import json
import logging

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.config import get_config_path
from tools.connectors.gateway.client import ConnectorClient
from tools.managed_gateway_identity import CallRef, connector_headers, user_agent


class _Response:
    def __init__(self, status, body, request_id="gw-req-1"):
        self.status_code = status
        self._body = body
        self.headers = {"x-request-id": request_id}

    def json(self):
        return self._body


class _Transport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.sent.append({"url": url, "headers": dict(headers or {}), "json": json})
        return self.responses.pop(0)


def _client(transport, **kwargs):
    return ConnectorClient(
        transport=transport, endpoint_resolver=lambda: "https://gw.test",
        header_provider=lambda url: {"Authorization": "Bearer t"}, **kwargs,
    )


def _consent(send: bool):
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"telemetry:\n  shared_metrics:\n    enabled: true\n    send: {str(send).lower()}\n")


@pytest.fixture
def platform():
    tokens = []

    def bind(name):
        tokens.extend(set_session_vars(platform=name, cron_session=""))

    yield bind
    clear_session_vars(tokens)


@pytest.mark.parametrize(("bound", "surface"), [
    ("", "cli"), ("tui", "tui"), ("desktop", "desktop"), ("telegram", "gateway"), ("api_server", "api"),
])
def test_user_agent_names_the_session_surface(platform, bound, surface):
    if bound:
        platform(bound)
    agent = user_agent()
    product, _, rest = agent.partition(" ")
    assert product.startswith("hermes-agent/") and len(product) > len("hermes-agent/")
    assert rest.startswith(f"({surface}; ") and rest.endswith(")")
    assert rest[len(surface) + 3:-1] in {"linux", "macos", "windows", "unknown"}


@pytest.mark.parametrize("send", [False, True])
def test_correlation_ids_go_only_with_send_consent(send):
    _consent(send)
    transport = _Transport(_Response(200, {"schemas": {}}), _Response(200, {"schemas": {}}))
    client = _client(transport, call=CallRef(turn_id="turn-1", tool_call_id="call-1"))
    client.schemas(["GMAIL_SEND_EMAIL"])
    client.schemas(["GMAIL_SEND_EMAIL"])

    first, second = (sent["headers"] for sent in transport.sent)
    assert first["User-Agent"].startswith("hermes-agent/")
    correlation = {k for k in first if k.startswith("x-nous-hermes-") or k == "x-request-id"}
    if not send:
        assert correlation == set()
        return
    assert first["x-nous-hermes-turn"] == "turn-1" and first["x-nous-hermes-tool-call"] == "call-1"
    assert first["x-request-id"] != second["x-request-id"]  # one id per HTTP attempt
    assert not any(k.startswith("x-hermes-") for k in first)  # the gateway strips only x-nous-


def test_session_header_is_the_conversation_root(monkeypatch):
    from agent.portal_tags import reset_conversation_context, set_conversation_context

    _consent(True)
    token = set_conversation_context("root-session")
    try:
        assert connector_headers()["x-nous-hermes-session"] == "root-session"
    finally:
        reset_conversation_context(token)


def test_gateway_request_id_is_logged_on_failure_and_success(caplog):
    transport = _Transport(
        _Response(200, {"schemas": {}}, request_id="ok-id"),
        _Response(403, {"error": {"code": "TOOL_NOT_ALLOWED", "message": "no"}}, request_id="bad-id"),
    )
    client = _client(transport, call=CallRef(tool_call_id="call-9"))
    with caplog.at_level(logging.DEBUG, logger="tools.managed_gateway_identity"):
        client.schemas(["X"])
        with pytest.raises(Exception) as raised:
            client.schemas(["X"])
    assert raised.value.request_id == "bad-id"
    lines = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (logging.DEBUG, "connector connectors.schemas 200 - request_id=ok-id dispatch=call-9") in lines
    assert (logging.WARNING,
            "connector connectors.schemas 403 TOOL_NOT_ALLOWED request_id=bad-id dispatch=call-9") in lines


def test_a_not_yet_created_account_stays_debug(caplog):
    transport = _Transport(_Response(404, {"error": "connection_not_found"}))
    with caplog.at_level(logging.DEBUG, logger="tools.managed_gateway_identity"):
        assert _client(transport).account_status("acct-1") is None
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]
    assert "connectors.accounts 404" in caplog.records[0].getMessage()


def _mint_ok(names):
    return _Response(200, {"results": [{"connector": n, "status": "initiated", "connectUrl": "https://l"} for n in names]})


def test_each_connect_path_names_its_initiator(monkeypatch):
    from tools.connectors import managed
    from tools.connectors.contract import Actor, TargetState
    from tools.connectors.operation import ConnectionOperation, Target
    from tools.connectors.run import reissue

    transport = _Transport(_mint_ok(["gmail"]), _mint_ok(["gmail"]), _mint_ok(["gmail"]))
    monkeypatch.setattr(managed, "ensure_guest_identity", lambda: None)
    real = ConnectorClient
    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", lambda **kw: real(
        transport=transport, endpoint_resolver=lambda: "https://gw.test",
        header_provider=lambda url: {"Authorization": "Bearer t"}, **kw))

    for initiator in (managed.AGENT_TOOL, managed.DESKTOP_PAGE):
        operation = ConnectionOperation([Target("gmail", "connector", "connect")])
        client = managed.managed_client() if initiator == managed.AGENT_TOOL else managed.managed_client(initiator=initiator)
        managed.mint(client, operation, ["gmail"], reinitiate=False, actor=Actor.backend_watcher)

    operation.transition("gmail", TargetState.failed, Actor.backend_watcher)
    assert reissue(operation, ["gmail"]) is None  # the card's Try again

    assert [sent["json"]["initiator"] for sent in transport.sent] == ["agent_tool", "desktop_page", "card_retry"]


def test_operation_logs_open_transitions_and_settle(caplog):
    from tools.connectors import live
    from tools.connectors.contract import Actor, SettleReason, TargetState
    from tools.connectors.operation import ConnectionOperation, Target

    operation = ConnectionOperation([Target("gmail", "connector", "connect")], session_key="s-log")
    with caplog.at_level(logging.INFO, logger="tools.connectors"):
        live.open(operation)
        operation.transition("gmail", TargetState.initiated, Actor.backend_watcher)
        operation.transition("gmail", TargetState.connected, Actor.backend_watcher)
        operation.settle(SettleReason.all_resolved)
        live.close(operation)
    op = operation.op_id
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.INFO] == [
        f"connection op {op} opened: connector:gmail:connect",
        f"connection op {op}: gmail pending -> initiated by backend_watcher",
        f"connection op {op}: gmail initiated -> connected by backend_watcher",
        f"connection op {op} settled by all_resolved: gmail=connected",
    ]
    assert json.dumps(operation.snapshot())  # snapshot still serializes
