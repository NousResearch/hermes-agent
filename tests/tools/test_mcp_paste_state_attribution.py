"""Callbacks are attributed to the login that printed their authorization URL (#134964).

The SDK generates the state and only compares it after ``callback_handler`` returns, so a callback
from a different surviving login used to look accepted the moment it landed ("Got authorization
code from paste — completing flow") and failed only later — reading to the operator as a
reauthorization loop. The redirect handler now captures the state off the authorization URL into a
per-flow slot, and both callback readers reject a mismatch before it poisons the result."""
import asyncio
import io
import socket
import threading
from http.client import HTTPConnection

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

import tools.mcp_oauth as mo


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


def _wait_listening(port: int) -> None:
    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            threading.Event().wait(0.02)
    raise AssertionError("callback listener never bound")


def _paste(monkeypatch, line: str, result: dict, flow_state: dict) -> None:
    monkeypatch.setattr(mo.sys, "stdin", io.StringIO(line + "\n"))
    mo._paste_callback_reader(result, flow_state)


def test_paste_from_another_login_is_rejected_before_the_result(monkeypatch, capsys):
    flow_state = {"expected_state": "s1"}
    result: dict = {"auth_code": None, "state": None, "error": None, "iss": None}

    _paste(monkeypatch, "/callback?code=cross&state=OTHER", result, flow_state)

    assert result["auth_code"] is None
    assert "different Hermes login" in capsys.readouterr().err


def test_paste_from_this_login_still_completes(monkeypatch, capsys):
    flow_state = {"expected_state": "s1"}
    result: dict = {"auth_code": None, "state": None, "error": None, "iss": None}

    _paste(monkeypatch, "/callback?code=mine&state=s1&iss=https://as.example", result, flow_state)

    assert result["auth_code"] == "mine"
    assert "completing flow" in capsys.readouterr().err


def test_paste_before_the_url_is_seen_keeps_the_old_lenient_shape(monkeypatch):
    flow_state = {"expected_state": None}
    result: dict = {"auth_code": None, "state": None, "error": None, "iss": None}

    _paste(monkeypatch, "/callback?code=early&state=anything", result, flow_state)

    assert result["auth_code"] == "early"


def test_redirect_handler_captures_the_state_off_the_url(monkeypatch):
    flow_state = {"expected_state": None}
    monkeypatch.setattr(mo, "_announce_authorization_url", lambda *a, **k: None)
    monkeypatch.setattr(mo, "_is_interactive", lambda: True)
    handler = mo._make_redirect_handler(27890, flow_state=flow_state)

    asyncio.run(handler("https://as.example/authorize?state=abc123&client_id=hermes"))

    assert flow_state["expected_state"] == "abc123"


def test_http_callback_from_another_login_gets_403_and_writes_nothing(monkeypatch):
    flow_state = {"expected_state": "s1"}
    handler_cls, result = mo._make_callback_handler(flow_state)
    assert result == {"auth_code": None, "state": None, "error": None, "iss": None}

    h = handler_cls.__new__(handler_cls)
    h.path = "/callback?code=cross&state=WRONG"
    sent: dict = {}

    def fake_send(status):
        sent["status"] = status

    h.send_response = fake_send
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h.wfile = io.BytesIO()
    h.do_GET()

    assert sent["status"] == 403
    assert result["auth_code"] is None
