"""Regression tests for the loopback OAuth callback handler (#116278).

A real browser follows the /callback redirect with speculative queryless
fetches (/favicon.ico), and the CLI waiter polls the result dict only every
500ms — so a handler that updates the result unconditionally loses the stored
code between polls and the login times out despite the success page the user
saw. These tests pin the two invariants that prevent that: a request without
``code``/``error`` never mutates the result, and the first terminal result
wins over later callbacks.
"""
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

from tools.mcp_oauth import _make_callback_handler


@pytest.fixture()
def callback_server():
    handler_cls, result = _make_callback_handler()
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield server.server_address[1], result
    server.shutdown()
    server.server_close()


def _get(port: int, path: str) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


def test_favicon_after_callback_keeps_stored_code(callback_server):
    port, result = callback_server
    assert _get(port, "/callback?code=synthetic-code&state=synthetic-state&iss=synthetic-iss") == 200
    assert _get(port, "/favicon.ico") == 404
    assert result["auth_code"] is not None
    assert result["state"] == "synthetic-state"
    assert result["iss"] == "synthetic-iss"


def test_favicon_before_callback_is_inert(callback_server):
    port, result = callback_server
    assert _get(port, "/favicon.ico") == 404
    assert result["auth_code"] is None
    assert result["error"] is None
    assert _get(port, "/callback?code=synthetic-code&state=synthetic-state") == 200
    assert result["auth_code"] is not None


def test_duplicate_callback_keeps_first_result(callback_server):
    port, result = callback_server
    assert _get(port, "/callback?code=first-code&state=first-state") == 200
    assert _get(port, "/callback?code=second-code&state=second-state") == 200
    assert result["auth_code"] == "first-code"
    assert result["state"] == "first-state"


def test_favicon_after_error_callback_keeps_error(callback_server):
    port, result = callback_server
    assert _get(port, "/callback?error=access_denied&state=synthetic-state") == 200
    assert _get(port, "/favicon.ico") == 404
    assert result["error"] == "access_denied"
    assert result["auth_code"] is None
