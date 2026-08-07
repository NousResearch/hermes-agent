"""Tests for MCPServerTask._preflight_content_type fast-fail behaviour.

These drive the REAL ``_preflight_content_type`` method against a real local
HTTP server (via httpx's ASGI/transport plumbing through a stdlib server),
rather than reimplementing the probe inline. That distinction matters: the
production probe must run on its own httpx client outside the MCP SDK's anyio
task group, and a faithful test must exercise that actual method so the
content-type allow-list, HEAD->GET fallback, POST probe fallback, and
best-effort pass-through are all covered as shipped.

OAuth note
----------
``MCPServerTask.run()`` skips the preflight entirely when ``auth_type=="oauth"``
(see ``test_run_skips_preflight_for_oauth``).  OAuth-protected MCP servers
return ``200 text/html`` (a login/landing page) on an unauthenticated probe,
which ``_preflight_content_type`` correctly rejects — the probe cannot tell
whether the page is a valid OAuth endpoint or a misconfigured URL.  The right
validator for OAuth servers is ``.well-known/oauth-protected-resource``, which
the OAuth handshake consults automatically.
"""

from __future__ import annotations

import asyncio
import http.server
import socketserver
import threading
from contextlib import contextmanager

import pytest

from tools.mcp_tool import MCPServerTask, NonMcpEndpointError


def _make_task(name: str = "probe_srv") -> MCPServerTask:
    """Minimal MCPServerTask without running the heavy __init__."""
    task = MCPServerTask.__new__(MCPServerTask)
    task.name = name
    return task


@contextmanager
def _serve(handler_cls):
    """Run *handler_cls* on a background thread; yield its base URL."""
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def _handler(status: int = 200,
             content_type: "str | None" = "text/html; charset=utf-8",
             body: bytes = b"<html>x</html>", head_status=None, record=None,
             post_content_type: "str | None" = None,
             post_body: bytes = b"",
             post_status: "int | None" = None):
    """Build a BaseHTTPRequestHandler that replies with the given shape.

    ``head_status`` lets HEAD return a different status than GET (to exercise
    the HEAD->GET fallback). ``record`` is an optional list that captures the
    HTTP methods the server actually saw.

    ``post_content_type`` / ``post_body`` / ``post_status`` let POST return a
    different response than HEAD/GET (to exercise the POST probe fallback for
    servers that serve HTML on GET but speak MCP via POST).
    """

    class _H(http.server.BaseHTTPRequestHandler):
        def _write(self, sc, ct, payload):
            self.send_response(sc)
            if ct is not None:
                self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def do_HEAD(self):
            if record is not None:
                record.append("HEAD")
            sc = head_status if head_status is not None else status
            self._write(sc, content_type, b"")

        def do_GET(self):
            if record is not None:
                record.append("GET")
            self._write(status, content_type, body)

        def do_POST(self):
            if record is not None:
                record.append("POST")
            # Read and discard request body to avoid broken pipe.
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            sc = post_status if post_status is not None else status
            ct = post_content_type if post_content_type is not None else content_type
            pb = post_body if post_body else body
            self._write(sc, ct, pb)

        def log_message(self, format, *args):  # noqa: A002
            pass

    return _H


# ---------------------------------------------------------------------------
# Reject: non-MCP content types on a 2xx response
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content_type", [
    "text/html; charset=utf-8",
    "text/html",
    "text/plain",
    "application/xml",
    "text/HTML",  # case-insensitivity
])
def test_non_mcp_content_type_raises(content_type):
    task = _make_task("bad_srv")
    with _serve(_handler(status=200, content_type=content_type)) as base:
        with pytest.raises(NonMcpEndpointError) as exc_info:
            asyncio.run(task._preflight_content_type(f"{base}/", timeout=5.0))
    msg = str(exc_info.value)
    assert "bad_srv" in msg
    assert "application/json" in msg and "text/event-stream" in msg


# ---------------------------------------------------------------------------
# Pass-through: valid MCP content types, ambiguous, and error responses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content_type", [
    "application/json",
    "application/json; charset=utf-8",
    "text/event-stream",
    "TEXT/EVENT-STREAM",
])
def test_valid_mcp_content_types_pass(content_type):
    task = _make_task()
    with _serve(_handler(status=200, content_type=content_type, body=b"{}")) as base:
        # Must not raise.
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


def test_missing_content_type_passes():
    task = _make_task()
    with _serve(_handler(status=200, content_type=None, body=b"")) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


# ---------------------------------------------------------------------------
# HEAD -> GET fallback
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ssl_verify / client_cert forwarding to the probe client
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OAuth server: why the run() guard is needed
# ---------------------------------------------------------------------------

def test_oauth_server_html_response_raises_without_skip():
    """_preflight_content_type raises NonMcpEndpointError for 200 text/html.

    This documents the failure mode that the ``self._auth_type != "oauth"``
    guard in ``MCPServerTask.run()`` prevents.  An OAuth-protected MCP server
    returns a login/landing page on an unauthenticated HEAD probe — identical
    to a misconfigured URL from the preflight's point of view — because it
    cannot serve a meaningful MCP response without a Bearer token.

    Real-world example: Hospitable's MCP server
    (``https://mcp.hospitable.com/mcp``) returns ``200 text/html`` to an
    unauthenticated httpx HEAD request.  With the guard removed, connecting
    via ``hermes mcp add/login`` raises ``NonMcpEndpointError`` before the
    OAuth browser flow can begin.  With the guard in place, 63 tools are
    discovered and the server connects successfully.
    """
    task = _make_task("hospitable")
    # HEAD returns 200 text/html — what Hospitable sends without a token.
    with _serve(_handler(status=200, content_type="text/html; charset=UTF-8")) as base:
        with pytest.raises(NonMcpEndpointError) as exc_info:
            asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))
    assert "hospitable" in str(exc_info.value)


def test_run_skips_preflight_for_oauth(monkeypatch):
    """MCPServerTask.run() must not call _preflight_content_type for OAuth servers.

    The ``self._auth_type != "oauth"`` guard in the preflight condition ensures
    that OAuth-protected servers never hit ``NonMcpEndpointError`` from the
    unauthenticated GET probe.  The probe is inapplicable to OAuth servers:
    their identity is established by the OAuth metadata discovery
    (``.well-known/oauth-protected-resource``), not by a GET content-type check.
    """
    import tools.mcp_tool as _mcp

    preflight_calls: list[str] = []

    async def _inner():
        # Patch at the class level: replacement receives (self, url, **kwargs).
        async def _fake_preflight(self, url, **kwargs):
            preflight_calls.append(url)

        async def _fake_run_http(self, config):
            # Abort immediately after the preflight gate — we only want to
            # verify the gate, not exercise the real transport.
            raise asyncio.CancelledError()

        # Bypass URL validation so the test doesn't need a live network.
        monkeypatch.setattr(_mcp, "_validate_remote_mcp_url", lambda n, u: None)
        monkeypatch.setattr(_mcp.MCPServerTask, "_preflight_content_type", _fake_preflight)
        monkeypatch.setattr(_mcp.MCPServerTask, "_run_http", _fake_run_http)

        task = _mcp.MCPServerTask("hospitable-test")
        with pytest.raises(asyncio.CancelledError):
            await task.run({"url": "https://mcp.hospitable.com/mcp", "auth": "oauth"})

    asyncio.run(_inner())
    assert preflight_calls == [], (
        "_preflight_content_type must not be called for OAuth servers; "
        "without the guard the OAuth flow is blocked by the 200 text/html "
        "landing page the server returns to an unauthenticated probe"
    )


def test_run_skips_preflight_when_skip_preflight_set(monkeypatch):
    """``skip_preflight: true`` in server config bypasses the probe entirely.

    Escape hatch for valid Streamable HTTP servers whose HEAD/GET answers a
    non-MCP content type (and whose POST probe still can't be validated, e.g.
    non-OAuth auth schemes the probe headers don't satisfy).
    """
    import tools.mcp_tool as _mcp

    preflight_calls: list[str] = []

    async def _inner():
        async def _fake_preflight(self, url, **kwargs):
            preflight_calls.append(url)

        async def _fake_run_http(self, config):
            raise asyncio.CancelledError()

        monkeypatch.setattr(_mcp, "_validate_remote_mcp_url", lambda n, u: None)
        monkeypatch.setattr(_mcp.MCPServerTask, "_preflight_content_type", _fake_preflight)
        monkeypatch.setattr(_mcp.MCPServerTask, "_run_http", _fake_run_http)

        task = _mcp.MCPServerTask("skip-preflight-test")
        with pytest.raises(asyncio.CancelledError):
            await task.run({
                "url": "https://mcp.example.com/mcp",
                "skip_preflight": True,
            })

    asyncio.run(_inner())
    assert preflight_calls == [], (
        "_preflight_content_type must not be called when skip_preflight is set"
    )


# ---------------------------------------------------------------------------
# POST probe fallback for POST-only MCP servers
# ---------------------------------------------------------------------------


def test_post_probe_not_attempted_for_valid_head():
    """When HEAD already returns application/json, no POST probe is needed."""
    task = _make_task()
    record: list[str] = []
    with _serve(_handler(
        status=200, content_type="application/json", body=b"{}",
        post_content_type="application/json",
        post_body=b'{}',
        record=record,
    )) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))
    assert record == ["HEAD"]
    assert "POST" not in record
