"""Tests for MCPServerTask._preflight_content_type fast-fail behaviour.

These drive the REAL ``_preflight_content_type`` method against a real local
HTTP server (via httpx's ASGI/transport plumbing through a stdlib server),
rather than reimplementing the probe inline. That distinction matters: the
production probe must run on its own httpx client outside the MCP SDK's anyio
task group, and a faithful test must exercise that actual method so the
content-type allow-list, HEAD->GET fallback, POST confirmation, and
best-effort pass-through are all covered as shipped.

Contract note (POST confirmation): MCP Streamable-HTTP is a POST protocol.
Some real MCP servers (e.g. Attio) serve an HTML marketing page on
``GET /mcp`` while answering ``POST /mcp`` with proper MCP JSON or an OAuth
challenge. The probe therefore must NOT reject on a non-MCP GET content type
alone -- it confirms with a POST first, and only rejects when the POST also
returns a clean 2xx non-MCP response (a genuine "this is a web page" signal).
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
             post_status: "int | None" = None,
             post_content_type: "str | None" = "__mirror__",
             post_body: bytes = b"{}"):
    """Build a BaseHTTPRequestHandler that replies with the given shape.

    ``head_status`` lets HEAD return a different status than GET (to exercise
    the HEAD->GET fallback). ``record`` is an optional list that captures the
    HTTP methods the server actually saw.

    The ``post_*`` knobs control the POST response used by the POST-confirmation
    step. By default POST mirrors a non-MCP web page (same status/content type
    as GET) so that a non-MCP GET is confirmed-and-rejected. Set
    ``post_content_type`` to an MCP type (or ``post_status`` to a 4xx auth
    challenge) to model a real MCP endpoint that only answers POST. Pass
    ``post_content_type=None`` to send a POST response with NO content-type
    header at all (the "mirror" sentinel is what makes ``None`` distinct from
    the default).
    """

    _post_status = post_status if post_status is not None else status
    _post_ct = content_type if post_content_type == "__mirror__" else post_content_type

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
            # Drain the request body so the client isn't left hanging.
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length:
                self.rfile.read(length)
            self._write(_post_status, _post_ct, post_body)

        def log_message(self, format, *args):  # noqa: A002
            pass

    return _H


# ---------------------------------------------------------------------------
# Reject: non-MCP content types confirmed by a clean 2xx non-MCP POST
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content_type", [
    "text/html; charset=utf-8",
    "text/html",
    "text/plain",
    "application/xml",
    "text/HTML",  # case-insensitivity
])
def test_non_mcp_content_type_raises(content_type):
    """GET returns non-MCP HTML AND POST also returns a clean 2xx web page ->
    genuine wrong-URL, must raise."""
    task = _make_task("bad_srv")
    with _serve(_handler(status=200, content_type=content_type)) as base:
        with pytest.raises(NonMcpEndpointError) as exc_info:
            asyncio.run(task._preflight_content_type(f"{base}/", timeout=5.0))
    msg = str(exc_info.value)
    assert "bad_srv" in msg
    assert "application/json" in msg and "text/event-stream" in msg


def test_non_mcp_error_is_non_retryable_connection_error():
    """NonMcpEndpointError must subclass ConnectionError (retry loop skips it
    via an explicit except; broad ConnectionError catchers still work)."""
    assert issubclass(NonMcpEndpointError, ConnectionError)


# ---------------------------------------------------------------------------
# POST confirmation: HTML on GET but a real MCP endpoint on POST must PASS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("post_content_type", [
    "application/json",
    "application/json; charset=utf-8",
    "text/event-stream",
])
def test_html_get_but_mcp_post_passes(post_content_type):
    """Attio-style: GET /mcp serves an HTML landing page, POST /mcp answers
    with an MCP content type. Must NOT raise."""
    task = _make_task()
    record: list[str] = []
    with _serve(_handler(
        status=200, content_type="text/html",
        post_status=200, post_content_type=post_content_type,
        record=record,
    )) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))
    assert "POST" in record  # the confirmation POST actually ran


@pytest.mark.parametrize("post_status", [400, 401, 403, 404, 500])
def test_html_get_but_post_challenge_passes(post_status):
    """GET serves HTML, POST returns an auth/error challenge (non-2xx) -> the
    URL is a real MCP endpoint that just needs credentials/a session. Pass."""
    task = _make_task()
    record: list[str] = []
    with _serve(_handler(
        status=200, content_type="text/html",
        post_status=post_status, post_content_type="text/html",
        record=record,
    )) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))
    assert "POST" in record


def test_html_get_but_post_no_content_type_passes():
    """GET HTML, POST returns 2xx with no content type -> ambiguous, pass."""
    task = _make_task()
    with _serve(_handler(
        status=200, content_type="text/html",
        post_status=200, post_content_type=None, post_body=b"",
    )) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


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
        # Must not raise. GET already looks like MCP -> no POST needed.
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


def test_missing_content_type_passes():
    task = _make_task()
    with _serve(_handler(status=200, content_type=None, body=b"")) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_non_2xx_responses_pass(status):
    """4xx/5xx on the GET are auth challenges or transient errors — let the SDK
    handle. The GET probe returns early without a POST confirmation."""
    task = _make_task()
    with _serve(_handler(status=status, content_type="text/html")) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))


def test_network_error_passes():
    """A connection failure (nothing listening) must pass through, not raise."""
    task = _make_task()
    # Reserve a port then close it so the connection is refused.
    s = socketserver.TCPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    dead_port = s.server_address[1]
    s.server_close()
    asyncio.run(
        task._preflight_content_type(
            f"http://127.0.0.1:{dead_port}/mcp", timeout=2.0
        )
    )


def test_cancelled_error_is_not_swallowed():
    """The best-effort except must NOT catch CancelledError (BaseException)."""
    task = _make_task()

    async def _run():
        import httpx
        orig = httpx.AsyncClient
        try:
            # Patch the client so entering it raises CancelledError.
            class _C(orig):
                async def __aenter__(self):
                    raise asyncio.CancelledError()

            httpx.AsyncClient = _C
            with pytest.raises(asyncio.CancelledError):
                await task._preflight_content_type("http://x/mcp", timeout=1.0)
        finally:
            httpx.AsyncClient = orig

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# HEAD -> GET fallback
# ---------------------------------------------------------------------------

def test_head_405_falls_back_to_get_and_rejects_html():
    """HEAD 405 -> GET HTML, and POST also a clean 2xx web page -> reject."""
    task = _make_task("fallback_srv")
    record: list[str] = []
    with _serve(_handler(
        status=200, content_type="text/html",
        head_status=405, record=record,
    )) as base:
        with pytest.raises(NonMcpEndpointError):
            asyncio.run(task._preflight_content_type(f"{base}/", timeout=5.0))
    # HEAD falls back to GET, then the non-MCP GET triggers a POST confirmation.
    assert record == ["HEAD", "GET", "POST"]


def test_head_501_falls_back_to_get_and_passes_json():
    task = _make_task()
    record: list[str] = []
    with _serve(_handler(
        status=200, content_type="application/json", body=b"{}",
        head_status=501, record=record,
    )) as base:
        asyncio.run(task._preflight_content_type(f"{base}/mcp", timeout=5.0))
    # GET already looks like MCP -> no POST confirmation needed.
    assert record == ["HEAD", "GET"]


# ---------------------------------------------------------------------------
# ssl_verify / client_cert forwarding to the probe client
# ---------------------------------------------------------------------------

def test_ssl_verify_and_cert_forwarded(monkeypatch):
    captured: dict = {}

    import httpx

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def head(self, url, headers=None):
            return httpx.Response(200, headers={"content-type": "application/json"})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    task = _make_task()
    asyncio.run(task._preflight_content_type(
        "https://mcp.example.com/mcp",
        ssl_verify=False,
        client_cert="/path/to/cert.pem",
        timeout=3.0,
    ))
    assert captured.get("verify") is False
    assert captured.get("cert") == "/path/to/cert.pem"
    assert captured.get("follow_redirects") is True
