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
from contextlib import asynccontextmanager, contextmanager

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_errors import NonMcpEndpointError


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
    from tools import mcp_tool_errors as _mcp_errors

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
        monkeypatch.setattr(_mcp_errors, "_validate_remote_mcp_url", lambda n, u: None)
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
    from tools import mcp_tool_errors as _mcp_errors

    preflight_calls: list[str] = []

    async def _inner():
        async def _fake_preflight(self, url, **kwargs):
            preflight_calls.append(url)

        async def _fake_run_http(self, config):
            raise asyncio.CancelledError()

        monkeypatch.setattr(_mcp_errors, "_validate_remote_mcp_url", lambda n, u: None)
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


def test_run_forwards_strict_redirect_headers_to_preflight(monkeypatch):
    """``strict_redirect_headers`` in server config must reach the probe — the
    boundary it promises is meaningless if the preflight leaks the configured
    headers on a redirect before the SDK client ever runs."""
    import tools.mcp_tool as _mcp
    from tools import mcp_tool_errors as _mcp_errors

    seen_kwargs: list[dict] = []

    async def _inner():
        async def _fake_preflight(self, url, **kwargs):
            seen_kwargs.append(kwargs)

        async def _fake_run_http(self, config):
            raise asyncio.CancelledError()

        monkeypatch.setattr(_mcp_errors, "_validate_remote_mcp_url", lambda n, u: None)
        monkeypatch.setattr(_mcp.MCPServerTask, "_preflight_content_type", _fake_preflight)
        monkeypatch.setattr(_mcp.MCPServerTask, "_run_http", _fake_run_http)

        task = _mcp.MCPServerTask("strict-preflight-test")
        with pytest.raises(asyncio.CancelledError):
            await task.run({
                "url": "https://mcp.example.com/mcp",
                "strict_redirect_headers": True,
            })

    asyncio.run(_inner())
    assert seen_kwargs and seen_kwargs[0].get("strict_redirect_headers") is True


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


# ---------------------------------------------------------------------------
# Redirect header scoping: credentials must not leave the configured origin
# ---------------------------------------------------------------------------


def _redirect_handler(target_base: str, record=None):
    """HEAD/GET/POST all 302 to ``target_base/mcp``."""

    class _H(http.server.BaseHTTPRequestHandler):
        def _redir(self):
            if record is not None:
                record.append(dict(self.headers))
            self.send_response(302)
            self.send_header("Location", f"{target_base}/mcp")
            self.end_headers()

        do_HEAD = do_GET = do_POST = _redir

        def log_message(self, format, *args):  # noqa: A002
            pass

    return _H


def _recording_mcp_handler(seen: dict):
    """200 application/json; captures request headers into ``seen``."""

    class _H(http.server.BaseHTTPRequestHandler):
        def _write_ok(self):
            seen.update({k.lower(): v for k, v in self.headers.items()})
            payload = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        do_HEAD = do_GET = do_POST = _write_ok

        def log_message(self, format, *args):  # noqa: A002
            pass

    return _H


def test_preflight_strips_credential_headers_on_cross_origin_redirect():
    """A configured ``Authorization``/``X-API-Key`` header must not reach a
    redirect target on another origin — the probe client previously followed
    redirects with no hook, forwarding every configured header verbatim."""
    task = _make_task()
    seen: dict = {}
    with _serve(_recording_mcp_handler(seen)) as target, \
            _serve(_redirect_handler(target)) as origin:
        asyncio.run(task._preflight_content_type(
            f"{origin}/mcp",
            headers={"Authorization": "Bearer s3cr3t", "X-API-Key": "k3y"},
            strict_redirect_headers=True,
            timeout=5.0,
        ))
    assert "s3cr3t" not in seen.values() and "k3y" not in seen.values()
    assert "authorization" not in seen and "x-api-key" not in seen


def test_preflight_keeps_headers_on_same_origin_redirect():
    """Relative redirects stay on the configured origin — headers must arrive."""
    task = _make_task()
    seen: dict = {}
    origin_hits: list = []

    class _Origin(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            if self.path == "/mcp":
                seen.update({k.lower(): v for k, v in self.headers.items()})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
            else:
                origin_hits.append(dict(self.headers))
                self.send_response(302)
                self.send_header("Location", "/mcp")
                self.end_headers()

        def log_message(self, format, *args):  # noqa: A002
            pass

    with _serve(_Origin) as origin:
        asyncio.run(task._preflight_content_type(
            f"{origin}/old",
            headers={"Authorization": "Bearer s3cr3t"},
            timeout=5.0,
        ))
    assert seen.get("authorization") == "Bearer s3cr3t"


def test_preflight_non_strict_forwards_configured_headers():
    """Non-strict is the documented compat contract: configured headers follow
    redirects (gateways that bounce to a same-service regional host). Only
    ``Authorization`` is withheld — matching the transport client's policy."""
    task = _make_task()
    seen: dict = {}
    with _serve(_recording_mcp_handler(seen)) as target, \
            _serve(_redirect_handler(target)) as origin:
        asyncio.run(task._preflight_content_type(
            f"{origin}/mcp",
            headers={"Authorization": "Bearer s3cr3t", "X-API-Key": "k3y"},
            strict_redirect_headers=False,
            timeout=5.0,
        ))
    assert seen.get("x-api-key") == "k3y"          # configured headers forward
    assert "authorization" not in seen              # bearer never leaves origin


def test_preflight_multi_hop_redirect_stays_stripped():
    """origin -> B -> C: once headers are stripped at the first cross-origin hop
    they must not reappear on a later hop (and are not restored if a redirect
    chains back toward the configured origin)."""
    task = _make_task()
    seen_b: dict = {}
    seen_c: dict = {}
    with _serve(_recording_mcp_handler(seen_c)) as c_base:
        class _B(http.server.BaseHTTPRequestHandler):
            def do_HEAD(self):
                seen_b.update({k.lower(): v for k, v in self.headers.items()})
                self.send_response(302)
                self.send_header("Location", f"{c_base}/mcp")
                self.end_headers()
            def log_message(self, format, *args):  # noqa: A002
                pass

        with _serve(_B) as b_base, _serve(_redirect_handler(b_base)) as origin:
            asyncio.run(task._preflight_content_type(
                f"{origin}/mcp",
                headers={"Authorization": "Bearer s3cr3t", "X-API-Key": "k3y"},
                strict_redirect_headers=True,
                timeout=5.0,
            ))
    assert "authorization" not in seen_b and "x-api-key" not in seen_b
    assert "authorization" not in seen_c and "x-api-key" not in seen_c


# ---------------------------------------------------------------------------
# Streamable HTTP transport client: same boundary on the real connection
# ---------------------------------------------------------------------------


def test_streamable_http_client_strips_credentials_on_redirect(monkeypatch):
    """Drive the REAL ``_streamable_http_transport`` construction: the client the
    SDK receives must be the stripping subclass — proven against a live redirect
    pair. Only the SDK's streams context manager is faked (a full MCP handshake
    is out of scope); the httpx client, its kwargs and the strip are all real."""
    import tools.mcp_tool as _mcp

    task = _make_task()
    seen: dict = {}
    captured: dict = {}

    @asynccontextmanager
    async def _fake_streams(url, http_client=None, **kwargs):
        captured["client"] = http_client
        yield (None, None, None)

    monkeypatch.setattr(_mcp, "streamable_http_client", _fake_streams)

    with _serve(_recording_mcp_handler(seen)) as target, \
            _serve(_redirect_handler(target)) as origin:
        async def _drive():
            cm = task._streamable_http_transport(
                f"{origin}/mcp",
                {"Authorization": "Bearer s3cr3t", "X-API-Key": "k3y"},
                connect_timeout=5.0, ssl_verify=True, client_cert=None,
                oauth_auth=None, strict_cfg_headers=True,
                configured_header_names={"x-api-key"})
            async with cm:
                await captured["client"].get(f"{origin}/mcp")

        asyncio.run(_drive())
    assert "authorization" not in seen and "x-api-key" not in seen
