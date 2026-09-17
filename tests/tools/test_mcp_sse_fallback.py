"""Invariant tests: automatic Streamable HTTP -> SSE transport fallback (#53676, #104343).

An SSE-only MCP server rejects the Streamable HTTP ``initialize`` POST (400-family status,
or the SDK's opaque -32603 "Server returned an error response"); the client must retry over
SSE on the initial connect only — never on reconnect after a proven session, never on
timeout, and a both-transports failure must say so actionably.
"""

import asyncio

import httpx
import pytest

from tools.mcp_tool import MCPServerTask


def _http_400(status=400):
    request = httpx.Request("POST", "http://127.0.0.1:1/mcp")
    return httpx.HTTPStatusError("Bad Request", request=request,
                                 response=httpx.Response(status, request=request))


class _SdkInternalError(Exception):
    """Shape of mcp.shared.exceptions.MCPError for an opaque initialize rejection."""

    def __init__(self):
        super().__init__("Server returned an error response")
        self.error = type("E", (), {"code": -32603})()


def _task(monkeypatch, http_exc, sse_result="shutdown", sse_exc=None):
    """MCPServerTask whose transports are recorded fakes: HTTP raises, SSE serves or raises."""
    task = MCPServerTask("t")
    task._config = {}
    calls = []

    async def fake_serve(self, cm, label, timeout):
        calls.append(label)
        if label != "SSE":
            raise http_exc
        if sse_exc is not None:
            raise sse_exc
        self._ever_connected = True
        return sse_result

    monkeypatch.setattr(MCPServerTask, "_serve_transport", fake_serve)
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_sse_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_build_oauth_auth", lambda self, *a: None)
    return task, calls


_CONFIG = {"url": "http://127.0.0.1:1/mcp", "connect_timeout": 1}


@pytest.mark.parametrize("exc", [_http_400(), _http_400(405),
                                 ExceptionGroup("g", [_SdkInternalError()])])
def test_sse_only_server_connects_via_fallback(monkeypatch, exc):
    """Initial connect: a Streamable HTTP rejection falls back to SSE and serves; the latch
    routes subsequent reconnects straight to SSE without re-trying Streamable HTTP."""
    task, calls = _task(monkeypatch, exc)
    assert asyncio.run(task._run_http(dict(_CONFIG))) == "shutdown"
    assert calls[-1] == "SSE" and len(calls) == 2
    assert asyncio.run(task._run_http(dict(_CONFIG))) == "shutdown"  # reconnect after latch
    assert calls[2:] == ["SSE"]


@pytest.mark.parametrize("exc,ever_connected", [
    (_http_400(), True),                # reconnect after a proven session: never mask the 400
    (asyncio.TimeoutError(), False),    # timeout is not a transport mismatch
    (_http_400(500), False),            # 5xx is a broken server, not SSE-only
])
def test_no_fallback_on_reconnect_timeout_or_server_error(monkeypatch, exc, ever_connected):
    task, calls = _task(monkeypatch, exc)
    task._ever_connected = ever_connected
    with pytest.raises(type(exc)):
        asyncio.run(task._run_http(dict(_CONFIG)))
    assert "SSE" not in calls


def test_both_transports_failing_names_both_and_suggests_config(monkeypatch):
    task, calls = _task(monkeypatch, _http_400(), sse_exc=ConnectionRefusedError("no sse"))
    with pytest.raises(ConnectionError, match="both Streamable HTTP and SSE.*transport: sse"):
        asyncio.run(task._run_http(dict(_CONFIG)))
    assert calls == ["HTTP", "SSE"] or calls == ["legacy HTTP", "SSE"]
    assert task._sse_fallback is False  # failed fallback must not latch
