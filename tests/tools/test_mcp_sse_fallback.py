"""Invariant tests: automatic Streamable HTTP -> SSE transport fallback (#53676, #104343).

An SSE-only MCP server rejects the Streamable HTTP ``initialize`` POST (400-family status,
or the SDK's opaque -32603 "Server returned an error response"); the client must retry over
SSE on the initial connect only — never on reconnect after a proven session, never on
timeout, never on a recorded 5xx (a server fault, not a transport mismatch), and a
both-transports failure must say so actionably.
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


def test_opaque_sdk_rejection_is_reported_with_the_servers_status_and_body(monkeypatch, caplog):
    """mcp >= 2.0 folds a non-JSON 4xx into ``-32603 Server returned an error response``; the
    warning and the both-transports ConnectionError must still name the HTTP status, the URL that
    was requested and the body the server sent (#114350, #113359). A root that already carries the
    status (httpx ``HTTPStatusError``) is left alone — no duplicated detail."""
    from tools.mcp_tool_errors import _make_http_rejection_recorder
    from tools.mcp_tool import sdk_httpx

    httpx2 = sdk_httpx()
    body = '{"jsonrpc":"2.0","error":{"code":-32020,"message":"Unsupported MCP-Protocol-Version"}}'

    async def _real_client_roundtrip(status, content_type):
        """The recorder on a real SDK-httpx client, through the streaming API the SDK uses; the
        SDK's own later ``aread()`` must still see the bytes."""
        sink: dict = {}
        transport = httpx2.MockTransport(
            lambda req: httpx2.Response(status, text=body, headers={"content-type": content_type}))
        async with httpx2.AsyncClient(transport=transport, event_hooks={
                "response": [_make_http_rejection_recorder(sink)]}) as client:
            async with client.stream("POST", "http://127.0.0.1:1/mcp", json={}) as resp:
                assert (await resp.aread()).decode() == body
        return sink

    assert asyncio.run(_real_client_roundtrip(200, "application/json")) == {}  # 2xx: nothing recorded
    recorded = asyncio.run(_real_client_roundtrip(400, "text/plain; charset=utf-8"))
    assert recorded == {"status": 400, "method": "POST", "url": "http://127.0.0.1:1/mcp", "body": body}

    def _connect_sees(rejection):
        task, _calls = _task(monkeypatch, ExceptionGroup("g", [_SdkInternalError()]),
                             sse_exc=ConnectionRefusedError("no sse"))
        monkeypatch.setattr(MCPServerTask, "_streamable_http_transport",
                            lambda self, *a, **k: self._http_rejection.update(rejection) or object())
        with pytest.raises(ConnectionError) as info:
            asyncio.run(task._run_http(dict(_CONFIG)))
        return str(info.value)

    with caplog.at_level("WARNING", logger="tools.mcp_tool"):
        message = _connect_sees(recorded)
    detail = "Server returned an error response (HTTP 400 from POST http://127.0.0.1:1/mcp: " + body + ")"
    assert detail in message and "SSE: no sse" in message
    assert any(detail in rec.getMessage() for rec in caplog.records), caplog.text

    # No rejection observed (the hook never fired): the SDK text stands alone, no fabricated detail.
    assert "Streamable HTTP: Server returned an error response; SSE" in _connect_sees({})


def test_opaque_rejection_without_fallback_surfaces_the_status(monkeypatch):
    """After a proven session the SSE fallback is off; the opaque SDK error still leaves ``_run_http``
    naming the recorded rejection instead of the bare ``Server returned an error response``."""
    task, calls = _task(monkeypatch, ExceptionGroup("g", [_SdkInternalError()]))
    task._ever_connected = True
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", lambda self, *a, **k: self._http_rejection.update(
        status=503, method="POST", url="http://127.0.0.1:1/mcp", body="upstream down") or object())
    with pytest.raises(ConnectionError, match=r"HTTP 503 from POST http://127\.0\.0\.1:1/mcp: upstream down"):
        asyncio.run(task._run_http(dict(_CONFIG)))
    assert "SSE" not in calls


def test_opaque_5xx_stays_on_streamable_http_instead_of_falling_back(monkeypatch):
    """mcp >= 2.0 folds a transient 5xx into the same opaque -32603 as a genuine transport
    rejection; the recorder's 5xx status must keep the failure on Streamable HTTP (the
    transient retry ladder) instead of latching a doomed SSE fallback for a
    Streamable-HTTP-only server (#121933)."""
    task, calls = _task(monkeypatch, ExceptionGroup("g", [_SdkInternalError()]))
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", lambda self, *a, **k: self._http_rejection.update(
        status=503, method="POST", url="http://127.0.0.1:1/mcp", body="upstream connect error") or object())
    with pytest.raises(ConnectionError, match=r"Streamable HTTP connect failed \(.*HTTP 503 from POST"):
        asyncio.run(task._run_http(dict(_CONFIG)))
    assert "SSE" not in calls
    assert task._sse_fallback is False  # a server fault must not latch the SSE path
