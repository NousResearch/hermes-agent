"""Regression tests for issue #105637.

The pinned ``mcp==2.0.0`` SDK's ``JSONRPCDispatcher.send_raw_request``
attaches ``params["_meta"] = {}`` unconditionally (SEP-414 trace-context
design keeps the key on the wire even with a no-op tracer). Strict MCP
servers — Meta's hosted Ads MCP server (``https://mcp.facebook.com/ads``) is
the first confirmed case — reject the empty object with HTTP 400
(``-32602 "_meta for Request must be a dict or null"``), making them
unreachable natively. ``_meta: null`` is rejected too: the field must be
omitted entirely.

The fix wraps the transport write stream handed to ``ClientSession`` so an
empty ``_meta`` dict is dropped from outgoing request/notification params
right before the frame reaches the wire. Populated ``_meta`` (progressToken,
W3C trace context) still flows untouched.

The helper tests drive the wrapper directly; the wiring tests drive the REAL
production paths (``MCPServerTask._serve_transport`` and ``_run_stdio``) with
fake SDK streams and assert on the stream instance actually handed to
``ClientSession``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")


# ── Fakes: transport CM + SessionMessage-shaped objects ─────────────────────

class _Frame:
    """Stand-in for ``mcp.shared.message.SessionMessage`` (duck-typed)."""

    def __init__(self, message):
        self.message = message
        self.metadata = None


class _Req:
    """Outgoing JSON-RPC request shape (``id`` + ``method`` + ``params``)."""

    def __init__(self, method: str, params=None, id=1):
        self.jsonrpc = "2.0"
        self.id = id
        self.method = method
        self.params = params


class _Notif:
    """Outgoing JSON-RPC notification shape (no ``id``)."""

    def __init__(self, method: str, params=None):
        self.jsonrpc = "2.0"
        self.method = method
        self.params = params


class _Resp:
    """Outgoing JSON-RPC response shape (``result``, no ``params``)."""

    def __init__(self, result=None):
        self.jsonrpc = "2.0"
        self.result = result


class _WriteStream:
    """Write stream capturing sent frames; async-CM so dispatcher teardown works."""

    def __init__(self):
        self.sent: list = []
        self.closed = False

    async def send(self, frame):
        self.sent.append(frame)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False


class _ReadStream:
    """Read stream that never yields (parks any receive loop)."""

    async def receive(self):
        await asyncio.sleep(3600)


class _FakeAsyncCM:
    """Minimal async context manager yielding a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_exc):
        return False


class _FakeSession:
    """Minimal ClientSession stand-in: handshake advertises no tools capability
    (so discovery skips ``tools/list``), then the caller pre-sets shutdown."""

    async def initialize(self):
        from types import SimpleNamespace
        return SimpleNamespace(capabilities=SimpleNamespace())  # no tools → skip tools/list


# ── Unit coverage for the strip helper itself ───────────────────────────────

class TestStripEmptyMetaHelper:
    @pytest.mark.asyncio
    async def test_empty_meta_dropped(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send(_Frame(_Req("initialize", params={"_meta": {}})))
        assert "_meta" not in raw.sent[0].message.params

    @pytest.mark.asyncio
    async def test_populated_meta_preserved(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send(_Frame(_Req("tools/call", params={"_meta": {"progressToken": 7}})))
        assert raw.sent[0].message.params["_meta"] == {"progressToken": 7}

    @pytest.mark.asyncio
    async def test_meta_with_trace_context_preserved(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        meta = {"traceparent": "00-abc-def-01"}
        await wrapped.send(_Frame(_Req("tools/call", params={"_meta": meta})))
        assert raw.sent[0].message.params["_meta"] == meta

    @pytest.mark.asyncio
    async def test_params_none_untouched(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send(_Frame(_Req("tools/list")))
        assert raw.sent[0].message.params is None

    @pytest.mark.asyncio
    async def test_other_params_preserved(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        params = {"name": "search", "arguments": {"q": 1}, "_meta": {}}
        await wrapped.send(_Frame(_Req("tools/call", params=params)))
        sent = raw.sent[0].message.params
        assert sent["name"] == "search" and sent["arguments"] == {"q": 1}
        assert "_meta" not in sent

    @pytest.mark.asyncio
    async def test_notification_empty_meta_dropped(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send(_Frame(_Notif("notifications/initialized", params={"_meta": {}})))
        assert "_meta" not in raw.sent[0].message.params

    @pytest.mark.asyncio
    async def test_response_frames_untouched(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send(_Frame(_Resp(result={"ok": True})))
        assert raw.sent[0].message.result == {"ok": True}
        assert not hasattr(raw.sent[0].message, "params") or raw.sent[0].message.params is None

    @pytest.mark.asyncio
    async def test_non_dict_params_untouched(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        req = _Req("odd/server", params=["not", "a", "dict"])
        await wrapped.send(_Frame(req))
        assert raw.sent[0].message.params == ["not", "a", "dict"]

    @pytest.mark.asyncio
    async def test_frame_without_message_attribute_untouched(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        await wrapped.send("raw-payload")
        assert raw.sent[0] == "raw-payload"

    @pytest.mark.asyncio
    async def test_wrapper_is_pass_through_async_cm(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        async with wrapped as w:
            assert w is wrapped
        assert raw.closed is True


# ── Real-SDK models: prove the strip fires on the actual wire path ──────────

class TestRealSdkModels:
    """Peer-review follow-up (PR #105738).

    The duck-typed fakes above always have a settable ``params``, so they
    cannot distinguish "the strip fired" from "the assignment raised and the
    fail-open fallback resent the frame with ``_meta`` still attached" — the
    capture list looks identical either way. These tests drive REAL SDK
    pydantic models, so a future SDK generation that freezes or validates
    ``params`` (making the assignment raise) fails loudly here instead of
    silently regressing to the HTTP 400 this fix exists to prevent.
    """

    def _real_sdk(self):
        try:
            from mcp.shared.message import SessionMessage
            from mcp.types import JSONRPCRequest
        except ImportError as exc:
            pytest.skip(f"SDK layout without SessionMessage/JSONRPCRequest: {exc}")
        return SessionMessage, JSONRPCRequest

    @pytest.mark.asyncio
    async def test_strip_fires_on_real_session_message(self):
        """The drop must survive contact with the SDK's own models: params is
        assigned in place on a real ``JSONRPCRequest`` inside a real
        ``SessionMessage`` — no exception, ``_meta`` gone, siblings intact."""
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream

        SessionMessage, JSONRPCRequest = self._real_sdk()
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        request = JSONRPCRequest(
            jsonrpc="2.0",
            id=7,
            method="tools/call",
            params={"name": "search", "arguments": {"q": 1}, "_meta": {}},
        )
        frame = SessionMessage(message=request)
        await wrapped.send(frame)

        assert len(raw.sent) == 1
        sent = raw.sent[0]
        assert sent is frame, "wrapper must mutate the frame in place, not rebuild it"
        assert (
            "_meta" not in sent.message.params
        ), "empty _meta reached the wire on a real SDK model"
        assert sent.message.params["name"] == "search"
        assert sent.message.params["arguments"] == {"q": 1}

    @pytest.mark.asyncio
    async def test_populated_meta_flows_on_real_session_message(self):
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream

        SessionMessage, JSONRPCRequest = self._real_sdk()
        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        meta = {"progressToken": 7, "traceparent": "00-abc-def-01"}
        request = JSONRPCRequest(
            jsonrpc="2.0",
            id=8,
            method="tools/call",
            params={"name": "search", "_meta": meta},
        )
        await wrapped.send(SessionMessage(message=request))

        assert raw.sent[0].message.params["_meta"] == meta

    @pytest.mark.asyncio
    async def test_unsetattrtable_params_fails_open_visibly(self):
        """Documented fail-open: if a model ever rejects the params assignment,
        the frame still flows — with ``_meta`` intact — so the server's own
        rejection surfaces instead of a silent drop. The two real-SDK tests
        above pin the settable case; this one pins the fallback's contract."""
        from tools.mcp_tool_transport import _strip_empty_meta_write_stream

        class _FrozenReq:
            """Params setter that always raises (pydantic ``frozen=True`` shape)."""

            def __init__(self, params):
                self.jsonrpc = "2.0"
                self.id = 1
                self.method = "tools/call"
                self._params = params

            @property
            def params(self):
                return self._params

            @params.setter
            def params(self, _value):
                raise TypeError("frozen model")

        raw = _WriteStream()
        wrapped = _strip_empty_meta_write_stream(raw)
        frame = _Frame(_FrozenReq({"name": "search", "_meta": {}}))
        await wrapped.send(frame)

        assert len(raw.sent) == 1, "fail-open must deliver the frame, not drop it"
        assert raw.sent[0] is frame
        assert raw.sent[0].message.params["_meta"] == {}


# ── Production-path coverage: the two ClientSession wiring sites ────────────

class TestSessionWiring:
    def test_serve_transport_wraps_write_stream(self):
        """``_serve_transport`` must hand ``ClientSession`` a wrapped write stream
        that strips empty ``_meta`` — the #105637 fix point for HTTP/SSE."""
        from tools import mcp_tool
        from tools.mcp_tool import MCPServerTask
        from tools.mcp_tool_transport import _MetaStrippingWriteStream

        server = MCPServerTask("wire-test")
        streams: list = []

        def _fake_client_session(read, write, **kwargs):
            streams.append((read, write))
            return _FakeAsyncCM(_FakeSession())

        async def drive():
            transport_cm = _FakeAsyncCM((_ReadStream(), _WriteStream()))
            server._shutdown_event.set()  # serve returns as soon as readiness is published
            with patch.object(mcp_tool, "ClientSession", _fake_client_session):
                await server._serve_transport(transport_cm, "http", 0.5)

        asyncio.run(drive())
        assert len(streams) == 1
        assert isinstance(streams[0][1], _MetaStrippingWriteStream)

    def test_run_stdio_wraps_write_stream(self):
        """``_run_stdio`` must hand ``ClientSession`` a wrapped write stream — the
        #105637 fix point for stdio servers."""
        from tools import mcp_tool
        from tools import mcp_tool_config as _mcp_config
        from tools.mcp_tool_transport import _MetaStrippingWriteStream

        server = mcp_tool.MCPServerTask("stdio-wire-test")
        streams: list = []

        def _fake_client_session(read, write, **kwargs):
            streams.append((read, write))
            return _FakeAsyncCM(_FakeSession())

        async def drive():
            with patch.object(mcp_tool, "stdio_client", lambda *a, **k: _FakeAsyncCM((object(), _WriteStream()))), \
                 patch.object(mcp_tool, "ClientSession", _fake_client_session), \
                 patch.object(_mcp_config, "_resolve_stdio_command", lambda c, e: (c, e)), \
                 patch.object(_mcp_config, "_write_stderr_log_header", lambda *a, **k: None), \
                 patch.object(mcp_tool, "_get_mcp_stderr_log", lambda: None), \
                 patch("tools.osv_check.check_package_for_malware", lambda *a, **k: None):
                server._shutdown_event.set()  # serve returns as soon as readiness is published
                await server._run_stdio({"command": "fake-mcp", "args": [], "connect_timeout": 0.2})

        asyncio.run(drive())
        assert len(streams) == 1
        assert isinstance(streams[0][1], _MetaStrippingWriteStream)
