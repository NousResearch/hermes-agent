"""Tests for MCP tool-handler stale-connection auto-reconnect (#90166).

A CDN / load balancer / proxy can close an idle keep-alive connection
out from under the MCP streamable-HTTP transport.  The SDK then surfaces
``MCPError: Connection closed`` or ``RemoteProtocolError: Server
disconnected without sending a response`` on the next tool call.  The
server itself is healthy — only the pooled connection is dead — so the
tool handler rebuilds the transport and retries once instead of
surfacing the transient failure to the model.

Before this fix, that error class fell through both the auth and
session-expired recovery paths (it is neither a credential problem nor a
server-side session GC) and landed as a plain tool error, so an idle
CDN kill turned into a visible ``MCPError: Connection closed`` that
self-healed only minutes later after parking.
"""

import asyncio
import json
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# _is_stale_connection_error — unit coverage
# ---------------------------------------------------------------------------


def _make_remote_protocol_error():
    """Build the exact exception class the SDK's httpx2 stack raises."""
    try:
        import httpx2
    except ImportError:  # pragma: no cover - dev extra always installs it
        pytest.skip("httpx2 not installed")
    return httpx2.RemoteProtocolError(
        "Server disconnected without sending a response"
    )


def test_is_stale_connection_detects_remote_protocol_error():
    """Reporter's exact traceback error (#90166)."""
    from tools.mcp_tool import _is_stale_connection_error
    assert _is_stale_connection_error(_make_remote_protocol_error()) is True


def test_is_stale_connection_detects_httpcore_remote_protocol_error():
    """The SDK may leak the underlying httpcore2 exception un-wrapped."""
    from tools.mcp_tool import _is_stale_connection_error
    try:
        import httpcore2
    except ImportError:  # pragma: no cover
        pytest.skip("httpcore2 not installed")
    exc = httpcore2.RemoteProtocolError(
        "Server disconnected without sending a response"
    )
    assert _is_stale_connection_error(exc) is True


def test_is_stale_connection_detects_connection_closed_message():
    """``MCPError: Connection closed`` — the SDK's user-facing wording."""
    from tools.mcp_tool import _is_stale_connection_error
    assert _is_stale_connection_error(RuntimeError("MCPError: Connection closed")) is True


def test_is_stale_connection_detects_wrapped_exception_group():
    """post_writer raises inside an anyio TaskGroup (issue traceback)."""
    from tools.mcp_tool import _is_stale_connection_error
    inner = _make_remote_protocol_error()
    eg = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    assert _is_stale_connection_error(eg) is True


def test_is_stale_connection_detects_chained_cause():
    """SDK wrappers raise a generic error *from* the transport error."""
    from tools.mcp_tool import _is_stale_connection_error
    root = _make_remote_protocol_error()
    wrapper = RuntimeError("MCP call failed")
    wrapper.__cause__ = root
    assert _is_stale_connection_error(wrapper) is True


def test_is_stale_connection_rejects_unrelated_errors():
    from tools.mcp_tool import _is_stale_connection_error
    assert _is_stale_connection_error(RuntimeError("Invalid params")) is False
    assert _is_stale_connection_error(ValueError("tool returned bad JSON")) is False
    # auth failures stay on the OAuth recovery path
    assert _is_stale_connection_error(RuntimeError("401 Unauthorized")) is False


def test_is_stale_connection_interrupt_override():
    from tools.mcp_tool import _is_stale_connection_error
    assert _is_stale_connection_error(InterruptedError("user stop")) is False


def test_is_stale_connection_traversal_is_budget_bounded():
    """Pathologically long chains stop at the node budget without spinning."""
    import tools.mcp_tool as mcp_mod
    from tools.mcp_tool import _is_stale_connection_error

    exc: BaseException = RuntimeError("leaf")
    for i in range(mcp_mod._EXC_TRAVERSAL_MAX_NODES * 2):
        wrapper = RuntimeError(f"layer {i}")
        wrapper.__cause__ = exc
        exc = wrapper
    assert _is_stale_connection_error(exc) is False


# ---------------------------------------------------------------------------
# Handler integration — recovery plumbing wires end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transport_config, expected_route",
    [
        ({"command": "cdr-mcp"}, "stdio"),
        ({"url": "https://qcc.example.test/mcp", "skip_preflight": True}, "http"),
    ],
    ids=["stdio", "http"],
)
def test_call_tool_handler_rebuilds_transport_on_stale_connection(
    monkeypatch, tmp_path, transport_config, expected_route
):
    """First call hits a dead keep-alive connection; the handler rebuilds the
    transport and retries once instead of surfacing the error."""
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask, _make_tool_handler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mcp_tool._ensure_mcp_loop()
    transport_ready = threading.Event()
    routes = []
    sessions = []
    call_count = {"n": 0}

    class _Session:
        async def call_tool(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_remote_protocol_error()
            result = type("R", (), {})()
            result.is_error = False
            result.content = [type("C", (), {})()]
            result.content[0].type = "text"
            result.content[0].text = "reconnected"
            result.structured_content = None
            return result

    class _LifecycleTask(MCPServerTask):
        async def _serve_transport(self, route, config):
            routes.append(route)
            self.session = _Session()
            sessions.append(self.session)
            self._ready.set()
            transport_ready.set()
            return await self._wait_for_lifecycle_event()

        async def _run_stdio(self, config):
            return await self._serve_transport("stdio", config)

        async def _run_http(self, config):
            return await self._serve_transport("http", config)

    server = _LifecycleTask("staleconn")
    mcp_tool._servers["staleconn"] = server
    mcp_tool._server_error_counts.pop("staleconn", None)
    mcp_tool._server_breaker_opened_at.pop("staleconn", None)
    loop = mcp_tool._mcp_loop
    assert loop is not None
    run_future = asyncio.run_coroutine_threadsafe(
        server.run(transport_config), loop
    )

    try:
        assert transport_ready.wait(3), "server lifecycle did not establish transport"
        handler = _make_tool_handler("staleconn", "lookup", 10.0)
        parsed = json.loads(handler({}))

        assert parsed == {"result": "reconnected"}
        assert call_count["n"] == 2
        assert routes == [expected_route, expected_route]
        assert len(sessions) == 2
    finally:
        loop.call_soon_threadsafe(server._shutdown_event.set)
        run_future.result(timeout=10)
        mcp_tool._servers.pop("staleconn", None)


# ---------------------------------------------------------------------------
# #91460 review follow-ups
# ---------------------------------------------------------------------------


def _spawn_reconnecting_server(mcp_tool, name, call_tool_impl):
    """Spawn a lifecycle-backed MCP server (stdio) whose ``call_tool``
    behaviour is provided by ``call_tool_impl(call_number)``.  Records the
    rebuild routes and sessions so tests can assert on reconnects.
    """
    transport_ready = threading.Event()
    routes = []
    sessions = []
    call_count = {"n": 0}

    class _Session:
        async def call_tool(self, *args, **kwargs):
            call_count["n"] += 1
            return await call_tool_impl(call_count["n"])

    class _LifecycleTask(mcp_tool.MCPServerTask):
        async def _serve_transport(self, route, config):
            routes.append(route)
            self.session = _Session()
            sessions.append(self.session)
            self._ready.set()
            transport_ready.set()
            return await self._wait_for_lifecycle_event()

        async def _run_stdio(self, config):
            return await self._serve_transport("stdio", config)

        async def _run_http(self, config):
            return await self._serve_transport("http", config)

    server = _LifecycleTask(name)
    mcp_tool._servers[name] = server
    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)
    loop = mcp_tool._mcp_loop
    assert loop is not None
    run_future = asyncio.run_coroutine_threadsafe(
        server.run({"command": "cdr-mcp"}), loop
    )
    assert transport_ready.wait(3), "server lifecycle did not establish transport"
    return server, run_future, routes, sessions, call_count


def _teardown_reconnecting_server(mcp_tool, name, server, run_future, loop):
    loop.call_soon_threadsafe(server._shutdown_event.set)
    run_future.result(timeout=10)
    mcp_tool._servers.pop(name, None)
    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)


def test_is_stale_connection_markers_can_be_disabled():
    """tools/call retries only on exact transport exception type names;
    message-marker matching stays available for the read-only resources/*
    paths (#91460 review)."""
    from tools.mcp_tool import _is_stale_connection_error
    marker_exc = RuntimeError("MCPError: Connection closed")
    assert _is_stale_connection_error(marker_exc) is True
    assert _is_stale_connection_error(
        marker_exc, allow_message_markers=False
    ) is False
    # Type-name evidence still counts with markers disabled.
    assert _is_stale_connection_error(
        _make_remote_protocol_error(), allow_message_markers=False
    ) is True


def test_call_tool_handler_does_not_retry_app_level_marker_error(
    monkeypatch, tmp_path,
):
    """An application-level failure merely *containing* a stale-connection
    marker phrase ("connection reset" — the reviewer's "connection reset
    by peer" case) must NOT trigger a transport rebuild + tools/call
    retry (#91460 review): the tool may already have partially executed,
    and re-running it would duplicate side effects."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    mcp_tool._ensure_mcp_loop()

    async def _impl(n):
        raise RuntimeError("backend connection reset unexpectedly")

    server, run_future, routes, sessions, call_count = (
        _spawn_reconnecting_server(mcp_tool, "appmarker", _impl)
    )
    try:
        handler = _make_tool_handler("appmarker", "lookup", 10.0)
        raw = handler({})
        parsed = json.loads(raw)
        assert "error" in parsed, parsed
        assert "backend connection reset" in parsed["error"], parsed
        assert call_count["n"] == 1, (
            f"app-level marker error must not retry tools/call; "
            f"got {call_count['n']} executions"
        )
        assert routes == ["stdio"], f"no transport rebuild expected: {routes}"
        assert len(sessions) == 1
    finally:
        _teardown_reconnecting_server(
            mcp_tool, "appmarker", server, run_future, mcp_tool._mcp_loop,
        )


def test_call_tool_handler_legacy_session_marker_still_retries(
    monkeypatch, tmp_path,
):
    """Pin the boundary of #91460 fix #1: "connection closed" also appears
    in the pre-existing session-expired marker list (#13383, predates the
    stale-connection PR), so an app-level error carrying *that* phrase is
    still retried once via the session-expired path.  The stale-connection
    handler itself must not add a second retry on top (type names only for
    tools/call).  This test documents the deliberately unchanged session
    behaviour; the exact executions count is the pinned contract."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    mcp_tool._ensure_mcp_loop()

    async def _impl(n):
        raise RuntimeError("backend connection closed unexpectedly")

    server, run_future, routes, sessions, call_count = (
        _spawn_reconnecting_server(mcp_tool, "sessmarker", _impl)
    )
    try:
        handler = _make_tool_handler("sessmarker", "lookup", 10.0)
        raw = handler({})
        parsed = json.loads(raw)
        assert "error" in parsed, parsed
        # Session-expired path retries once; the stale-connection path
        # must not stack a second reconnect+retry on top.
        assert call_count["n"] == 2, (
            f"expected session-path retry only (2 executions), "
            f"got {call_count['n']}"
        )
        assert routes == ["stdio", "stdio"], routes
        assert len(sessions) == 2
    finally:
        _teardown_reconnecting_server(
            mcp_tool, "sessmarker", server, run_future, mcp_tool._mcp_loop,
        )


def test_call_tool_handler_retry_budget_caps_total_executions(
    monkeypatch, tmp_path,
):
    """One user call chains the auth / session-expired / stale-connection
    recovery handlers.  Without a shared budget, a session-expired retry
    that raises a stale-connection error would reach the stale handler
    for a second reconnect+retry — up to three executions of a
    non-idempotent tool.  The shared per-invocation budget caps it at
    initial + one retry (#91460 review)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    mcp_tool._ensure_mcp_loop()

    async def _impl(n):
        if n == 1:
            # Matches the session-expired classifier via the message
            # marker, and the stale-connection classifier via the chained
            # transport exception type.
            root = _make_remote_protocol_error()
            exc = RuntimeError("closed resource")
            exc.__cause__ = root
            raise exc
        raise _make_remote_protocol_error()

    server, run_future, routes, sessions, call_count = (
        _spawn_reconnecting_server(mcp_tool, "budgetcap", _impl)
    )
    try:
        handler = _make_tool_handler("budgetcap", "lookup", 10.0)
        raw = handler({})
        parsed = json.loads(raw)
        assert "error" in parsed, parsed
        assert call_count["n"] == 2, (
            f"expected exactly 2 executions (initial + one retry), "
            f"got {call_count['n']}"
        )
        # Session-expired handler rebuilt the transport once; the stale
        # handler must not stack a second reconnect+retry on top.
        assert routes == ["stdio", "stdio"], routes
    finally:
        _teardown_reconnecting_server(
            mcp_tool, "budgetcap", server, run_future, mcp_tool._mcp_loop,
        )


def test_stale_handler_unparseable_retry_result_does_not_reset_breaker(
    monkeypatch, tmp_path,
):
    """A retry result that fails json.loads after a successful reconnect
    must NOT reset the circuit-breaker error counter (#91460 review): an
    unparseable response is exactly the half-broken state the breaker
    tracks.  The raw string is still returned to preserve behaviour."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _handle_stale_connection_and_retry

    mcp_tool._ensure_mcp_loop()

    async def _impl(n):
        raise AssertionError("unused")

    server, run_future, routes, sessions, call_count = (
        _spawn_reconnecting_server(mcp_tool, "staleconn", _impl)
    )
    mcp_tool._server_error_counts["staleconn"] = 2
    try:
        out = _handle_stale_connection_and_retry(
            "staleconn",
            _make_remote_protocol_error(),
            lambda: "this is not json",
            "tools/call lookup",
            allow_message_markers=False,
        )
        assert out == "this is not json"
        assert mcp_tool._server_error_counts.get("staleconn") == 2, (
            "unparseable retry payload must not reset the breaker"
        )

        # A well-formed success payload on a later recovery still resets.
        out2 = _handle_stale_connection_and_retry(
            "staleconn",
            _make_remote_protocol_error(),
            lambda: json.dumps({"ok": 1}),
            "tools/call lookup",
            allow_message_markers=False,
        )
        assert out2 == json.dumps({"ok": 1})
        assert mcp_tool._server_error_counts.get("staleconn") == 0
    finally:
        _teardown_reconnecting_server(
            mcp_tool, "staleconn", server, run_future, mcp_tool._mcp_loop,
        )


def test_stale_handler_error_payload_returns_none_without_reset(
    monkeypatch, tmp_path,
):
    """A retry payload carrying an "error" key is not a success: the
    handler returns None (caller's generic path bumps the breaker) and
    must not reset the counter (#91460 review)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _handle_stale_connection_and_retry

    mcp_tool._ensure_mcp_loop()

    async def _impl(n):
        raise AssertionError("unused")

    server, run_future, routes, sessions, call_count = (
        _spawn_reconnecting_server(mcp_tool, "staleconn", _impl)
    )
    mcp_tool._server_error_counts["staleconn"] = 2
    try:
        out = _handle_stale_connection_and_retry(
            "staleconn",
            _make_remote_protocol_error(),
            lambda: json.dumps({"error": "tool blew up"}),
            "tools/call lookup",
            allow_message_markers=False,
        )
        assert out is None
        assert mcp_tool._server_error_counts.get("staleconn") == 2
    finally:
        _teardown_reconnecting_server(
            mcp_tool, "staleconn", server, run_future, mcp_tool._mcp_loop,
        )


def test_stale_handler_reconnect_timeout_derives_from_deadline(
    monkeypatch, tmp_path,
):
    """The reconnect wait must be capped at the caller's remaining budget,
    not a fixed 15s (#91460 review): a 10s tool call must not stall 15s
    waiting for a slow reconnect."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _handle_stale_connection_and_retry

    mcp_tool._ensure_mcp_loop()
    captured = {}

    class _Server:
        pass

    srv = _Server()
    srv._reconnect_event = threading.Event()
    srv.session = object()
    mcp_tool._servers["tmo"] = srv

    def _fake_reconnect_and_wait(server_name, srv_arg, *, op_description, timeout):
        captured["timeout"] = timeout
        return True

    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait", _fake_reconnect_and_wait,
    )

    try:
        start = time.monotonic()
        out = _handle_stale_connection_and_retry(
            "tmo",
            _make_remote_protocol_error(),
            lambda: json.dumps({"ok": 1}),
            "tools/call t",
            allow_message_markers=False,
            deadline=start + 3.0,
        )
        assert out == json.dumps({"ok": 1})
        assert 0 < captured["timeout"] <= 3.0, captured

        captured.clear()
        out2 = _handle_stale_connection_and_retry(
            "tmo",
            _make_remote_protocol_error(),
            lambda: json.dumps({"ok": 1}),
            "tools/call t",
            allow_message_markers=False,
            deadline=None,
        )
        assert out2 == json.dumps({"ok": 1})
        assert captured["timeout"] == 15.0, captured

        captured.clear()
        out3 = _handle_stale_connection_and_retry(
            "tmo",
            _make_remote_protocol_error(),
            lambda: json.dumps({"ok": 1}),
            "tools/call t",
            allow_message_markers=False,
            deadline=start - 1.0,  # already expired
        )
        assert out3 == json.dumps({"ok": 1})
        assert captured["timeout"] == 0.0, captured
    finally:
        mcp_tool._servers.pop("tmo", None)
        mcp_tool._server_error_counts.pop("tmo", None)
        mcp_tool._server_breaker_opened_at.pop("tmo", None)
