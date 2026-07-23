"""Tests for MCP tool-handler transport-session auto-reconnect.

When a Streamable HTTP MCP server garbage-collects its server-side
session (idle TTL, server restart, pod rotation, …) it rejects
subsequent requests with a JSON-RPC error containing phrases like
``"Invalid or expired session"``.  The OAuth token remains valid —
only the transport session state needs rebuilding.

Before the #13383 fix, this class of failure fell through as a plain
tool error with no recovery path, so every subsequent call on the
affected MCP server failed until the gateway was manually restarted.
"""
import asyncio
import json
import threading
import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _is_session_expired_error — unit coverage
# ---------------------------------------------------------------------------


def test_is_session_expired_detects_invalid_or_expired_session():
    """Reporter's exact wpcom-mcp error message (#13383)."""
    from tools.mcp_tool import _is_session_expired_error
    exc = RuntimeError("Invalid params: Invalid or expired session")
    assert _is_session_expired_error(exc) is True


def test_is_session_expired_detects_expired_session_variant():
    """Generic ``session expired`` / ``expired session`` phrasings used
    by other SDK servers."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("Session expired")) is True
    assert _is_session_expired_error(RuntimeError("expired session: abc")) is True


def test_is_session_expired_detects_session_not_found():
    """Server-side GC produces ``session not found`` / ``unknown session``
    on some implementations."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("session not found")) is True
    assert _is_session_expired_error(RuntimeError("Unknown session: abc123")) is True


def test_is_session_expired_detects_session_terminated():
    """Remote Playwright MCP reports transport loss as ``Session terminated``."""
    from tools.mcp_tool import _is_session_expired_error

    assert _is_session_expired_error(RuntimeError("Session terminated")) is True


def test_is_session_expired_detects_stale_pipe_and_closed_transport_variants():
    """Stdio/AnyIO stale-pipe failures usually surface as closed-resource
    or broken-pipe text, not an HTTP session-expired JSON-RPC error."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("ClosedResourceError")) is True
    assert _is_session_expired_error(RuntimeError("closed resource in MCP child")) is True
    assert _is_session_expired_error(RuntimeError("transport is closed")) is True
    assert _is_session_expired_error(RuntimeError("Broken pipe while writing request")) is True
    assert _is_session_expired_error(RuntimeError("End of file from MCP server")) is True


def test_is_session_expired_is_case_insensitive():
    """Match uses lower-cased comparison so servers that emit the
    message in different cases (SDK formatter quirks) still trigger."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("INVALID OR EXPIRED SESSION")) is True
    assert _is_session_expired_error(RuntimeError("Session Expired")) is True


def test_is_session_expired_rejects_unrelated_errors():
    """Narrow scope: only the specific session-expired markers trigger.
    A regular RuntimeError / ValueError does not."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("Tool failed to execute")) is False
    assert _is_session_expired_error(ValueError("Missing parameter")) is False
    assert _is_session_expired_error(Exception("Connection refused")) is False
    # 401 is handled by the sibling _is_auth_error path, not here.
    assert _is_session_expired_error(RuntimeError("401 Unauthorized")) is False


def test_is_session_expired_rejects_interrupted_error():
    """InterruptedError is the user-cancel signal — must never route
    through the session-reconnect path."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(InterruptedError()) is False
    assert _is_session_expired_error(InterruptedError("Invalid or expired session")) is False


def test_is_session_expired_detects_message_less_anyio_transport_failures():
    """Recognized stream failures have no text for marker matching."""
    from anyio import BrokenResourceError, EndOfStream
    from tools.mcp_tool import _is_session_expired_error

    assert _is_session_expired_error(BrokenResourceError()) is True
    assert _is_session_expired_error(EndOfStream()) is True


def test_is_session_expired_detects_wrapped_closed_resource():
    """AnyIO task groups may wrap a message-less transport close."""
    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    exc = ExceptionGroup("MCP transport failed", [ClosedResourceError()])
    assert _is_session_expired_error(exc) is True


def test_is_session_expired_rejects_mixed_group_with_user_interruption():
    """Cancellation anywhere in the tree takes precedence over transport loss."""
    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    exc = ExceptionGroup(
        "cancelled MCP transport",
        [InterruptedError("cancel"), ClosedResourceError()],
    )
    assert _is_session_expired_error(exc) is False


def test_is_session_expired_finds_closed_resource_beyond_recursion_limit():
    """The full classifier must handle arbitrarily deep transport wrappers."""
    import sys

    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    class NestedException(Exception):
        exceptions: tuple[BaseException, ...]

    exc = ClosedResourceError()
    for _ in range(sys.getrecursionlimit() + 100):
        wrapper = NestedException("wrapped")
        wrapper.exceptions = (exc,)
        exc = wrapper

    assert _is_session_expired_error(exc) is True


def test_is_session_expired_handles_cyclic_graph_without_transport_error():
    """A cyclic non-transport graph must terminate and classify false."""
    from tools.mcp_tool import _is_session_expired_error

    class CyclicException(Exception):
        exceptions: tuple[BaseException, ...]

    first = CyclicException("first")
    second = CyclicException("second")
    first.exceptions = (second,)
    second.exceptions = (first,)

    assert _is_session_expired_error(first) is False


def test_is_session_expired_finds_transport_error_in_cyclic_graph():
    """Cycle detection must not prevent scanning reachable transport errors."""
    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    class CyclicException(Exception):
        exceptions: tuple[BaseException, ...]

    first = CyclicException("first")
    second = CyclicException("second")
    first.exceptions = (second, ClosedResourceError())
    second.exceptions = (first,)

    assert _is_session_expired_error(first) is True


def test_is_session_expired_rejects_empty_message():
    """Bare exceptions with no message shouldn't match."""
    from tools.mcp_tool import _is_session_expired_error
    assert _is_session_expired_error(RuntimeError("")) is False
    assert _is_session_expired_error(Exception()) is False


def test_is_session_expired_follows_cause_chain():
    """A transport close reachable only via ``__cause__`` must classify."""
    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    try:
        try:
            raise ClosedResourceError()
        except ClosedResourceError as inner:
            raise RuntimeError("MCP request failed") from inner
    except RuntimeError as exc:
        assert _is_session_expired_error(exc) is True


def test_is_session_expired_follows_context_chain():
    """Implicit ``__context__`` chaining must also be scanned."""
    from anyio import BrokenResourceError
    from tools.mcp_tool import _is_session_expired_error

    try:
        try:
            raise BrokenResourceError()
        except BrokenResourceError:
            raise RuntimeError("while handling transport write")
    except RuntimeError as exc:
        assert _is_session_expired_error(exc) is True


def test_is_session_expired_interruption_in_cause_chain_wins():
    """User cancellation buried in the chain overrides transport signals."""
    from anyio import ClosedResourceError
    from tools.mcp_tool import _is_session_expired_error

    root = InterruptedError("cancel")
    mid = ClosedResourceError()
    mid.__cause__ = root
    top = RuntimeError("transport is closed")
    top.__cause__ = mid
    assert _is_session_expired_error(top) is False


def test_is_session_expired_handles_cyclic_cause_context_chain():
    """Cycles through __cause__/__context__ must terminate (visited set)."""
    from tools.mcp_tool import _is_session_expired_error

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__context__ = a  # cycle back through the other link
    assert _is_session_expired_error(a) is False

    from anyio import ClosedResourceError

    c = RuntimeError("c")
    d = ClosedResourceError()
    c.__cause__ = d
    d.__context__ = c  # cycle, but transport error is reachable
    assert _is_session_expired_error(c) is True


def test_is_session_expired_traversal_is_budget_bounded():
    """Pathologically long chains stop at the node budget without spinning."""
    import tools.mcp_tool as mcp_mod
    from tools.mcp_tool import _is_session_expired_error

    exc: BaseException = RuntimeError("leaf")
    for i in range(mcp_mod._EXC_TRAVERSAL_MAX_NODES * 2):
        wrapper = RuntimeError(f"layer {i}")
        wrapper.__cause__ = exc
        exc = wrapper

    # Terminates quickly and classifies false (no transport signal within
    # budget). The exact outcome past the budget is unspecified; the
    # invariant under test is termination.
    assert _is_session_expired_error(exc) is False


# ---------------------------------------------------------------------------
# Handler integration — verify the recovery plumbing wires end-to-end
# ---------------------------------------------------------------------------


def _install_stub_server(name: str = "wpcom"):
    """Register a minimal server stub that _handle_session_expired_and_retry
    can signal via _reconnect_event, and that reports ready+session after
    the event fires."""
    from tools import mcp_tool

    mcp_tool._ensure_mcp_loop()

    server = MagicMock()
    server.name = name

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    server._ready = _ReadyAdapter()

    # _reconnect_event is called via loop.call_soon_threadsafe(…set); use
    # a threading-safe substitute.  The production reconnect path must not
    # treat the old stale session as fresh, so this test double swaps in a
    # distinct session object when reconnect is requested.
    reconnect_flag = threading.Event()

    class _EventAdapter:
        def set(self):
            reconnect_flag.set()
            old_session = server.session
            new_session = MagicMock()
            for method_name in (
                "call_tool",
                "list_resources",
                "read_resource",
                "list_prompts",
                "get_prompt",
            ):
                if hasattr(old_session, method_name):
                    setattr(new_session, method_name, getattr(old_session, method_name))
            server.session = new_session
            ready_flag.set()

    server._reconnect_event = _EventAdapter()

    # session attr must be truthy for the handler's initial check
    # (``if not server or not server.session``) and for the post-
    # reconnect readiness probe (``srv.session is not None``).
    server.session = MagicMock()
    return server, reconnect_flag


@pytest.mark.parametrize(
    "transport_config, expected_route",
    [
        ({"command": "librarian-mcp"}, "stdio"),
        ({"url": "https://neo4j.example.test/mcp", "skip_preflight": True}, "http"),
    ],
    ids=["stdio", "http"],
)
def test_call_tool_handler_rebuilds_configured_server_transport(
    monkeypatch, tmp_path, transport_config, expected_route
):
    """The real server run loop selects and rebuilds its configured transport."""
    from anyio import ClosedResourceError
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask, _make_tool_handler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mcp_tool._ensure_mcp_loop()
    transport_ready = threading.Event()
    routes = []
    configs = []
    sessions = []
    call_count = {"n": 0}

    class _Session:
        async def call_tool(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ClosedResourceError
            result = MagicMock()
            result.isError = False
            result.content = [MagicMock(type="text", text="reconnected")]
            result.structuredContent = None
            return result

    class _LifecycleTask(MCPServerTask):
        async def _serve_transport(self, route, config):
            routes.append(route)
            configs.append(dict(config))
            self.session = _Session()
            sessions.append(self.session)
            self._ready.set()
            transport_ready.set()
            return await self._wait_for_lifecycle_event()

        async def _run_stdio(self, config):
            return await self._serve_transport("stdio", config)

        async def _run_http(self, config):
            return await self._serve_transport("http", config)

    server = _LifecycleTask("resumed")
    mcp_tool._servers["resumed"] = server
    mcp_tool._server_error_counts.pop("resumed", None)
    mcp_tool._server_breaker_opened_at.pop("resumed", None)
    loop = mcp_tool._mcp_loop
    assert loop is not None
    run_future = asyncio.run_coroutine_threadsafe(
        server.run(transport_config), loop
    )

    try:
        assert transport_ready.wait(3), "server lifecycle did not establish transport"
        handler = _make_tool_handler("resumed", "health", 10.0)
        parsed = json.loads(handler({}))

        assert parsed == {"result": "reconnected"}
        assert call_count["n"] == 2
        assert routes == [expected_route, expected_route]
        assert configs == [transport_config, transport_config]
        assert len(sessions) == 2
        assert sessions[0] is not sessions[1]
    finally:
        loop.call_soon_threadsafe(server._shutdown_event.set)
        run_future.result(timeout=5)
        mcp_tool._servers.pop("resumed", None)
        mcp_tool._server_error_counts.pop("resumed", None)
        mcp_tool._server_breaker_opened_at.pop("resumed", None)


def test_call_tool_handler_reconnects_on_session_expired(monkeypatch, tmp_path):
    """Reporter's exact repro: call_tool raises "Invalid or expired
    session", handler triggers reconnect, retries once, and returns
    the retry's successful JSON (not the generic error)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    server, reconnect_flag = _install_stub_server("wpcom")
    mcp_tool._servers["wpcom"] = server
    mcp_tool._server_error_counts.pop("wpcom", None)

    # First call raises session-expired; second call (post-reconnect)
    # returns a proper MCP tool result.
    call_count = {"n": 0}

    async def _call_sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Invalid params: Invalid or expired session")
        # Second call: mimic the MCP SDK's structured success response.
        result = MagicMock()
        result.isError = False
        result.content = [MagicMock(type="text", text="tool completed")]
        result.structuredContent = None
        return result

    server.session.call_tool = _call_sequence

    try:
        handler = _make_tool_handler("wpcom", "wpcom-mcp-content-authoring", 10.0)
        out = handler({"slug": "hello"})
        parsed = json.loads(out)
        # Retry succeeded — no error surfaced to caller.
        assert "error" not in parsed, (
            f"Expected retry to succeed after reconnect; got: {parsed}"
        )
        # _reconnect_event was signalled exactly once.
        assert reconnect_flag.is_set(), (
            "Handler did not trigger transport reconnect on session-expired "
            "error — the reconnect flow is the whole point of this fix."
        )
        # Exactly 2 call attempts (original + one retry).
        assert call_count["n"] == 2, (
            f"Expected 1 original + 1 retry = 2 calls; got {call_count['n']}"
        )
    finally:
        mcp_tool._servers.pop("wpcom", None)
        mcp_tool._server_error_counts.pop("wpcom", None)


def test_session_expired_retry_waits_for_new_session(monkeypatch, tmp_path):
    """Regression for long-lived HTTP/stream MCP sessions.

    If the reconnect helper only checks ``_ready.is_set()`` and
    ``session is not None``, it can return immediately while ``session`` still
    points at the stale transport. The retry then hits the same dead session
    and the circuit breaker eventually reports the server as unreachable. The
    handler must wait for a distinct session object before retrying.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    mcp_tool._ensure_mcp_loop()
    server = MagicMock()
    server.name = "hindsight"
    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    old_session = MagicMock()

    async def _old_call(*a, **kw):
        raise RuntimeError("Session terminated")

    old_session.call_tool = _old_call
    new_session = MagicMock()

    async def _new_call(*a, **kw):
        result = MagicMock()
        result.isError = False
        result.content = [MagicMock(type="text", text="bank ok")]
        result.structuredContent = None
        return result

    new_session.call_tool = _new_call
    server.session = old_session
    server._ready = _ReadyAdapter()

    class _ReconnectAdapter:
        def set(self):
            server.session = new_session
            ready_flag.set()

    server._reconnect_event = _ReconnectAdapter()
    mcp_tool._servers["hindsight"] = server
    mcp_tool._server_error_counts["hindsight"] = 7
    # Stamp the breaker "open" far enough in the past that the cooldown has
    # provably elapsed, so this call is a half-open probe. The breaker compares
    # against time.monotonic() (tools/mcp_tool.py), whose origin is arbitrary and
    # small on a freshly-booted CI container — a hardcoded literal like 123.0
    # only looked "elapsed" on a long-uptime dev box and flaked under CI.
    mcp_tool._server_breaker_opened_at["hindsight"] = (
        time.monotonic() - mcp_tool._CIRCUIT_BREAKER_COOLDOWN_SEC - 1.0
    )

    try:
        handler = _make_tool_handler("hindsight", "get_bank", 10.0)
        parsed = json.loads(handler({}))
        assert parsed.get("result") == "bank ok", parsed
        assert mcp_tool._server_error_counts.get("hindsight", 0) == 0
        assert "hindsight" not in mcp_tool._server_breaker_opened_at
    finally:
        mcp_tool._servers.pop("hindsight", None)
        mcp_tool._server_error_counts.pop("hindsight", None)
        mcp_tool._server_breaker_opened_at.pop("hindsight", None)


def test_call_tool_handler_non_session_expired_error_falls_through(
    monkeypatch, tmp_path
):
    """Preserved-behaviour canary: a non-session-expired exception must
    NOT trigger reconnect — it must fall through to the generic error
    path so the caller sees the real failure."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    server, reconnect_flag = _install_stub_server("srv")
    mcp_tool._servers["srv"] = server
    mcp_tool._server_error_counts.pop("srv", None)

    async def _raises(*a, **kw):
        raise RuntimeError("Tool execution failed — unrelated error")

    server.session.call_tool = _raises

    try:
        handler = _make_tool_handler("srv", "mytool", 10.0)
        out = handler({"arg": "v"})
        parsed = json.loads(out)
        # Generic error path surfaced the failure.
        assert "MCP call failed" in parsed.get("error", "")
        # Reconnect was NOT triggered for this unrelated failure.
        assert not reconnect_flag.is_set(), (
            "Reconnect must not fire for non-session-expired errors — "
            "this would cause spurious transport churn on every tool "
            "failure."
        )
    finally:
        mcp_tool._servers.pop("srv", None)
        mcp_tool._server_error_counts.pop("srv", None)


def test_session_expired_handler_returns_none_without_loop(monkeypatch):
    """Defensive: if the MCP loop isn't running (cold start / shutdown
    race), the handler must fall through cleanly instead of hanging
    or raising."""
    from tools import mcp_tool
    from tools.mcp_tool import _handle_session_expired_and_retry

    # Install a server stub but make the event loop unavailable.
    server = MagicMock()
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set = MagicMock(return_value=True)
    server.session = MagicMock()
    mcp_tool._servers["srv-noloop"] = server

    monkeypatch.setattr(mcp_tool, "_mcp_loop", None)

    try:
        out = _handle_session_expired_and_retry(
            "srv-noloop",
            RuntimeError("Invalid or expired session"),
            lambda: '{"ok": true}',
            "tools/call",
        )
        assert out is None, (
            "Without an event loop, session-expired handler must fall "
            "through to caller's generic error path — not hang or raise."
        )
    finally:
        mcp_tool._servers.pop("srv-noloop", None)


def test_session_expired_handler_returns_none_without_server_record():
    """If the server has been torn down / isn't in _servers, fall
    through cleanly — nothing to reconnect to."""
    from tools.mcp_tool import _handle_session_expired_and_retry
    out = _handle_session_expired_and_retry(
        "does-not-exist",
        RuntimeError("Invalid or expired session"),
        lambda: '{"ok": true}',
        "tools/call",
    )
    assert out is None


def test_session_expired_handler_returns_none_when_retry_also_fails(
    monkeypatch, tmp_path
):
    """If the retry after reconnect also raises, fall through to the
    generic error path (don't loop forever, don't mask the second
    failure)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _handle_session_expired_and_retry

    server, _ = _install_stub_server("srv-retry-fail")
    mcp_tool._servers["srv-retry-fail"] = server

    def _retry_raises():
        raise RuntimeError("retry blew up too")

    try:
        out = _handle_session_expired_and_retry(
            "srv-retry-fail",
            RuntimeError("Invalid or expired session"),
            _retry_raises,
            "tools/call",
        )
        assert out is None, (
            "When the retry itself fails, the handler must return None "
            "so the caller's generic error path runs — no retry loop."
        )
    finally:
        mcp_tool._servers.pop("srv-retry-fail", None)


# ---------------------------------------------------------------------------
# Parallel coverage for resources/list, resources/read, prompts/list,
# prompts/get — all four handlers share the same exception path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_factory, handler_kwargs, session_method, op_label",
    [
        ("_make_list_resources_handler", {"tool_timeout": 10.0}, "list_resources", "list_resources"),
        ("_make_read_resource_handler", {"tool_timeout": 10.0}, "read_resource", "read_resource"),
        ("_make_list_prompts_handler", {"tool_timeout": 10.0}, "list_prompts", "list_prompts"),
        ("_make_get_prompt_handler", {"tool_timeout": 10.0}, "get_prompt", "get_prompt"),
    ],
)
def test_non_tool_handlers_also_reconnect_on_session_expired(
    monkeypatch, tmp_path, handler_factory, handler_kwargs, session_method, op_label
):
    """All four non-``tools/call`` MCP handlers share the recovery
    pattern and must reconnect the same way on session-expired."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    server, reconnect_flag = _install_stub_server(f"srv-{op_label}")
    mcp_tool._servers[f"srv-{op_label}"] = server
    mcp_tool._server_error_counts.pop(f"srv-{op_label}", None)

    call_count = {"n": 0}

    async def _sequence(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Invalid or expired session")
        # Return something with the shapes each handler expects.
        # Explicitly set primitive attrs — MagicMock's default auto-attr
        # behaviour surfaces ``MagicMock`` values for optional fields
        # like ``description``, which break ``json.dumps`` downstream.
        result = MagicMock()
        result.resources = []
        result.prompts = []
        result.contents = []
        result.messages = []  # get_prompt
        result.description = None  # get_prompt optional field
        return result

    setattr(server.session, session_method, _sequence)

    factory = getattr(mcp_tool, handler_factory)
    # list_resources / list_prompts take (server_name, timeout).
    # read_resource / get_prompt take the same signature.
    try:
        handler = factory(f"srv-{op_label}", **handler_kwargs)
        if op_label == "read_resource":
            out = handler({"uri": "file://foo"})
        elif op_label == "get_prompt":
            out = handler({"name": "p1"})
        else:
            out = handler({})
        parsed = json.loads(out)
        assert "error" not in parsed, (
            f"{op_label}: expected retry success, got {parsed}"
        )
        assert reconnect_flag.is_set(), (
            f"{op_label}: reconnect should fire for session-expired"
        )
        assert call_count["n"] == 2, (
            f"{op_label}: expected 1 original + 1 retry"
        )
    finally:
        mcp_tool._servers.pop(f"srv-{op_label}", None)
        mcp_tool._server_error_counts.pop(f"srv-{op_label}", None)
