"""A completed MCP round trip must not trip the transport circuit breaker,
even when the tool itself returns a functional/business error.

``_make_tool_handler`` used to parse the JSON result and bump
``_server_error_counts`` whenever it contained an ``"error"`` key — which is
also how ``tool_error()`` reports an ordinary functional failure (unknown
tool, bad argument, business-logic rejection, ...). That conflated "the
remote tool said no" with "the transport is down", so a chatty tool
returning ordinary errors could eventually trip a breaker meant only for
connectivity failures.

A successfully *received* response — whether it carries a functional error
or not — proves the transport round-tripped, so it must always close the
breaker. Only transport-level failures (timeouts, exceptions raised while
talking to the session) may bump it.
"""
import asyncio
import json
import threading

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl):
    """Install a fake, always-connected MCP server in the module's registry."""
    mcp_tool_module._ensure_mcp_loop()

    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session
    server._is_recycled_stdio.return_value = False

    # The handlers serialize JSON-RPC under ``async with server._rpc_lock``.
    # Use a real asyncio.Lock rather than MagicMock's auto-generated async
    # context manager, so acquire/release is genuinely exercised. It must be
    # built on the MCP loop that awaits it — asyncio primitives bind to the
    # loop of first use.
    async def _make_lock():
        return asyncio.Lock()

    server._rpc_lock = mcp_tool_module._run_on_mcp_loop(_make_lock, timeout=5)

    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _make_reconnectable(server):
    """Give a stub server the ``_ready`` / ``_reconnect_event`` handshake that
    ``_signal_reconnect_and_wait`` drives during auth and session recovery.

    Reconnecting swaps in a *distinct* session object (carrying the same
    ``call_tool``) because ``_wait_for_server_session_ready`` explicitly
    refuses to accept the pre-reconnect session as proof of readiness.
    """
    ready = threading.Event()
    ready.set()
    reconnected = threading.Event()

    class _ReadyAdapter:
        def is_set(self):
            return ready.is_set()

        def clear(self):
            ready.clear()

        def set(self):
            ready.set()

    class _ReconnectAdapter:
        def set(self):
            fresh = MagicMock()
            fresh.call_tool = server.session.call_tool
            server.session = fresh
            ready.set()
            reconnected.set()

    server._ready = _ReadyAdapter()
    server._reconnect_event = _ReconnectAdapter()
    return reconnected


def _cleanup(mcp_tool_module, name: str) -> None:
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)


def _functional_result(*, is_error: bool, text: str):
    return SimpleNamespace(
        is_error=is_error,
        content=[SimpleNamespace(text=text)],
        structured_content=None,
    )


def test_successful_response_resets_the_breaker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _call_tool(*a, **kw):
        return _functional_result(is_error=False, text="ok")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()
    mcp_tool._server_error_counts["srv"] = 2  # prior failures, still below threshold

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({})

        assert json.loads(result) == {"result": "ok"}
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_functional_error_response_is_visible_but_resets_the_breaker(monkeypatch, tmp_path):
    """The MCP round trip succeeded — the tool just declined the request.

    That still proves the transport works, so the breaker must reset,
    while the error text remains in the returned payload for the model.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _call_tool(*a, **kw):
        return _functional_result(is_error=True, text="unknown tool 'frobnicate'")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()
    mcp_tool._server_error_counts["srv"] = 2  # prior failures, still below threshold

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)

        assert "error" in parsed
        assert "frobnicate" in parsed["error"]
        assert mcp_tool._server_error_counts.get("srv", 0) == 0, (
            "a functional tool error must not be counted as a transport failure"
        )
    finally:
        _cleanup(mcp_tool, "srv")


def test_repeated_functional_errors_never_trip_the_breaker(monkeypatch, tmp_path):
    """Calling a tool that always returns a functional error must not
    accumulate toward the threshold, however many times it's called."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _call_tool(*a, **kw):
        return _functional_result(is_error=True, text="missing required argument")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 5):
            parsed = json.loads(handler({}))
            assert "error" in parsed

        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_timeout_bumps_the_breaker(monkeypatch, tmp_path):
    """A transport timeout is a real connectivity failure and must count."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _call_tool(*a, **kw):
        await asyncio.sleep(10)
        return _functional_result(is_error=False, text="unreachable")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 0.05)
        result = handler({})
        parsed = json.loads(result)

        assert "error" in parsed
        assert mcp_tool._server_error_counts.get("srv", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv")


def test_transport_exception_bumps_the_breaker(monkeypatch, tmp_path):
    """An exception raised while talking to the session (not a functional
    tool error) is a genuine transport failure and must count."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    async def _call_tool(*a, **kw):
        raise ConnectionResetError("peer closed the connection")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)

        assert "error" in parsed
        assert mcp_tool._server_error_counts.get("srv", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv")


def test_breaker_recovers_after_transport_failure_once_a_response_arrives(monkeypatch, tmp_path):
    """A transport failure bumps the counter; the very next completed round
    trip — functional error or not — must reset it back to zero."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    state = {"fail": True}

    async def _call_tool(*a, **kw):
        if state["fail"]:
            raise ConnectionResetError("peer closed the connection")
        return _functional_result(is_error=True, text="bad request")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    mcp_tool._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)

        json.loads(handler({}))
        assert mcp_tool._server_error_counts.get("srv", 0) == 1

        state["fail"] = False
        parsed = json.loads(handler({}))
        assert "error" in parsed  # functional error, still visible
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


# ---------------------------------------------------------------------------
# The same rule on the two recovered-retry paths.
#
# ``_handle_auth_error_and_retry`` and ``_handle_session_expired_and_retry``
# used to accept a retry only when its payload had no ``"error"`` key. A retry
# that round-tripped but returned a *functional* error was therefore treated as
# a failed recovery: the auth path replaced it with a ``needs_reauth`` response
# and bumped the breaker, the session path fell through to the caller's generic
# error handler which bumped the breaker and masked the payload. Both hid the
# real error from the model and counted a working transport as broken.
# ---------------------------------------------------------------------------


@pytest.fixture
def _oauth_recovery_succeeds(monkeypatch):
    """Force ``handle_401`` to report that credentials were recovered."""
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    reset_manager_for_tests()
    manager = get_manager()

    async def _handle_401(server_name, token=None):
        return True

    monkeypatch.setattr(manager, "handle_401", _handle_401)
    return manager


def test_auth_recovery_retry_with_functional_error_resets_breaker(
    monkeypatch, tmp_path, _oauth_recovery_succeeds
):
    """Auth recovery works, then the retried tool call returns a functional
    error: the model must see that error, not ``needs_reauth``, and the
    transport breaker must be reset rather than bumped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    calls = {"n": 0}

    async def _call_tool(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OAuthFlowError("401 Unauthorized")
        return _functional_result(is_error=True, text="unknown tool 'frobnicate'")

    server = _install_stub_server(mcp_tool, "srv", _call_tool)
    reconnected = _make_reconnectable(server)
    mcp_tool._server_error_counts["srv"] = 2  # prior failures, still below threshold

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        parsed = json.loads(handler({}))

        assert calls["n"] == 2, "expected the original call plus one retry"
        assert reconnected.is_set()
        assert "frobnicate" in parsed["error"], parsed
        assert "needs_reauth" not in parsed, (
            "a completed retry proves the credentials work — telling the model "
            "to re-authenticate would hide the real, functional error"
        )
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_session_recovery_retry_with_functional_error_resets_breaker(
    monkeypatch, tmp_path
):
    """Same contract on the session-expired path: reconnect succeeds, the
    retried call returns a functional error, and that error is what the
    caller gets — with the breaker reset, not bumped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    calls = {"n": 0}

    async def _call_tool(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Invalid params: Invalid or expired session")
        return _functional_result(is_error=True, text="missing required argument 'path'")

    server = _install_stub_server(mcp_tool, "srv", _call_tool)
    reconnected = _make_reconnectable(server)
    mcp_tool._server_error_counts["srv"] = 2

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        parsed = json.loads(handler({}))

        assert calls["n"] == 2, "expected the original call plus one retry"
        assert reconnected.is_set()
        assert "missing required argument 'path'" in parsed["error"], parsed
        assert "MCP call failed" not in parsed["error"], (
            "the functional error must not be replaced by the generic "
            "transport-failure message"
        )
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_retry_helpers_return_the_functional_payload_verbatim(
    monkeypatch, tmp_path, _oauth_recovery_succeeds
):
    """Both helpers hand the retry's response back untouched, so no detail
    of the server's functional error is rewritten or dropped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool

    payload = json.dumps({
        "error": "invalid arguments: 'limit' must be an integer",
        "details": {"field": "limit", "code": 422},
    })

    async def _unused(*a, **kw):  # pragma: no cover - retry_call is stubbed
        raise AssertionError("session.call_tool should not be reached")

    server = _install_stub_server(mcp_tool, "srv", _unused)
    _make_reconnectable(server)

    try:
        for helper, exc in (
            (mcp_tool._handle_auth_error_and_retry, OAuthFlowError("401")),
            (
                mcp_tool._handle_session_expired_and_retry,
                RuntimeError("Invalid or expired session"),
            ),
        ):
            mcp_tool._server_error_counts["srv"] = 3
            out = helper("srv", exc, lambda: payload, "tools/call t")
            assert out == payload, f"{helper.__name__} rewrote the payload"
            assert mcp_tool._server_error_counts.get("srv", 0) == 0, (
                f"{helper.__name__} must reset the transport breaker"
            )
    finally:
        _cleanup(mcp_tool, "srv")


def test_auth_retry_that_raises_still_reports_needs_reauth(
    monkeypatch, tmp_path, _oauth_recovery_succeeds
):
    """Unchanged behaviour for a genuinely failing retry: no response came
    back, so it stays a transport failure and the model is told to
    re-authenticate."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.client.auth import OAuthFlowError
    from tools import mcp_tool

    async def _unused(*a, **kw):  # pragma: no cover - retry_call is stubbed
        raise AssertionError("session.call_tool should not be reached")

    server = _install_stub_server(mcp_tool, "srv", _unused)
    _make_reconnectable(server)

    def _retry_call():
        raise OAuthFlowError("still failing post-reconnect")

    try:
        out = mcp_tool._handle_auth_error_and_retry(
            "srv", OAuthFlowError("initial"), _retry_call, "tools/call t",
        )
        parsed = json.loads(out)

        assert parsed.get("needs_reauth") is True, parsed
        # Reconnect reset the counter, the failing retry bumped it once.
        assert mcp_tool._server_error_counts.get("srv", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv")


def test_session_retry_that_raises_still_bumps_the_breaker(monkeypatch, tmp_path):
    """A reconnect whose retry also fails at the transport level falls
    through to the generic error path and counts as one failure."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    calls = {"n": 0}

    async def _call_tool(*a, **kw):
        calls["n"] += 1
        raise RuntimeError("Invalid or expired session")

    server = _install_stub_server(mcp_tool, "srv", _call_tool)
    _make_reconnectable(server)

    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)
        parsed = json.loads(handler({}))

        assert calls["n"] == 2, "expected the original call plus one retry"
        assert "error" in parsed
        assert mcp_tool._server_error_counts.get("srv", 0) == 1
    finally:
        _cleanup(mcp_tool, "srv")
