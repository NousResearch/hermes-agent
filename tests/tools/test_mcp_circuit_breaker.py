"""Tests for MCP tool-handler circuit-breaker recovery.

The circuit breaker in ``tools/mcp_tool.py`` is intended to short-circuit
calls to an MCP server that has failed ``_CIRCUIT_BREAKER_THRESHOLD``
consecutive times, then *transition back to a usable state* once the
server has had time to recover (or an explicit reconnect succeeds).

The original implementation only had two states — closed and open — with
no mechanism to transition back to closed, so a tripped breaker stayed
tripped for the lifetime of the process. These tests lock in the
half-open / cooldown / reconnect-resets-breaker behavior that fixes
that.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl):
    """Install a fake MCP server in the module's registry.

    ``call_tool_impl`` is an async function stored at ``session.call_tool``
    (it's what the tool handler invokes).
    """
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session
    server._reconnect_event = MagicMock()
    server._ready = MagicMock()
    server._ready.is_set.return_value = True

    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name: str) -> None:
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    if hasattr(mcp_tool_module, "_server_breaker_opened_at"):
        mcp_tool_module._server_breaker_opened_at.pop(name, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_circuit_breaker_half_opens_after_cooldown(monkeypatch, tmp_path):
    """After a tripped breaker's cooldown elapses, the *next* call must
    actually execute against the session (half-open probe). When the
    probe succeeds, the breaker resets to fully closed.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_success(*a, **kw):
        call_count["n"] += 1
        result = MagicMock()
        result.isError = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structuredContent = None
        return result

    _install_stub_server(mcp_tool, "srv", _call_tool_success)
    mcp_tool._ensure_mcp_loop()

    try:
        # Trip the breaker by setting the count at/above threshold and
        # stamping the open-time to "now".
        mcp_tool._server_error_counts["srv"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        # The breaker-open timestamp dict is introduced by the fix; on
        # a pre-fix build it won't exist, which will cause the test to
        # fail at the .get() inside the gate (correct — the fix is
        # required for this state to be tracked at all).
        if hasattr(mcp_tool, "_server_breaker_opened_at"):
            mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]
        cooldown = getattr(mcp_tool, "_CIRCUIT_BREAKER_COOLDOWN_SEC", 60.0)

        handler = _make_tool_handler("srv", "tool1", 10.0)

        # Before cooldown: must short-circuit (no session call).
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert "unreachable" in parsed["error"].lower()
        assert call_count["n"] == 0, (
            "breaker should short-circuit before cooldown elapses"
        )

        # Advance past cooldown → next call is a half-open probe that
        # actually hits the session.
        fake_now[0] += cooldown + 1.0

        result = handler({})
        parsed = json.loads(result)
        assert parsed.get("result") == "ok", parsed
        assert call_count["n"] == 1, "half-open probe should invoke session"

        # On probe success the breaker must close (count reset to 0).
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")


def test_circuit_breaker_reopens_on_probe_failure(monkeypatch, tmp_path):
    """If the half-open probe fails, the breaker must re-arm the
    cooldown (not let every subsequent call through).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    call_count = {"n": 0}

    async def _call_tool_fails(*a, **kw):
        call_count["n"] += 1
        raise RuntimeError("still broken")

    _install_stub_server(mcp_tool, "srv", _call_tool_fails)
    mcp_tool._ensure_mcp_loop()

    try:
        mcp_tool._server_error_counts["srv"] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        if hasattr(mcp_tool, "_server_breaker_opened_at"):
            mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]
        cooldown = getattr(mcp_tool, "_CIRCUIT_BREAKER_COOLDOWN_SEC", 60.0)

        handler = _make_tool_handler("srv", "tool1", 10.0)

        # Advance past cooldown, run probe, expect failure.
        fake_now[0] += cooldown + 1.0
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed
        assert call_count["n"] == 1, "probe should invoke session once"

        # The probe failure must have re-armed the cooldown — another
        # immediate call should short-circuit, not invoke session again.
        result = handler({})
        parsed = json.loads(result)
        assert "unreachable" in parsed.get("error", "").lower()
        assert call_count["n"] == 1, (
            "breaker should re-open and block further calls after probe failure"
        )
    finally:
        _cleanup(mcp_tool, "srv")


def test_repeated_oauth_recovery_failures_trip_breaker(monkeypatch, tmp_path):
    """A recoverable 401 is not a success signal until the retry succeeds."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from mcp.client.auth import OAuthFlowError

    reset_manager_for_tests()

    async def _call_tool_unused(*a, **kw):  # pragma: no cover
        raise AssertionError("session.call_tool should not be reached in this test")

    _install_stub_server(mcp_tool, "srv", _call_tool_unused)
    mcp_tool._ensure_mcp_loop()

    mgr = get_manager()

    async def _h401(name, token=None):
        return True

    monkeypatch.setattr(mgr, "handle_401", _h401)

    try:
        def _retry_call():
            raise OAuthFlowError("still failing after recovery was deemed viable")

        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
            result = mcp_tool._handle_auth_error_and_retry(
                "srv",
                OAuthFlowError("initial"),
                _retry_call,
                "tools/call test",
            )
            parsed = json.loads(result)
            assert parsed.get("needs_reauth") is True, parsed

        count = mcp_tool._server_error_counts.get("srv", 0)
        assert count >= mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        assert "srv" in mcp_tool._server_breaker_opened_at
    finally:
        _cleanup(mcp_tool, "srv")
