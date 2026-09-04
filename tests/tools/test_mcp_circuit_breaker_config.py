"""Tests for the config-driven MCP circuit breaker settings.

The breaker's behavior is controlled by ``mcp.circuit_breaker.*`` in config.yaml
(``enabled`` / ``threshold`` / ``cooldown_seconds``), read at each check by
``tools.mcp_tool._get_circuit_breaker_config``. The handler gate lives in
``tools.mcp_tool_handlers._check_circuit_breaker`` and honors those settings.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")
from tools import mcp_tool_loop as _mcp_loop  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_stub_server(mcp_tool_module, name: str, call_tool_impl):
    """Install a fake MCP server in the module's registry."""
    import threading

    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    class _ReconnectAdapter:
        def __init__(self):
            self.set_calls = 0

        def set(self):
            self.set_calls += 1
            old_session = server.session
            new_session = MagicMock()
            if old_session is not None:
                new_session.call_tool = old_session.call_tool
            elif call_tool_impl is not None:
                new_session.call_tool = call_tool_impl
            server.session = new_session
            ready_flag.set()

        def assert_called_once(self):
            assert self.set_calls == 1, f"set() called {self.set_calls} times"

    server._reconnect_event = _ReconnectAdapter()
    server._ready = _ReadyAdapter()
    server._is_recycled_stdio.return_value = False

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


def _ok_call():
    """Return an async ``session.call_tool`` stub that yields a success result."""
    async def _call_tool_success(*a, **kw):
        result = MagicMock()
        result.is_error = False
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        result.structured_content = None
        return result
    return _call_tool_success


# ---------------------------------------------------------------------------
# _get_circuit_breaker_config
# ---------------------------------------------------------------------------


def test_get_config_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    assert mcp_tool._get_circuit_breaker_config() == (True, 3, 60.0)


def test_get_config_reads_custom_values_and_coerces_false_string(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "mcp:\n"
        "  circuit_breaker:\n"
        "    enabled: \"false\"\n"      # string must coerce to False, not stay truthy
        "    threshold: 5\n"
        "    cooldown_seconds: 120.5\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    assert mcp_tool._get_circuit_breaker_config() == (False, 5, 120.5)


def test_get_config_clamps_invalid_values(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "mcp:\n"
        "  circuit_breaker:\n"
        "    threshold: 0\n"
        "    cooldown_seconds: -5\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    enabled, threshold, cooldown = mcp_tool._get_circuit_breaker_config()
    assert enabled is True
    assert threshold == 1
    assert cooldown == 0.0


# ---------------------------------------------------------------------------
# _bump_server_error honors enabled + configured threshold
# ---------------------------------------------------------------------------


def test_bump_server_error_opens_at_config_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    monkeypatch.setattr(mcp_tool, "_get_circuit_breaker_config", lambda: (True, 2, 60.0))
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._server_breaker_opened_at.pop("srv", None)
    try:
        mcp_tool._bump_server_error("srv")
        assert "srv" not in mcp_tool._server_breaker_opened_at  # 1 < threshold 2
        mcp_tool._bump_server_error("srv")
        assert "srv" in mcp_tool._server_breaker_opened_at  # hits threshold 2
        assert mcp_tool._server_error_counts["srv"] == 2
    finally:
        mcp_tool._server_error_counts.pop("srv", None)
        mcp_tool._server_breaker_opened_at.pop("srv", None)


def test_bump_server_error_disabled_tracks_but_never_opens(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    monkeypatch.setattr(mcp_tool, "_get_circuit_breaker_config", lambda: (False, 3, 60.0))
    mcp_tool._server_error_counts.pop("srv", None)
    mcp_tool._server_breaker_opened_at.pop("srv", None)
    try:
        for _ in range(5):  # far past the threshold
            mcp_tool._bump_server_error("srv")
        assert mcp_tool._server_error_counts["srv"] == 5  # still tracked for diagnostics
        assert "srv" not in mcp_tool._server_breaker_opened_at  # never opened
    finally:
        mcp_tool._server_error_counts.pop("srv", None)
        mcp_tool._server_breaker_opened_at.pop("srv", None)


# ---------------------------------------------------------------------------
# Handler gate (_check_circuit_breaker via _make_tool_handler)
# ---------------------------------------------------------------------------


def test_handler_short_circuits_at_config_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    monkeypatch.setattr(mcp_tool, "_get_circuit_breaker_config", lambda: (True, 3, 60.0))

    call_count = {"n": 0}
    impl = _ok_call()

    async def _call_and_count(*a, **kw):
        call_count["n"] += 1
        return await impl(*a, **kw)

    _install_stub_server(mcp_tool, "srv", _call_and_count)
    _mcp_loop._ensure_mcp_loop()
    try:
        mcp_tool._server_error_counts["srv"] = 3  # at configured threshold
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]

        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert "unreachable" in parsed["error"].lower()
        assert call_count["n"] == 0, "breaker should short-circuit before cooldown elapses"
    finally:
        _cleanup(mcp_tool, "srv")


def test_handler_disabled_never_short_circuits(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    monkeypatch.setattr(mcp_tool, "_get_circuit_breaker_config", lambda: (False, 3, 60.0))

    call_count = {"n": 0}
    impl = _ok_call()

    async def _call_and_count(*a, **kw):
        call_count["n"] += 1
        return await impl(*a, **kw)

    _install_stub_server(mcp_tool, "srv", _call_and_count)
    _mcp_loop._ensure_mcp_loop()
    try:
        # Trip the breaker state far past the threshold; with enabled=false the
        # gate must still let the call through to the session.
        mcp_tool._server_error_counts["srv"] = 10
        mcp_tool._server_breaker_opened_at["srv"] = 0.0

        handler = _make_tool_handler("srv", "tool1", 10.0)
        result = handler({})
        parsed = json.loads(result)
        assert parsed.get("result") == "ok", parsed
        assert call_count["n"] == 1, "disabled breaker must not short-circuit"
    finally:
        _cleanup(mcp_tool, "srv")


def test_handler_half_opens_after_config_cooldown(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    # Custom cooldown shorter than the 60s default — proves the value comes from config.
    monkeypatch.setattr(mcp_tool, "_get_circuit_breaker_config", lambda: (True, 3, 5.0))

    call_count = {"n": 0}
    impl = _ok_call()

    async def _call_and_count(*a, **kw):
        call_count["n"] += 1
        return await impl(*a, **kw)

    _install_stub_server(mcp_tool, "srv", _call_and_count)
    _mcp_loop._ensure_mcp_loop()
    try:
        mcp_tool._server_error_counts["srv"] = 3
        fake_now = [1000.0]

        def _fake_monotonic():
            return fake_now[0]

        monkeypatch.setattr(mcp_tool.time, "monotonic", _fake_monotonic)
        mcp_tool._server_breaker_opened_at["srv"] = fake_now[0]

        handler = _make_tool_handler("srv", "tool1", 10.0)

        # Before the (custom, 5s) cooldown: short-circuit.
        result = handler({})
        parsed = json.loads(result)
        assert "error" in parsed, parsed
        assert call_count["n"] == 0

        # Advance past cooldown → next call is a half-open probe that hits the session.
        fake_now[0] += 6.0
        result = handler({})
        parsed = json.loads(result)
        assert parsed.get("result") == "ok", parsed
        assert call_count["n"] == 1

        # Probe success closes the breaker.
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")