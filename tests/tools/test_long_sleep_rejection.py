"""Salvage of #8612 by @SHL0MS — long foreground sleep must not hang the agent."""

from __future__ import annotations

import json

from tools.terminal_tool import _long_foreground_sleep_rejection, terminal_tool


def test_sleep_over_30s_rejected_in_helper():
    err = _long_foreground_sleep_rejection("sleep 180", background=False)
    assert err is not None
    assert "sleep 180" in err
    assert "background=true" in err


def test_short_sleep_allowed_in_helper():
    assert _long_foreground_sleep_rejection("sleep 30", background=False) is None
    assert _long_foreground_tool_sleep_ok() is True


def _long_foreground_tool_sleep_ok() -> bool:
    return _long_foreground_sleep_rejection("  sleep 5  ", background=False) is None


def test_background_long_sleep_allowed_in_helper():
    assert _long_foreground_sleep_rejection("sleep 600", background=True) is None


def test_non_pure_sleep_not_blocked():
    # Keep path for readiness scripts using sleep as part of a pipeline.
    assert _long_foreground_sleep_rejection("sleep 120 && echo ok", background=False) is None


def test_terminal_tool_rejects_long_foreground_sleep(monkeypatch):
    # Avoid creating real environments if rejection fails.
    called = {"env": False}

    def _boom(*_a, **_k):
        called["env"] = True
        raise AssertionError("should not create env for rejected sleep")

    monkeypatch.setattr("tools.terminal_tool._get_or_create_env", _boom, raising=False)
    # Prefer patching create path inside module if present
    import tools.terminal_tool as tt

    if hasattr(tt, "_get_or_create_env"):
        monkeypatch.setattr(tt, "_get_or_create_env", _boom)

    raw = terminal_tool(command="sleep 90", background=False)
    data = json.loads(raw)
    assert data.get("exit_code") == -1
    assert "sleep 90" in data.get("error", "")
    assert "Rejected" in data.get("error", "")
