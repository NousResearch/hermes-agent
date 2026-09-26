"""Materialized tool-schema arguments on an ordinary foreground call (#119196).

A provider that materializes every advertised ``terminal`` property sends
``background=false, notify=false, heartbeat=60`` (60 was the only schema-valid
number) for a plain ``pwd``. That shape has no foreground meaning, so it must
execute once instead of coming back as a validation error the model repeats —
or worse, as an invitation to move a short command to ``background=true`` and
flood the session with completion notices.
"""
import json

import pytest

# The complete schema-materialized argument object reported in #119196.
MATERIALIZED_FOREGROUND_CALL = {
    "command": "pwd",
    "background": False,
    "notify": False,
    "heartbeat": 60,
    "pty": False,
    "timeout": 20,
}


def _capture_terminal(monkeypatch):
    """Route the real handler's spawn through a recorder instead of a process."""
    from tools import terminal_tool as tt

    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "captured", "session_id": None, "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    return tt, captured


def _dispatch(args, **kwargs):
    from tools.registry import registry

    raw = registry.dispatch("terminal", dict(args), **kwargs)
    return json.loads(raw) if isinstance(raw, str) else raw


def test_materialized_foreground_call_executes_once_without_armed_notifications(monkeypatch):
    """The literal #119196 shape reaches terminal dispatch, in the foreground,
    with nothing notification-shaped left on the call."""
    _, captured = _capture_terminal(monkeypatch)

    result = _dispatch(MATERIALIZED_FOREGROUND_CALL)

    assert not result.get("error"), result
    assert result.get("exit_code") == 0
    # No tracked process, no completion notice, no heartbeat interval armed.
    assert captured["background"] is False
    assert captured["heartbeat"] == 0
    assert captured["notify_on_complete"] is False
    assert captured["watch_patterns"] is None
    # The rest of the materialized shape survives dispatch untouched.
    assert captured["timeout"] == 20
    assert captured["pty"] is False


def test_terminal_schema_heartbeat_admits_a_disabled_value():
    """``0`` must be schema-valid (and the advertised default): a schema whose
    only legal number is ``60`` forces a materializing provider to request a
    heartbeat it never meant to ask for."""
    from tools.registry import registry

    heartbeat = registry.get_entry("terminal").schema["parameters"]["properties"]["heartbeat"]

    assert heartbeat["type"] == "integer"
    assert heartbeat["minimum"] == 0
    assert heartbeat["default"] == 0


def test_foreground_notify_still_refuses_with_the_corrected_call(monkeypatch):
    """Notification *intent* on a foreground call keeps its teaching error —
    only the meaningless heartbeat field is normalized away."""
    _, captured = _capture_terminal(monkeypatch)

    result = _dispatch({"command": "pwd", "background": False, "notify": True})

    assert "background" in result.get("error", "")
    assert "background=true" in result["error"]
    assert not captured


def test_negative_heartbeat_reports_the_non_negative_contract(monkeypatch):
    """With ``0`` as the schema floor, a negative value is the invalid one —
    the error has to say so instead of pointing at the 60s runtime clamp."""
    _capture_terminal(monkeypatch)

    result = _dispatch({"command": "pwd", "background": True, "heartbeat": -1})

    assert "non-negative" in result.get("error", "")
    assert not _dispatch({"command": "pwd", "background": True, "heartbeat": 0}).get("error")


@pytest.mark.platforms("linux")
def test_materialized_foreground_call_really_runs_the_command(tmp_path, monkeypatch):
    """End-to-end form of the issue repro: the reported shape executes the
    command once, in the foreground, and never starts a tracked session."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = _dispatch(MATERIALIZED_FOREGROUND_CALL, task_id="t-119196")

    assert not result.get("error"), result
    assert result["exit_code"] == 0
    assert result["output"].strip()
    assert not result.get("session_id")
