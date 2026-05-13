"""Tests for tools/runtime_tool.py — runtime trace inspection tool."""

import json


def test_runtime_inspect_registered_in_debugging_toolset():
    import tools.runtime_tool  # noqa: F401 - registers tool
    from tools.registry import registry
    from toolsets import resolve_toolset

    entry = registry.get_entry("runtime_inspect")
    assert entry is not None
    assert entry.toolset == "debugging"
    assert "runtime_inspect" in resolve_toolset("debugging")


def test_runtime_inspect_returns_recent_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event
    from tools.runtime_tool import runtime_inspect

    emit_runtime_event("assign_agent.resolved", session_id="inspect-session", agent={"name": "a"})
    emit_runtime_event("assign_agent.completed", session_id="inspect-session", agent={"name": "a"}, success=True)
    emit_runtime_event("assign_agent.completed", session_id="other-session", agent={"name": "a"}, success=True)

    parsed = json.loads(runtime_inspect(session_id="inspect-session", limit=10))

    assert parsed["success"] is True
    assert parsed["session_id"] == "inspect-session"
    assert parsed["count"] == 2
    assert [event["event"] for event in parsed["events"]] == [
        "assign_agent.resolved",
        "assign_agent.completed",
    ]


def test_runtime_inspect_filters_by_agent_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event
    from tools.runtime_tool import runtime_inspect

    emit_runtime_event("assign_agent.completed", session_id="inspect-session", agent={"name": "a"})
    emit_runtime_event("assign_agent.completed", session_id="inspect-session", agent={"name": "b"})

    parsed = json.loads(runtime_inspect(session_id="inspect-session", agent_name="b"))

    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["events"][0]["data"]["agent"]["name"] == "b"


def test_runtime_inspect_clamps_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.runtime_trace import emit_runtime_event
    from tools.runtime_tool import runtime_inspect

    for idx in range(3):
        emit_runtime_event("assign_agent.completed", session_id="inspect-session", seq=idx)

    parsed = json.loads(runtime_inspect(session_id="inspect-session", limit=1))

    assert parsed["success"] is True
    assert parsed["count"] == 1
    assert parsed["events"][0]["data"]["seq"] == 2
