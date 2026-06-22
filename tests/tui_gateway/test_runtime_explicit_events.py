from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_runtime_explicit_events")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list[tuple[str, str, dict | None]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    monkeypatch.setattr(server, "_tool_ctx", lambda name, args: args.get("command", ""))
    monkeypatch.setattr(
        server,
        "_block",
        lambda event, sid, payload, **_kwargs: captured.append((event, sid, payload)) or None,
    )
    return captured


def test_tool_callbacks_emit_explicit_agent_and_tool_lifecycle(server, emits):
    sid = "runtime-sid"
    server._sessions[sid] = {"tool_started_at": {}, "edit_snapshots": {}}

    server._on_tool_start(sid, "tc-1", "terminal", {"command": "npm test"})
    server._on_tool_complete(sid, "tc-1", "terminal", {"command": "npm test"}, '{"ok": true}')

    assert [event for event, _, _ in emits] == [
        "agent.status.changed",
        "tool_call.started",
        "tool.start",
        "agent.status.changed",
        "tool_call.completed",
        "tool.complete",
    ]
    assert emits[0][2] == {"status": "running_tool", "tool_id": "tc-1", "tool_name": "terminal"}
    assert emits[1][2]["name"] == "terminal"
    assert emits[3][2] == {"status": "completed", "tool_id": "tc-1", "tool_name": "terminal"}


def test_agent_callbacks_emit_explicit_generation_reasoning_waiting_and_task_events(server, emits):
    callbacks = server._agent_cbs("sid-2")

    callbacks["thinking_callback"]("thinking")
    callbacks["reasoning_callback"]("reasoning")
    callbacks["clarify_callback"]("Need input?", ["yes"])
    callbacks["notice_callback"](MagicMock(text="Careful", level="warning", kind="warning", ttl_ms=1000, key="k", id="n1"))

    names = [event for event, _, _ in emits]
    assert "agent.status.changed" in names
    assert ("agent.status.changed", "sid-2", {"status": "thinking"}) in emits
    assert ("agent.status.changed", "sid-2", {"status": "reasoning"}) in emits
    assert ("agent.status.changed", "sid-2", {"status": "waiting_user"}) in emits
    assert ("warning.emitted", "sid-2", {"text": "Careful", "level": "warning", "kind": "warning", "key": "k", "id": "n1"}) in emits


def test_subagent_progress_emits_explicit_handoff_lifecycle(server, emits):
    server._on_tool_progress(
        "parent-sid",
        "subagent.start",
        preview="Build the parser",
        child_session_id="child-session",
        subagent_id="worker-1",
        parent_id="parent-sid",
        goal="Build the parser",
    )
    server._on_tool_progress(
        "parent-sid",
        "subagent.complete",
        preview="Done",
        child_session_id="child-session",
        subagent_id="worker-1",
        parent_id="parent-sid",
        summary="Done",
    )

    assert [event for event, _, _ in emits[:4]] == [
        "handoff.created",
        "subagent.start",
        "handoff.completed",
        "subagent.complete",
    ]
    assert emits[0][2]["confidence"] == "explicit"
    assert emits[0][2]["target_session_id"] == "child-session"
    assert emits[2][2]["confidence"] == "explicit"
    assert emits[2][2]["target_session_id"] == "child-session"
