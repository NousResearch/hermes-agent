"""Cross-layer regression for Codex app-server tool cards in the TUI.

Drives the merged app-server event bridge through the real TUI gateway
callbacks and asserts stable tool ids flow into tool.start/tool.complete
TUI events. Grafted from PR #65412 (@HaiderSultanArc).
"""

from types import SimpleNamespace

from agent.codex_runtime import make_codex_app_server_event_bridge
from tui_gateway import server


def test_codex_bridge_emits_one_authoritative_tui_tool_lifecycle(monkeypatch):
    sid = "codex-live-events"
    events = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event_type, session_id, payload=None: events.append((
            event_type,
            session_id,
            payload,
        )),
    )
    monkeypatch.setitem(
        server._sessions,
        sid,
        {
            "tool_progress_mode": "all",
            "tool_started_at": {},
            "edit_snapshots": {},
        },
    )
    callbacks = server._agent_cbs(sid)
    agent = SimpleNamespace(
        tool_progress_callback=callbacks["tool_progress_callback"],
        tool_start_callback=callbacks["tool_start_callback"],
        tool_complete_callback=callbacks["tool_complete_callback"],
        _emit_interim_assistant_message=None,
        show_commentary=True,
    )
    bridge = make_codex_app_server_event_bridge(agent)
    started = {
        "type": "commandExecution",
        "id": "tool-1",
        "command": "pwd",
        "cwd": "/tmp",
    }

    bridge({"method": "item/started", "params": {"item": started}})
    bridge({
        "method": "item/completed",
        "params": {"item": dict(started, aggregatedOutput="/tmp\n", exitCode=0)},
    })

    lifecycle = [
        (event_type, payload)
        for event_type, _, payload in events
        if event_type in {"tool.start", "tool.complete"}
    ]
    assert [event_type for event_type, _ in lifecycle] == [
        "tool.start",
        "tool.complete",
    ]
    assert lifecycle[0][1]["tool_id"] == "codex_exec_tool-1"
    assert lifecycle[1][1]["tool_id"] == "codex_exec_tool-1"
