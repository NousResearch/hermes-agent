"""Diagnostic wake execution and structured controls survive a presentation veto."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.notification_presentation import notification_turn
from tui_gateway import server


@pytest.mark.parametrize("muted", [False, True])
def test_real_tui_emitter_keeps_control_frames_and_restores_callbacks(monkeypatch, muted):
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    callback = Mock()
    agent = SimpleNamespace(status_callback=callback, clarify_callback=callback)
    with notification_turn(agent, muted=muted, session_id="session"):
        server._emit("message.delta", "session", {"text": "diagnostic echoed by model"})
        server._emit("message.complete", "session", {"text": "diagnostic echoed by model"})
        agent.clarify_callback("question", ["choice"])
        server._emit("notification.clear", "session", {"key": "cleared"})
    assert [frame["params"]["type"] for frame in frames] == (
        ["notification.clear"] if muted else ["message.delta", "message.complete", "notification.clear"])
    assert agent.status_callback is callback
    callback.assert_called_once_with("question", ["choice"])
    server._emit("message.delta", "session", {"text": "next human result"})
    assert frames[-1]["params"]["payload"]["text"] == "next human result"


@pytest.mark.parametrize("suppress", [False, True])
def test_tui_kanban_splits_diagnostics_from_results_before_wake(tmp_path, monkeypatch, suppress):
    from gateway.warning_notifications import DiagnosticText
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not suppress).lower()}}}")
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}")
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda session: [])
    emitted, submitted = [], []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_notif_submit", lambda *args, **kwargs: submitted.append((args, kwargs)))
    session = {"profile_home": str(owner), "history_lock": threading.RLock(), "_kanban_pending": [DiagnosticText("worker crash"), "requested result"]}
    server._notif_poll_kanban("session", session)
    if not suppress:
        assert submitted[0][0][3] == "worker crash\nrequested result"
        assert submitted[0][1] == {}
        assert session["_kanban_pending"] == []
        return
    assert submitted[0][0][3] == "worker crash"
    assert submitted[0][1]["display_metadata"]["notification_category"] == "diagnostic"
    assert session["_kanban_pending"] == ["requested result"]
    session["running"] = False
    server._notif_poll_kanban("session", session)
    assert submitted[1][0][3] == "requested result"
    assert submitted[1][1] == {}
