"""Diagnostic wake exercises the actual TUI turn, event transport, and control callback."""
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tests.tui_gateway.test_auto_continue import turn_env, marker_home, _session


@pytest.mark.parametrize("suppress", [False, True])
def test_diagnostic_turn_runs_but_never_echoes_on_wire(turn_env, marker_home, monkeypatch, suppress):
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not suppress).lower()}}}")
    owner = marker_home / "owner"
    owner.mkdir()
    (owner / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}")
    frames, work, controls = [], [], []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
                            clarify_callback=lambda *a: controls.append(a))
    def run(message, **kwargs):
        work.append(message)
        kwargs["stream_callback"]("technical diagnostic echo")
        if agent.interim_assistant_callback:
            agent.interim_assistant_callback("technical interim")
        agent.clarify_callback("Recovery approval?", ["yes", "no"])
        return {"final_response": "technical final", "messages": []}
    agent.run_conversation = run
    session = _session(agent=agent, running=True)
    session["profile_home"] = str(owner)
    server._run_prompt_submit("request", "session", session, "engine failure",
                              display_metadata={"notification_category": "diagnostic"})
    assert work == ["engine failure"]
    assert controls == [("Recovery approval?", ["yes", "no"])]
    content_frames = [f for f in frames if (f.get("params") or {}).get("type") in {"message.delta", "message.interim", "message.complete", "error"}]
    assert bool(content_frames) is not suppress
    assert session["running"] is False


def test_restart_preserves_diagnostic_category_through_marker(turn_env, marker_home, monkeypatch):
    from tui_gateway.turn_marker import read_turn_marker
    session = _session()
    server._record_turn_marker(session, "engine diagnostic",
                               notification_category="diagnostic")
    marker = read_turn_marker(marker_home, "session-key")
    assert marker["notification_category"] == "diagnostic"
    calls = []
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_auto_continue_config", lambda: (True, 3600, 2))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: calls.append(k))
    server._maybe_schedule_auto_continue("session", session, "session-key")
    assert calls[0]["display_metadata"]["notification_category"] == "diagnostic"


def test_next_human_followup_is_outside_diagnostic_presentation_scope(turn_env, marker_home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(marker_home))
    (marker_home / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    agent = SimpleNamespace(session_id="session-key", clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: {"final_response": "diagnostic echo", "messages": []})
    session = _session(agent=agent, running=True)
    def followups(*a):
        server._emit("message.complete", "session", {"text": "next requested result"})
    monkeypatch.setattr(server, "_run_post_turn_followups", followups)
    server._run_prompt_submit("request", "session", session, "engine failure",
                              display_metadata={"notification_category": "diagnostic"})
    finals = [f["params"]["payload"]["text"] for f in frames if f.get("params", {}).get("type") == "message.complete"]
    assert finals == ["next requested result"]
