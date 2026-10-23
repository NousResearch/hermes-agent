"""Process exit must not book a live turn as a user Stop."""
import threading

from tui_gateway import server as tui_server


def test_exit_interrupt_names_shutdown_not_the_user(monkeypatch):
    calls = []

    def fake_interrupt(sid, session, **kwargs):
        calls.append((sid, kwargs))

    monkeypatch.setattr(tui_server, "_interrupt_session_turn", fake_interrupt)
    monkeypatch.setattr(
        tui_server, "_sessions", {"sid": {"running": True}}, raising=False,
    )
    monkeypatch.setattr(tui_server, "_sessions_lock", threading.RLock(), raising=False)
    monkeypatch.setattr(
        "tools.environments.base.kill_live_foreground_processes",
        lambda **_kwargs: None,
    )

    tui_server._stop_turns_before_exit(budget_s=0)

    assert calls
    sid, kwargs = calls[0]
    assert sid == "sid"
    assert kwargs.get("tool_reason") == "gateway shutdown"
    assert kwargs.get("hook_reason") == "gateway_shutdown"
    assert kwargs.get("tool_reason") not in {
        "explicit stop requested", "user interrupt", "user sent a new message",
    }
