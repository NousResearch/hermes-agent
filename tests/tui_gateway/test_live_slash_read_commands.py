"""Read-only slash commands must stay on the live desktop session."""

import threading

from tui_gateway import server


def test_read_only_slash_commands_use_live_session_for_desktop(monkeypatch):
    session = {
        "agent": None,
        "session_key": "desktop-key",
        "cwd": "",
        "history": [{"role": "user", "content": "hello"}],
        "history_lock": threading.Lock(),
    }
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda current_session: False)

    output = server._live_slash_command_output("desktop-sid", session, "context", "")

    assert output.startswith("Conversation: 1 messages")


def test_read_only_slash_commands_without_session_keep_no_session_reply():
    assert server._live_slash_command_output("desktop-sid", None, "context", "") == (
        "Conversation is empty (no messages yet)."
    )
