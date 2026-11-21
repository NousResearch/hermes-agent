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


def test_tools_with_live_agent_answer_from_live_session():
    """#128874: /tools on a desktop session with an active agent must not fall
    through to the slash worker, which cannot see the serve process's agent."""
    session = {
        "agent": object(),
        "session_key": "desktop-key",
        "cwd": "",
        "history": [],
        "history_lock": threading.Lock(),
    }

    output = server._live_slash_command_output("desktop-sid", session, "tools", "")

    assert output is not None
    assert "No active agent" not in output


def test_tools_without_a_live_agent_keep_the_worker_routing():
    """Agentless /tools consults nothing the live session can answer with, so the
    old worker routing stays (same scope decision as upstream's /tools guard)."""
    session = {
        "agent": None,
        "session_key": "desktop-key",
        "cwd": "",
        "history": [],
        "history_lock": threading.Lock(),
    }

    assert server._live_slash_command_output("desktop-sid", session, "tools", "") is None


def test_help_with_a_session_answers_from_live_session():
    """/help is static content, so any available session serves it in-process."""
    session = {
        "agent": None,
        "session_key": "desktop-key",
        "cwd": "",
        "history": [],
        "history_lock": threading.Lock(),
    }

    output = server._live_slash_command_output("desktop-sid", session, "help", "")

    assert output is not None
    assert "Available commands" in output
