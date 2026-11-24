"""Regression coverage for crashed TUI gateway turns."""

import threading
import types

from tui_gateway import server


def test_turn_error_restores_agent_transcript_to_gateway_history():
    agent = types.SimpleNamespace(
        _session_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "completed work"},
        ]
    )
    session = {
        "history": [{"role": "user", "content": "first"}],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    assert server._restore_agent_history_after_turn_error(session, agent) is True
    assert session["history"] == agent._session_messages
    assert session["history_version"] == 1


def test_turn_error_with_no_agent_transcript_does_not_overwrite_history():
    agent = types.SimpleNamespace(_session_messages=None)
    session = {
        "history": [{"role": "user", "content": "first"}],
        "history_lock": threading.Lock(),
        "history_version": 4,
    }

    assert server._restore_agent_history_after_turn_error(session, agent) is False
    assert session["history"] == [{"role": "user", "content": "first"}]
    assert session["history_version"] == 4
