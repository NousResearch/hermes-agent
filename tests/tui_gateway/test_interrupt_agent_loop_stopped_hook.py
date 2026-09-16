"""The TUI/desktop ``session.interrupt`` path is a sibling of the gateway's
``_interrupt_and_clear_session``: when a live turn is stopped, plugins holding
per-turn external resources need the same ``agent_loop_stopped`` signal.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_session(running: bool) -> dict:
    return {
        "history_lock": threading.Lock(),
        "running": running,
        "queued_prompt": None,
        "session_key": "agent:main:tui:dm:s1",
        "agent": MagicMock(),
        "_run_thread": None,
    }


def _hook_calls(mock_invoke_hook):
    return [
        call
        for call in mock_invoke_hook.call_args_list
        if call.args and call.args[0] == "agent_loop_stopped"
    ]


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupt_running_turn_fires_agent_loop_stopped(mock_invoke_hook):
    from tui_gateway import server

    session = _make_session(running=True)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    calls = _hook_calls(mock_invoke_hook)
    assert len(calls) == 1
    assert calls[0].kwargs == {
        "session_key": "agent:main:tui:dm:s1",
        "platform": "tui",
        "reason": "user_stop",
        "invalidation_reason": "session_interrupt",
    }


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupt_idle_session_does_not_fire_hook(mock_invoke_hook):
    """No live turn -> nothing for a plugin to cancel -> no hook noise."""
    from tui_gateway import server

    session = _make_session(running=False)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    assert _hook_calls(mock_invoke_hook) == []


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_failure_does_not_break_interrupt(mock_invoke_hook):
    """A misbehaving plugin must never prevent the interrupt itself."""
    mock_invoke_hook.side_effect = RuntimeError("plugin exploded")
    from tui_gateway import server

    session = _make_session(running=True)
    with patch.object(server, "_clear_pending"):
        server._interrupt_session_turn("s1", session)

    # The cancel flag was still set despite the hook blowing up.
    assert session["_turn_cancel_requested"] is True
