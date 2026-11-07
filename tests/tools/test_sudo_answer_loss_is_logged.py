"""An answer that never reaches sudo leaves a trace (#148460).

Each of these ends in ``sudo: a password is required``, the same text as when nobody was asked, so without a log
line a lost answer reads as a wrong password. The command still continues without one.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from gateway.config import Platform
from gateway.session_events import SessionEvents
from gateway.session_sudo_prompt import ask_sudo, sudo_prompt_scope
from tools import terminal_tool, terminal_tool_sudo

_SECRET = "hunter2-do-not-log"


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "sudo" in r.getMessage().lower()]


def test_a_failing_prompt_callback_is_logged_with_its_cause(monkeypatch, caplog):
    def broken_bridge():
        raise ConnectionError("session-scoped RPC rejected")

    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: broken_bridge)

    assert terminal_tool_sudo._prompt_for_sudo_password(timeout_seconds=1) == ""

    assert _warnings(caplog)
    assert any(r.exc_info and r.exc_info[0] is ConnectionError for r in caplog.records)


def test_a_declined_prompt_stays_quiet(monkeypatch, caplog):
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: lambda: "")

    assert terminal_tool_sudo._prompt_for_sudo_password(timeout_seconds=1) == ""

    assert _warnings(caplog) == []


def test_a_delivered_password_is_never_logged(monkeypatch, caplog):
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: lambda: _SECRET)

    with caplog.at_level(logging.DEBUG):
        assert terminal_tool_sudo._prompt_for_sudo_password(timeout_seconds=1) == _SECRET

    assert _SECRET not in caplog.text


def test_no_attached_viewer_is_logged(caplog):
    assert ask_sudo(SessionEvents(), "s1", "sudo true", timeout=1) == ""

    assert any("s1" in message for message in _warnings(caplog))


def test_a_session_gone_from_memory_is_logged(caplog):
    previous = terminal_tool._get_sudo_password_callback()
    turn = SimpleNamespace(_approval_owner=(SimpleNamespace(sessions={}), "stale-id", 0),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))

    with sudo_prompt_scope(turn, SimpleNamespace()):
        callback = terminal_tool._get_sudo_password_callback()
        assert callback is not previous
        assert callback() == ""

    assert any("stale-id" in message for message in _warnings(caplog))
