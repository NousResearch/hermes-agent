"""A system-driven TUI interrupt must not be booked as a user stop (#112647).

Observed defect (em-board t_3f73de7d, 7/7 kills): the desktop app retires its pooled remote backend, the
tui_gateway process runs its shutdown fan-out (``session_reaper._stop_turns_before_exit`` ->
``session_lifecycle._interrupt_session_turn``), and every live turn is hard-interrupted through the
USER-attributed default ("explicit stop requested"). The transcript and the durable turn record then say
``interrupted_by_user`` and the UI shows "Operation interrupted." — an app-driven teardown impersonating
the user, so cancelled work is booked as user intent.

``agent.interrupt_control`` already separates the two: ``tool_reason`` names the system producer and
``interrupt_issuer()`` reads it back (a USER_INTERRUPT_REASONS reason -> ``None``, i.e. a human stop). Every
other non-user producer in the tree passes one (gateway timeout/shutdown, api_server, subagents); the TUI
lifecycle path never did. These tests pin the wiring for the non-user callers, and pin that an explicit
user stop keeps its user attribution.
"""

from __future__ import annotations

import threading

import pytest

from agent.interrupt_control import interrupt_issuer, _REASON_HARD_STOP


class _RecordingAgent:
    """Minimal AIAgent stand-in: records what ``hard_interrupt(tool_reason=...)`` latched.

    The real mixin stores the reason on ``_tool_interrupt_reason`` (defaulting to the hard-stop reason when
    no ``tool_reason`` is supplied), which is exactly what ``interrupt_issuer`` reads.
    """

    def __init__(self) -> None:
        self._tool_interrupt_reason = None

    def hard_interrupt(self, message=None, *, tool_reason=None):  # noqa: D102 - mirrors AIAgent.mixin
        self._tool_interrupt_reason = tool_reason or _REASON_HARD_STOP

    def interrupt(self, message=None):  # noqa: D102 - legacy ABI fallback
        self._tool_interrupt_reason = _REASON_HARD_STOP


def _session(agent, sid: str) -> dict:
    return {
        "_sid": sid,
        "session_key": "agent:main:tui:dm:issuer",
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "queued_prompt": None,
        "agent": agent,
        "_run_thread": None,
    }


@pytest.fixture()
def server(monkeypatch):
    from tui_gateway import server as _server

    monkeypatch.setattr(_server, "_clear_pending", lambda *a, **k: None)
    monkeypatch.setattr(_server, "_session_uses_compute_host", lambda *a, **k: False)
    monkeypatch.setattr(_server, "_finalize_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "tools.environments.base.kill_live_foreground_processes", lambda *a, **k: None
    )
    return _server


def test_shutdown_fan_out_names_the_producer(server):
    """The 7/7 caller: process shutdown hard-interrupts every live turn through the shared helper.

    Called through the published ``server`` binding (the same rebound-globals entry point
    ``_shutdown_sessions`` uses), so the module-level session registry is the server's.
    """
    agent = _RecordingAgent()
    session = _session(agent, "sid-shutdown")
    with server._sessions_lock:
        server._sessions[session["_sid"]] = session
    try:
        server._stop_turns_before_exit(0.0)
    finally:
        with server._sessions_lock:
            server._sessions.pop(session["_sid"], None)

    assert agent._tool_interrupt_reason, "the shutdown fan-out must reach the agent"
    assert interrupt_issuer(agent) == "gateway_shutting_down"


def test_ws_orphan_reap_names_the_producer(server, monkeypatch):
    """The graceful-detach caller: a clientless session interrupted at grace is not a user stop either."""
    sid = "sid-orphan"
    agent = _RecordingAgent()
    session = _session(agent, sid)
    session["transport"] = server._detached_ws_transport

    class _Timer:
        def __init__(self, delay, callback):
            self.callback, self.daemon = callback, False

        def start(self):
            self.callback()

        def cancel(self):
            pass

    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a, **k: False)
    monkeypatch.setattr(server.threading, "Timer", _Timer)

    server._schedule_ws_orphan_reap(sid, delay_s=0)

    assert agent._tool_interrupt_reason, "the orphan reaper must reach the agent"
    assert interrupt_issuer(agent) == "client_disconnected"


def test_explicit_session_interrupt_stays_a_user_stop(server):
    """``session.interrupt`` (and a client interrupt control frame) must keep user attribution."""
    agent = _RecordingAgent()
    session = _session(agent, "sid-user")

    server._interrupt_session_turn("sid-user", session)

    assert agent._tool_interrupt_reason, "the explicit stop must still reach the agent"
    assert interrupt_issuer(agent) is None, "a deliberate stop is interrupted_by_user, not by_system"
