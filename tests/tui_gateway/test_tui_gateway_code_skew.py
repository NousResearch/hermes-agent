"""Tests: tui_gateway code-skew guard during deferred agent build.

When the checkout drifts while a long-lived dashboard/Desktop backend is running,
_start_agent_build must refuse to build into a stale sys.modules graph and report
an actionable restart requirement before lazy imports trigger unhandled ImportErrors.
"""

from __future__ import annotations

import os
import threading

import pytest

from tui_gateway import server
from tui_gateway import user_messages as um
from tui_gateway.user_messages import AGENT_BUILD_ABANDONED


class _InlineThread:
    """Run the build synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent=None, **extra):
    return {
        "agent": agent,
        "agent_error": None,
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


def test_code_skew_aborts_agent_build_with_actionable_error(monkeypatch, tmp_path):
    """When boot revision ≠ disk revision, agent build stops before lazy imports and emits error."""
    emitted: list[tuple] = []
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr("gateway.code_skew.detect_code_skew", lambda: ("boot123456", "disk654321"))

    sid = "code-skew-sid"
    session = _session(agent_ready=threading.Event(), cwd=str(tmp_path), profile_home=None)
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
    finally:
        server._sessions.pop(sid, None)

    assert session["agent"] is None
    assert session["agent_ready"].is_set()
    assert session["agent_error"] is not None
    assert "boot123456" in session["agent_error"]
    assert "disk654321" in session["agent_error"]
    assert "stale-module crash" in session["agent_error"]

    error_events = [p for e, s, p in emitted if e == "error" and s == sid]
    assert len(error_events) == 1
    assert error_events[0].get("code") == "code_skew_restart_required"
    assert "boot123456" in error_events[0].get("message", "")


def test_code_skew_desktop_hint_customization(monkeypatch, tmp_path):
    """Under HERMES_SERVE_HEADLESS=1 (Desktop app), hint specifically mentions Restart backend / Desktop."""
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr("gateway.code_skew.detect_code_skew", lambda: ("rev_a", "rev_b"))

    sid = "headless-skew-sid"
    session = _session(agent_ready=threading.Event(), cwd=str(tmp_path), profile_home=None)
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
    finally:
        server._sessions.pop(sid, None)

    assert "restart the Desktop-owned backend" in session["agent_error"]


def test_no_code_skew_proceeds_to_build(monkeypatch, tmp_path):
    """When detect_code_skew returns None, build proceeds past the guard."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr("gateway.code_skew.detect_code_skew", lambda: None)
    # Stop build at _await_resume_history so it doesn't try to invoke the full agent factory in this unit test
    monkeypatch.setattr(server, "_await_resume_history", lambda sid, current: False)

    sid = "clean-skew-sid"
    session = _session(agent_ready=threading.Event(), cwd=str(tmp_path), profile_home=None)
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
    finally:
        server._sessions.pop(sid, None)

    # Proceeded past code-skew guard and hit the _await_resume_history stub
    assert session["agent_error"] == AGENT_BUILD_ABANDONED


def test_turn_error_text_renders_code_skew_surface():
    """Turn error formatter produces clean non-technical titles and actionable hint for code skew."""
    surface = {"layer": "runtime", "code": "code_skew_detected", "retryable": False}
    text = um.turn_error_text("This process is running code from abc but disk is now xyz", surface)
    lines = text.split("\n")

    assert lines[0] == "Backend restart required after code update. Your message was not answered."
    assert "Details: This process is running code from abc but disk is now xyz" in lines[1]
    assert lines[2] == "Restart Hermes Desktop or the gateway service to load updated code."
