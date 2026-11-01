"""Tests for /loop handling in tui_gateway.

The TUI routes ``/loop`` through ``command.dispatch`` (same rationale as
``/goal`` — the CLI handler drives ``_pending_input``, which the slash
worker has no reader for). State mutations go through the shared
``dispatch_loop_command``; the per-session notification poller fires due
wakeups via ``_maybe_fire_tui_loop_tick``.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


@pytest.fixture()
def session(server):
    sid = "sid-loop-test"
    session_key = "tui-loop-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── command.dispatch /loop ────────────────────────────────────────────




def test_loop_set_persists(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="loop", arg="5m check the deploy", session_id=sid)
    result = r["result"]
    assert result["type"] == "exec"

    from hermes_cli.loops import LoopManager

    mgr = LoopManager(session_key)
    assert mgr.state is not None
    assert mgr.state.prompt == "check the deploy"
    assert mgr.state.status == "active"
    assert mgr.state.interval_seconds == 300.0




def test_loop_pause_resume_stop(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="loop", arg="5m poll CI", session_id=sid)

    r = _call(server, "command.dispatch", name="loop", arg="pause", session_id=sid)
    assert "paused" in r["result"]["output"].lower()

    r = _call(server, "command.dispatch", name="loop", arg="resume", session_id=sid)
    assert "resumed" in r["result"]["output"].lower()

    r = _call(server, "command.dispatch", name="loop", arg="stop", session_id=sid)
    assert "stopped" in r["result"]["output"].lower()

    from hermes_cli.loops import LoopManager

    assert not LoopManager(session_key).has_loop()


def test_loop_requires_session(server):
    r = _call(server, "command.dispatch", name="loop", arg="5m x", session_id="unknown")
    assert "error" in r
    assert r["error"]["code"] == 4001


# ── idle wakeup driver ────────────────────────────────────────────────


def test_tui_tick_fires_when_idle_and_due(server, session):
    sid, session_key, s = session
    from hermes_cli.loops import LoopManager, save_loop

    mgr = LoopManager(session_key)
    mgr.set("poll the build", interval_seconds=60)
    mgr.state.next_due_at = time.time() - 1
    save_loop(session_key, mgr.state)

    fired = {}

    def fake_submit(rid, sid_, session_, text, **kwargs):
        fired["text"] = text

    with patch.object(server, "_run_prompt_submit", fake_submit), \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    assert "poll the build" in fired.get("text", "")
    # Session claimed for the wakeup turn.
    assert s["running"] is True


def test_tui_tick_defers_when_running(server, session):
    sid, session_key, s = session
    from hermes_cli.loops import LoopManager, save_loop

    mgr = LoopManager(session_key)
    mgr.set("poll", interval_seconds=60)
    mgr.state.next_due_at = time.time() - 1
    save_loop(session_key, mgr.state)
    s["running"] = True

    with patch.object(server, "_run_prompt_submit") as submit, \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_not_called()
    # Tick not consumed — still due for the next poll.
    assert LoopManager(session_key).state.ticks_fired == 0


def test_tui_tick_leaves_gateway_routed_loop_for_gateway(server, session):
    """A /loop set from a messaging chat (route pinned by the gateway) must not be consumed by a TUI/Desktop
    viewer of the same session: the gateway's wakeup scanner owns delivery back to that chat (#111841)."""
    sid, session_key, s = session
    from hermes_cli.loops import LoopManager, save_loop

    mgr = LoopManager(session_key)
    mgr.set("poll", interval_seconds=60, route={"platform": "telegram", "chat_id": "42"})
    mgr.state.next_due_at = time.time() - 1
    save_loop(session_key, mgr.state)

    with patch.object(server, "_run_prompt_submit") as submit, \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_not_called()
    assert s["running"] is False
    state = LoopManager(session_key).state
    assert state.ticks_fired == 0 and not state.awaiting_response and state.next_due_at <= time.time()


def test_tui_tick_defers_to_active_goal(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager
    from hermes_cli.loops import LoopManager, save_loop

    GoalManager(session_id=session_key).set("finish the feature")
    mgr = LoopManager(session_key)
    mgr.set("poll", interval_seconds=60)
    mgr.state.next_due_at = time.time() - 1
    save_loop(session_key, mgr.state)

    with patch.object(server, "_run_prompt_submit") as submit, \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_not_called()
    assert s["running"] is False


def test_tui_tick_noop_when_not_due(server, session):
    sid, session_key, s = session
    from hermes_cli.loops import LoopManager

    mgr = LoopManager(session_key)
    mgr.set("poll", interval_seconds=300)
    # New loops are due immediately; push the wakeup out to model "not due".
    from hermes_cli.loops import save_loop
    mgr.state.next_due_at = time.time() + 300
    save_loop(session_key, mgr.state)

    with patch.object(server, "_run_prompt_submit") as submit, \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_not_called()
    assert s["running"] is False


def test_slash_loop_noop_ticks_pause_the_loop(server, session):
    """A /loop whose prompt resolves to no output (unknown slash command) pauses at the no-op
    streak threshold instead of silently burning the whole tick budget."""
    sid, session_key, s = session
    from hermes_cli.loops import SLASH_NOOP_PAUSE_AFTER, LoopManager, save_loop

    mgr = LoopManager(session_key)
    mgr.set("/bogus-command", interval_seconds=60)

    statuses = []

    def fake_emit(event, sid_, payload):
        if payload.get("kind") == "loop":
            statuses.append(payload["text"])

    with patch.object(server, "_emit", fake_emit), \
         patch.dict(server._methods, {"command.dispatch": lambda rid, params: {
             "error": {"code": 4004, "message": "unknown command: bogus-command"}}}):
        for i in range(SLASH_NOOP_PAUSE_AFTER):
            mgr = LoopManager(session_key)  # reload: the poller's writes must not be clobbered by a stale copy
            mgr.state.next_due_at = time.time() - 1
            save_loop(session_key, mgr.state)
            server._maybe_fire_tui_loop_tick(sid, s)
            assert LoopManager(session_key).state.slash_noop_streak == i + 1
            assert LoopManager(session_key).state.status == (
                "paused" if i + 1 == SLASH_NOOP_PAUSE_AFTER else "active")

    state = LoopManager(session_key).state
    assert any("unknown command" in t for t in statuses)  # the dispatch error is surfaced, not swallowed
    assert any("/loop resume" in t for t in statuses)
    assert "no output" in (state.paused_reason or "")


def test_slash_loop_skill_command_runs_the_turn_and_clears_the_streak(server, session):
    """A skill command dispatches back type 'skill' with a message (methods_tools.py): the tick
    must run it as a turn, not class the bead's own repro (/loop 5m /<skill>) as a no-op."""
    sid, session_key, s = session
    from hermes_cli.loops import LoopManager, save_loop

    mgr = LoopManager(session_key)
    mgr.set("/my-skill", interval_seconds=60)
    mgr.state.slash_noop_streak = 2
    mgr.state.next_due_at = time.time() - 1
    save_loop(session_key, mgr.state)

    with patch.object(server, "_emit"), \
         patch.object(server, "_run_prompt_submit", return_value=True) as submit, \
         patch.dict(server._methods, {"command.dispatch": lambda rid, params: {
             "result": {"type": "skill", "message": "run the skill"}}}):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_called_once()
    assert submit.call_args[0][3] == "run the skill"
    state = LoopManager(session_key).state
    assert state.slash_noop_streak == 0 and state.status == "active"
