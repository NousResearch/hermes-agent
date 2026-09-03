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
        mod._pending.clear()
        mod._answers.clear()


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


def test_loop_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="loop", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No loop set" in r["result"]["output"]


def test_loop_set_persists(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="loop", arg="5m check the deploy", session_id=sid)
    result = r["result"]
    assert result["type"] == "exec"
    assert "Loop set" in result["output"]

    from hermes_cli.loops import LoopManager

    mgr = LoopManager(session_key)
    assert mgr.state is not None
    assert mgr.state.prompt == "check the deploy"
    assert mgr.state.status == "active"
    assert mgr.state.interval_seconds == 300.0


def test_loop_proactive_alias_resolves(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="proactive", arg="5m ping", session_id=sid)
    assert "Loop set" in r["result"]["output"]


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
    assert "[/loop wakeup #1" in fired["text"]
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
