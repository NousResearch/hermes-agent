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

    LoopManager(session_key).set("poll", interval_seconds=300)

    with patch.object(server, "_run_prompt_submit") as submit, \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    submit.assert_not_called()
    assert s["running"] is False


# ── profile-scoped loop state ─────────────────────────────────────────


def test_tui_tick_fires_profile_home_loop(server, session, tmp_path):
    """A dashboard session with profile_home must read its loop from that
    profile's state.db, not the launch profile's."""
    sid, session_key, s = session
    profile_home = tmp_path / "profiles" / "dev"
    profile_home.mkdir(parents=True)

    from hermes_cli import loops
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    # Seed a due loop inside the profile's own state.db.
    token = set_hermes_home_override(str(profile_home))
    try:
        mgr = loops.LoopManager(session_key)
        mgr.set("poll the profile build", interval_seconds=60)
        mgr.state.next_due_at = time.time() - 1
        loops.save_loop(session_key, mgr.state)
    finally:
        reset_hermes_home_override(token)

    s["profile_home"] = str(profile_home)

    fired = {}

    def fake_submit(rid, sid_, session_, text, **kwargs):
        fired["text"] = text

    with patch.object(server, "_run_prompt_submit", fake_submit), \
         patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)

    # The loop was found in the session's profile, so the tick fired.
    assert "poll the profile build" in fired.get("text", "")
    assert s["running"] is True


def test_loop_set_persists_to_profile_home(server, session, tmp_path):
    """`/loop` created from a dashboard session lands in that session's
    profile state.db, not the launch profile's."""
    sid, session_key, s = session
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    s["profile_home"] = str(profile_home)

    r = _call(server, "command.dispatch", name="loop", arg="5m check profile", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "Loop set" in r["result"]["output"]

    from hermes_cli import loops
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    # The loop must be readable from the profile's home, not the launch home.
    token = set_hermes_home_override(str(profile_home))
    try:
        mgr = loops.LoopManager(session_key)
        assert mgr.state is not None
        assert mgr.state.prompt == "check profile"
    finally:
        reset_hermes_home_override(token)
