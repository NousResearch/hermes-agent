"""A parked /goal resumes from the TUI/Desktop session-owner poller once its barrier lifts.

The barrier is otherwise re-checked only after a turn (``evaluate_after_turn``). With nothing else
arriving, a Desktop session stayed parked for hours after a timed judge ``wait`` expired, the goal
still shown active. The classic CLI's ``_maybe_resume_parked_goal`` idle hook never reached this
poller, and a detached session was reapable while parked, which stopped the poller with it.
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
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()


@pytest.fixture()
def session(server):
    sid, key = "sid-goal-park", "tui-goal-park-1"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock()}
    server._sessions[sid] = s
    return sid, key, s


def _park(key: str, *, expired: bool):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id=key)
    mgr.set("ship the release")
    mgr.wait_for_seconds(60, reason="cron at 01:40")
    if expired:
        mgr.state.waiting_until = time.time() - 1
        mgr._save()
    return mgr


def _load(key: str):
    from hermes_cli.goals import load_goal

    return load_goal(key)


def test_poller_resumes_a_goal_whose_timed_wait_expired(server, session):
    sid, key, s = session
    server._get_db().create_session(key, source="desktop")
    _park(key, expired=True)
    dispatched: list[str] = []

    def submit(rid, sid_, session_, text, **kw):
        dispatched.append(text)
        return True

    stop = threading.Event()
    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        t = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, s), daemon=True)
        t.start()
        deadline = time.monotonic() + 8
        while not dispatched and time.monotonic() < deadline:
            time.sleep(0.1)
        stop.set()
        t.join(timeout=5)

    assert len(dispatched) == 1 and "ship the release" in dispatched[0]
    assert s["running"] is True  # claimed for the continuation turn
    assert _load(key).waiting_until == 0.0 and _load(key).status == "active"


@pytest.mark.parametrize("case", ["still_waiting", "busy", "paused"])
def test_resume_is_a_no_op_while_parked_busy_or_inactive(server, session, case):
    sid, key, s = session
    mgr = _park(key, expired=case != "still_waiting")
    if case == "paused":
        mgr.pause()
    s["running"] = case == "busy"
    with patch.object(server, "_run_prompt_submit", MagicMock()) as submit, patch.object(server, "_emit"):
        server._maybe_resume_tui_parked_goal(sid, s)
    submit.assert_not_called()
    assert s["running"] is (case == "busy")  # a busy session is never released by the poller
    if case == "still_waiting":
        assert _load(key).waiting_until > time.time()  # barrier kept


def test_refused_dispatch_releases_the_claim(server, session):
    sid, key, s = session
    _park(key, expired=True)

    def submit(*a, **k):
        with s["history_lock"]:
            s["running"] = False
        return False

    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        server._maybe_resume_tui_parked_goal(sid, s)
    assert s["running"] is False


def test_a_parked_goal_keeps_the_detached_session_alive(server, session):
    """The poller is the only driver of the resume: reaping the session while parked stops it."""
    _, key, s = session
    mgr = _park(key, expired=False)
    assert server._session_owns_live_wakeup_schedule(s) is True
    mgr.clear()
    assert server._session_owns_live_wakeup_schedule(s) is False
