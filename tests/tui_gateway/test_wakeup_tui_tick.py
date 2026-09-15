"""schedule_wakeup firing from the TUI/Desktop session-owner poller (same driver as /heartbeat)."""

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
    sid, key = "sid-wk-test", "tui-wk-session-1"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock()}
    server._sessions[sid] = s
    return sid, key, s


def _arm_due(key: str):
    from hermes_cli.wakeups import WakeupManager

    mgr = WakeupManager(key)
    wakeup, err, _ = mgr.schedule("poll the deploy", delay="10s", now=time.time() - 3600)
    assert err is None
    return wakeup


@pytest.mark.parametrize("outcome", ["started", "refused"])
def test_tui_wakeup_fires_once_or_rearms_when_no_turn_starts(server, session, outcome):
    """A due wakeup re-enters the live session through _run_prompt_submit exactly once; a refused
    dispatch releases the claim AND puts the wakeup back so it is not silently consumed."""
    sid, key, s = session
    wakeup = _arm_due(key)
    dispatched: list[str] = []

    def submit(rid, sid_, session_, text, **kw):
        dispatched.append(text)
        if outcome == "refused":
            with s["history_lock"]:
                s["running"] = False
            return False
        return True

    with patch.object(server, "_run_prompt_submit", submit), patch.object(server, "_emit"):
        server._maybe_fire_tui_wakeup(sid, s)
        server._maybe_fire_tui_wakeup(sid, s)  # second poll: claimed (started) or busy-guarded (refused re-arm fires again)

    from hermes_cli.wakeups import WakeupManager

    pending = [w.id for w in WakeupManager(key).load().pending]
    assert "poll the deploy" in dispatched[0]
    if outcome == "started":
        assert len(dispatched) == 1 and s["running"] is True and pending == []
    else:
        # Re-armed after the first refusal, so the second idle poll tried again; still pending after both.
        assert len(dispatched) == 2 and s["running"] is False and pending == [wakeup.id]
