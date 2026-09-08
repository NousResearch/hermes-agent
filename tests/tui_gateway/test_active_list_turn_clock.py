"""Tests: session.active_list reports each live turn's clock.

The desktop sidebar's opt-in "Elapsed" row figure needs to know WHEN a running
turn started for rows this window never watched start (a turn kicked off from
another window, or before a reconnect). ``session.info`` already carries
``turn_started_at`` for the focused session; the live snapshot every row is
rehydrated from must speak the same clock.

Contract:
- a running session's row carries ``turn_started_at`` read from
  ``inflight_turn.started_at`` (epoch seconds, same source as session.info);
- an idle session's row carries ``None``, so the client clears its clock;
- the snapshot and ``session.info`` agree on the value.
"""

from __future__ import annotations

import threading

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _record(**extra):
    base = {
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "last_active": 1_700_000_100.0,
        "created_at": 1_700_000_000.0,
        "running": False,
        "session_key": "20260906_000000_abc123",
        "source": "desktop",
        "inflight_turn": None,
    }
    base.update(extra)
    return base


@pytest.fixture
def live_sessions(home):
    running = _record(
        running=True,
        session_key="running-key",
        inflight_turn={
            "assistant": "",
            "started_at": 1_700_000_042.5,
            "streaming": True,
            "updated_at": 1_700_000_050.0,
            "user": "do the thing",
        },
    )
    idle = _record(session_key="idle-key")
    srv._sessions["rt-running"] = running
    srv._sessions["rt-idle"] = idle
    yield running, idle
    srv._sessions.pop("rt-running", None)
    srv._sessions.pop("rt-idle", None)


def _active_list():
    out = srv._methods["session.active_list"](1, {})
    assert "error" not in out, out
    return {row["id"]: row for row in out["result"]["sessions"]}


def test_running_row_carries_turn_started_at(live_sessions):
    rows = _active_list()
    assert rows["rt-running"]["status"] == "working"
    assert rows["rt-running"]["turn_started_at"] == pytest.approx(1_700_000_042.5)


def test_idle_row_carries_no_turn_clock(live_sessions):
    rows = _active_list()
    assert rows["rt-idle"]["status"] == "idle"
    assert rows["rt-idle"]["turn_started_at"] is None


def test_snapshot_and_session_info_agree(live_sessions):
    running, _idle = live_sessions
    rows = _active_list()
    assert rows["rt-running"]["turn_started_at"] == srv._turn_started_at(running)


def test_turn_clock_ignores_malformed_inflight():
    assert srv._turn_started_at(None) is None
    assert srv._turn_started_at({}) is None
    assert srv._turn_started_at({"inflight_turn": "not-a-dict"}) is None
    assert srv._turn_started_at({"inflight_turn": {"started_at": 0}}) is None
    assert srv._turn_started_at({"inflight_turn": {"started_at": 12.0}}) == 12.0
