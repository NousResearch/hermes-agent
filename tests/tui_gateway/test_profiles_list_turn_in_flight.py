"""Tests: profiles.list ``turn_in_flight`` (live session turn lease).

Why: the desktop's busy state only covers turns that desktop dispatched, and
the client-side activity window both lags a tool-heavy turn and lingers ~90s
after it ends. A turn another bot started (bot-to-bot delegation) or a
messaging platform started therefore read idle in rosters for its whole run.
The turn lease is exact on both edges: acquired in the turn prologue,
refreshed until the turn ends, released on completion.

Contract under test:
- ``turn_in_flight`` is True while the profile holds an unexpired lease.
- False with no lease, and False again once the lease expires.
- ``include_sessions: false`` skips the field entirely.
- Best-effort: a db without the probe reads False rather than failing the
  whole profiles.list call.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _db(profile_dir):
    from hermes_state import SessionDB

    return SessionDB(db_path=profile_dir / "state.db")


def _add_session(db, sid, *, source="cli", ts, text):
    db.create_session(sid, source)
    db.append_message(sid, "user", text, timestamp=ts)


def _profiles(params):
    envelope = srv._methods["profiles.list"](1, params)
    return envelope["result"]["profiles"]


def _row(profiles, name):
    return next(p for p in profiles if p["name"] == name)


def test_turn_in_flight_true_while_lease_is_live(home):
    db = _db(home)
    _add_session(db, "chat1", ts=1000, text="hello")
    assert db.try_acquire_session_turn_lease("chat1", "pid=1:turn=t1:platform=test")
    db.close()

    assert _row(_profiles({}), "default")["turn_in_flight"] is True


def test_turn_in_flight_false_without_a_lease(home):
    db = _db(home)
    _add_session(db, "chat1", ts=1000, text="hello")
    db.close()

    assert _row(_profiles({}), "default")["turn_in_flight"] is False


def test_turn_in_flight_false_once_the_lease_expires(home):
    db = _db(home)
    _add_session(db, "chat1", ts=1000, text="hello")
    assert db.try_acquire_session_turn_lease("chat1", "pid=1:turn=t1:platform=test")
    # Expire it in place: a crashed process never runs release, and the
    # roster must stop reporting the turn once the TTL passes.
    with db._lock:
        db._conn.execute("UPDATE session_turn_leases SET expires_at = expires_at - 100000")
        db._conn.commit()
    db.close()

    assert _row(_profiles({}), "default")["turn_in_flight"] is False


def test_turn_in_flight_cleared_by_release(home):
    db = _db(home)
    _add_session(db, "chat1", ts=1000, text="hello")
    holder = "pid=1:turn=t1:platform=test"
    assert db.try_acquire_session_turn_lease("chat1", holder)
    db.release_session_turn_lease("chat1", holder)
    db.close()

    assert _row(_profiles({}), "default")["turn_in_flight"] is False


def test_include_sessions_false_omits_turn_in_flight(home):
    db = _db(home)
    _add_session(db, "chat1", ts=1000, text="hello")
    db.close()

    row = _row(_profiles({"include_sessions": False}), "default")
    assert "turn_in_flight" not in row
