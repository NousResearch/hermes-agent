"""``session.resume`` must not abandon the profile-scoped SessionDB it opens.

In app-global remote mode a resume for another local profile opens a DEDICATED
``SessionDB(db_path=<profile>/state.db)`` handle (the ``session.resume`` handler
in tui_gateway/methods_session.py). That handle is the caller's to close until
it is handed to the long-lived agent by ``_init_session`` — and
``_init_session`` never closes a caller-supplied ``session_db`` (its
``_init_owns_db`` stays False for that case).

Every early return before that transfer used to drop the handle on the floor,
so its SQLite fds stayed open for as long as anything kept the instance
reachable — and a ``SessionDB`` pins ITSELF once its background token writer
starts (``atexit.register(self._drain_token_queue_at_exit)``, which only
``close()`` unregisters).

Pinned here, in both directions:

* the pre-transfer early returns (session-not-found, "resume failed", the
  live-session fast path, the deferred cold-resume return) all close it;
* a resume that COMPLETES the transfer leaves it open — closing there would
  fault every later turn with "Cannot operate on a closed database";
* an ``_init_session`` that raises AFTER registering the session must drop that
  half-built registration, otherwise the live-session fast path serves a
  session whose db we just closed on every later resume of the same id;
* the shared launch-profile handle (``_get_db()``) is never closed, since it
  outlives the RPC.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _RecordingDB:
    """Stand-in for ``hermes_state.SessionDB`` that counts ``close()`` calls.

    Implements only the surface ``session.resume`` touches.
    """

    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path
        self.closed = 0
        self.rows: dict = {}
        self.reopen_error: Exception | None = None

    def close(self):
        self.closed += 1

    def get_session(self, target):
        return self.rows.get(target)

    def get_session_by_title(self, _target):
        return None

    def resolve_resume_session_id(self, target):
        return target

    def reopen_session(self, _target):
        if self.reopen_error is not None:
            raise self.reopen_error

    def get_resume_conversations(self, _target):
        return ([], [])

    def get_ancestor_display_prefix(self, _target):
        return []

    def get_messages_as_conversation(self, _target, **_kwargs):
        return []


@pytest.fixture()
def profile_dbs(monkeypatch, tmp_path):
    """Route profile-scoped opens to _RecordingDB; yield the list of opens.

    ``params['profile']`` selects the profile scope; omitting it resolves to
    the launch profile (``_profile_home`` -> None) and the shared handle.
    """
    opened: list[_RecordingDB] = []
    profile_home = tmp_path / "work"
    profile_home.mkdir()

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(
        server, "_profile_home", lambda profile: profile_home if profile else None
    )
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    # The handler builds nothing on the paths under test; keep it hermetic and
    # off the real agent/secret/HERMES_HOME machinery.
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    known = set(server._sessions)
    yield opened
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _resume(**params):
    return server.handle_request(
        {"id": "1", "method": "session.resume", "params": params}
    )


def test_resume_closes_profile_db_when_session_not_found(profile_dbs):
    """The 'session not found' early return must not leak the handle."""
    resp = _resume(session_id="missing", profile="work")

    assert resp["error"]["code"] == 4007
    assert len(profile_dbs) == 1
    assert profile_dbs[0].closed == 1


def test_resume_closes_profile_db_when_reopen_fails(profile_dbs, monkeypatch):
    """The 'resume failed' early return must not leak the handle."""

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": ""}
        db.reopen_error = RuntimeError("database is locked")
        profile_dbs.append(db)
        return db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)

    resp = _resume(session_id="s1", profile="work")

    assert resp["error"]["code"] == 5000
    assert "resume failed" in resp["error"]["message"]
    assert profile_dbs[0].closed == 1


def test_resume_closes_profile_db_on_live_session_fast_path(profile_dbs, monkeypatch):
    """Re-resuming an already-live session returns early — and must close.

    This is the hottest leak in practice: every reconnect/tile-paint resume of
    a chat that is already live takes this path, so the fd growth tracked
    reconnect count rather than anything rare.
    """

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": ""}
        profile_dbs.append(db)
        return db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key: ("live-sid", {}))
    monkeypatch.setattr(
        server,
        "_live_session_payload",
        lambda sid, session, **_kwargs: {"session_id": sid},
    )
    monkeypatch.setattr(server, "_child_run_active", lambda _key: False)

    resp = _resume(session_id="s1", profile="work")

    assert resp["result"]["resumed"] == "s1"
    assert profile_dbs[0].closed == 1


def test_resume_closes_profile_db_on_deferred_cold_resume(profile_dbs, monkeypatch):
    """The DEFAULT resume path returns before any transfer — and must close.

    A cold resume without ``eager_build`` registers a deferred session record
    (no agent, no db reference) and builds the agent later off the response
    path, so the handle opened here is never handed to anyone.
    """

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": ""}
        profile_dbs.append(db)
        return db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})

    resp = _resume(session_id="s1", profile="work")

    assert resp["result"]["session_key"] == "s1"
    assert resp["result"]["status"] == "idle"
    assert profile_dbs[0].closed == 1


def test_resume_hands_profile_db_to_deferred_history_worker(profile_dbs, monkeypatch):
    """Incremental hydration owns the profile handle until its read completes."""
    history_started = threading.Event()
    release_history = threading.Event()
    close_completed = threading.Event()

    class _BlockingDB(_RecordingDB):
        def close(self):
            super().close()
            close_completed.set()

        def get_resume_conversations(self, _target):
            history_started.set()
            assert release_history.wait(timeout=2.0)
            assert self.closed == 0
            return ([], [])

    def _factory(db_path=None, **kwargs):
        db = _BlockingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": "", "message_count": 0}
        profile_dbs.append(db)
        return db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})

    try:
        resp = _resume(
            session_id="s1", profile="work", defer_history=True
        )
        sid = resp["result"]["session_id"]
        db = profile_dbs[0]
        assert history_started.wait(timeout=1.0)
        assert db.closed == 0

        release_history.set()
        assert server._sessions[sid]["resume_history_ready"].wait(timeout=1.0)
        assert close_completed.wait(timeout=1.0)
        assert db.closed == 1
    finally:
        release_history.set()


def test_resume_keeps_profile_db_open_after_ownership_transfer(profile_dbs, monkeypatch):
    """A COMPLETED resume transfers the handle to the agent — do not close it.

    Guards the other direction: closing here would hand the live session a dead
    connection and fault every subsequent turn.
    """
    captured: dict = {}

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": ""}
        profile_dbs.append(db)
        return db

    def _fake_make_agent(sid, key, session_db=None, **_kwargs):
        captured["agent_db"] = session_db
        return types.SimpleNamespace(model="test")

    def _fake_init_session(sid, key, agent, history, session_db=None, **_kwargs):
        captured["init_db"] = session_db

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    monkeypatch.setattr(server, "_init_session", _fake_init_session)
    monkeypatch.setattr(server, "_set_session_context", lambda _target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})
    monkeypatch.setattr(server, "_session_info", lambda agent, *a: {"model": "test"})

    resp = _resume(session_id="s1", profile="work", eager_build=True)

    assert resp["result"]["session_key"] == "s1"
    db = profile_dbs[0]
    # The agent and the live session both took THIS handle...
    assert captured["agent_db"] is db
    assert captured["init_db"] is db
    # ...so the handler must have released ownership instead of closing it.
    assert db.closed == 0


def test_resume_drops_half_built_session_when_init_session_raises(
    profile_dbs, monkeypatch
):
    """Closing the handle is only safe if the failed registration goes with it.

    ``_init_session`` publishes ``_sessions[sid]`` BEFORE its first read through
    the handle. If that read raises, the handle is still ours (and gets closed),
    so the half-built session must not stay registered — otherwise the
    live-session fast path serves that dead session on every later resume of the
    same id, forever, with "'NoneType' object has no attribute 'execute'".
    """
    captured: dict = {}

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        db.rows["s1"] = {"id": "s1", "cwd": ""}
        profile_dbs.append(db)
        return db

    def _fake_init_session(sid, key, agent, history, session_db=None, **_kwargs):
        # Same ordering as the real one: register, THEN read through the db.
        captured["sid"] = sid
        with server._sessions_lock:
            server._sessions[sid] = {"agent": agent, "session_key": key}
        raise RuntimeError("database is locked")

    monkeypatch.setattr("hermes_state.SessionDB", _factory)
    monkeypatch.setattr(
        server, "_make_agent", lambda *a, **k: types.SimpleNamespace(model="test")
    )
    monkeypatch.setattr(server, "_init_session", _fake_init_session)
    monkeypatch.setattr(server, "_set_session_context", lambda _target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _found: {})

    resp = _resume(session_id="s1", profile="work", eager_build=True)

    assert resp["error"]["code"] == 5000
    assert profile_dbs[0].closed == 1
    assert captured["sid"] not in server._sessions


def test_resume_never_closes_shared_launch_db(profile_dbs, monkeypatch):
    """No profile scope -> the shared ``_get_db()`` handle, which we never close."""
    shared = _RecordingDB(db_path="launch")
    monkeypatch.setattr(server, "_get_db", lambda: shared)

    resp = _resume(session_id="missing")

    assert resp["error"]["code"] == 4007
    assert profile_dbs == []  # no dedicated handle was opened
    assert shared.closed == 0
