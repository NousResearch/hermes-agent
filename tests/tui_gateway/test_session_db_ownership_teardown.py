"""Dedicated profile ``SessionDB`` handles must be closed by whoever ends up owning them.

Companion to ``test_session_resume_db_ownership.py``, which pins the paths that
return BEFORE a handle reaches an agent. This file pins the other half — the two
gaps that survived that change:

1.  **Teardown.** Once ownership transfers, the agent is the owner, and
    ``AIAgent.close()`` (reached from ``_teardown_session`` on ``session.close``
    and the orphaned-session reaper) only called ``session_db.end_session()`` —
    which finalizes the session ROW, not the connection. A successfully resumed
    profile session therefore kept its dedicated SQLite handle, its db/-wal/-shm
    fds and its background token-writer thread alive for the life of the gateway.
    Ownership is explicit (``_owns_session_db``) precisely so this close can
    happen without ever touching the SHARED launch handle, which outlives every
    agent.

2.  **The other pre-transfer build paths.** ``session.resume`` was not the only
    profile-scoped open. The deferred builder (``_start_agent_build``), the
    branch handler and the compute host all open a dedicated handle, pass it to
    ``_make_agent``, and had no close on their failure paths.

The direction that must NOT regress is asserted everywhere: the shared launch
handle is never closed, and a handle that WAS transferred is not closed twice.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _RecordingDB:
    """Stand-in for ``hermes_state.SessionDB`` that counts ``close()`` calls."""

    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path
        self.closed = 0

    def close(self):
        self.closed += 1

    def end_session(self, *_a, **_k):
        pass


# ---------------------------------------------------------------------------
# 1. AIAgent.close() — the teardown owner
# ---------------------------------------------------------------------------


def _bare_agent(**attrs):
    """An AIAgent with __init__ bypassed, carrying only what close() reads."""
    from unittest.mock import patch

    with patch("run_agent.AIAgent.__init__", return_value=None):
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sid"
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_close_closes_a_dedicated_handle_it_owns():
    """The gap the review found: end_session() is not close()."""
    db = _RecordingDB()
    agent = _bare_agent(_session_db=db, _owns_session_db=True)

    agent.close()

    assert db.closed == 1


def test_close_never_closes_a_shared_handle():
    """The direction that must not regress.

    Almost every agent is handed the SHARED launch handle, which outlives it and
    is used by every other live session. Closing that on teardown would break
    every other chat in the gateway, so ownership defaults to False and only the
    dedicated-open sites set it.
    """
    db = _RecordingDB()
    agent = _bare_agent(_session_db=db, _owns_session_db=False)

    agent.close()

    assert db.closed == 0


def test_close_without_ownership_attribute_does_not_close():
    """Agents built before this flag existed (and test doubles) must be safe."""
    db = _RecordingDB()
    agent = _bare_agent(_session_db=db)  # no _owns_session_db at all

    agent.close()  # must not raise

    assert db.closed == 0


def test_close_is_idempotent_for_an_owned_handle():
    """close() is documented as safe to call repeatedly.

    Teardown can genuinely reach an agent twice (session.close racing the
    orphaned-session reaper), so the second call must not double-close.
    """
    db = _RecordingDB()
    agent = _bare_agent(_session_db=db, _owns_session_db=True)

    agent.close()
    agent.close()

    assert db.closed == 1


def test_raising_close_is_swallowed_and_not_retried():
    """A raising ``session_db.close()`` must not escape ``agent.close()``,
    and the flag stays cleared so a second ``agent.close()`` does not
    re-attempt the close (the flag is dropped BEFORE the close call —
    the documented-idempotency ordering)."""
    attempts: list[int] = []

    class _Raising(_RecordingDB):
        def close(self):
            attempts.append(1)
            raise RuntimeError("disk gone")

    db = _Raising()
    agent = _bare_agent(_session_db=db, _owns_session_db=True)

    agent.close()  # must not raise
    agent.close()  # flag already cleared — no second attempt

    assert attempts == [1]
    assert getattr(agent, "_owns_session_db") is False


def test_close_still_ends_the_session_row_before_closing():
    """Ordering matters: the row is finalized THROUGH the handle we then close."""
    calls: list[str] = []

    class _Ordered(_RecordingDB):
        def end_session(self, *_a, **_k):
            calls.append("end_session")

        def close(self):
            calls.append("close")
            super().close()

    db = _Ordered()
    agent = _bare_agent(
        _session_db=db, _owns_session_db=True, _end_session_on_close=True
    )

    agent.close()

    assert calls == ["end_session", "close"]


def test_lazy_recall_open_is_owned_by_the_agent(monkeypatch):
    """The agent's own lazy open has no other owner, so close() must release it.

    ``_get_session_db_for_recall`` opens a handle when no frontend supplied one.
    Nothing else ever holds a reference to it, so before this change it was
    unconditionally abandoned.
    """
    opened: list[_RecordingDB] = []

    def _factory(*_a, **_k):
        db = _RecordingDB()
        opened.append(db)
        return db

    monkeypatch.setattr("hermes_state_registry.acquire", _factory)

    agent = _bare_agent(_session_db=None, _persist_disabled=False)
    got = agent._get_session_db_for_recall()

    assert got is opened[0]
    assert agent._owns_session_db is True

    agent.close()
    assert opened[0].closed == 1


# ---------------------------------------------------------------------------
# 2. _transfer_db_to_agent — the transfer contract
# ---------------------------------------------------------------------------


def test_transfer_marks_the_agent_that_holds_the_handle():
    db = _RecordingDB()
    agent = types.SimpleNamespace(_session_db=db, _owns_session_db=False)

    assert server._transfer_db_to_agent(agent, db) is True
    assert agent._owns_session_db is True


def test_transfer_is_refused_when_the_agent_holds_a_different_handle():
    """A refusal is the signal that the caller still owns the handle.

    If the build handed the agent some other db, marking it would make the agent
    close a handle it does not hold while the real one leaks.
    """
    db, other = _RecordingDB(), _RecordingDB()
    agent = types.SimpleNamespace(_session_db=other, _owns_session_db=False)

    assert server._transfer_db_to_agent(agent, db) is False
    assert agent._owns_session_db is False


@pytest.mark.parametrize(
    "agent, db",
    [
        (None, _RecordingDB()),
        (types.SimpleNamespace(_session_db=None), None),
    ],
)
def test_transfer_is_refused_for_missing_operands(agent, db):
    assert server._transfer_db_to_agent(agent, db) is False


def test_transfer_is_refused_for_the_shared_launch_handle(monkeypatch):
    """Defense in depth for #91610: identity alone passes for the SHARED
    launch handle — a launch-profile agent IS holding it — so ownership
    would make session.close() tear down the process-wide database under
    every other session. The transfer must refuse it even when a caller
    invokes the transfer incorrectly."""
    shared = _RecordingDB()
    monkeypatch.setattr(server, "_get_db", lambda: shared)
    agent = types.SimpleNamespace(_session_db=shared, _owns_session_db=False)

    assert server._transfer_db_to_agent(agent, shared) is False
    assert agent._owns_session_db is False


def test_get_db_returns_the_cached_instance(monkeypatch):
    """The identity defense (``db is _get_db()``) only works while _get_db
    hands out ONE process-wide instance. Pin the caching semantics: once a
    handle exists, repeated calls return the same object rather than
    constructing per-call wrappers (review finding on #91631)."""
    sentinel = types.SimpleNamespace(closed=0)
    monkeypatch.setattr(server, "_db", sentinel)
    monkeypatch.setattr(server, "_db_error", None)

    assert server._get_db() is sentinel
    assert server._get_db() is server._get_db()


# ---------------------------------------------------------------------------
# 3. The deferred builder — _start_agent_build
# ---------------------------------------------------------------------------


@pytest.fixture()
def build_env(monkeypatch, tmp_path):
    """Neutralize everything the deferred build touches except db ownership."""
    profile_home = tmp_path / "work"
    profile_home.mkdir()

    opened: list[_RecordingDB] = []

    def _factory(db_path=None, **kwargs):
        db = _RecordingDB(db_path=db_path, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr("hermes_state_registry.acquire", _factory)
    for name, value in [
        ("_set_session_context", lambda _key: []),
        ("_clear_session_context", lambda _tokens: None),
        ("_wire_callbacks", lambda _sid: None),
        ("_config_model_target", lambda: None),
        ("_load_memory_notifications", lambda: False),
        ("_start_notification_poller", lambda _sid, _session: None),
        ("_notify_session_boundary", lambda *a, **k: None),
        ("_session_info", lambda *a, **k: {}),
        ("_probe_config_health", lambda _cfg: None),
        ("_load_cfg", lambda: {}),
        ("_emit", lambda *a, **k: None),
        ("_schedule_mcp_late_refresh", lambda *a, **k: None),
        ("_session_source", lambda _current: None),
        ("_child_run_active", lambda _key: False),
    ]:
        if hasattr(server, name):
            monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "set_hermes_home_override", lambda _home: None)
    monkeypatch.setattr(server, "reset_hermes_home_override", lambda _tok: None)
    yield types.SimpleNamespace(opened=opened, profile_home=str(profile_home))


def _run_build(sid, session):
    """Drive _start_agent_build to completion (it builds on a daemon thread)."""
    server._start_agent_build(sid, session)
    assert session["agent_ready"].wait(timeout=10), "build thread did not finish"


def _session(profile_home):
    return {
        "session_key": "key-1",
        "agent_ready": threading.Event(),
        "profile_home": profile_home,
    }


@pytest.fixture()
def registered(monkeypatch):
    """Register/unregister sessions in the module-global _sessions map."""
    added: list[str] = []

    def _add(sid, session):
        with server._sessions_lock:
            server._sessions[sid] = session
        added.append(sid)

    yield _add
    with server._sessions_lock:
        for sid in added:
            server._sessions.pop(sid, None)


def test_deferred_build_closes_the_handle_when_the_build_fails(
    build_env, registered, monkeypatch
):
    """The failure path the review named: nothing takes the handle, so close it."""
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no provider")),
    )
    sid, session = "sid-fail", _session(build_env.profile_home)
    registered(sid, session)

    _run_build(sid, session)

    assert session.get("agent") is None
    assert len(build_env.opened) == 1
    assert build_env.opened[0].closed == 1


def test_deferred_build_transfers_the_handle_on_success(
    build_env, registered, monkeypatch
):
    """A built, retained agent becomes the owner — and the builder must not close."""
    captured: dict = {}

    def _fake_make_agent(sid, key, session_db=None, **_kwargs):
        captured["db"] = session_db
        return types.SimpleNamespace(_session_db=session_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    sid, session = "sid-ok", _session(build_env.profile_home)
    registered(sid, session)

    _run_build(sid, session)

    db = build_env.opened[0]
    assert captured["db"] is db
    assert db.closed == 0
    # Ownership landed on the agent, so _teardown_session releases it later.
    assert session["agent"]._owns_session_db is True


def test_deferred_build_uses_the_preserved_desktop_workspace_provenance(
    build_env, registered, monkeypatch
):
    captured: dict = {}

    def _fake_make_agent(*_args, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(_session_db=session_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    monkeypatch.setattr(
        server, "_session_source", lambda current: current.get("source")
    )
    sid, session = "sid-workspace", _session(build_env.profile_home)
    session.update(
        {
            "cwd": "C:/picked/repo",
            "explicit_cwd": True,
            "source": "desktop",
        }
    )
    registered(sid, session)

    _run_build(sid, session)

    assert captured["context_cwd_is_launch_artifact"] is False


def test_deferred_build_closes_the_handle_when_the_session_is_reaped_midbuild(
    build_env, registered, monkeypatch
):
    """A discarded agent is never torn down, so transferring to it would leak.

    ``_build`` already computes ``replaced`` for the approval-notifier cleanup.
    When the session was swapped out from under the build, the agent it produced
    is unreachable — ``_teardown_session`` will never call close() on it — so the
    handle has to be closed right here instead of handed over.
    """

    def _fake_make_agent(sid, key, session_db=None, **_kwargs):
        # Simulate a concurrent reap landing while the agent was being built.
        with server._sessions_lock:
            server._sessions[sid] = {"session_key": "someone-else"}
        return types.SimpleNamespace(_session_db=session_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    sid, session = "sid-reaped", _session(build_env.profile_home)
    registered(sid, session)

    _run_build(sid, session)

    db = build_env.opened[0]
    assert db.closed == 1
    assert session["agent"]._owns_session_db is False


def test_deferred_build_never_opens_or_closes_for_the_launch_profile(
    build_env, registered, monkeypatch
):
    """No profile scope -> no dedicated handle; the shared one is untouched."""
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *a, **k: types.SimpleNamespace(_session_db=None, _owns_session_db=False),
    )
    sid, session = "sid-launch", _session(None)
    registered(sid, session)

    _run_build(sid, session)

    assert build_env.opened == []
