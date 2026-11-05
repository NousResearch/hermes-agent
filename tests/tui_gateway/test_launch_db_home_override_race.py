"""Regression for #102526.

The launch backend's lazy ``_get_db()`` handle must bind to the import-time
launch home, not whatever ``get_hermes_home()`` resolves to at first-touch
time. The desktop multiplex cron ticker installs per-profile override windows
at startup; if the first ``session.*`` RPC races into a foreign window, the
backend permanently serves the wrong profile's state.db.
"""

from __future__ import annotations

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import close_all_stores, live_stores, open_persistence, profile_scope, store_location
from tui_gateway import server


def _store_file(handle):
    """The file holding the store ``handle`` is bound to."""
    return store_location(handle.scope).resolve()


def _open_store_count() -> int:
    """How many stores this process holds open."""
    with live_stores() as handles:
        return len(handles)


def _close_launch_handle():
    """Release the launch handle a test opened through ``_get_db``."""
    if server._db is not None:
        server._db.close()


@pytest.fixture()
def launch_db_env(monkeypatch, tmp_path):
    launch_home = tmp_path / "launch"
    foreign_home = tmp_path / "foreign"
    launch_home.mkdir()
    foreign_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", str(launch_home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)
    try:
        yield launch_home, foreign_home
    finally:
        _close_launch_handle()
        close_all_stores()


def test_get_db_first_touch_under_foreign_override_uses_launch_path(launch_db_env):
    launch_home, foreign_home = launch_db_env
    token = set_hermes_home_override(str(foreign_home))
    try:
        db = server._get_db()
        assert db is not None
        assert _store_file(db) == (launch_home / "state.db").resolve()
        assert not (foreign_home / "state.db").exists()
        assert server._get_db() is db
    finally:
        reset_hermes_home_override(token)


def test_get_db_follows_a_process_home_redirected_after_import(monkeypatch, tmp_path):
    """The launch handle resolves ``HERMES_HOME`` at first use, not at import (#112692): a harness
    that redirects the env after ``tui_gateway.server`` is imported must not open the import-time
    home's state.db. Only the context-local override is ignored (#102526), never the process env."""
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setenv("HERMES_HOME", str(redirected))
    try:
        db = server._get_db()
        assert db is not None
        assert _store_file(db) == (redirected / "state.db").resolve()
    finally:
        _close_launch_handle()
        close_all_stores()


def test_insights_get_reads_the_requested_profile_store_not_the_launch_handle(launch_db_env, monkeypatch, tmp_path):
    """``insights.get {profile}`` was ``scoped=True`` then ``_get_db()``: a scoped first touch pinned
    the launch handle to the foreign store. It must count the requested profile's sessions through
    ``_profile_db`` and leave the launch handle on the launch home."""
    launch_home, _foreign = launch_db_env
    profiles_root = tmp_path / "profiles"
    work = profiles_root / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles_root / name)
    monkeypatch.setattr(server, "_canonical_profile_request", lambda name: name or None)

    with open_persistence(profile_scope(home=work)) as seeded:
        seeded.sessions.create(NewSession(session_id="work-only", source="tui", model="m"))

    result = server._methods["insights.get"]("rid", {"profile": "work", "days": 30})

    assert result["result"]["sessions"] == 1
    assert _store_file(server._get_db()) == (launch_home / "state.db").resolve()
    assert server._get_db().sessions.get("work-only") is None


def test_background_side_agent_persists_into_the_parent_agent_store(launch_db_env, monkeypatch):
    """``prompt.background`` side agents write ``bg_*`` rows next to their parent's transcript: a
    named-profile chat's parent holds a dedicated profile handle, and handing the launch handle
    instead made those rows show up in the default profile's history."""
    import types

    parent_db = object()
    agent = types.SimpleNamespace(model="m", provider="p", _fallback_chain=[], _session_db=parent_db)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])

    assert server._background_agent_kwargs(agent, "bg_1")["session_db"] is parent_db


def test_background_side_agent_holds_its_own_store_reference(launch_db_env, tmp_path):
    """The parent releases its store reference from ``AIAgent.close()``; a side agent sharing
    that object without its own reference had its store closed under a live background turn.
    ``prompt.background`` must open (and release) a separate reference on the same store."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent_db = open_persistence(profile_scope(home=profile_home), eager=True)

    with server._side_agent_session_db(parent_db) as side_db:
        assert side_db is not parent_db
        assert _store_file(side_db) == _store_file(parent_db)
        parent_db.close()  # parent closes mid-turn
        assert _open_store_count() == 1
        side_db.sessions.create(NewSession(session_id="bg_1", source="tui", model="m"))
    assert _open_store_count() == 0  # side agent's reference released on exit


def test_prompt_background_turn_survives_parent_close(launch_db_env, tmp_path, monkeypatch):
    """End to end through the RPC: the side agent's ``run_conversation`` keeps a live store after
    the parent agent released its own reference."""
    from unittest.mock import patch

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent_db = open_persistence(profile_scope(home=profile_home), eager=True)
    parent = type("Parent", (), {"model": "m", "provider": "p", "_fallback_chain": [], "_session_db": parent_db})()
    session = {"agent": parent, "session_key": "k", "profile_home": None}
    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen["db"] = kwargs["session_db"]

        def run_conversation(self, **_kw):
            parent_db.close()  # parent closes / resets mid-turn
            # Still the one shared store, and open: the side agent's own reference kept it alive
            # (no #94736 emergency reopen of a closed store).
            seen["still_shared"] = seen["db"] is not parent_db and _open_store_count() == 1
            seen["db"].sessions.create(NewSession(session_id="bg_1", source="tui", model="m"))
            return {"final_response": "ok"}

    class InlineThread:
        def __init__(self, target=None, **_kw):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_reasoning_config", lambda *_a, **_kw: None)
    with patch("tui_gateway.server.threading.Thread", InlineThread), \
            patch("run_agent.AIAgent", FakeAgent), \
            patch("tui_gateway.server._sess", return_value=(session, None)), \
            patch("tui_gateway.server._set_session_context", return_value=None), \
            patch("tui_gateway.server._clear_session_context"), \
            patch("tui_gateway.server._session_cwd", return_value=str(tmp_path)), \
            patch("tui_gateway.server._emit"):
        server._methods["prompt.background"]("rid", {"text": "hi", "session_id": "ui1"})

    # Every handle on one profile shares ONE store; the side agent's own reference is what kept it open.
    assert _store_file(seen["db"]) == _store_file(parent_db)
    assert seen["still_shared"] is True
    assert _open_store_count() == 0


def test_notification_owner_gate_resolves_rotated_key_in_the_session_profile_store(launch_db_env, tmp_path):
    """A compression-rotated NAMED-PROFILE session must still claim events keyed by its compressed
    parent: the lineage lives in ``profiles/<x>/state.db``, which the launch handle cannot see, so
    the fail-closed owner gate silently dropped every post-compression notification."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    with open_persistence(profile_scope(home=profile_home)) as db:
        db.sessions.create(NewSession(session_id="parent", source="tui", model="m"))
        db.messages.append("parent", NewMessage(role="user", content="hello"))
        db.sessions.end("parent", "compression")
        db.sessions.create(NewSession(session_id="child", source="tui", model="m", parent_session_id="parent"))
        db.messages.append("child", NewMessage(role="user", content="later"))

    session = {"profile_home": str(profile_home), "session_key": "child", "agent": None}
    evt = {"type": "async_delegation", "session_key": "parent"}

    assert server._session_owns_notification_event("ui1", session, evt) is True
    assert server._get_db().sessions.get("parent") is None  # never looked up through the launch store


def test_restored_delegation_requires_a_live_notification_consumer():
    """Regression for #138343: a persisted completion cannot wake a detached runtime."""
    session = {"session_key": "chat", "transport": None}
    event = {"type": "async_delegation", "session_key": "chat", "restored": True, "startup_replay": True}

    assert server._session_owns_notification_event("ui1", session, event) is False

    session["transport"] = object()
    assert server._session_owns_notification_event("ui1", session, event) is True


def test_foreign_profile_poller_requeues_event_owned_through_another_profiles_lineage(launch_db_env, tmp_path):
    """Two profiles share one completion queue. Profile B's poller dequeues an event keyed on profile
    A's compressed parent: B cannot resolve A's lineage in its own store, so both of B's ownership
    checks were false and ``_notif_handle_event`` dropped the event. B must recognise A's live
    continuation as the owner and hand the event back."""
    import threading
    from tools.process_registry import process_registry

    a_home, b_home = tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"
    a_home.mkdir(parents=True)
    b_home.mkdir(parents=True)
    with open_persistence(profile_scope(home=a_home)) as db:
        db.sessions.create(NewSession(session_id="parent", source="tui", model="m"))
        db.sessions.end("parent", "compression")
        db.sessions.create(NewSession(session_id="child", source="tui", model="m", parent_session_id="parent"))

    def _sess(home, key):
        return {"profile_home": str(home), "session_key": key, "agent": None,
                "history_lock": threading.RLock(), "running": False}
    sess_a, sess_b = _sess(a_home, "child"), _sess(b_home, "other")
    evt = {"type": "async_delegation", "session_key": "parent", "delegation_id": "d1", "results": []}
    queue = process_registry.completion_queue
    while not queue.empty():
        queue.get_nowait()
    with server._sessions_lock:
        saved = dict(server._sessions)
        server._sessions.clear()
        server._sessions.update({"uiA": sess_a, "uiB": sess_b})
    try:
        assert server._notif_handle_event("uiB", sess_b, dict(evt), set(), process_registry, lambda e: "t", None) is True
        assert queue.qsize() == 1  # requeued for A, not dropped
        assert server._notification_event_belongs_elsewhere("uiA", sess_a, queue.get_nowait()) is False
    finally:
        with server._sessions_lock:
            server._sessions.clear()
            server._sessions.update(saved)
