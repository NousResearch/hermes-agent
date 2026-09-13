"""Regression for #102526.

The launch backend's lazy ``_get_db()`` handle must bind to the import-time
launch home, not whatever ``get_hermes_home()`` resolves to at first-touch
time. The desktop multiplex cron ticker installs per-profile override windows
at startup; if the first ``session.*`` RPC races into a foreign window, the
backend permanently serves the wrong profile's state.db.
"""

from __future__ import annotations

import pytest

import hermes_state_registry as registry
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


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
        registry.close_all()


def test_get_db_first_touch_under_foreign_override_uses_launch_path(launch_db_env):
    launch_home, foreign_home = launch_db_env
    token = set_hermes_home_override(str(foreign_home))
    try:
        db = server._get_db()
        assert db is not None
        assert db.db_path.resolve() == (launch_home / "state.db").resolve()
        assert not (foreign_home / "state.db").exists()
        assert server._get_db() is db
    finally:
        reset_hermes_home_override(token)


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

    seeded = registry.acquire(work / "state.db")
    seeded.create_session("work-only", source="tui", model="m")
    registry.release(seeded)

    result = server._methods["insights.get"]("rid", {"profile": "work", "days": 30})

    assert result["result"]["sessions"] == 1
    assert server._get_db().db_path.resolve() == (launch_home / "state.db").resolve()
    assert server._get_db().get_session("work-only") is None


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


def test_background_side_agent_holds_its_own_registry_reference(launch_db_env, tmp_path):
    """The parent releases its registry reference from ``AIAgent.close()``; a side agent sharing
    that object without its own reference had its store torn down under a live background turn.
    ``prompt.background`` must acquire (and release) a separate reference on the same file."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent_db = registry.acquire(profile_home / "state.db")

    with server._side_agent_session_db(parent_db) as side_db:
        assert side_db.db_path == parent_db.db_path
        registry.release_or_close(parent_db)  # parent closes mid-turn
        assert registry.stats()["live_generations"] == 1
        assert side_db._conn is not None
        side_db.create_session("bg_1", source="tui", model="m")
    assert registry.stats()["live_generations"] == 0  # side agent's reference released on exit


def test_prompt_background_turn_survives_parent_close(launch_db_env, tmp_path, monkeypatch):
    """End to end through the RPC: the side agent's ``run_conversation`` keeps a live store after
    the parent agent released its own reference."""
    from unittest.mock import patch

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent_db = registry.acquire(profile_home / "state.db")
    parent = type("Parent", (), {"model": "m", "provider": "p", "_fallback_chain": [], "_session_db": parent_db})()
    session = {"agent": parent, "session_key": "k", "profile_home": None}
    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen["db"] = kwargs["session_db"]

        def run_conversation(self, **_kw):
            registry.release_or_close(parent_db)  # parent closes / resets mid-turn
            # Still registry-owned and open: the side agent's own reference kept the generation alive
            # (no #94736 emergency reopen of a torn-down connection).
            seen["still_shared"] = seen["db"]._shared_registry_owned and seen["db"]._conn is not None
            seen["db"].create_session("bg_1", source="tui", model="m")
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

    # The registry lends ONE shared object per path; the side agent's own refcount is what kept it open.
    assert seen["db"].db_path == parent_db.db_path
    assert seen["still_shared"] is True
    assert registry.stats()["live_generations"] == 0


def test_notification_owner_gate_resolves_rotated_key_in_the_session_profile_store(launch_db_env, tmp_path):
    """A compression-rotated NAMED-PROFILE session must still claim events keyed by its compressed
    parent: the lineage lives in ``profiles/<x>/state.db``, which the launch handle cannot see, so
    the fail-closed owner gate silently dropped every post-compression notification."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    db = registry.acquire(profile_home / "state.db")
    db.create_session("parent", source="tui", model="m")
    db.append_message("parent", "user", "hello")
    db.end_session("parent", "compression")
    db.create_session("child", source="tui", model="m", parent_session_id="parent")
    db.append_message("child", "user", "later")
    registry.release(db)

    session = {"profile_home": str(profile_home), "session_key": "child", "agent": None}
    evt = {"type": "async_delegation", "session_key": "parent"}

    assert server._session_owns_notification_event("ui1", session, evt) is True
    assert server._get_db().get_session("parent") is None  # never looked up through the launch store
