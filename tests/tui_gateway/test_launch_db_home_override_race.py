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
    import hermes_state
    # Exercise dynamic temp-home resolution, not conftest's fixed DB constant.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
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
