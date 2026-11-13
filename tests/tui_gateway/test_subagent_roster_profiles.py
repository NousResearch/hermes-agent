"""``subagent.roster`` across two profiles served by one process: a background process lights only the
conversation of the profile that owns it, even when the other profile's store holds an identical route."""

import threading
from pathlib import Path

import pytest

ROUTE = "agent:main:telegram:dm:42"  # unmultiplexed / legacy route: identical in every profile's store


@pytest.fixture()
def two_profiles(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_state import SessionDB
    from tools import delegate_tool_registry as reg
    from tools.process_registry import process_registry
    from tui_gateway import launch_profile_policy, server

    launch = tmp_path / "launch-home"
    bot2 = launch / "profiles" / "bot2"
    bot2.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())

    launch_db = SessionDB(db_path=launch / "state.db")
    launch_db.create_session("default-conv", source="telegram", session_key=ROUTE)
    bot2_db = SessionDB(db_path=bot2 / "state.db")
    bot2_db.create_session("bot2-legacy", source="telegram", session_key=ROUTE)
    bot2_db.create_session("bot2-conv", source="telegram", session_key="agent:bot2:telegram:dm:42")
    bot2_db.close()
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(reg, "_active_subagents", {})
    monkeypatch.setattr(reg, "_active_subagents_lock", threading.RLock())

    procs = {}
    monkeypatch.setattr(process_registry, "list_sessions", lambda **_: [{"session_id": k} for k in procs])
    monkeypatch.setattr(process_registry, "get", procs.get)

    def spawn(pid, *, parent="", key=ROUTE):
        from tools.process_registry import ProcessSession
        procs[pid] = ProcessSession(id=pid, command="sleep 90", session_key=key, parent_session_id=parent)

    def roster(profile=None):
        params = {"profile": profile} if profile else {}
        resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": params})
        return {e["session_id"]: e["background"] for e in resp["result"]["sessions"]}

    yield spawn, roster
    launch_db.close()


def test_a_stored_parent_id_is_authoritative_across_profiles(two_profiles):
    """The default profile's process names its conversation; the identical legacy route in bot2's store
    must not light bot2's row."""
    spawn, roster = two_profiles
    spawn("proc_default", parent="default-conv")

    assert roster("bot2") == {}
    assert roster() == {"default-conv": 1}
    assert roster("bot2") == {}


def test_route_fallback_is_fenced_to_the_profile_namespace(two_profiles):
    """Without a stored parent id the route decides, but only inside the requested profile's namespace:
    an unnamespaced route is the launch profile's, a namespaced one its own profile's."""
    spawn, roster = two_profiles
    spawn("proc_legacy")
    spawn("proc_bot2", key="agent:bot2:telegram:dm:42")

    assert roster("bot2") == {"bot2-conv": 1}
    assert roster() == {"default-conv": 1}
