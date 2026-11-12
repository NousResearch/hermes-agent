"""``subagent.roster``: the sessionless, profile-scoped feed the Desktop sidebar hydrates from."""

import json
import threading

import pytest


@pytest.fixture()
def roster(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    from tools import delegate_tool_registry as reg
    from tui_gateway import server

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-a", source="tui")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(reg, "_active_subagents", {})
    monkeypatch.setattr(reg, "_active_subagents_lock", threading.RLock())

    def call():
        resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": {}})
        return {e["session_id"]: e for e in resp["result"]["sessions"]}

    yield reg._active_subagents, call
    db.close()


def test_roster_reports_live_children_without_an_attached_session(roster):
    """A conversation nobody has open (no live session record, no transport) still reports its running
    top-level children, so its sidebar row can light up after a reload."""
    active, call = roster
    active["c1"] = {"subagent_id": "c1", "depth": 0, "owner_agent_session_id": "parent-a", "status": "running"}
    active["c2"] = {"subagent_id": "c2", "depth": 0, "owner_agent_session_id": "parent-a", "status": "running"}
    active["g1"] = {"subagent_id": "g1", "depth": 1, "owner_agent_session_id": "child-x", "status": "running"}

    assert call() == {"parent-a": {"session_id": "parent-a", "subagents": 2, "background": 0}}


def test_roster_omits_conversations_stored_in_another_profile(roster):
    """One process serves several profiles: a child whose conversation is not in the requested
    profile's state.db is never reported, so its session id does not leak across profiles."""
    active, call = roster
    active["c1"] = {"subagent_id": "c1", "depth": 0, "owner_agent_session_id": "other-profile-sid"}

    assert json.dumps(call()) == "{}"


def test_roster_resolves_a_background_process_route_key_to_its_stored_conversation(roster, monkeypatch):
    """On the gateway path a ``terminal(background=true)`` process carries the conversation's ROUTE
    (``agent:main:local:dm:...``) as its ``session_key``, not the stored id; the roster still reports it
    under the stored conversation that route belongs to, so a background-only row lights after a reload."""
    from tools.process_registry import ProcessSession, process_registry
    from tui_gateway import server

    _, call = roster
    route = "agent:main:local:dm:desktop-chat-1"
    server._get_db().create_session("local-abc123", source="local", session_key=route)
    proc = ProcessSession(id="proc_route", command="sleep 90", session_key=route)
    monkeypatch.setattr(process_registry, "list_sessions", lambda **_: [{"session_id": proc.id}])
    monkeypatch.setattr(process_registry, "get", lambda pid: proc if pid == proc.id else None)

    assert call() == {"local-abc123": {"session_id": "local-abc123", "subagents": 0, "background": 1}}
