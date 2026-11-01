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
