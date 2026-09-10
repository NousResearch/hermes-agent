"""The composer receives its durable binding, not the foreground workspace."""
import json
import subprocess

import pytest

import tui_gateway.server as server
from hermes_state import SessionDB


def call(method, **params):
    response = server._methods[method](1, params)
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("mode", ["folder", "current", "worktree"])
@pytest.mark.parametrize("live_method", ["session.resume", "session.activate"])
def test_binding_is_exposed_on_create_cold_resume_and_live_info(tmp_path, monkeypatch, mode, live_method):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    folder = tmp_path / "project"
    folder.mkdir()
    if mode != "folder":
        subprocess.run(["git", "init", "-b", "main", str(folder)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(folder), "-c", "user.name=Test", "-c", "user.email=test@localhost",
                        "commit", "--allow-empty", "-m", "base"], check=True, capture_output=True)
    prepared = call("projects.workspace.prepare", path=str(folder), mode=mode, requestId="summary")
    project = call("projects.get", id=prepared["projectId"])["project"]
    assert prepared.get("projectName") == project["name"]
    assert prepared.get("mode") == mode
    # Existing chats predate display metadata. Their original binding must work.
    prepared.pop("mode", None)
    prepared.pop("projectName", None)
    created = call("session.create", source="desktop", cwd=prepared["cwd"], coding_workspace=prepared)
    stored_id = created["stored_session_id"]
    persisted = json.loads(db.get_session(stored_id)["model_config"])["coding_workspace"]
    assert created["info"].get("coding_workspace") == persisted
    server._sessions.pop(created["session_id"])
    db.close()
    db = SessionDB(db_path)

    def fail_agent_build(*args, **kwargs):
        pytest.fail("Lazy resume and activation must not construct or schedule an agent")

    with monkeypatch.context() as lazy_patch:
        lazy_patch.setattr(server, "_schedule_agent_build", fail_agent_build)
        lazy_patch.setattr(server, "_start_agent_build", fail_agent_build)
        resumed = call("session.resume", session_id=stored_id, source="desktop", lazy=True)
        assert resumed["info"].get("coding_workspace") == persisted
        live = server._sessions[resumed["session_id"]]
        assert live["agent"] is None
        assert server._session_info(None, live).get("coding_workspace") == persisted
        target = stored_id if live_method == "session.resume" else resumed["session_id"]
        reattached = call(live_method, session_id=target, source="desktop", lazy=True)
        assert reattached["session_id"] == resumed["session_id"]
        assert reattached["session_key"] == stored_id
        assert server._sessions[reattached["session_id"]] is live
        assert live["agent"] is None
        assert reattached["info"].get("coding_workspace") == persisted
    ordinary = call("session.create", source="desktop", cwd=str(folder))
    assert ordinary["info"].get("coding_workspace") is None
    assert json.loads(db.get_session(stored_id)["model_config"])["coding_workspace"] == persisted
    db.close()
