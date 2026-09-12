"""Desktop-provided project workspaces remain durable session metadata."""

from hermes_state import SessionDB
from tui_gateway import server


def test_desktop_provided_workspace_cwd_persists_when_local_probe_cannot_see_it(
    monkeypatch, tmp_path
):
    db = SessionDB(db_path=tmp_path / "state.db")
    workspace = tmp_path / "desktop-only-workspace"
    assert not workspace.exists()

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)

    sid = None
    try:
        response = server.handle_request(
            {
                "id": "create",
                "method": "session.create",
                "params": {"source": "desktop", "cwd": str(workspace)},
            }
        )
        assert "result" in response, response
        sid = response["result"]["session_id"]
        stored_id = response["result"]["stored_session_id"]

        session = server._sessions[sid]
        assert session["cwd"] == str(workspace)
        assert session["explicit_cwd"] is True

        assert server._persist_session_row_for_submit("rid", session) is None
        assert db.get_session(stored_id)["cwd"] == str(workspace)
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()
