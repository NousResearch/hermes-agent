"""Desktop-provided project workspaces remain durable session metadata (#108205).

A desktop client names a workspace its gateway host cannot ``isdir``-probe (Docker/remote
backend topology). The failed probe is a topology artifact, not a verdict on the path, so a
nonblank desktop-sourced cwd with the client's own provenance (``cwd_explicit``, #52589)
persists to the session row; an inherited workspace only persists when the cwd resolution
actually adopted it, and a launch-dir fallback still persists nothing.
"""

import os

from hermes_state import SessionDB
from tui_gateway import server


def _gateway_with_db(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    return db


def _create_desktop_session(params):
    response = server.handle_request(
        {"id": "create", "method": "session.create",
         "params": {"source": "desktop", **params}}
    )
    assert "result" in response, response
    return response["result"]


def test_desktop_explicit_cwd_persists_when_local_probe_cannot_see_it(monkeypatch, tmp_path):
    db = _gateway_with_db(monkeypatch, tmp_path)
    workspace = tmp_path / "desktop-only-workspace"
    assert not workspace.exists()

    sid = None
    try:
        result = _create_desktop_session({"cwd": str(workspace), "cwd_explicit": True})
        sid, stored_id = result["session_id"], result["stored_session_id"]

        session = server._sessions[sid]
        assert session["cwd"] == str(workspace)
        assert session["explicit_cwd"] is True
        assert session["cwd_chosen"] is True

        assert server._persist_session_row_for_submit("rid", session) is None
        assert db.get_session(stored_id)["cwd"] == str(workspace)
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_desktop_inherited_cwd_persists_only_when_the_resolution_adopted_it(
    monkeypatch, tmp_path
):
    """An inherited (cwd_explicit=false) workspace under a container backend: the cwd
    resolution adopts the client path (no host isdir on a container path), so it is the
    session's workspace and persists — while still yielding to a named profile's
    configured terminal.cwd inside _completion_cwd (#52589)."""
    db = _gateway_with_db(monkeypatch, tmp_path)
    container_cwd = "/workspace/repo"

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(server, "_launch_configured_cwd", lambda: None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    sid = None
    try:
        result = _create_desktop_session({"cwd": container_cwd})
        sid, stored_id = result["session_id"], result["stored_session_id"]

        session = server._sessions[sid]
        assert session["cwd"] == container_cwd
        assert session["explicit_cwd"] is True
        # A workspace the client SENT but did not claim: the settle may still re-home the chat.
        assert session["cwd_chosen"] is False

        assert server._persist_session_row_for_submit("rid", session) is None
        assert db.get_session(stored_id)["cwd"] == container_cwd
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_desktop_launch_fallback_cwd_still_persists_nothing(monkeypatch, tmp_path):
    """The inherited cwd resolves to the gateway's launch directory (os.getcwd) — an
    artifact of how the app started, not a picked workspace: the row stays NULL and the
    sidebar keeps filing it under "No workspace" (the #98924 rule)."""
    db = _gateway_with_db(monkeypatch, tmp_path)
    launch_dir = tmp_path / "gateway-launch-dir"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert os.path.realpath(server._completion_cwd({})) == os.path.realpath(launch_dir)

    sid = None
    try:
        result = _create_desktop_session({"cwd": "relative-inherited-notebook"})
        sid, stored_id = result["session_id"], result["stored_session_id"]

        session = server._sessions[sid]
        assert session["explicit_cwd"] is False

        assert server._persist_session_row_for_submit("rid", session) is None
        assert db.get_session(stored_id)["cwd"] is None
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()


def test_a_surface_that_sends_no_workspace_is_not_recorded_as_chosen(monkeypatch, tmp_path):
    """Only a client that NAMED a workspace can say whether it picked it (#52589 provenance).

    The TUI sends no ``cwd`` at all: recording that as ``cwd_explicit=false`` made the settle read
    the chat's own launch repo as an inherited default, and re-home a chat the user had started
    inside a project.
    """
    db = _gateway_with_db(monkeypatch, tmp_path)

    sid = None
    try:
        response = server.handle_request(
            {"id": "create", "method": "session.create", "params": {"source": "tui", "cols": 80}}
        )
        assert "result" in response, response
        sid = response["result"]["session_id"]

        assert server._sessions[sid]["cwd_chosen"] is None
    finally:
        if sid:
            server._sessions.pop(sid, None)
        db.close()
