"""A workspace change on a session with a BUILT agent settles: result frame and one typed session.info.

``_cwd_info`` returned ``server._session_info(...)`` — a base ``SessionLiveInfo`` — for a session with an
agent, and both consumers validated it into a subclass (``SessionCwdSetResult``, ``SessionInfoPayload``).
Pydantic rejects a base-class instance there, so ``session.cwd.set`` / ``session.workspace.move`` changed
and persisted the cwd, then answered an error with no event. Lazy sessions took the mapping branch and
never showed it; these tests keep both branches."""

from __future__ import annotations

import contextlib

import pytest

from tui_gateway import server
from tui_gateway.contracts.common import SessionLiveInfo


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    new_cwd = tmp_path / "workspace"
    new_cwd.mkdir()
    events, rows = [], []

    class DB:
        def get_session(self, _sid):
            return {"id": _sid}

        def update_session_cwd(self, session_id, cwd, *_a, **_k):
            rows.append((session_id, cwd))

        def close(self):
            pass

    @contextlib.contextmanager
    def _profile_db(_params, *, writer=False):
        yield DB()

    import tools.terminal_tool_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "cleanup_vm", lambda _key: None)
    monkeypatch.setattr(server, "_get_db", lambda: DB())
    monkeypatch.setattr(server, "_profile_db", _profile_db)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "main")
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda cwd: str(new_cwd))
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    # Production shape: a real SessionLiveInfo (the base class), never a dict.
    monkeypatch.setattr(server, "_session_info", lambda agent, session: SessionLiveInfo(cwd=session["cwd"], branch="main"))
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)) or True)
    return new_cwd, events, rows


def _session(sid: str, tmp_path, *, agent):
    session = {"session_key": sid, "cwd": str(tmp_path / "old"), "running": False, "agent": agent}
    server._sessions[sid] = session
    return session


@pytest.mark.parametrize("agent", [object(), None], ids=["built-agent", "lazy"])
def test_session_cwd_set_settles_with_one_typed_session_info(workspace, tmp_path, agent):
    new_cwd, events, _rows = workspace
    session = _session("cwd-sid", tmp_path, agent=agent)
    try:
        response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "session.cwd.set",
                                          "params": {"session_id": "cwd-sid", "cwd": str(new_cwd)}})
    finally:
        server._sessions.pop("cwd-sid", None)

    assert response["result"]["cwd"] == str(new_cwd) == session["cwd"], response
    assert [(e, s, p.cwd) for e, s, p in events] == [("session.info", "cwd-sid", str(new_cwd))]


@pytest.mark.parametrize("agent", [object(), None], ids=["built-agent", "lazy"])
def test_session_workspace_move_settles_with_one_typed_session_info(workspace, tmp_path, agent):
    new_cwd, events, rows = workspace
    live = _session("move-sid", tmp_path, agent=agent)
    try:
        response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "session.workspace.move",
                                          "params": {"session_key": "move-sid", "cwd": str(new_cwd)}})
    finally:
        server._sessions.pop("move-sid", None)

    assert response["result"]["cwd"] == str(new_cwd) == live["cwd"], response
    assert ("move-sid", str(new_cwd)) in rows
    assert [(e, s, p.cwd) for e, s, p in events] == [("session.info", "move-sid", str(new_cwd))]
