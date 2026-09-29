"""The coding-workspace wire goes through the REAL dispatch path (``handle_request``): unknown-key
admission, then the handler, then the strict result check. Handler-level tests call
``server._methods[...]`` directly and cannot see a param the contract forgot to declare."""
import subprocess

import pytest

import tui_gateway.server as server


def rpc(method, **params):
    return server.handle_request({"jsonrpc": "2.0", "id": "1", "method": method, "params": params})


def ok(method, **params):
    response = rpc(method, **params)
    assert "error" not in response, response
    return response["result"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "repo"
    path.mkdir()
    for args in (["init", "-b", "main"],
                 ["-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base"]):
        subprocess.check_output(["git", "-C", str(path), *args])
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    known = set(server._sessions)
    yield path
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def test_desktop_workspace_flow_is_accepted_by_the_contracts(repo):
    inspection = ok("projects.workspace.inspect", path=str(repo))
    assert inspection["repoRoot"] == str(repo)
    project = ok("projects.workspace.register", path=str(repo))["project"]
    prepared = ok("projects.workspace.prepare", path=str(repo), mode="worktree",
                  projectId=project["id"], requestId="contract-draft")
    assert prepared["requestId"] == "contract-draft"

    # Exactly the desktop's session.create shape: the prepare receipt rides back verbatim.
    created = ok("session.create", source="desktop", cols=80, cwd=prepared["cwd"], cwd_explicit=True,
                 coding_workspace=prepared)
    info = created["info"]
    assert info["cwd"] == prepared["cwd"]
    assert info["coding_workspace"]["requestId"] == "contract-draft"
    assert info["coding_workspace"]["artifactsPath"]

    sid = created["session_id"]
    verified = ok("session.workspace.verify", session_id=sid, cwd=prepared["cwd"])
    assert verified == {"cwd": prepared["cwd"], "gatewayCwd": prepared["cwd"]}
    remapped = ok("session.workspace.references", session_id=sid, paths=[], text="plain text",
                  reference_cwd=str(repo))
    assert remapped == {"paths": [], "text": "plain text"}
    assert "sourceCwd" in ok("complete.path", word="")


def test_unknown_keys_are_still_rejected(repo):
    binding = {"requestId": "r", "cwd": str(repo), "unexpected": 1}
    response = rpc("session.create", cwd=str(repo), coding_workspace=binding)
    assert response["error"]["code"] == 4000
    assert "coding_workspace.unexpected" in response["error"]["message"]
    response = rpc("session.create", cwd=str(repo), not_a_session_create_param=True)
    assert response["error"]["code"] == 4000
    assert "not_a_session_create_param" in response["error"]["message"]
    response = rpc("projects.workspace.prepare", path=str(repo), mode="folder", extra=True)
    assert response["error"]["code"] == 4000
