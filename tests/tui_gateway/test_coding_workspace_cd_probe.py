"""The legacy session.workspace.verify probe after the agent's shell cd'd out of the checkout.

The probe must pass (the next turn starts at the root; tests/tui_gateway/test_coding_workspace_worker.py)
without moving the shell: a running turn may be the one that cd'd.
"""
import json
import shlex
import subprocess

import pytest

from tui_gateway import server


@pytest.mark.parametrize("dest", ["outside", "deleted"])
def test_verify_probe_passes_and_leaves_an_escaped_shell_alone(tmp_path, monkeypatch, dest):
    from tools.terminal_tool import clear_session_cwd, get_session_cwd, terminal_tool
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-b", "main")
    git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    checkout = tmp_path / "selected"
    git("worktree", "add", "-b", "selected", str(checkout))
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    def call(method, params):
        response = server._methods[method](1, params)
        assert "error" not in response, response
        return response["result"]
    prepared = call("projects.workspace.prepare", {
        "path": str(repo), "mode": "existing", "existingPath": str(checkout), "requestId": "probe"})
    created = call("session.create", {"cwd": str(checkout), "coding_workspace": prepared, "source": "desktop"})
    key = server._sessions[created["session_id"]]["session_key"]
    target = tmp_path / "session-artifacts" if dest == "outside" else checkout / "gone"
    target.mkdir()
    try:
        moved = json.loads(terminal_tool("cd -- " + shlex.quote(str(target)) + " && pwd", task_id=key, timeout=10))
        assert moved["output"].strip() == str(target), moved
        if dest == "deleted":
            target.rmdir()
        verified = call("session.workspace.verify", {"session_id": created["session_id"], "cwd": str(checkout)})
        assert verified["cwd"] == str(checkout)
        assert get_session_cwd(key) == str(target)
    finally:
        clear_session_cwd(key)
