"""Lost creation replies recover from SQLite across real backend-process restarts."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


CREATE = r'''
import json, os, sys
from tui_gateway import server
server._schedule_agent_build = lambda sid: None
server._schedule_session_cap_enforcement = lambda: None
params = json.loads(sys.stdin.read())
if 'coding_workspace' not in params:
    response = server._methods['projects.workspace.prepare'](1, {
        'profile': params['profile'], 'path': params['cwd'],
        'mode': 'folder', 'requestId': 'same-draft'})
    assert 'error' not in response, response
    params['coding_workspace'] = response['result']
result = server._methods['session.create'](2, params)
assert 'error' not in result, result
from tui_gateway.coding_workspaces import workspace_instructions
sid = result['result']['session_id']
session = server._sessions[sid]
verified = server._methods['session.workspace.verify'](3, {'session_id': sid, 'cwd': params['cwd']})
assert 'error' not in verified, verified
# Capture only the outbound worker transport; the gateway submit guard stays real.
frames = []
server._session_uses_compute_host = lambda *a: True
def send(rid, sid, session, text, **kw):
    frames.append(server._compute_host_turn_frame(rid, sid, session, text))
    session['running'] = False
    return server._ok(rid, {'status': 'streaming'})
server._submit_prompt_to_compute_host = send
submitted = server._methods['prompt.submit'](4, {'session_id': sid, 'text': 'continue in this checkout'})
assert 'error' not in submitted, submitted
assert len(frames) == 1
binding = session['coding_workspace']
assert frames[0]['coding_workspace'] == binding
sys.__stdout__.write(json.dumps({
    'params': params, 'created': result['result'], 'pid': os.getpid(),
    'binding': binding, 'instructions': workspace_instructions(binding)}) + '\n')
'''


@pytest.mark.parametrize("git_change", [None, "branch", "detach"], ids=["folder", "branch", "detach"])
def test_lost_response_restart_reuses_receipt_with_profile_isolation(tmp_path, git_change):
    home = tmp_path / "deployment"
    folder = tmp_path / "folder"
    folder.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()
    if git_change:
        git("init", "-b", "main")
        git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    for profile in ("alpha", "beta"):
        (home / "profiles" / profile).mkdir(parents=True)
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home)}
    def run(params):
        proc = subprocess.run([sys.executable, "-c", CREATE], input=json.dumps(params),
                              capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])
    alpha = run({"profile": "alpha", "cwd": str(folder), "source": "desktop"})
    if git_change == "branch":
        git("checkout", "-b", "next-task")
    elif git_change == "detach":
        git("checkout", "--detach", "HEAD")
    # The first backend exited. Client never had to receive the original result:
    # prepared binding alone is sufficient to recover the exact durable session.
    recovered = run(alpha["params"])
    beta = run({"profile": "beta", "cwd": str(folder), "source": "desktop"})
    assert recovered["binding"] == alpha["binding"]
    assert recovered["instructions"] == alpha["instructions"]
    assert recovered["params"] == alpha["params"]
    if git_change:
        assert recovered["binding"]["branch"] != (git("branch", "--show-current") or None)
    assert alpha["pid"] != recovered["pid"]
    assert alpha["created"]["stored_session_id"] == recovered["created"]["stored_session_id"]
    assert alpha["created"]["stored_session_id"] != beta["created"]["stored_session_id"]
    for profile, result in (("alpha", alpha), ("beta", beta)):
        owned = home / "profiles" / profile
        with sqlite3.connect(owned / "state.db") as db:
            rows = db.execute("SELECT id, cwd, model_config FROM sessions").fetchall()
        assert len(rows) == 1 and rows[0][0] == result["created"]["stored_session_id"]
        assert rows[0][1] == str(folder)
        binding = json.loads(rows[0][2])["coding_workspace"]
        assert binding == result["binding"]
        assert binding["requestId"] == "same-draft"
        artifacts = Path(binding["artifactsPath"])
        assert artifacts.is_relative_to(owned) and artifacts.is_dir()
        assert list((owned / "cache" / "session-artifacts").iterdir()) == [artifacts]
    if (home / "state.db").exists():
        with sqlite3.connect(home / "state.db") as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone():
                assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


PREPARE = r'''
import json, sys
from tui_gateway import server
params = json.loads(sys.stdin.read())
response = server._methods['projects.workspace.prepare'](1, params)
assert 'error' not in response, response
sys.__stdout__.write(json.dumps(response['result']) + '\n')
'''


def test_prepared_but_unbound_worktree_is_reclaimed_after_process_restart(tmp_path):
    """The reviewer's window: prepare mutated Git, the client died before session.create, and
    the renderer's in-memory requestId is gone. A new draft must adopt the orphan, not make a second one."""
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    git("init", "-b", "main")
    git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base")
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home)}

    def prepare(params):
        proc = subprocess.run([sys.executable, "-c", PREPARE], input=json.dumps(params),
                              capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    first = prepare({"path": str(repo), "mode": "worktree", "requestId": "draft-before-crash"})
    # Process 1 is gone; no session.create ever happened; the renderer minted a new UUID.
    second = prepare({"path": str(repo), "mode": "worktree", "requestId": "draft-after-restart"})
    managed = sorted(p.name for p in (repo / ".worktrees").iterdir())
    assert second["cwd"] == first["cwd"], (first, second)
    assert len(managed) == 1, managed
    assert len([l for l in git("worktree", "list", "--porcelain").splitlines() if l.startswith("worktree ")]) == 2
