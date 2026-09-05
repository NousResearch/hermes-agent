"""Lost creation replies recover from SQLite across real backend-process restarts."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


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
sys.__stdout__.write(json.dumps({'params': params, 'created': result['result'], 'pid': os.getpid()}) + '\n')
'''


def test_lost_response_restart_reuses_receipt_with_profile_isolation(tmp_path):
    home = tmp_path / "deployment"
    folder = tmp_path / "folder"
    folder.mkdir()
    for profile in ("alpha", "beta"):
        (home / "profiles" / profile).mkdir(parents=True)
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home)}
    def run(params):
        proc = subprocess.run([sys.executable, "-c", CREATE], input=json.dumps(params),
                              capture_output=True, text=True, env=env, timeout=30)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])
    alpha = run({"profile": "alpha", "cwd": str(folder), "source": "desktop"})
    # The first backend exited. Client never had to receive the original result:
    # prepared binding alone is sufficient to recover the exact durable session.
    recovered = run(alpha["params"])
    beta = run({"profile": "beta", "cwd": str(folder), "source": "desktop"})
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
        assert binding["requestId"] == "same-draft"
        artifacts = Path(binding["artifactsPath"])
        assert artifacts.is_relative_to(owned) and artifacts.is_dir()
        assert list((owned / "cache" / "session-artifacts").iterdir()) == [artifacts]
    if (home / "state.db").exists():
        with sqlite3.connect(home / "state.db") as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone():
                assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
