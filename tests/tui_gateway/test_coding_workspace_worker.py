"""The actual ComputeHost dispatch boundary in a fresh process; no provider calls."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tui_gateway import server


WORKER = r'''
import io, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost
from tools.terminal_tool import terminal_tool
frame, sabotage, receipt = json.loads(sys.stdin.read())
# Only provider construction and unrelated UI services are replaced. Session
# hydration, terminal registration, compute admission and local pwd stay real.
built = {}
def make_agent(*a, **kw):
    built.update(kw)
    return SimpleNamespace()
server._make_agent = make_agent
server._wire_session_agent = lambda *a: None
server._start_session_services = lambda *a: None
server._schedule_mcp_late_refresh = lambda *a: None
server._session_info = lambda *a: {}
def dispatch(rid, sid, session, text, **kw):
    Path(receipt).write_text(json.dumps({
        'pid': os.getpid(), 'explicit': session.get('explicit_cwd'),
        'binding': session.get('coding_workspace'), 'agentBinding': built.get('coding_workspace'),
        'pwd': json.loads(terminal_tool('pwd', task_id=session['session_key'], timeout=10))}))
    session['running'] = False
server._run_prompt_submit = dispatch
out = io.StringIO()
host = ComputeHost(stdout=out, heartbeat_secs=0)
ensure = host._ensure_server_session
def tampered(server, frame):
    session = ensure(server, frame)
    if sabotage == 'explicit':
        session['explicit_cwd'] = False
    elif sabotage == 'cwd':
        session['cwd'] = str(Path(frame['cwd']).parent)
    elif sabotage == 'deleted':
        Path(frame['cwd']).rename(frame['cwd'] + '-moved')
    elif sabotage == 'branch':
        import subprocess
        subprocess.check_call(['git', '-C', frame['cwd'], 'checkout', '-b', 'unexpected'])
    return session
host._ensure_server_session = tampered
host._run_real_turn(frame)
host.close()
sys.__stdout__.write(json.dumps([json.loads(line) for line in out.getvalue().splitlines()]) + "\n")
'''


@pytest.mark.parametrize("sabotage", ["none", "explicit", "cwd", "deleted", "branch"])
def test_compute_process_verifies_checkout_before_dispatch(tmp_path, monkeypatch, sabotage):
    from hermes_constants import get_hermes_home
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "--allow-empty", "-m", "base"], check=True, capture_output=True)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    def call(method, params):
        response = server._methods[method](1, params)
        assert "error" not in response, response
        return response["result"]
    prepared = call("projects.workspace.prepare", {"path": str(repo), "mode": "current", "requestId": "worker"})
    created = call("session.create", {"cwd": str(repo), "coding_workspace": prepared, "source": "desktop"})
    session = server._sessions[created["session_id"]]
    call("session.workspace.verify", {"session_id": created["session_id"], "cwd": str(repo)})
    frame = server._compute_host_turn_frame("turn", created["session_id"], session, "never call a provider")
    receipt = tmp_path / "dispatch.json"
    env = {**os.environ, "HERMES_HOME": str(get_hermes_home())}
    child = subprocess.run([sys.executable, "-c", WORKER], input=json.dumps([frame, sabotage, str(receipt)]), text=True, capture_output=True, env=env, timeout=30)
    assert child.returncode == 0, child.stderr
    frames = json.loads(child.stdout.splitlines()[-1])
    if sabotage == "none":
        assert receipt.exists(), frames
        result = json.loads(receipt.read_text())
        assert result["pid"] != os.getpid()
        assert result["explicit"] is True
        assert result["binding"] == session["coding_workspace"]
        assert result["agentBinding"] == session["coding_workspace"]
        assert result["pwd"]["exit_code"] == 0 and result["pwd"]["output"].strip() == str(repo)
    else:
        assert not receipt.exists(), frames
        assert any(f["type"] == "turn.error" for f in frames), frames
        assert not any(f["type"] == "turn.started" for f in frames), frames
