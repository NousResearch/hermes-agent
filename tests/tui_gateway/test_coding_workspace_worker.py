"""The actual ComputeHost dispatch boundary in a fresh process; no provider calls."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tui_gateway import server


WORKER = r'''
import io, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost
from tui_gateway.coding_workspaces import workspace_instructions
from tools.terminal_tool import register_task_env_overrides, terminal_tool
frame, change, warm, receipt = json.loads(sys.stdin.read())
# Only provider construction/dispatch and unrelated UI services are replaced.
# Hydration, terminal registration, compute admission and local pwd stay real.
built = []
dispatched = []
def make_agent(*a, **kw):
    built.append(kw)
    return SimpleNamespace()
server._make_agent = make_agent
server._wire_session_agent = lambda *a: None
server._start_session_services = lambda *a: None
server._schedule_mcp_late_refresh = lambda *a: None
server._session_info = lambda *a: {}
def dispatch(rid, sid, session, text, **kw):
    dispatched.append({
        'pid': os.getpid(), 'explicit': session.get('explicit_cwd'),
        'binding': session.get('coding_workspace'), 'agentBinding': built[0].get('coding_workspace'),
        'builds': len(built), 'agentId': id(session['agent']),
        'instructions': workspace_instructions(session.get('coding_workspace')),
        'pwd': json.loads(terminal_tool('pwd', task_id=session['session_key'], timeout=10))})
    Path(receipt).write_text(json.dumps(dispatched), encoding='utf-8')
    session['running'] = False
server._run_prompt_submit = dispatch
out = io.StringIO()
host = ComputeHost(stdout=out, heartbeat_secs=0)
if warm:
    host._run_real_turn({**frame, 'request_id': 'warmup'})
    assert len(dispatched) == 1, out.getvalue()
ensure = host._ensure_server_session
def tampered(server, frame):
    session = ensure(server, frame)
    cwd = Path(frame['cwd'])
    def git(*args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()
    if change == 'explicit':
        session['explicit_cwd'] = False
    elif change == 'cwd':
        session['cwd'] = str(cwd.parent)
    elif change == 'deleted':
        cwd.rename(str(cwd) + '-moved')
    elif change == 'root':
        # A real linked checkout now belongs to a relocated primary repository.
        root = Path(frame['coding_workspace']['repoRoot'])
        relocated = root.with_name('relocated')
        root.rename(relocated)
        gitdir = relocated / '.git' / 'worktrees' / cwd.name
        (cwd / '.git').write_text(f'gitdir: {gitdir}\n', encoding='utf-8')
    elif change == 'terminal-cwd':
        register_task_env_overrides(session['session_key'], {'cwd': str(cwd.parent)})
    elif change == 'nonlocal':
        Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
            json.dumps({'terminal': {'backend': 'ssh'}}), encoding='utf-8')
    elif change == 'branch':
        git('checkout', '-b', 'another-task')
    elif change == 'detach':
        git('checkout', '--detach', 'HEAD')
        assert git('branch', '--show-current') == ''
    elif change == 'rebase':
        # Stop a real rebase on a conflict: Git reports detached HEAD until resolved.
        base = git('rev-parse', 'HEAD')
        git('checkout', '-b', 'topic')
        (cwd / 'tracked.txt').write_text('topic\n', encoding='utf-8')
        git('commit', '-am', 'topic')
        git('checkout', '-b', 'upstream', base)
        (cwd / 'tracked.txt').write_text('upstream\n', encoding='utf-8')
        git('commit', '-am', 'upstream')
        git('checkout', 'topic')
        rebased = subprocess.run(['git', '-C', str(cwd), 'rebase', 'upstream'], capture_output=True, text=True)
        assert rebased.returncode != 0, rebased
        assert Path(git('rev-parse', '--path-format=absolute', '--git-path', 'rebase-merge')).is_dir()
        assert git('branch', '--show-current') == ''
    return session
host._ensure_server_session = tampered
host._run_real_turn(frame)
host.close()
sys.__stdout__.write(json.dumps([json.loads(line) for line in out.getvalue().splitlines()]) + "\n")
'''


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("change", ["none", "explicit", "cwd", "deleted", "root", "terminal-cwd", "nonlocal", "branch", "detach", "rebase"])
def test_compute_process_verifies_checkout_before_dispatch(tmp_path, monkeypatch, change, warm):
    from hermes_constants import get_hermes_home
    from tui_gateway.coding_workspaces import workspace_instructions
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@localhost")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "base")
    checkout = tmp_path / "selected"
    git("worktree", "add", "-b", "selected", str(checkout))
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    def call(method, params):
        response = server._methods[method](1, params)
        assert "error" not in response, response
        return response["result"]
    prepared = call("projects.workspace.prepare", {
        "path": str(repo), "mode": "existing", "existingPath": str(checkout), "requestId": "worker"})
    created = call("session.create", {"cwd": str(checkout), "coding_workspace": prepared, "source": "desktop"})
    session = server._sessions[created["session_id"]]
    binding = json.loads(json.dumps(session["coding_workspace"]))
    instructions = workspace_instructions(binding)
    call("session.workspace.verify", {"session_id": created["session_id"], "cwd": str(checkout)})
    frame = server._compute_host_turn_frame("turn", created["session_id"], session, "never call a provider")
    receipt = tmp_path / "dispatch.json"
    env = {**os.environ, "HERMES_HOME": str(get_hermes_home())}
    child = subprocess.run([sys.executable, "-c", WORKER], input=json.dumps([frame, change, warm, str(receipt)]), text=True, capture_output=True, env=env, timeout=30)
    assert child.returncode == 0, child.stderr
    frames = json.loads(child.stdout.splitlines()[-1])
    results = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else []
    accepted = change in {"none", "branch", "detach", "rebase"}
    assert len(results) == int(warm) + int(accepted), json.dumps(frames, indent=2)
    turn = [f for f in frames if f.get("request_id") == frame["request_id"]]
    if accepted:
        assert any(f["type"] == "turn.started" for f in turn), frames
        assert any(f["type"] == "turn.end" for f in turn), frames
        assert not any(f["type"] == "turn.error" for f in turn), frames
        for result in results:
            assert result["pid"] != os.getpid()
            assert result["explicit"] is True
            assert result["binding"] == binding
            assert result["agentBinding"] == binding
            assert result["instructions"] == instructions
            assert result["builds"] == 1
            assert result["agentId"] == results[0]["agentId"]
            assert result["pwd"]["exit_code"] == 0 and result["pwd"]["output"].strip() == str(checkout)
    else:
        assert any(f["type"] == "turn.error" for f in turn), frames
        assert not any(f["type"] == "turn.started" for f in turn), frames
