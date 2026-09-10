"""Real shell/CLI ingress: inherited context is not a board-write grant."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from tools import kanban_tools
from tools.environments.local import LocalEnvironment


ROOT = Path(__file__).resolve().parents[2]


def _worker_board(tmp_path, monkeypatch):
    db = tmp_path / "assigned.db"
    conn = connect(db)
    own, foreign = [kb.create_task(conn, title=title) for title in ("own", "foreign")]
    for tid in (own, foreign):
        kb.claim_task(conn, tid)
    task = kb.get_task(conn, own)
    for key, value in {
        "HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock, "HOME": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, own, foreign


def test_terminal_descendants_cannot_mutate_even_after_task_is_removed(tmp_path, monkeypatch):
    conn, own, foreign = _worker_board(tmp_path, monkeypatch)
    script = tmp_path / "descendant.py"
    script.write_text(
        "import os, sys, json, subprocess\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tools import kanban_tools as kt\n"
        "from agent.delegation_context import is_dispatcher_owned_worker_context\n"
        f"own, foreign = {own!r}, {foreign!r}\n"
        "out = {'owner': is_dispatcher_owned_worker_context(), 'default': kt._default_task_id(None),"
        " 'db': os.getenv('HERMES_KANBAN_DB'), 'board': os.getenv('HERMES_KANBAN_BOARD')}\n"
        "out['show'] = json.loads(kt._handle_show({'task_id':own}))\n"
        "out['tools'] = [json.loads(kt._handle_complete({'task_id': t, 'summary':'must refuse'})) for t in (own,foreign)]\n"
        "os.environ.pop('HERMES_KANBAN_TASK', None)\n"
        f"p = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', 'complete', foreign, '--result', 'must refuse'], cwd={str(ROOT)!r}, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=45)\n"
        "out['later_cli'] = {'rc': p.returncode, 'out':p.stdout, 'err':p.stderr}\n"
        "print('SCOPE_RESULT=' + json.dumps(out))\n"
    )
    terminal = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = terminal.execute(f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    finally:
        terminal.cleanup()
    from agent.skill_preprocessing import run_inline_shell
    from agent.shell_hooks import ShellHookSpec, _spawn
    from tools.code_execution_env import _build_child_env
    from tools.mcp_tool_config import _build_safe_env

    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    outputs = [result.get("output", ""), run_inline_shell(command, tmp_path, 45),
               _spawn(ShellHookSpec(event="session:start", command=command, timeout=45), "{}")['stdout']]
    child_envs = [
        _build_child_env(rpc_endpoint="fixture", rpc_token="fixture", tmpdir=str(tmp_path), child_python=sys.executable),
        _build_safe_env({"HERMES_HOME": os.environ["HERMES_HOME"]}),
    ]
    for env in child_envs:
        proc = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    for output in outputs:
        row = json.loads(next(line.split("SCOPE_RESULT=", 1)[1] for line in output.splitlines() if "SCOPE_RESULT=" in line))
        assert row["show"]["task"]["id"] == own, row
        assert not row["owner"] and row["default"] is None, row
        assert row["db"] == str(tmp_path / "assigned.db") and row["board"] == "default"
        assert all("error" in value for value in row["tools"]), row
        assert row["later_cli"]["rc"] != 0, row
    assert [kb.get_task(conn, t).status for t in (own, foreign)] == ["running", "running"]
    assert json.loads(kanban_tools._handle_complete({"summary": "parent handoff"}))["ok"]
    assert kb.get_task(conn, own).status == "done"
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_worker_cli_cannot_use_foreign_task_to_drop_run_scope(tmp_path, monkeypatch):
    conn, own, foreign = _worker_board(tmp_path, monkeypatch)
    assert "error" in json.loads(kanban_tools._handle_complete({"task_id": foreign, "summary": "no"}))
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "complete", foreign, "--result", "no"],
        cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=45,
    )
    assert proc.returncode != 0 and "worker is scoped to task" in proc.stderr, (proc.stdout, proc.stderr)
    assert kb.get_task(conn, foreign).status == "running"
    attachment = tmp_path / "note.txt"
    attachment.write_text("fixture")
    assert kb.block_task(conn, foreign, reason="fixture awaiting orchestrator")
    for arguments in (["attach", foreign, str(attachment)], ["unblock", foreign]):
        proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *arguments],
                              cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=45)
        assert proc.returncode != 0, (arguments, proc.stdout, proc.stderr)
    assert not kb.list_attachments(conn, foreign)
    assert json.loads(kanban_tools._handle_complete({"task_id": own, "summary": "parent"}))["ok"]
    conn.close()
