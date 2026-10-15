"""A deleted working directory must not wedge the terminal, and the model must be told.

``BaseEnvironment.execute()`` embeds ``builtin cd -- <cwd> || exit 126`` in the
wrapper script. When the directory is gone — a previous call ``rm -rf``'d it
(#17558), or the session's recorded ``cd`` state points at a removed path — the
command never ran and every later call exited 126 the same way. These tests
drive a real bash so the wrapper's own ``cd`` is the thing under test.
"""

import json
import os
import shutil

import pytest

from tools.environments.local import LocalEnvironment


def test_dead_cwd_runs_in_fallback_and_reports_it_once(tmp_path):
    """Both the env's own cwd and an explicit (session-recorded) cwd recover."""
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LocalEnvironment, "init_session", lambda self, *a, **k: None)
        env = LocalEnvironment(cwd=str(doomed), timeout=30)
    try:
        shutil.rmtree(doomed)

        first = env.execute("pwd", timeout=30)
        assert first["returncode"] == 0, first["output"]
        assert first["output"].strip() == str(tmp_path)
        assert first["cwd_fallback"] == {"requested": str(doomed), "used": str(tmp_path)}

        explicit = env.execute("pwd", cwd=str(doomed), timeout=30)
        assert explicit["returncode"] == 0, explicit["output"]
        assert explicit["cwd_fallback"]["requested"] == str(doomed)

        env.cwd = str(doomed)  # a stale env cwd must not taint a command given a live one
        healthy = env.execute("pwd", cwd=str(tmp_path), timeout=30)
        assert healthy["returncode"] == 0 and "cwd_fallback" not in healthy
    finally:
        env.cleanup()


def test_terminal_tool_recovers_a_dead_session_cwd_with_a_model_notice(tmp_path, monkeypatch):
    """The session record carries the dead path on every call; the tool must run, warn, then heal."""
    from tools.terminal_tool import terminal_tool
    from tools.terminal_tool_lifecycle import cleanup_vm

    sub = tmp_path / "proj" / "sub"
    sub.mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(sub))
    task = "dead-cwd-notice"
    try:
        assert json.loads(terminal_tool(command="pwd", task_id=task))["output"].strip() == str(sub)
        shutil.rmtree(sub)

        recovered = json.loads(terminal_tool(command="pwd", task_id=task))
        assert recovered["exit_code"] == 0, recovered
        assert recovered["output"].strip() == str(sub.parent)
        assert str(sub) in recovered["cwd_fallback"] and str(sub.parent) in recovered["cwd_fallback"]

        healed = json.loads(terminal_tool(command="pwd", task_id=task))
        assert healed["exit_code"] == 0 and "cwd_fallback" not in healed
        assert os.path.isdir(healed["output"].strip())
    finally:
        cleanup_vm(task)
