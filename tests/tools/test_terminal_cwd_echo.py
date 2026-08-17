"""Tests for the terminal result cwd echo (feat/terminal-cwd-echo).

When a command changes the session working directory, the tool result
gains a "cwd" field so the model stops prefixing every command with
'cd X && ' (60% of production terminal calls) and stops running pwd
diagnostics after directory changes.
"""

import json
import os
import tempfile

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


class TestCwdEcho:
    def test_cd_reports_new_cwd(self, isolated_home, tmp_path):
        target = tmp_path / "projdir"
        target.mkdir()
        r = json.loads(terminal_tool(f"cd {target}", task_id="t-cwd-1"))
        assert r["exit_code"] == 0
        assert "cwd" in r
        assert os.path.realpath(r["cwd"]) == os.path.realpath(str(target))

    def test_non_cd_command_has_no_cwd_field(self, isolated_home):
        r = json.loads(terminal_tool("echo hello", task_id="t-cwd-2"))
        assert r["exit_code"] == 0
        assert "cwd" not in r

    def test_cwd_persists_and_stops_reporting_when_stable(self, isolated_home, tmp_path):
        target = tmp_path / "stable"
        target.mkdir()
        r1 = json.loads(terminal_tool(f"cd {target}", task_id="t-cwd-3"))
        assert "cwd" in r1
        # Next command runs IN the new cwd without changing it: no echo.
        r2 = json.loads(terminal_tool("pwd", task_id="t-cwd-3"))
        assert "cwd" not in r2
        assert os.path.realpath(r2["output"].strip()) == os.path.realpath(str(target))

    def test_cd_within_chain_reports_final_dir(self, isolated_home, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        r = json.loads(terminal_tool(f"cd {a} && cd {b} && echo done", task_id="t-cwd-4"))
        assert r["exit_code"] == 0
        assert os.path.realpath(r.get("cwd", "")) == os.path.realpath(str(b))

    def test_workdir_override_does_not_echo(self, isolated_home, tmp_path):
        # Per-command workdir is transient by contract; it must not surface
        # a cwd field (the session cwd did not change).
        sub = tmp_path / "transient"
        sub.mkdir()
        r = json.loads(terminal_tool("echo hi", workdir=str(sub), task_id="t-cwd-5"))
        assert r["exit_code"] == 0
        assert "cwd" not in r

    def test_failed_cd_no_echo(self, isolated_home):
        r = json.loads(terminal_tool("cd /nonexistent_dir_zzz_42", task_id="t-cwd-6"))
        assert r["exit_code"] != 0
        assert "cwd" not in r
