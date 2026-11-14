"""Literal-PID kills of the supervised gateway itself are refused before execution (#149944)."""

import os

import pytest

from cron.lifecycle_guard import scan_gateway_lifecycle
from tools.terminal_tool_guards import gateway_lifecycle_block

PID = os.getpid()
OTHER = PID + 100000


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)

    def _run(command, env_type="local"):
        return gateway_lifecycle_block(
            command=command, env=None, env_type=env_type, cwd=str(tmp_path),
            workdir=None, session_key="pid-test",
        )
    return _run


@pytest.mark.parametrize("command", [
    "kill {pid}", "kill -9 {pid}", "kill -TERM 42 {pid}", "kill -s KILL {pid}", "kill -- {pid}",
    "/bin/kill -9 {pid}", "sudo kill -9 {pid}", "nohup kill {pid}", "env -u X kill {pid}",
    "taskkill /F /PID {pid}", "taskkill.exe /PID {pid} /T", "Stop-Process -Id {pid} -Force",
    "bash -c 'kill -9 {pid}'", "su -c 'kill {pid}'", "true && kill {pid}",
    "if true; then kill {pid}; fi", "for x in a; do kill {pid}; done", "time kill {pid}",
])
def test_literal_gateway_pid_kill_is_blocked(run, command):
    assert run(command.format(pid=PID)) is not None


def test_referenced_script_killing_gateway_pid_is_blocked(run, tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text(f"#!/bin/sh\nkill -9 {PID}\n")
    assert run(f"bash {script}") is not None


@pytest.mark.parametrize("command", [
    "kill {other}", "kill -9 {other}", "taskkill /F /PID {other}",
    "echo kill {pid}", "grep 'kill {pid}' input.txt",
    "kill -l {pid}", "kill -0 {pid}", "kill -s 0 {pid}", "kill -s {pid} {other}",
    "taskkill /S remotehost /PID {pid}",
])
def test_other_pids_probes_and_data_stay_allowed(run, command):
    assert run(command.format(pid=PID, other=OTHER)) is None


def test_pid_check_is_scoped_to_local_supervised_gateway(run, monkeypatch):
    # Remote/sandbox backends have their own PID table; cron callers pass no gateway_pid.
    assert run(f"kill {PID}", env_type="docker") is None
    assert run(f"kill {PID}", env_type="ssh") is None
    assert scan_gateway_lifecycle(f"kill {PID}") == (False, None)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    assert run(f"kill {PID}") is None
