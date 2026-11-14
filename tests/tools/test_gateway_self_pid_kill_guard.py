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
    # `+N` is a PID; `-N` after a signal or `--` is the gateway's own process group.
    "kill +{pid}", "kill -9 -{pid}", "kill -- -{pid}", "kill -s TERM -{pid}", "kill -SIGKILL -{pid}",
    # PowerShell: positional Id, colon-bound Id, comma list split across spaces.
    "Stop-Process {pid} -Force", "Stop-Process -Force {pid}", "Stop-Process -Id:{pid}",
    "Stop-Process -Id 1, {pid}", "Stop-Process 1 , {pid}", "taskkill /PID:{pid}", "taskkill /F /pid:{pid}",
    # String runners and command substitution.
    "eval kill {pid}", "eval 'kill -9 {pid}'", "echo `kill {pid}`", "x=\"$(kill {pid})\"",
    "powershell -c 'Stop-Process -Id {pid}'", "pwsh.exe -Command \"Stop-Process {pid}\"",
    "cmd /c taskkill /PID {pid}", "cmd.exe /C \"taskkill /F /PID {pid}\"",
    # A redirection before the command word; a lethal signal after a signal-0 lookalike group target.
    ">/dev/null kill -9 {pid}", "2>/dev/null kill {pid}", "> /dev/null kill {pid}", "kill -9 -0 {pid}",
    "kill -USR1 {pid}", "kill -9 " + "1" * 5000 + " {pid}", "kill 00{pid}",
    # A PID operand ends option parsing, so a later `-N` is a group target; unlisted value options.
    "kill {other} -{pid}", "kill {other} -- {pid}",
    "powershell -c 'Stop-Process -ErrorAction Stop -Id {pid}'", "Stop-Process -ea SilentlyContinue {pid}",
])
def test_literal_gateway_pid_kill_is_blocked(run, command):
    assert run(command.format(pid=PID, other=OTHER)) is not None


def test_referenced_script_killing_gateway_pid_is_blocked(run, tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text(f"#!/bin/sh\nkill -9 {PID}\n")
    assert run(f"bash {script}") is not None


@pytest.mark.parametrize("command", [
    "kill {other}", "kill -9 {other}", "taskkill /F /PID {other}",
    "echo kill {pid}", "grep 'kill {pid}' input.txt",
    "kill -l {pid}", "kill -0 {pid}", "kill -s 0 {pid}", "kill -s {pid} {other}",
    "taskkill /S remotehost /PID {pid}",
    # A leading `-N` with no signal before it is a signal number, not a target.
    "kill -{pid} {other}", "kill +{other}", "kill -9 -{other}", "kill -- -{other}",
    "Stop-Process {other} -Force", "Stop-Process -Id:{other}", "Stop-Process -Id 1, {other}",
    "Stop-Process -Name {pid}", "Stop-Process -Name:{pid} -Force", "taskkill /PID:{other}",
    "eval echo {pid}", "echo `date` {pid}", "powershell -c 'Get-Process -Id {pid}'",
    "cmd /c echo {pid}", "sleep {pid}", ">kill {pid}", "grep -F '$(bash /dev/null)' notes.txt",
])
def test_other_pids_probes_and_data_stay_allowed(run, command):
    assert run(command.format(pid=PID, other=OTHER)) is None


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", [
    "kill -USR2 {pid}", "kill -SIGUSR2 {pid}", "kill -s USR2 {pid}", "kill --signal=usr2 {pid}", "kill -{signum} {pid}",
    "kill -sUSR2 {pid}", "kill -n{signum} {pid}",
])
def test_documented_sigusr2_stack_dump_stays_allowed(run, command):
    # `kill -USR2 <gateway pid>` is the documented faulthandler dump; the gateway keeps running.
    import signal
    assert run(command.format(pid=PID, signum=int(signal.SIGUSR2))) is None


@pytest.mark.parametrize("command", [
    "eval 'bash -c \"kill -9 {pid}\"'", "cmd /c \"bash -c 'kill {pid}'\"",
    "bash -c \"eval 'sh -c \\\"kill {pid}\\\"'\"", "eval 'bash {script}'",
])
def test_string_runner_payloads_open_shell_payloads_and_scripts(run, tmp_path, command):
    script = tmp_path / "k.sh"
    script.write_text(f"#!/bin/sh\nkill -9 {PID}\n")
    assert run(command.format(pid=PID, script=script)) is not None


def test_oversized_numeric_operand_keeps_the_script_walk(run, tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n")
    assert run(f"kill {'1' * 5000} 2>/dev/null; bash {script}") is not None


def test_pid_check_is_scoped_to_local_supervised_gateway(run, monkeypatch):
    # Remote/sandbox backends have their own PID table; cron callers pass no gateway_pid.
    assert run(f"kill {PID}", env_type="docker") is None
    assert run(f"kill {PID}", env_type="ssh") is None
    assert scan_gateway_lifecycle(f"kill {PID}") == (False, None)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    assert run(f"kill {PID}") is None
