"""Diagnostics must distinguish policy rejection from inspected job properties."""

import json
import os
import plistlib
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_tool_guards import gateway_lifecycle_block


def test_bootstrap_rejection_does_not_invent_keepalive(tmp_path, monkeypatch):
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    plist = tmp_path / "com.example.schedule.plist"
    plist.write_bytes(plistlib.dumps({
        "Label": "com.example.schedule",
        "ProgramArguments": ["/bin/true"],
        "RunAtLoad": False,
        "StartCalendarInterval": {"Hour": 9, "Minute": 0},
    }))
    blocked = gateway_lifecycle_block(
        command=f"launchctl bootstrap gui/501 {plist}",
        env=None, env_type="local", cwd=str(tmp_path), workdir=None,
        session_key="diagnostic-test",
    )
    assert blocked is not None
    result = json.loads(blocked)
    assert result["exit_code"] == 1
    assert result["error"]


def test_interpreter_kill_rejection_names_the_owned_process_route(tmp_path, monkeypatch):
    """Scheduled-Task topology (#113667): only ``HERMES_SUPERVISED_CHILD`` marks the launch, and an
    image-name kill of the interpreter is refused with the ``proc_*`` / explicit-PID route named.
    Other images and explicit PIDs pass."""
    import os

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    for marker in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_S6_SUPERVISED_CHILD", "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: os.getpid())

    def run(command):
        return gateway_lifecycle_block(
            command=command, env=None, env_type="local", cwd=str(tmp_path), workdir=None,
            session_key="diagnostic-test",
        )

    blocked = json.loads(run("taskkill /F /IM python.exe 2>/dev/null | head -2"))
    assert blocked["exit_code"] == 1
    assert "proc_" in blocked["error"] and "explicit PID" in blocked["error"]
    assert "proc_" in json.loads(run("pkill -9 python3"))["error"]
    assert run("taskkill /F /IM agent-browser.exe /T") is None
    assert run("taskkill /F /PID 46544") is None
    # Absence on the wrong side: a plain foreground `hermes gateway run` carries no launch marker.
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD")
    assert run("pkill -9 python3") is None


@pytest.mark.parametrize("command", [
    "kill {pid}", "kill -9 {pid}", "kill -TERM 42 {pid}",
    "taskkill /F /PID {pid}", "bash -c 'kill -9 {pid}'",
    "if true; then kill {pid}; fi", "for x in a; do kill {pid}; done",
    "if ! kill {pid}; then true; fi",
    "if true; then {{ kill {pid}; }}; fi",
    "if ! {{ kill {pid}; }}; then true; fi",
    "time kill {pid}", "time -p kill {pid}",
    "if ! bash -c 'kill {pid}'; then true; fi",
])
def test_supervised_local_gateway_pid_is_blocked(command, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    result = gateway_lifecycle_block(
        command=command.format(pid=os.getpid()), env=None, env_type="local",
        cwd=str(tmp_path), workdir=None, session_key="pid-test",
    )
    assert result is not None


def test_pid_scope_and_referenced_script(tmp_path, monkeypatch):
    from cron.lifecycle_guard import scan_gateway_lifecycle

    pid = os.getpid()
    script = tmp_path / "worker.sh"
    script.write_text(f"#!/bin/sh\nkill -9 {pid}\n")
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)

    def run(command, env_type="local"):
        return gateway_lifecycle_block(
            command=command, env=None, env_type=env_type, cwd=str(tmp_path),
            workdir=None, session_key="pid-test",
        )

    assert run(f"bash {script}") is not None
    assert run(f"if true; then bash {script}; fi") is not None
    assert run(f"if ! bash {script}; then true; fi") is not None
    assert run(f"time bash {script}") is not None
    assert run(f"kill {pid + 100000}") is None
    assert run(f"echo kill {pid}") is None
    assert run(f"grep 'kill {pid}' input.txt") is None
    assert run(f"echo 'if ! kill {pid}; then true; fi'") is None
    assert run(f"printf '%s' '{{ kill {pid}; }}'") is None
    assert run(f"if true; then echo '{{ kill {pid}; }}'; fi") is None
    assert run(f"if ! grep 'kill {pid}' input.txt; then true; fi") is None
    assert run(f"'!' kill {pid}") is None
    assert run(f"'{{' kill {pid}") is None
    assert run(f"\\! kill {pid}") is None
    assert run(f"if true; then {{ kill {pid + 100000}; }}; fi") is None
    assert run(f"kill -l {pid}") is None
    assert run(f"kill -0 {pid}") is None
    assert run(f"kill -s 0 {pid}") is None
    assert run(f"kill -s0 {pid}") is None
    assert run(f"kill -n0 {pid}") is None
    assert run(f"command -v kill {pid}") is None
    assert run(f"command -V kill {pid}") is None
    assert run(f"if true; then command -v bash {script}; fi") is None
    assert run(f"echo time kill {pid}") is None
    assert run(f"taskkill /S remotehost /PID {pid}") is None
    assert run(f"kill -s {pid} 42") is None
    assert run(f"echo 'bash {script}'") is None
    assert run(f"taskkill /PID {pid + 100000}") is None
    assert run(f"kill {pid}", "ssh") is None
    assert run(f"bash {script}", "docker") is None
    assert scan_gateway_lifecycle(f"kill {pid}") == (False, None)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    assert run(f"kill {pid}") is None


@pytest.mark.parametrize("command", ["command -v kill {pid}", "command -V kill {pid}",
                                   "kill -s0 {pid}", "kill -n0 {pid}"])
def test_pid_queries_do_not_signal(command, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    assert gateway_lifecycle_block(
        command=command.format(pid=os.getpid()), env=None, env_type="local",
        cwd=str(tmp_path), workdir=None, session_key="pid-test",
    ) is None


def test_pid_scan_command_substitution_executes():
    """``$(...)`` and backticks EXECUTE their payload, in double quotes and unquoted."""
    from cron.lifecycle_guard import scan_gateway_lifecycle

    pid = os.getpid()
    assert scan_gateway_lifecycle(f'echo "$(kill {pid})"', gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"echo `kill {pid}`", gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f'echo "`kill {pid}`"', gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"echo $(kill {pid})", gateway_pid=pid) == (True, None)
    # Single quotes make the same characters literal data: nothing executes.
    assert scan_gateway_lifecycle(f"echo '$(kill {pid})'", gateway_pid=pid) == (False, None)
    assert scan_gateway_lifecycle(f"echo '`kill {pid}`'", gateway_pid=pid) == (False, None)
    # A substitution aimed at another PID stays allowed.
    assert scan_gateway_lifecycle(f"echo $(kill {pid + 100000})", gateway_pid=pid) == (False, None)


def test_pid_scan_multilayer_wrapper_string_payloads():
    """``sudo env -S '…'`` / ``sudo su -c '…'``: string command options behind a wrapper chain."""
    from cron.lifecycle_guard import scan_gateway_lifecycle

    pid = os.getpid()
    assert scan_gateway_lifecycle(f"sudo env -S 'kill {pid}'", gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"sudo su -c 'kill {pid}'", gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"sudo su -c 'kill {pid + 100000}'", gateway_pid=pid) == (False, None)
    assert scan_gateway_lifecycle(f"sudo env -u X kill {pid}", gateway_pid=pid) == (True, None)


def test_pid_scan_kill_end_of_options_marker():
    """After ``--`` every token is an operand: ``-0`` is not the signal-0 probe there."""
    from cron.lifecycle_guard import scan_gateway_lifecycle

    pid = os.getpid()
    assert scan_gateway_lifecycle(f"kill -- -0 {pid}", gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"kill -- -0 {pid + 100000}", gateway_pid=pid) == (False, None)
    # Before the marker the probe reading stands: -0 sends nothing.
    assert scan_gateway_lifecycle(f"kill -0 -- {pid}", gateway_pid=pid) == (False, None)
    assert scan_gateway_lifecycle(f"kill -0 {pid}", gateway_pid=pid) == (False, None)


def test_pid_scan_quoted_first_word_is_not_a_reserved_word():
    """Quoting a word removes reserved-word status: it names a (missing) command instead."""
    from cron.lifecycle_guard import scan_gateway_lifecycle

    pid = os.getpid()
    assert scan_gateway_lifecycle(f"'time' kill {pid}", gateway_pid=pid) == (False, None)
    assert scan_gateway_lifecycle(f"'then' kill {pid}", gateway_pid=pid) == (False, None)
    assert scan_gateway_lifecycle(f'"time" kill {pid}', gateway_pid=pid) == (False, None)
    # Unquoted prefixes still peel to the real kill.
    assert scan_gateway_lifecycle(f"time kill {pid}", gateway_pid=pid) == (True, None)
    assert scan_gateway_lifecycle(f"if true; then kill {pid}; fi", gateway_pid=pid) == (True, None)

def test_pid_scan_exception_does_not_allow_unscanned_script(tmp_path, monkeypatch):
    from cron import lifecycle_guard

    script = tmp_path / "worker.sh"
    script.write_text(f"kill {os.getpid()}\n")
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script", MagicMock(side_effect=OSError("read failed")))
    result = gateway_lifecycle_block(
        command=f"bash {script}", env=None, env_type="local", cwd=str(tmp_path),
        workdir=None, session_key="pid-test",
    )
    assert result is not None


def test_terminal_tool_blocks_gateway_pid_before_execution(tmp_path, monkeypatch):
    from tools import terminal_tool as terminal_module

    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    env = MagicMock()
    env.cwd = str(tmp_path)
    config = {"env_type": "local", "timeout": 180, "cwd": str(tmp_path),
              "host_cwd": None, "modal_mode": "auto", "docker_image": "",
              "modal_image": "", "daytona_image": ""}
    with patch.object(terminal_module, "_get_env_config", return_value=config), \
         patch.object(terminal_module, "_start_cleanup_thread"), \
         patch.object(terminal_module, "_active_environments", {"default": env}), \
         patch.object(terminal_module, "_last_activity", {"default": 0}), \
         patch.object(terminal_module, "_session_cwd", {}), \
         patch.object(terminal_module, "_check_all_guards", return_value={"approved": True}):
        result = json.loads(terminal_module.terminal_tool(f"kill -9 {os.getpid()}", force=True))
    assert result["status"] == "error"
    assert "gateway" in result["error"]
    env.execute.assert_not_called()
