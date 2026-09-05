"""Real Bash startup/routing contracts for background pipes and PTYs."""
import json
import os
import shlex
import signal
import sys
import time

import pytest

from hermes_cli.session_execution import (
    SessionExecutionContext,
    register_session_execution_context,
    remove_session_execution_context,
    resolve_session_execution_context,
)
from tools.process_registry import ProcessRegistry


@pytest.fixture
def shell_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BASH_ENV", raising=False)
    startup = tmp_path / "startup.sh"
    shell = tmp_path / "fixture-bash"
    # Bash -lic doesn't use --rcfile. Explicitly source the inert initializer
    # before the production command, keeping real interactive Bash semantics.
    shell.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        'os.execv("/bin/bash", ["/bin/bash", "--noprofile", "--norc", "-ic", '
        f"{('source ' + shlex.quote(str(startup)) + '; ')!r} + sys.argv[-1]])\n"
    )
    shell.chmod(0o700)
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: str(shell))
    registry = ProcessRegistry()
    sessions = []

    def spawn(command, *, use_pty, context=None, readonly=False):
        startup.write_text(
            "export ROUTING_VALUE=startup ORDINARY_VALUE=ordinary\n"
            "probe_function() { printf 'FUNCTION_OK\\n'; }\n"
            + ("readonly ROUTING_VALUE\n" if readonly else "")
        )
        lease = None
        if context is not None:
            register_session_execution_context("background-routing", context)
            lease = resolve_session_execution_context(session_id="background-routing")
        session = registry.spawn_local(
            command, cwd=str(tmp_path), task_id="background-routing",
            execution_context=lease, use_pty=use_pty,
        )
        sessions.append(session)
        # No silently successful pipe fallback in a PTY regression test.
        assert (session._pty is not None) == use_pty
        return registry, session

    yield spawn

    for session in sessions:
        if not session.exited:
            registry.kill_process(session.id)
        session._reader_thread.join(timeout=5)
        assert not session._reader_thread.is_alive()
        if session.process is not None:
            session.process.wait(timeout=5)
            session.process.stdout.close()
        if session._pty is not None:
            session._pty.close()
    remove_session_execution_context("background-routing")


@pytest.mark.linux_only
@pytest.mark.parametrize("use_pty", [False, True], ids=["pipe", "pty"])
@pytest.mark.parametrize("operation", ["set", "unset", "unregistered"])
@pytest.mark.parametrize("readonly", [False, True], ids=["mutable", "readonly"])
def test_startup_routing_restores_or_aborts_before_user_command(
    tmp_path, shell_probe, use_pty, operation, readonly,
):
    expected = "owner's value\nwith $literal ; shell syntax"
    context = None if operation == "unregistered" else SessionExecutionContext(
        env_set={"ROUTING_VALUE": expected} if operation == "set" else {},
        env_unset={"ROUTING_VALUE"} if operation == "unset" else set(),
    )
    marker = tmp_path / "user-command.json"
    code = (
        "import json,os,pathlib; "
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps("
        '[os.getenv("ROUTING_VALUE"), os.getenv("ORDINARY_VALUE"), '
        "os.getcwd(), os.isatty(1)]))"
    )
    command = f"probe_function; {shlex.quote(sys.executable)} -c {shlex.quote(code)}; exit 23"
    registry, session = shell_probe(
        command, use_pty=use_pty, context=context, readonly=readonly,
    )
    result = registry.wait(session.id, timeout=15)
    assert result["status"] == "exited", result
    if readonly and context is not None:
        assert not marker.exists(), {"observed": marker.read_text(), "result": result}
        assert "FUNCTION_OK" not in result["output"], result
        assert result["exit_code"] == 126, result
        assert "readonly variable" in result["output"], result
    else:
        routing = "startup" if context is None else expected if operation == "set" else None
        assert json.loads(marker.read_text()) == [routing, "ordinary", str(tmp_path), use_pty]
        assert "FUNCTION_OK" in result["output"], result
        assert result["exit_code"] == 23, result
    assert session.cwd == str(tmp_path)
    assert session.command == command


@pytest.mark.linux_only
@pytest.mark.parametrize("use_pty", [False, True], ids=["pipe", "pty"])
@pytest.mark.parametrize("registered", [False, True], ids=["unregistered", "registered"])
def test_background_routing_preserves_signal_delivery_and_exit_status(
    tmp_path, shell_probe, use_pty, registered,
):
    ready = tmp_path / "ready"
    context = SessionExecutionContext(env_set={"ROUTING_VALUE": "owner"}) if registered else None
    # Install the trap before announcing readiness; signal only this private
    # shell's pid, never the test runner's group or a desktop process.
    command = f"trap 'exit 42' TERM; printf ready > {shlex.quote(str(ready))}; while :; do :; done"
    registry, session = shell_probe(command, use_pty=use_pty, context=context)
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), registry.poll(session.id)
    os.kill(session.pid, signal.SIGTERM)
    result = registry.wait(session.id, timeout=15)
    assert result["status"] == "exited", result
    assert result["exit_code"] == 42, result
