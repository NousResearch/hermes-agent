"""Contract tests for the direct POSIX stdio MCP child watchdog."""

import os
import signal
import subprocess
import sys
import time

import pytest

from tools import mcp_stdio_watchdog, mcp_tool


def test_is_orphaned_is_false_while_direct_parent_is_unchanged():
    original_ppid = 1234

    assert mcp_stdio_watchdog._is_orphaned(
        original_ppid,
        getppid=lambda: original_ppid,
    ) is False


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_watchdog_module_wrap_command_uses_calling_parent_pid():
    parent_pid = os.getpid()

    wrapped_command, wrapped_args = mcp_stdio_watchdog.wrap_command(
        "/opt/hermes/bin/cua-driver",
        ["mcp", "--no-overlay"],
    )

    assert wrapped_command == sys.executable
    assert wrapped_args == [
        os.path.abspath(mcp_stdio_watchdog.__file__),
        "--ppid",
        str(parent_pid),
        "--",
        "/opt/hermes/bin/cua-driver",
        "mcp",
        "--no-overlay",
    ]


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_wrap_command_uses_stable_parent_pid_and_preserves_command_tail():
    parent_pid = os.getpid()
    command = "/opt/hermes/bin/mcp-server"
    command_args = ["--label", "value with spaces", "--", "literal-tail"]

    wrapped_command, wrapped_args = mcp_tool._wrap_command_with_watchdog(
        command,
        command_args,
    )

    assert wrapped_command == sys.executable
    assert wrapped_args == [
        os.path.join(os.path.dirname(mcp_tool.__file__), "mcp_stdio_watchdog.py"),
        "--ppid",
        str(parent_pid),
        "--",
        command,
        *command_args,
    ]
    assert "--create-time" not in wrapped_args


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
@pytest.mark.live_system_guard_bypass
def test_sigterm_resistant_child_is_killed_before_sdk_cleanup_deadline(tmp_path):
    """The supervisor must finish escalation before MCP SDK kills the wrapper."""
    pid_file = tmp_path / "child.pid"
    child_script = tmp_path / "ignore_term.py"
    child_script.write_text(
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    watchdog = subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(mcp_stdio_watchdog.__file__),
            "--ppid",
            str(os.getpid()),
            "--",
            sys.executable,
            str(child_script),
            str(pid_file),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 2.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists(), "watchdog child never reached its entry point"
        child_pid = int(pid_file.read_text(encoding="utf-8"))

        started = time.monotonic()
        watchdog.send_signal(signal.SIGTERM)
        watchdog.wait(timeout=1.8)
        assert time.monotonic() - started < 1.8
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if watchdog.poll() is None:
            watchdog.kill()
            watchdog.wait(timeout=2.0)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
