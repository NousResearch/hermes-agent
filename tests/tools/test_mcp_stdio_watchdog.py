"""Contract tests for the direct POSIX stdio MCP child watchdog."""

import os
import sys

import pytest

from tools import mcp_stdio_watchdog, mcp_tool


def test_is_orphaned_is_false_while_direct_parent_is_unchanged():
    original_ppid = 1234

    assert mcp_stdio_watchdog._is_orphaned(
        original_ppid,
        getppid=lambda: original_ppid,
    ) is False


def _fake_proc_stat(comm: str, starttime: str = "17123") -> str:
    # Fields 1..21 minimal (only the tail's alignment matters); field 22 is
    # starttime, the value we assert on.
    tail = [
        "R",      # 3 state
        "1000",   # 4 ppid
        "1000",   # 5 pgrp
        "1000",   # 6 session
        "0",      # 7 tty_nr
        "-1",     # 8 tpgid
        "0",      # 9 flags
        "0",      # 10 minflt
        "0",      # 11 cminflt
        "0",      # 12 majflt
        "0",      # 13 cmajflt
        "0",      # 14 utime
        "0",      # 15 stime
        "0",      # 16 cutime
        "0",      # 17 cstime
        "20",     # 18 priority
        "0",      # 19 nice
        "1",      # 20 num_threads
        "0",      # 21 itrealvalue
        starttime,  # 22 starttime
    ]
    return f"100 ({comm}) " + " ".join(tail) + " 0\n"


def test_parse_starttime_from_proc_stat_handles_normal_comm():
    line = _fake_proc_stat("sleep")
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat(line) == "17123"


def test_parse_starttime_from_proc_stat_handles_comm_with_closing_paren():
    # Kernel escapes ')' inside comm as '\\)', but some names (e.g. shell
    # pipelines, "(sd-pam)") legitimately contain a raw ')' — splitting on
    # the LAST ')' must still land on field 3.
    line = _fake_proc_stat("bash (sd-pam)")
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat(line) == "17123"


def test_parse_starttime_from_proc_stat_handles_comm_with_spaces():
    line = _fake_proc_stat("my process (daemon)", starttime="4242")
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat(line) == "4242"


def test_parse_starttime_from_proc_stat_returns_none_on_malformed_data():
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat("") is None
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat("100 sleep R") is None
    assert mcp_stdio_watchdog._parse_starttime_from_proc_stat("garbage") is None


def test_is_orphaned_true_when_parent_pid_is_recycled(monkeypatch):
    # The PID still reports our original parent — but the process occupying
    # it is a different incarnation (OS recycled the PID after a crash).
    monkeypatch.setattr(mcp_stdio_watchdog, "_original_parent_identity", "old-incarnation")
    monkeypatch.setattr(
        mcp_stdio_watchdog,
        "_read_process_identity",
        lambda pid: "new-incarnation",
    )

    assert mcp_stdio_watchdog._is_orphaned(
        1234,
        getppid=lambda: 1234,
    ) is True


def test_is_orphaned_false_when_parent_identity_is_unchanged(monkeypatch):
    monkeypatch.setattr(mcp_stdio_watchdog, "_original_parent_identity", "same-incarnation")
    monkeypatch.setattr(
        mcp_stdio_watchdog,
        "_read_process_identity",
        lambda pid: "same-incarnation",
    )

    assert mcp_stdio_watchdog._is_orphaned(
        1234,
        getppid=lambda: 1234,
    ) is False


def test_is_orphaned_false_when_parent_identity_is_unreadable(monkeypatch):
    # Transient identity read failure must never kill — fall back to the
    # legacy PID-only verdict instead of guessing.
    monkeypatch.setattr(mcp_stdio_watchdog, "_original_parent_identity", "old-incarnation")
    monkeypatch.setattr(mcp_stdio_watchdog, "_read_process_identity", lambda pid: None)

    assert mcp_stdio_watchdog._is_orphaned(
        1234,
        getppid=lambda: 1234,
    ) is False


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
