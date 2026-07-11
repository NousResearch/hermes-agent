"""Regression tests for MCP stdio watchdog orphan detection.

Issue #62505: the POSIX stdio MCP watchdog incorrectly classified a live
Hermes parent as orphaned and terminated the MCP server process group when
psutil.Process.create_time() drifted due to system clock changes.

The fix removes the create_time comparison and relies on direct PPID
equality instead, which is sufficient for POSIX direct-child detection.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_watchdog_no_create_time_argument():
    """Watchdog script does not require --create-time argument."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mcp_stdio_watchdog",
            "--ppid",
            str(os.getpid()),
            "--",
            "echo",
            "test",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # The command should run successfully (echo prints "test")
    assert result.returncode == 0
    assert "test" in result.stdout


def test_watchdog_create_time_argument_rejected():
    """Watchdog script rejects the old --create-time argument."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mcp_stdio_watchdog",
            "--ppid",
            "12345",
            "--create-time",
            "100.0",
            "--",
            "echo",
            "test",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Should error because --create-time is no longer accepted
    assert result.returncode != 0
    assert "unrecognized arguments: --create-time" in result.stderr


def test_watchdog_requires_ppid():
    """Watchdog script requires --ppid argument."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mcp_stdio_watchdog",
            "--",
            "echo",
            "test",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Should error because --ppid is required
    assert result.returncode != 0
    assert "required: --ppid" in result.stderr


def test_watchdog_terminates_when_parent_dies():
    """Watchdog terminates child when parent PPID doesn't match.

    If we provide a PPID that doesn't match the actual parent of the
    watchdog, the watchdog should terminate the child immediately.
    """
    # Use a non-existent PPID
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mcp_stdio_watchdog",
            "--ppid",
            "99999",
            "--",
            "sleep",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Should terminate immediately (exit code 143 = SIGTERM, or similar)
    assert result.returncode != 0
    # Child should not complete the 10-second sleep
    assert "test" not in result.stdout