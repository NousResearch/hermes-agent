"""Regression tests for the stdio MCP parent-death watchdog."""

import sys

import psutil
import pytest

from tools import mcp_stdio_watchdog


def test_is_orphaned_tolerates_small_parent_create_time_drift():
    """WSL can move psutil's wall-clock create_time by a few seconds mid-process."""
    parent = psutil.Process()
    recorded_create_time = parent.create_time() + 2.0

    assert (
        mcp_stdio_watchdog._is_orphaned(
            parent.pid,
            recorded_create_time,
            getppid=lambda: parent.pid,
        )
        is False
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc start ticks are Linux-only"
)
def test_is_orphaned_uses_stable_proc_start_ticks_on_linux():
    """A matching procfs start token outranks drifting wall-clock create_time."""
    parent = psutil.Process()
    start_ticks = mcp_stdio_watchdog._read_proc_start_ticks(parent.pid)
    assert start_ticks is not None

    assert (
        mcp_stdio_watchdog._is_orphaned(
            parent.pid,
            parent.create_time() + 60.0,
            getppid=lambda: parent.pid,
            parent_start_ticks=start_ticks,
        )
        is False
    )


def test_is_orphaned_rejects_large_parent_create_time_mismatch():
    """A materially different create_time still identifies PID reuse."""
    parent = psutil.Process()
    recorded_create_time = parent.create_time() + 60.0

    assert (
        mcp_stdio_watchdog._is_orphaned(
            parent.pid,
            recorded_create_time,
            getppid=lambda: parent.pid,
        )
        is True
    )
