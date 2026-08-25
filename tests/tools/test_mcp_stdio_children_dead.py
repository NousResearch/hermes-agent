"""Regression tests for ``_stdio_children_dead`` (#94335).

The liveness check was inverted: the loop returned True ("all children dead")
on the first LIVE pid, so the #81995 pre-call fast-fail raised
``TimeoutError: MCP stdio subprocess ... has exited`` for every tools/call in
oneshot (-z) sessions even though the subprocess was demonstrably alive.
"""

from unittest.mock import patch

from tools.mcp_tool import MCPServerTask


def _task_with_pids(pids, *, http=False):
    task = object.__new__(MCPServerTask)
    task._stdio_child_pids = pids
    task._config = {"url": "http://example.invalid"} if http else {"command": "x"}
    return task


def test_live_child_reports_not_dead():
    """The reported bug: an alive tracked pid must NOT report all-dead."""
    with patch("psutil.pid_exists", return_value=True):
        assert _task_with_pids([60634])._stdio_children_dead() is False


def test_all_children_dead_reports_dead():
    with patch("psutil.pid_exists", return_value=False):
        assert _task_with_pids([111, 222])._stdio_children_dead() is True


def test_mixed_liveness_reports_not_dead():
    """One live sibling is enough — dead others must not flip the verdict."""
    with patch("psutil.pid_exists", side_effect=lambda pid: pid != 111):
        assert _task_with_pids([111, 222])._stdio_children_dead() is False


def test_no_captured_pids_stays_fail_open():
    """Unknown (no tracked pids / HTTP transport) must not fail fast."""
    assert _task_with_pids([])._stdio_children_dead() is False
    assert _task_with_pids([1], http=True)._stdio_children_dead() is False
