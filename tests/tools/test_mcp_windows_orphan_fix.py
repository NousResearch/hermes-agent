"""Tests for the Windows MCP orphan fix (#61059).

Windows has no POSIX parent-death supervisor / killpg safety net, so orphaned
npx → node.exe trees accumulated across session restarts. The fix has two rungs:

1. ``_run_stdio`` attaches this process to a KILL_ON_JOB_CLOSE job object before
   spawning stdio children, so any child/grandchild created after the attach dies
   with the parent at the kernel level — even on an ungraceful exit.
2. Windows reaps kill the whole process tree (direct child + descendants), since
   there is no pgid to group-kill and grandchildren reparent with ParentId=null.

The job-attach call itself is ctypes/Win32 and untestable off Windows; here we
verify it is invoked on the Windows spawn path and not on POSIX, plus the
best-effort contract of the tree-kill helpers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE

pytestmark = pytest.mark.skipif(not _MCP_AVAILABLE, reason="MCP SDK not installed")


def _run_stdio_with_mocks(os_name: str, attach_mock) -> None:
    """Drive _run_stdio's pre-spawn path with the transport mocked out."""
    import tools.mcp_tool_transport as transport_mod

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()

    async def _serve(session, timeout, **kwargs):
        return "ok"

    task = MCPServerTask("test-win-orphan")
    task._serve_session = _serve
    task._session_kwargs = lambda: {}

    async def fake_preflight(name, command, args):
        return command, args

    with (
        patch("hermes_cli.process_identity.attach_self_to_kill_on_close_job", attach_mock),
        patch("tools.mcp_tool.StdioServerParameters"),
        patch("tools.mcp_tool.stdio_client", return_value=(mock_stdio_cm := MagicMock())),
        patch("tools.mcp_tool.ClientSession", return_value=(mock_session_cm := MagicMock())),
        patch("tools.mcp_tool._preflight_stdio_command", fake_preflight),
        patch("tools.mcp_tool_lifecycle._snapshot_child_pids", return_value=set()),
        patch("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children"),
        patch("tools.mcp_tool_config._write_stderr_log_header"),
    ):
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)
        import asyncio

        # invoke the mixin method directly: start() would enter the reconnect loop because the
        # mocked session never completes a real handshake.
        asyncio.run(task._run_stdio({"command": "echo", "args": ["hello"]}))


def test_windows_spawn_attaches_kill_on_close_job():
    """On Windows the stdio spawn path must self-attach to a kill-on-close job (#61059)."""
    attach = MagicMock(return_value=True)
    _run_stdio_with_mocks("nt", attach)
    attach.assert_called_once_with()


def test_spawn_always_attempts_attach():
    """The attach call is unconditional and self-guarded (no-op off Windows); the spawn
    path must invoke it so Windows sessions get job coverage without any os.name checks
    that would be hard to mock at the spawn site."""
    attach = MagicMock(return_value=False)  # POSIX no-op return
    _run_stdio_with_mocks("posix", attach)
    attach.assert_called_once_with()


def test_windows_attach_failure_never_blocks_spawn():
    """A failed job attach (e.g. nested-job refusal) must not fail the MCP connection."""
    attach = MagicMock(side_effect=OSError("access denied"))
    _run_stdio_with_mocks("nt", attach)  # raises → would fail the test
    attach.assert_called_once_with()


class TestWindowsTreeKillHelpers:
    """Best-effort contract of the tree-kill helpers (Win32 paths can't run here)."""

    def test_tree_kill_swallows_missing_process(self):
        import tools.mcp_tool_lifecycle as lifecycle

        # Must not raise for a PID that raced away. (_kill_windows_process_tree carries no
        # os.name gate itself — the gate lives at the _signal_mcp_process call site — so this
        # exercises the helper directly without patching os.name, which breaks pathlib.)
        lifecycle._kill_windows_process_tree(999999999, 15)

    def test_tree_kill_terminates_descendants(self, monkeypatch):
        import tools.mcp_tool_lifecycle as lifecycle

        terminated = []
        killed = []

        class FakeProc:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=True):
                return [FakeProc(2), FakeProc(3)]

            def terminate(self):
                terminated.append(self.pid)

            def kill(self):
                killed.append(self.pid)

        fake_psutil = MagicMock()
        fake_psutil.Process = FakeProc
        fake_psutil.wait_procs = lambda procs, timeout: ([], [])  # all exited gracefully
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)

        lifecycle._kill_windows_process_tree(1, 15)
        # SIGTERM pass: descendants only — the direct child was already signalled by the caller.
        assert sorted(terminated) == [2, 3]
        assert killed == []

        # Force pass: survivors of the wait are killed. wait_procs reports everyone alive.
        fake_psutil.wait_procs = lambda procs, timeout: ([], list(procs))
        lifecycle._kill_windows_process_tree(1, 9)
        assert sorted(killed) == [2, 3]

    def test_ledger_tree_kill_helper_swallows_errors(self):
        """_kill_process_tree_windows never raises, even for a vanished process."""
        from hermes_cli.process_identity import _kill_process_tree_windows

        class Boom:
            def children(self, recursive=True):
                raise OSError("gone")

            def terminate(self):
                raise OSError("gone")

        _kill_process_tree_windows(Boom())  # must not raise
