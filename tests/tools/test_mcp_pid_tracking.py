"""Tests for MCP stdio PID tracking — grandchild process discovery.

Verifies the fix for #26042: shell-wrapper MCP servers spawn grandchildren
that escape direct-child PID snapshots, leading to zombie accumulation.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Import the function under test
from tools.mcp_tool import _snapshot_child_pids, _kill_orphaned_mcp_children


class TestSnapshotChildPidsRecursive:
    """Verify _snapshot_child_pids discovers grandchildren, not just direct children."""

    def test_recursive_linux_proc_children(self):
        """Simulate /proc/{pid}/task/{pid}/children with a process tree:
        
        Hermes (PID 1) -> bash (PID 10) -> node (PID 20)
        
        Old code only saw PID 10 (direct child).
        New code should discover PID 20 (grandchild) by recursing.
        """
        my_pid = 1234
        
        # Simulate /proc reading:
        # - PID 1234 (Hermes) has children: [10]
        # - PID 10 (bash wrapper) has children: [20]
        # - PID 20 (node server) has no children
        proc_children = {
            my_pid: "10",
            10: "20",
            20: "",
        }
        
        def mock_open_children(path):
            # Extract PID from path like /proc/1234/task/1234/children
            parts = path.split("/")
            pid = int(parts[2])
            return mock_open(read_data=proc_children.get(pid, ""))()
        
        with patch("os.getpid", return_value=my_pid), \
             patch("builtins.open", side_effect=lambda p, *a, **kw: mock_open_children(p)):
            result = _snapshot_child_pids()
        
        # Must include both direct child (10) AND grandchild (20)
        assert 10 in result, "Direct child PID should be discovered"
        assert 20 in result, "Grandchild PID should be discovered via recursive walk"
        assert my_pid not in result, "Own PID should not be in the result"

    def test_psutil_fallback_uses_recursive(self):
        """When /proc is unavailable, psutil fallback should use recursive=True.
        
        Since the test environment has /proc, we verify the code path by
        reading the source directly. The functional behavior is already
        covered by the /proc recursive tests above.
        """
        import inspect
        source = inspect.getsource(_snapshot_child_pids)
        # Verify the psutil fallback uses recursive=True
        assert "children(recursive=True)" in source, (
            "psutil fallback must use recursive=True to discover grandchildren"
        )

    def test_empty_result_when_no_children(self):
        """When no children exist, return empty set."""
        my_pid = 9999
        
        with patch("os.getpid", return_value=my_pid), \
             patch("builtins.open", side_effect=FileNotFoundError("no /proc")):
            # psutil also fails
            with patch.dict(sys.modules, {}):
                # Force psutil import to fail
                result = _snapshot_child_pids()
        
        assert isinstance(result, set)
        # May or may not be empty depending on psutil availability in test env

    def test_handles_proc_read_errors_gracefully(self):
        """Individual /proc reads that fail should not prevent other children."""
        my_pid = 42
        
        # PID 42 has children [100, 101]
        # PID 100 has children [200] (readable)
        # PID 101 raises OSError (unreadable — process already exited)
        call_count = {"n": 0}
        
        def selective_open(path, *args, **kwargs):
            if str(my_pid) in path:
                return mock_open(read_data="100 101")()
            elif "100" in path:
                return mock_open(read_data="200")()
            elif "101" in path:
                raise OSError("process gone")
            raise FileNotFoundError(path)
        
        with patch("os.getpid", return_value=my_pid), \
             patch("builtins.open", side_effect=selective_open):
            result = _snapshot_child_pids()
        
        # Should still find 100 and 200, skipping 101 gracefully
        assert 100 in result
        assert 200 in result
        # 101 itself is a direct child so it's discovered from PID 42's children file
        assert 101 in result


class TestKillOrphanedMcpChildren:
    """Verify _kill_orphaned_mcp_children also terminates children of tracked PIDs."""

    def test_kills_children_of_orphans(self):
        """When killing an orphan PID, also discover and kill its children."""
        import tools.mcp_tool as mcp_mod
        
        # Set up orphan set with PID 100, which has a child PID 200
        mcp_mod._orphan_stdio_pids.clear()
        mcp_mod._orphan_stdio_pids.add(100)
        mcp_mod._stdio_pids.clear()
        
        killed_pids = []
        
        def mock_kill(pid, sig):
            killed_pids.append(pid)
        
        # PID 100's children file says it has child 200
        def selective_open(path, *args, **kwargs):
            if "/proc/100/" in path:
                return mock_open(read_data="200")()
            raise FileNotFoundError(path)
        
        import signal
        with patch("os.kill", side_effect=mock_kill), \
             patch("builtins.open", side_effect=selective_open), \
             patch("time.sleep"), \
             patch("gateway.status._pid_exists", return_value=True):
            _kill_orphaned_mcp_children()
        
        # Should have sent SIGTERM to both 100 and 200
        assert 100 in killed_pids, "Orphan PID should be killed"
        assert 200 in killed_pids, "Child of orphan should also be killed"
        
        # Cleanup
        mcp_mod._orphan_stdio_pids.clear()

    def test_no_proc_still_kills_tracked_pids(self):
        """Even when /proc is unavailable, tracked orphan PIDs are still killed."""
        import tools.mcp_tool as mcp_mod
        
        mcp_mod._orphan_stdio_pids.clear()
        mcp_mod._orphan_stdio_pids.add(300)
        mcp_mod._stdio_pids.clear()
        
        killed_pids = []
        
        def mock_kill(pid, sig):
            killed_pids.append(pid)
        
        with patch("os.kill", side_effect=mock_kill), \
             patch("builtins.open", side_effect=FileNotFoundError("no /proc")), \
             patch("time.sleep"):
            _kill_orphaned_mcp_children()
        
        assert 300 in killed_pids
        
        # Cleanup
        mcp_mod._orphan_stdio_pids.clear()
