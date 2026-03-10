"""Tests for file injection (host → sandbox).

All tests use mocks -- no real Docker/SSH environments are needed.
"""

import base64
import gzip
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.file_transfer import (
    inject_file_to_sandbox,
    queue_injection,
    process_pending_injections,
    _pending_injections,
    MAX_FILE_SIZE,
)


# =========================================================================
# Local Injection
# =========================================================================

class TestLocalInjection:
    """Tests for local backend file injection."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_inject_local_success(self, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("a,b\n1,2\n")
        dest = tmp_path / "workspace" / "uploads" / "upload.csv"

        result = inject_file_to_sandbox(str(source), str(dest))
        assert result["success"] is True
        assert os.path.exists(dest)
        assert dest.read_text() == "a,b\n1,2\n"

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_inject_local_missing_source(self, tmp_path):
        result = inject_file_to_sandbox("/nonexistent/file.csv", str(tmp_path / "dest.csv"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_inject_local_too_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.file_transfer.MAX_FILE_SIZE", 10)
        source = tmp_path / "big.bin"
        source.write_bytes(b"x" * 100)
        result = inject_file_to_sandbox(str(source), str(tmp_path / "dest.bin"))
        assert result["success"] is False
        assert "too large" in result["error"].lower()


# =========================================================================
# Docker Injection (mocked)
# =========================================================================

class TestDockerInjection:
    """Tests for Docker backend injection with mocked subprocess."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    @patch("tools.file_transfer.subprocess.run")
    @patch("tools.terminal_tool._active_environments")
    def test_docker_cp_reverse_success(self, mock_envs, mock_run, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        mock_env._container_id = "abc123"
        mock_envs.get = MagicMock(return_value=mock_env)

        mock_run.return_value = SimpleNamespace(returncode=0, stderr="")

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is True
        assert result["remote_path"] == "/workspace/uploads/upload.csv"

        # Verify docker cp was called
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "docker" in cmd[0]
        assert "cp" in cmd[1]

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    @patch("tools.file_transfer.subprocess.run")
    @patch("tools.terminal_tool._active_environments")
    def test_docker_cp_reverse_failure(self, mock_envs, mock_run, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        mock_env._container_id = "abc123"
        mock_envs.get = MagicMock(return_value=mock_env)

        mock_run.return_value = SimpleNamespace(returncode=1, stderr="Permission denied")

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is False


# =========================================================================
# SSH Injection (mocked)
# =========================================================================

class TestSSHInjection:
    """Tests for SSH backend injection with mocked subprocess."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "ssh"})
    @patch("tools.file_transfer.subprocess.run")
    @patch("tools.terminal_tool._active_environments")
    def test_scp_reverse_success(self, mock_envs, mock_run, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        mock_env.control_socket = "/tmp/ctrl.sock"
        mock_env.host = "remote.host"
        mock_env.user = "testuser"
        mock_env.port = 22
        mock_envs.get = MagicMock(return_value=mock_env)

        mock_run.return_value = SimpleNamespace(returncode=0, stderr="")

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is True

    @patch.dict(os.environ, {"TERMINAL_ENV": "ssh"})
    @patch("tools.file_transfer.subprocess.run")
    @patch("tools.terminal_tool._active_environments")
    def test_scp_reverse_failure(self, mock_envs, mock_run, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        mock_env.control_socket = "/tmp/ctrl.sock"
        mock_env.host = "remote.host"
        mock_env.user = "testuser"
        mock_env.port = 22
        mock_envs.get = MagicMock(return_value=mock_env)

        mock_run.return_value = SimpleNamespace(returncode=1, stderr="Connection refused")

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is False


# =========================================================================
# Base64 Injection (mocked)
# =========================================================================

class TestBase64Injection:
    """Tests for gzip+base64 fallback injection."""

    @patch.dict(os.environ, {"TERMINAL_ENV": "modal"})
    @patch("tools.terminal_tool._active_environments")
    def test_base64_upload_success(self, mock_envs, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("col1,col2\n1,2\n")

        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "", "returncode": 0}
        mock_envs.get = MagicMock(return_value=mock_env)

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is True

        # Verify execute was called (mkdir + write chunks + decode)
        assert mock_env.execute.call_count >= 3

    @patch.dict(os.environ, {"TERMINAL_ENV": "modal"})
    @patch("tools.terminal_tool._active_environments")
    def test_base64_upload_decode_failure(self, mock_envs, tmp_path):
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        # mkdir succeeds, write succeeds, decode fails
        mock_env.execute.side_effect = [
            {"output": "", "returncode": 0},  # mkdir
            {"output": "", "returncode": 0},  # echo chunk
            {"output": "gunzip: error", "returncode": 1},  # decode
        ]
        mock_envs.get = MagicMock(return_value=mock_env)

        result = inject_file_to_sandbox(
            str(source), "/workspace/uploads/upload.csv", "task1"
        )
        assert result["success"] is False


# =========================================================================
# Deferred Injection Queue
# =========================================================================

class TestDeferredInjection:
    """Tests for the queue_injection / process_pending_injections mechanism."""

    def setup_method(self):
        """Clear the pending queue before each test."""
        _pending_injections.clear()

    def test_queue_injection_stores_entry(self):
        queue_injection("task-abc", "/host/file.csv", "/workspace/uploads/file.csv")
        assert "task-abc" in _pending_injections
        assert len(_pending_injections["task-abc"]) == 1
        assert _pending_injections["task-abc"][0]["host_path"] == "/host/file.csv"

    def test_queue_multiple_files(self):
        queue_injection("task-abc", "/host/a.csv", "/workspace/uploads/a.csv")
        queue_injection("task-abc", "/host/b.csv", "/workspace/uploads/b.csv")
        assert len(_pending_injections["task-abc"]) == 2

    def test_process_clears_queue(self, tmp_path):
        """After processing, the queue for that task_id should be empty."""
        source = tmp_path / "file.csv"
        source.write_text("data")
        dest = tmp_path / "uploads" / "file.csv"

        queue_injection("task-x", str(source), str(dest))

        with patch.dict(os.environ, {"TERMINAL_ENV": "local"}):
            results = process_pending_injections("task-x")

        assert len(results) == 1
        assert results[0]["success"] is True
        assert "task-x" not in _pending_injections

    def test_process_empty_queue_is_noop(self):
        results = process_pending_injections("nonexistent-task")
        assert results == []

    @patch.dict(os.environ, {"TERMINAL_ENV": "docker"})
    @patch("tools.file_transfer.subprocess.run")
    @patch("tools.terminal_tool._active_environments")
    def test_process_injects_to_docker(self, mock_envs, mock_run, tmp_path):
        """Deferred injection should work with Docker backend."""
        source = tmp_path / "upload.csv"
        source.write_text("data")

        mock_env = MagicMock()
        mock_env._container_id = "cnt123"
        mock_envs.get = MagicMock(return_value=mock_env)
        mock_run.return_value = SimpleNamespace(returncode=0, stderr="")

        queue_injection("task-d", str(source), "/workspace/uploads/upload.csv")
        results = process_pending_injections("task-d")

        assert len(results) == 1
        assert results[0]["success"] is True
        mock_run.assert_called_once()

    def test_second_upload_same_session(self, tmp_path):
        """Queue + flush + queue again + flush again must work (env reuse)."""
        f1 = tmp_path / "first.csv"
        f1.write_text("a")
        f2 = tmp_path / "second.csv"
        f2.write_text("b")
        dest1 = tmp_path / "uploads" / "first.csv"
        dest2 = tmp_path / "uploads" / "second.csv"

        # First upload cycle
        queue_injection("reuse-task", str(f1), str(dest1))
        with patch.dict(os.environ, {"TERMINAL_ENV": "local"}):
            r1 = process_pending_injections("reuse-task")
        assert len(r1) == 1
        assert r1[0]["success"] is True
        assert "reuse-task" not in _pending_injections

        # Second upload cycle (env already exists)
        queue_injection("reuse-task", str(f2), str(dest2))
        with patch.dict(os.environ, {"TERMINAL_ENV": "local"}):
            r2 = process_pending_injections("reuse-task")
        assert len(r2) == 1
        assert r2[0]["success"] is True
        assert dest2.exists()


# =========================================================================
# file_tools reuse path flush
# =========================================================================

class TestFileToolsReuseFlush:
    """Verify _get_file_ops() flushes pending injections on cache-hit reuse.

    Bug context: the cache-hit fast path returned directly without calling
    process_pending_injections(), so uploads arriving after the sandbox was
    already created were never injected when file_tools was the first caller.
    """

    def setup_method(self):
        _pending_injections.clear()

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_cache_hit_flushes_pending(self, tmp_path):
        """_get_file_ops cache-hit must call process_pending_injections."""
        from tools.file_tools import _get_file_ops, _file_ops_cache, _file_ops_lock
        from tools.terminal_tool import _active_environments, _env_lock, _last_activity

        task_id = "flush-cache-test"
        mock_env = MagicMock()
        cached_ops = MagicMock()

        # Pre-populate env and file_ops cache to simulate prior access
        with _env_lock:
            _active_environments[task_id] = mock_env
            _last_activity[task_id] = time.time()
        with _file_ops_lock:
            _file_ops_cache[task_id] = cached_ops

        try:
            with patch("tools.file_transfer.process_pending_injections") as mock_flush:
                result = _get_file_ops(task_id)
                assert result is cached_ops  # confirmed cache hit
                mock_flush.assert_called_once_with(task_id)
        finally:
            with _env_lock:
                _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
            with _file_ops_lock:
                _file_ops_cache.pop(task_id, None)

    @patch.dict(os.environ, {"TERMINAL_ENV": "local"})
    def test_cache_hit_actually_injects_file(self, tmp_path):
        """End-to-end: queue upload, then _get_file_ops must inject it."""
        from tools.file_tools import _get_file_ops, _file_ops_cache, _file_ops_lock
        from tools.terminal_tool import _active_environments, _env_lock, _last_activity

        task_id = "flush-e2e-test"
        mock_env = MagicMock()
        cached_ops = MagicMock()

        with _env_lock:
            _active_environments[task_id] = mock_env
            _last_activity[task_id] = time.time()
        with _file_ops_lock:
            _file_ops_cache[task_id] = cached_ops

        # Queue an upload
        source = tmp_path / "upload.csv"
        source.write_text("hello")
        dest = tmp_path / "injected" / "upload.csv"
        queue_injection(task_id, str(source), str(dest))
        assert task_id in _pending_injections

        try:
            result = _get_file_ops(task_id)
            assert result is cached_ops
            # Queue must be drained and file must exist
            assert task_id not in _pending_injections
            assert dest.exists()
            assert dest.read_text() == "hello"
        finally:
            with _env_lock:
                _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
            with _file_ops_lock:
                _file_ops_cache.pop(task_id, None)


# =========================================================================
# Thread-safe queue
# =========================================================================

class TestQueueThreadSafety:
    """Verify that the pending injection queue uses locking."""

    def test_queue_uses_lock(self):
        """queue_injection should acquire _pending_lock."""
        import tools.file_transfer as ft
        tid = "lock_test"
        ft._pending_injections.pop(tid, None)

        # Replace the module-level lock with a tracking wrapper
        import threading
        real_lock = threading.Lock()
        acquired = []

        class TrackingLock:
            def __enter__(self):
                acquired.append("enter")
                return real_lock.__enter__()
            def __exit__(self, *a):
                return real_lock.__exit__(*a)

        original_lock = ft._pending_lock
        ft._pending_lock = TrackingLock()
        try:
            ft.queue_injection(tid, "/fake/src", "/fake/dst")
            assert len(acquired) >= 1, "Lock was not acquired during queue_injection"
        finally:
            ft._pending_lock = original_lock
            ft._pending_injections.pop(tid, None)

    def test_process_uses_lock(self):
        """process_pending_injections should acquire _pending_lock."""
        import tools.file_transfer as ft
        import threading
        real_lock = threading.Lock()
        acquired = []

        class TrackingLock:
            def __enter__(self):
                acquired.append("enter")
                return real_lock.__enter__()
            def __exit__(self, *a):
                return real_lock.__exit__(*a)

        original_lock = ft._pending_lock
        ft._pending_lock = TrackingLock()
        try:
            ft.process_pending_injections("nonexistent_task")
            assert len(acquired) >= 1, "Lock was not acquired during process"
        finally:
            ft._pending_lock = original_lock


# =========================================================================
# is_safe_file_path guard
# =========================================================================

class TestSafeFilePathGuard:
    """Tests for is_safe_file_path() allowlist function."""

    def test_rejects_outside_cache(self):
        from tools.file_transfer import is_safe_file_path
        assert is_safe_file_path("/etc/passwd") is None

    def test_rejects_dotenv(self):
        from tools.file_transfer import is_safe_file_path
        home = os.path.expanduser("~")
        assert is_safe_file_path(os.path.join(home, ".env")) is None

    def test_accepts_file_in_cache(self):
        from tools.file_transfer import is_safe_file_path, get_file_cache_dir
        cache_dir = get_file_cache_dir()
        probe = cache_dir / "test_guard_probe.txt"
        probe.write_text("test")
        try:
            result = is_safe_file_path(str(probe))
            assert result is not None
            assert result.name == "test_guard_probe.txt"
        finally:
            probe.unlink(missing_ok=True)

    def test_rejects_traversal(self):
        from tools.file_transfer import is_safe_file_path, FILE_CACHE_DIR
        evil = str(FILE_CACHE_DIR / ".." / ".." / ".bashrc")
        assert is_safe_file_path(evil) is None

    def test_rejects_nonexistent(self):
        from tools.file_transfer import is_safe_file_path, FILE_CACHE_DIR
        assert is_safe_file_path(str(FILE_CACHE_DIR / "does_not_exist.csv")) is None
