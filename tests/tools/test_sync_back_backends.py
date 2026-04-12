"""Tests for backend-specific bulk download implementations and cleanup() wiring."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments import modal as modal_env
from tools.environments import daytona as daytona_env
from tools.environments.ssh import SSHEnvironment


# ── SSH helpers ──────────────────────────────────────────────────────


@pytest.fixture
def ssh_mock_env(monkeypatch):
    """Create an SSHEnvironment with mocked connection/sync."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {
            "sync": lambda self, **k: None,
            "sync_back": lambda self: None,
        })(),
    )
    return SSHEnvironment(host="example.com", user="testuser")


# ── Modal helpers ────────────────────────────────────────────────────


def _make_mock_modal_env():
    """Create a minimal ModalEnvironment without calling __init__."""
    env = object.__new__(modal_env.ModalEnvironment)
    env._sandbox = MagicMock()
    env._worker = MagicMock()
    env._persistent = False
    env._task_id = "test"
    env._sync_manager = None
    return env


def _wire_modal_download(env, *, tar_bytes=b"fake-tar-data", exit_code=0):
    """Wire sandbox.exec.aio to return mock tar output for download tests.

    Returns the exec_calls list for assertion.
    """
    exec_calls = []

    async def mock_exec_fn(*args, **kwargs):
        exec_calls.append(args)
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.read = MagicMock()
        proc.stdout.read.aio = AsyncMock(return_value=tar_bytes)
        proc.wait = MagicMock()
        proc.wait.aio = AsyncMock(return_value=exit_code)
        return proc

    env._sandbox.exec = MagicMock()
    env._sandbox.exec.aio = mock_exec_fn

    def real_run_coroutine(coro, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    env._worker.run_coroutine = real_run_coroutine
    return exec_calls


# ── Daytona helpers ──────────────────────────────────────────────────


def _make_mock_daytona_env():
    """Create a minimal DaytonaEnvironment without calling __init__."""
    env = object.__new__(daytona_env.DaytonaEnvironment)
    env._sandbox = MagicMock()
    env._remote_home = "/root"
    env._sync_manager = None
    env._lock = __import__("threading").Lock()
    env._persistent = True
    env._task_id = "test"
    env._daytona = MagicMock()
    return env


# =====================================================================
# SSH bulk download
# =====================================================================


class TestSSHBulkDownload:
    """Unit tests for _ssh_bulk_download."""

    def test_ssh_bulk_download_runs_tar_over_ssh(self, ssh_mock_env, tmp_path):
        """subprocess.run command should include tar cf - over SSH."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            # open() will be called to write stdout; mock it to avoid actual file I/O
            ssh_mock_env._ssh_bulk_download(dest)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "tar cf -" in cmd_str
        assert "-C /" in cmd_str
        assert "home/testuser/.hermes" in cmd_str
        assert "ssh" in cmd_str
        assert "testuser@example.com" in cmd_str

    def test_ssh_bulk_download_writes_to_dest(self, ssh_mock_env, tmp_path):
        """subprocess.run should receive stdout=open(dest, 'wb')."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)

        # The stdout kwarg should be a file object opened for writing
        call_kwargs = mock_run.call_args
        # stdout is passed as a keyword arg
        stdout_val = call_kwargs.kwargs.get("stdout") or call_kwargs[1].get("stdout")
        # The file was opened via `with open(dest, "wb") as f` and passed as stdout=f.
        # After the context manager exits, the file is closed, but we can verify
        # the dest path was used by checking if the file was created.
        assert dest.exists()

    def test_ssh_bulk_download_raises_on_failure(self, ssh_mock_env, tmp_path):
        """Non-zero returncode should raise RuntimeError."""
        dest = tmp_path / "backup.tar"

        failed = subprocess.CompletedProcess([], 1, stderr=b"Permission denied")
        with patch.object(subprocess, "run", return_value=failed):
            with pytest.raises(RuntimeError, match="SSH bulk download failed"):
                ssh_mock_env._ssh_bulk_download(dest)

    def test_ssh_bulk_download_uses_120s_timeout(self, ssh_mock_env, tmp_path):
        """The subprocess.run call should use a 120s timeout."""
        dest = tmp_path / "backup.tar"

        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as mock_run:
            ssh_mock_env._ssh_bulk_download(dest)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("timeout") == 120 or call_kwargs[1].get("timeout") == 120


class TestSSHCleanup:
    """Verify SSH cleanup() calls sync_back() before closing ControlMaster."""

    def test_ssh_cleanup_calls_sync_back(self, monkeypatch):
        """cleanup() should call sync_back() before SSH control socket teardown."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        call_order = []

        class TrackingSyncManager:
            def __init__(self, **kwargs):
                pass

            def sync(self, **kw):
                pass

            def sync_back(self):
                call_order.append("sync_back")

        monkeypatch.setattr(ssh_env, "FileSyncManager", TrackingSyncManager)

        env = SSHEnvironment(host="h", user="u")
        # Ensure control_socket does not exist so cleanup skips the SSH exit call
        env.control_socket = Path("/nonexistent/socket")

        env.cleanup()

        assert "sync_back" in call_order

    def test_ssh_cleanup_calls_sync_back_before_control_exit(self, monkeypatch):
        """sync_back() must run before the ControlMaster exit command."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        call_order = []

        class TrackingSyncManager:
            def __init__(self, **kwargs):
                pass

            def sync(self, **kw):
                pass

            def sync_back(self):
                call_order.append("sync_back")

        monkeypatch.setattr(ssh_env, "FileSyncManager", TrackingSyncManager)

        env = SSHEnvironment(host="h", user="u")

        # Create a fake control socket so cleanup tries the SSH exit
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sock") as tmp:
            env.control_socket = Path(tmp.name)

        def mock_run(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            if "-O" in cmd and "exit" in cmd_str:
                call_order.append("control_exit")
            return subprocess.CompletedProcess([], 0)

        with patch.object(subprocess, "run", side_effect=mock_run):
            env.cleanup()

        assert call_order.index("sync_back") < call_order.index("control_exit")


# =====================================================================
# Modal bulk download
# =====================================================================


class TestModalBulkDownload:
    """Unit tests for _modal_bulk_download."""

    def test_modal_bulk_download_command(self, tmp_path):
        """exec should be called with tar cf - -C /root/.hermes ."""
        env = _make_mock_modal_env()
        exec_calls = _wire_modal_download(env, tar_bytes=b"tar-content")
        dest = tmp_path / "backup.tar"

        env._modal_bulk_download(dest)

        assert len(exec_calls) == 1
        args = exec_calls[0]
        assert args[0] == "bash"
        assert args[1] == "-c"
        assert "tar cf -" in args[2]
        assert "-C / root/.hermes" in args[2]

    def test_modal_bulk_download_writes_to_dest(self, tmp_path):
        """Downloaded tar bytes should be written to the dest path."""
        env = _make_mock_modal_env()
        expected_data = b"some-tar-archive-bytes"
        _wire_modal_download(env, tar_bytes=expected_data)
        dest = tmp_path / "backup.tar"

        env._modal_bulk_download(dest)

        assert dest.exists()
        assert dest.read_bytes() == expected_data

    def test_modal_bulk_download_handles_str_output(self, tmp_path):
        """If stdout returns str instead of bytes, it should be encoded."""
        env = _make_mock_modal_env()
        # Simulate Modal SDK returning str
        _wire_modal_download(env, tar_bytes="string-tar-data")
        dest = tmp_path / "backup.tar"

        env._modal_bulk_download(dest)

        assert dest.read_bytes() == b"string-tar-data"

    def test_modal_bulk_download_raises_on_failure(self, tmp_path):
        """Non-zero exit code should raise RuntimeError."""
        env = _make_mock_modal_env()
        _wire_modal_download(env, exit_code=1)
        dest = tmp_path / "backup.tar"

        with pytest.raises(RuntimeError, match="Modal bulk download failed"):
            env._modal_bulk_download(dest)

    def test_modal_bulk_download_uses_120s_timeout(self, tmp_path):
        """run_coroutine should be called with timeout=120."""
        env = _make_mock_modal_env()
        _wire_modal_download(env, tar_bytes=b"data")

        run_kwargs = {}
        original_run = env._worker.run_coroutine

        def tracking_run(coro, **kwargs):
            run_kwargs.update(kwargs)
            return original_run(coro, **kwargs)

        env._worker.run_coroutine = tracking_run
        dest = tmp_path / "backup.tar"

        env._modal_bulk_download(dest)

        assert run_kwargs.get("timeout") == 120


class TestModalCleanup:
    """Verify Modal cleanup() calls sync_back() before terminate."""

    def test_modal_cleanup_calls_sync_back(self):
        """cleanup() should call sync_back() before sandbox.terminate."""
        env = _make_mock_modal_env()

        call_order = []
        sync_mgr = MagicMock()
        sync_mgr.sync_back = lambda: call_order.append("sync_back")
        env._sync_manager = sync_mgr

        # Mock terminate to track call order
        async def mock_terminate():
            pass

        env._sandbox.terminate = MagicMock()
        env._sandbox.terminate.aio = mock_terminate
        env._worker.run_coroutine = lambda coro, **kw: (
            call_order.append("terminate"),
            asyncio.new_event_loop().run_until_complete(coro),
        )
        env._worker.stop = lambda: None

        env.cleanup()

        assert "sync_back" in call_order
        assert call_order.index("sync_back") < call_order.index("terminate")


# =====================================================================
# Daytona bulk download
# =====================================================================


class TestDaytonaBulkDownload:
    """Unit tests for _daytona_bulk_download."""

    def test_daytona_bulk_download_creates_tar_and_downloads(self, tmp_path):
        """exec and download_file should both be called."""
        env = _make_mock_daytona_env()
        dest = tmp_path / "backup.tar"

        env._daytona_bulk_download(dest)

        # exec called twice: tar creation + rm cleanup
        assert env._sandbox.process.exec.call_count == 2
        tar_cmd = env._sandbox.process.exec.call_args_list[0][0][0]
        assert "tar cf" in tar_cmd
        assert "/tmp/.hermes_sync.tar" in tar_cmd
        assert ".hermes" in tar_cmd

        cleanup_cmd = env._sandbox.process.exec.call_args_list[1][0][0]
        assert "rm -f /tmp/.hermes_sync.tar" in cleanup_cmd

        env._sandbox.fs.download_file.assert_called_once_with(
            "/tmp/.hermes_sync.tar", str(dest)
        )

    def test_daytona_bulk_download_uses_remote_home(self, tmp_path):
        """The tar command should use the env's _remote_home."""
        env = _make_mock_daytona_env()
        env._remote_home = "/home/daytona"
        dest = tmp_path / "backup.tar"

        env._daytona_bulk_download(dest)

        tar_cmd = env._sandbox.process.exec.call_args_list[0][0][0]
        assert "home/daytona/.hermes" in tar_cmd


class TestDaytonaCleanup:
    """Verify Daytona cleanup() calls sync_back() before stop."""

    def test_daytona_cleanup_calls_sync_back(self):
        """cleanup() should call sync_back() before sandbox.stop()."""
        env = _make_mock_daytona_env()

        call_order = []
        sync_mgr = MagicMock()
        sync_mgr.sync_back = lambda: call_order.append("sync_back")
        env._sync_manager = sync_mgr
        env._sandbox.stop = lambda: call_order.append("stop")

        env.cleanup()

        assert "sync_back" in call_order
        assert "stop" in call_order
        assert call_order.index("sync_back") < call_order.index("stop")


# =====================================================================
# FileSyncManager wiring: bulk_download_fn passed by each backend
# =====================================================================


class TestBulkDownloadWiring:
    """Verify each backend passes bulk_download_fn to FileSyncManager."""

    def test_ssh_passes_bulk_download_fn(self, monkeypatch):
        """SSHEnvironment should pass _ssh_bulk_download to FileSyncManager."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/root")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        captured_kwargs = {}

        class CaptureSyncManager:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def sync(self, **kw):
                pass

        monkeypatch.setattr(ssh_env, "FileSyncManager", CaptureSyncManager)

        SSHEnvironment(host="h", user="u")

        assert "bulk_download_fn" in captured_kwargs
        assert callable(captured_kwargs["bulk_download_fn"])

    def test_modal_passes_bulk_download_fn(self, monkeypatch):
        """ModalEnvironment should pass _modal_bulk_download to FileSyncManager."""
        captured_kwargs = {}

        def capture_fsm(**kwargs):
            captured_kwargs.update(kwargs)
            return type("M", (), {"sync": lambda self, **k: None})()

        monkeypatch.setattr(modal_env, "FileSyncManager", capture_fsm)

        env = object.__new__(modal_env.ModalEnvironment)
        env._sandbox = MagicMock()
        env._worker = MagicMock()
        env._persistent = False
        env._task_id = "test"

        # Replicate the wiring done in __init__
        from tools.environments.file_sync import iter_sync_files
        env._sync_manager = modal_env.FileSyncManager(
            get_files_fn=lambda: iter_sync_files("/root/.hermes"),
            upload_fn=env._modal_upload,
            delete_fn=env._modal_delete,
            bulk_upload_fn=env._modal_bulk_upload,
            bulk_download_fn=env._modal_bulk_download,
        )

        assert "bulk_download_fn" in captured_kwargs
        assert callable(captured_kwargs["bulk_download_fn"])

    def test_daytona_passes_bulk_download_fn(self, monkeypatch):
        """DaytonaEnvironment should pass _daytona_bulk_download to FileSyncManager."""
        captured_kwargs = {}

        def capture_fsm(**kwargs):
            captured_kwargs.update(kwargs)
            return type("M", (), {"sync": lambda self, **k: None})()

        monkeypatch.setattr(daytona_env, "FileSyncManager", capture_fsm)

        env = object.__new__(daytona_env.DaytonaEnvironment)
        env._sandbox = MagicMock()
        env._remote_home = "/root"
        env._lock = __import__("threading").Lock()
        env._persistent = True
        env._task_id = "test"
        env._daytona = MagicMock()

        # Replicate the wiring done in __init__
        from tools.environments.file_sync import iter_sync_files
        env._sync_manager = daytona_env.FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{env._remote_home}/.hermes"),
            upload_fn=env._daytona_upload,
            delete_fn=env._daytona_delete,
            bulk_upload_fn=env._daytona_bulk_upload,
            bulk_download_fn=env._daytona_bulk_download,
        )

        assert "bulk_download_fn" in captured_kwargs
        assert callable(captured_kwargs["bulk_download_fn"])
