"""Tests for file handle management in LocalEnvironment._update_cwd.

_update_cwd must close the file handle after reading the CWD temp file.
Using open() without a context manager leaks the file descriptor.
"""

import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import LocalEnvironment


def _make_local_env(cwd_file: str) -> LocalEnvironment:
    """Create a minimal LocalEnvironment with a given cwd_file path."""
    env = object.__new__(LocalEnvironment)
    env._cwd_file = cwd_file
    env.cwd = "/original"
    env._cwd_marker = "__HERMES_CWD_test__"
    return env


class TestLocalEnvFileHandle:
    def test_update_cwd_closes_file_handle(self):
        """_update_cwd must not leak file handles over repeated calls."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cwd", delete=False) as f:
            f.write("/test/path")
            cwd_file = f.name

        try:
            env = _make_local_env(cwd_file)

            import subprocess
            pid = os.getpid()
            fd_before = subprocess.run(
                ["lsof", "-p", str(pid)],
                capture_output=True, text=True
            ).stdout.count("\n")

            for _ in range(50):
                env._update_cwd({"output": ""})

            fd_after = subprocess.run(
                ["lsof", "-p", str(pid)],
                capture_output=True, text=True
            ).stdout.count("\n")

            # Should not have 50 new fds open
            assert fd_after - fd_before < 10, (
                f"File descriptor leak: {fd_after - fd_before} new fds after 50 calls"
            )
            assert env.cwd == "/test/path"
        finally:
            os.unlink(cwd_file)

    def test_update_cwd_uses_context_manager(self):
        """_update_cwd should use a context manager for file access."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cwd", delete=False) as f:
            f.write("/ctx/path")
            cwd_file = f.name

        try:
            env = _make_local_env(cwd_file)

            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.read.return_value = "/ctx/path"
                mock_open.return_value = mock_file

                env._update_cwd({"output": ""})

                # open() should have been called (with context manager support)
                mock_open.assert_called_once_with(cwd_file)
                # __enter__ should have been called (context manager used)
                mock_file.__enter__.assert_called()
                # __exit__ should also have been called (context manager cleaned up)
                mock_file.__exit__.assert_called()
        finally:
            os.unlink(cwd_file)

    def test_update_cwd_closes_file_handle_on_read_exception(self):
        """_update_cwd must close the file descriptor even when read() raises."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix=".cwd", delete=False) as f:
            # Write invalid UTF-8 bytes so open(...).read() raises UnicodeDecodeError
            f.write(b"/test/\xff\xfe/path")
            cwd_file = f.name

        try:
            env = _make_local_env(cwd_file)

            captured_fh = []
            real_open = open

            def tracking_open(path, *args, **kwargs):
                fh = real_open(path, *args, **kwargs)
                captured_fh.append(fh)
                return fh

            with patch("builtins.open", tracking_open):
                try:
                    env._update_cwd({"output": ""})
                except UnicodeDecodeError:
                    pass  # pre-existing: except clause doesn't catch UnicodeDecodeError

            assert captured_fh, "open() was not called"
            assert captured_fh[0].closed, (
                "File descriptor was not closed after UnicodeDecodeError"
            )
        finally:
            os.unlink(cwd_file)
