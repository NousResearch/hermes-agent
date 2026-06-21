"""Tests for tools.admin_executor — Windows elevated command execution."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call


class TestAdminExecutorPlatformChecks(unittest.TestCase):
    """Platform detection and capability checks."""

    @patch("tools.admin_executor.sys")
    def test_is_windows_true(self, mock_sys):
        mock_sys.platform = "win32"
        from tools.admin_executor import is_windows
        self.assertTrue(is_windows())

    @patch("tools.admin_executor.sys")
    def test_is_windows_false(self, mock_sys):
        mock_sys.platform = "linux"
        from tools.admin_executor import is_windows
        self.assertFalse(is_windows())

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_is_running_as_admin_non_windows(self, _):
        from tools.admin_executor import is_running_as_admin
        self.assertFalse(is_running_as_admin())

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_can_elevate_non_windows(self, _):
        from tools.admin_executor import can_elevate
        self.assertFalse(can_elevate())


class TestExecuteElevatedValidation(unittest.TestCase):
    """Input validation for execute_elevated."""

    @patch("tools.admin_executor.is_windows", return_value=False)
    def test_rejects_non_windows(self, _):
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("only supported on Windows", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=True)
    def test_rejects_already_admin(self, *_):
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Already running as administrator", result["error"])


class TestCanElevate(unittest.TestCase):
    """can_elevate() logic."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=True)
    def test_already_admin_returns_false(self, *_):
        from tools.admin_executor import can_elevate
        self.assertFalse(can_elevate())

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_can_elevate_on_windows(self, mock_ctypes, *_):
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock()
        from tools.admin_executor import can_elevate
        self.assertTrue(can_elevate())


class TestShellExecuteErrorCodes(unittest.TestCase):
    """ShellExecuteW error code handling — especially ERROR_CANCELLED."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_error_cancelled_1223_immediate_return(self, mock_ctypes, *_):
        """ERROR_CANCELLED (1223) must return immediately, not poll for timeout."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(return_value=1223)
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello", timeout=5)
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("cancelled", result["error"].lower())
        self.assertIn("1223", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_error_access_denied_5(self, mock_ctypes, *_):
        """Access denied (5) returns immediately."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(return_value=5)
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Access denied", result["error"])
        self.assertIn("5", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_error_unknown_code(self, mock_ctypes, *_):
        """Unknown error code returns descriptive message."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(return_value=15)
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("Unknown error", result["error"])
        self.assertIn("15", result["error"])

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_shellexecute_exception(self, mock_ctypes, *_):
        """ShellExecuteW raising an exception is caught."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(
            side_effect=OSError("access violation")
        )
        from tools.admin_executor import execute_elevated
        result = execute_elevated("echo hello")
        self.assertEqual(result["exit_code"], -1)
        self.assertIn("ShellExecuteW failed", result["error"])


class TestTempDirectoryCleanup(unittest.TestCase):
    """Temp directory must be cleaned up after execution."""

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_cleanup_on_shellexecute_failure(self, mock_ctypes, *_):
        """Temp dir is cleaned up even when ShellExecuteW returns error."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(return_value=5)

        created_dirs = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("tools.admin_executor.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            from tools.admin_executor import execute_elevated
            result = execute_elevated("echo hello")

        self.assertEqual(result["exit_code"], -1)
        # Verify temp dir was cleaned up
        for d in created_dirs:
            self.assertFalse(os.path.exists(d), f"Temp dir not cleaned: {d}")

    @patch("tools.admin_executor.is_windows", return_value=True)
    @patch("tools.admin_executor.is_running_as_admin", return_value=False)
    @patch("tools.admin_executor.ctypes")
    def test_cleanup_on_cancelled_uac(self, mock_ctypes, *_):
        """Temp dir is cleaned up when user cancels UAC (1223)."""
        mock_ctypes.windll.shell32.ShellExecuteW = MagicMock(return_value=1223)

        created_dirs = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("tools.admin_executor.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            from tools.admin_executor import execute_elevated
            result = execute_elevated("echo hello")

        self.assertIn("cancelled", result["error"].lower())
        for d in created_dirs:
            self.assertFalse(os.path.exists(d), f"Temp dir not cleaned: {d}")


if __name__ == "__main__":
    unittest.main()
