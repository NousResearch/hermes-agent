"""Tests for tools.admin_executor — Windows elevated command execution."""

import unittest
from unittest.mock import patch, MagicMock


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


if __name__ == "__main__":
    unittest.main()
