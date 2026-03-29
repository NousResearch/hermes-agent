"""Tests for system theme detection."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from hermes_cli.display import get_system_theme


class TestGetSystemTheme:

    def test_returns_valid_value(self):
        result = get_system_theme()
        assert result in ("dark", "light", "unknown")

    def test_windows_dark_mode(self):
        mock_winreg = MagicMock()
        mock_key = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key
        mock_winreg.QueryValueEx.return_value = (0, 4)  # 0 = dark
        mock_winreg.HKEY_CURRENT_USER = 0x80000001
        with patch.object(sys, "platform", "win32"):
            with patch.dict("sys.modules", {"winreg": mock_winreg}):
                # Force re-import inside the function
                result = get_system_theme()
        assert result == "dark"

    def test_windows_light_mode(self):
        mock_winreg = MagicMock()
        mock_key = MagicMock()
        mock_winreg.OpenKey.return_value = mock_key
        mock_winreg.QueryValueEx.return_value = (1, 4)  # 1 = light
        mock_winreg.HKEY_CURRENT_USER = 0x80000001
        with patch.object(sys, "platform", "win32"):
            with patch.dict("sys.modules", {"winreg": mock_winreg}):
                result = get_system_theme()
        assert result == "light"

    def test_windows_registry_error_returns_dark(self):
        mock_winreg = MagicMock()
        mock_winreg.OpenKey.side_effect = FileNotFoundError
        mock_winreg.HKEY_CURRENT_USER = 0x80000001
        with patch.object(sys, "platform", "win32"):
            with patch.dict("sys.modules", {"winreg": mock_winreg}):
                result = get_system_theme()
        assert result == "dark"

    def test_macos_dark(self):
        mock_result = MagicMock()
        mock_result.stdout = "Dark\n"
        with patch.object(sys, "platform", "darwin"):
            with patch("subprocess.run", return_value=mock_result):
                result = get_system_theme()
        assert result == "dark"

    def test_macos_light(self):
        mock_result = MagicMock()
        mock_result.stdout = "\n"  # no output = light mode on macOS
        with patch.object(sys, "platform", "darwin"):
            with patch("subprocess.run", return_value=mock_result):
                result = get_system_theme()
        assert result == "light"

    def test_linux_gtk_dark(self):
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"GTK_THEME": "Adwaita:dark"}):
                result = get_system_theme()
        assert result == "dark"

    def test_linux_gtk_light(self):
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"GTK_THEME": "Adwaita:light"}):
                result = get_system_theme()
        assert result == "light"

    def test_fallback_is_dark(self):
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"GTK_THEME": ""}, clear=False):
                result = get_system_theme()
        assert result == "dark"
