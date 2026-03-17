"""Tests for tools/infsh_tool.py — inference.sh CLI integration."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.infsh_tool import (
    check_infsh_requirements,
    infsh_tool,
    infsh_install,
)


class TestCheckRequirements:
    def test_returns_bool(self):
        result = check_infsh_requirements()
        assert isinstance(result, bool)

    def test_returns_true_when_infsh_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/infsh" if cmd == "infsh" else None)
        assert check_infsh_requirements() is True

    def test_returns_false_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        assert check_infsh_requirements() is False


class TestInfshTool:
    def test_not_installed_returns_error(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: False)
        result = json.loads(infsh_tool("app list"))
        assert result["success"] is False
        assert "not installed" in result["error"].lower()

    def test_successful_command(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: True)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"apps": ["flux", "veo"]}'
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = json.loads(infsh_tool("app list --search flux"))
            assert result["success"] is True
            mock_run.assert_called_once()
            call_cmd = mock_run.call_args[0][0]
            assert "infsh app list --search flux" in call_cmd

    def test_failed_command(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: True)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "unknown command"

        with patch("subprocess.run", return_value=mock_result):
            result = json.loads(infsh_tool("badcommand"))
            assert result["success"] is False
            assert result["exit_code"] == 1

    def test_timeout_handled(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: True)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("infsh", 300)):
            result = json.loads(infsh_tool("app run something", timeout=300))
            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    def test_json_output_parsed(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: True)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"url": "https://example.com/image.png"}'
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = json.loads(infsh_tool("app run flux --json"))
            assert result["success"] is True
            assert isinstance(result["output"], dict)
            assert result["output"]["url"] == "https://example.com/image.png"


class TestInfshInstall:
    def test_already_installed(self, monkeypatch):
        monkeypatch.setattr("tools.infsh_tool.check_infsh_requirements", lambda: True)
        monkeypatch.setattr("tools.infsh_tool._get_infsh_path", lambda: "/usr/local/bin/infsh")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "infsh v1.2.3"

        with patch("subprocess.run", return_value=mock_result):
            result = json.loads(infsh_install())
            assert result["success"] is True
            assert result["already_installed"] is True


class TestToolRegistration:
    def test_tools_registered(self):
        from tools.registry import registry
        assert "infsh" in registry._tools
        assert "infsh_install" in registry._tools

    def test_infsh_in_inference_toolset(self):
        from toolsets import TOOLSETS
        assert "inference" in TOOLSETS
        assert "infsh" in TOOLSETS["inference"]["tools"]
        assert "infsh_install" in TOOLSETS["inference"]["tools"]

    def test_infsh_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "infsh" not in _HERMES_CORE_TOOLS
        assert "infsh_install" not in _HERMES_CORE_TOOLS
