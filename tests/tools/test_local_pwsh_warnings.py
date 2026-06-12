"""Tests for pwsh_transform warning propagation through LocalEnvironment.

Verifies that _run_powershell captures warnings from pwsh_transform (always-on),
execute() attaches them to the result dict, and terminal_tool surfaces them in JSON.
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_popen_for_powershell(stdout_data="output"):
    """Return a fake Popen that captures what was passed to it."""

    def fake_popen(args, **kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = MagicMock()
        proc.stdout.__iter__ = lambda s: iter([stdout_data])
        proc.stdin = MagicMock()
        proc.poll.return_value = 0
        return proc

    return fake_popen


# ---------------------------------------------------------------------------
# _run_powershell captures warnings (always-on pwsh_transform)
# ---------------------------------------------------------------------------

class TestRunPowershellCapturesWarnings:
    """Test that _run_powershell always stores pwsh_transform warnings on self._pwsh_warnings."""

    def test_refresh_env_from_registry_called(self):
        """refresh_env_from_registry is called before pwsh_transform."""
        from tools.environments.local import LocalEnvironment

        with patch(
            "tools.environments.local.refresh_env_from_registry",
        ) as mock_refresh, patch(
            "tools.environments.local.pwsh_transform",
            return_value=("code", []),
        ), patch(
            "tools.environments.local.subprocess.Popen",
            _fake_popen_for_powershell(),
        ):
            env = LocalEnvironment(cwd=r"C:\tmp", timeout=30)
            env._shell_path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            env._run_powershell("Write-Output hello")

            mock_refresh.assert_called_once()

    def test_warnings_stored_on_instance(self):
        """When pwsh_transform returns warnings, they are stored on self._pwsh_warnings."""
        from tools.environments.local import LocalEnvironment

        # Mock to return tuple with warnings
        mock_transform_result = ("transformed code", ["Line 1: ternary operator `$a ? $b : $c` rewritten"])

        with patch(
            "tools.environments.local.pwsh_transform",
            return_value=mock_transform_result,
        ), patch(
            "tools.environments.local.subprocess.Popen",
            _fake_popen_for_powershell(),
        ):
            env = LocalEnvironment(cwd=r"C:\tmp", timeout=30)
            env._shell_path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            env._run_powershell("$a ? $b : $c")

            assert hasattr(env, "_pwsh_warnings")
            assert env._pwsh_warnings == ["Line 1: ternary operator `$a ? $b : $c` rewritten"]

    def test_pwsh_transform_always_called(self):
        """pwsh_transform is now always-on — called for every command."""
        from tools.environments.local import LocalEnvironment

        with patch(
            "tools.environments.local.pwsh_transform",
            return_value=("code", []),
        ) as mock_transform, patch(
            "tools.environments.local.subprocess.Popen",
            _fake_popen_for_powershell(),
        ):
            env = LocalEnvironment(cwd=r"C:\tmp", timeout=30)
            env._shell_path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            env._run_powershell("Write-Output hello")

            # pwsh_transform should always be called (no conditional guard)
            mock_transform.assert_called_once()


# ---------------------------------------------------------------------------
# execute() attaches warnings to result dict
# ---------------------------------------------------------------------------

class TestExecuteAttachesWarnings:
    """Test that execute() propagates _pwsh_warnings to the result dict."""

    def test_execute_attaches_pwsh_warnings_to_result(self):
        from tools.environments.local import LocalEnvironment

        mock_transform_result = ("transformed code", ["Line 2: null-coalescing `??` rewritten"])

        with patch(
            "tools.environments.local.pwsh_transform",
            return_value=mock_transform_result,
        ), patch(
            "tools.environments.local.subprocess.Popen",
            _fake_popen_for_powershell("foo"),
        ), patch(
            "tools.environments.local.os.path.isdir", return_value=True
        ):
            env = LocalEnvironment(cwd=r"C:\tmp", timeout=30)
            env._shell_path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            result = env.execute("$a ?? $b")

            assert "pwsh_warnings" in result
            assert result["pwsh_warnings"] == ["Line 2: null-coalescing `??` rewritten"]


# ---------------------------------------------------------------------------
# terminal_tool surfaces warnings in JSON
# ---------------------------------------------------------------------------

class TestTerminalToolSurfacesWarnings:
    """Test that terminal_tool() includes pwsh_warnings in JSON response."""

    def test_terminal_tool_json_includes_pwsh_warnings(self):
        from tools.terminal_tool import terminal_tool

        mock_result = {
            "output": "hello world",
            "cwd": r"C:\tmp",
            "exit_code": 0,
            "pwsh_warnings": ["Line 1: ternary operator rewritten"],
        }

        raw = terminal_tool(mock_result, task_id="test-task")
        parsed = json.loads(raw)

        assert parsed["result"]["success"] is True
        assert "pwsh_warnings" in parsed["result"]
        assert parsed["result"]["pwsh_warnings"] == ["Line 1: ternary operator rewritten"]

    def test_no_warnings_key_when_empty(self):
        from tools.terminal_tool import terminal_tool

        mock_result = {
            "output": "hello world",
            "cwd": r"C:\tmp",
            "exit_code": 0,
            "pwsh_warnings": [],
        }

        raw = terminal_tool(mock_result, task_id="test-task")
        parsed = json.loads(raw)

        assert parsed["result"]["success"] is True
        assert "pwsh_warnings" not in parsed["result"]
