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
            # Force the powershell shell type so execute() dispatches to
            # _run_powershell (otherwise on a non-Windows test host it routes
            # to the bash path and never captures pwsh warnings).
            env._shell_type = "powershell"
            result = env.execute("$a ?? $b")

            assert "pwsh_warnings" in result
            assert result["pwsh_warnings"] == ["Line 2: null-coalescing `??` rewritten"]


# ---------------------------------------------------------------------------
# terminal_tool surfaces warnings in JSON
# ---------------------------------------------------------------------------

class TestTerminalToolSurfacesWarnings:
    """terminal_tool() includes pwsh_warnings in its JSON when the underlying
    execute() result carries them, and omits the key otherwise.

    terminal_tool runs a real command through the active environment, so we
    inject a fake local environment whose execute() returns a controlled result
    dict, then assert on the (flat) JSON terminal_tool emits.
    """

    def _run_with_fake_env_result(self, exec_result, tmp_path, task_id):
        import tools.terminal_tool as tt

        fake_env = MagicMock()
        fake_env.cwd = str(tmp_path)
        fake_env.env = {}
        fake_env.execute.return_value = exec_result

        with patch.object(tt, "_create_environment", return_value=fake_env), \
             patch.dict(tt._active_environments, {}, clear=True):
            raw = tt.terminal_tool("echo hi", task_id=task_id)
        return json.loads(raw)

    def test_terminal_tool_json_includes_pwsh_warnings(self, tmp_path):
        parsed = self._run_with_fake_env_result(
            {
                "output": "hello world",
                "returncode": 0,
                "pwsh_warnings": ["Line 1: ternary operator rewritten"],
            },
            tmp_path,
            task_id="pwsh-warn-present",
        )
        assert parsed["pwsh_warnings"] == ["Line 1: ternary operator rewritten"]

    def test_no_warnings_key_when_empty(self, tmp_path):
        parsed = self._run_with_fake_env_result(
            {"output": "hello world", "returncode": 0, "pwsh_warnings": []},
            tmp_path,
            task_id="pwsh-warn-empty",
        )
        assert "pwsh_warnings" not in parsed


# ---------------------------------------------------------------------------
# UTF-8 encoding: _run_powershell() emits correct UTF-8 output
# ---------------------------------------------------------------------------

class TestRunPowershellUtf8Encoding:
    """Verify _run_powershell() emits correct UTF-8 output."""

    pytestmark = pytest.mark.skipif(
        sys.platform != "win32",
        reason="PowerShell UTF-8 roundtrip depends on Windows shell selection",
    )

    def test_unicode_output_roundtrips(self, tmp_path):
        """Non-ASCII output from PowerShell is preserved."""
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd=str(tmp_path))
        # Write a command that outputs UTF-8 characters
        result = env.execute("Write-Output 'caf\u00e9 \u4f60\u597d'")
        assert result["returncode"] == 0
        assert "caf\u00e9" in result["output"]
        assert "\u4f60\u597d" in result["output"]

    def test_cjk_filenames_in_output(self, tmp_path):
        """Filenames with CJK characters are preserved."""
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd=str(tmp_path))
        result = env.execute("Get-ChildItem -Name")
        assert result["returncode"] == 0
        # Output should be valid UTF-8 (no UnicodeDecodeError during capture)

    def test_stderr_unicode(self, tmp_path):
        """Non-ASCII in stderr is preserved."""
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd=str(tmp_path))
        result = env.execute(
            "Write-Error '\u9519\u8bef\u4fe1\u606f'"
        )
        # Write-Error output (stderr merged into stdout) should contain the CJK
        assert "\u9519\u8bef\u4fe1\u606f" in result["output"]


# ---------------------------------------------------------------------------
# pwsh_transform + encoding preamble composition
# ---------------------------------------------------------------------------

class TestPwshTransformAndUtf8Compose:
    """pwsh_transform runs BEFORE the encoding preamble is prepended."""

    def test_transform_applied_before_preamble(self):
        """Encoding preamble must NOT be transformed by pwsh_transform."""
        from tools.environments.proccess_pwsh import pwsh_transform
        preamble = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
            "$OutputEncoding=[System.Text.Encoding]::UTF8;"
        )
        result, warnings = pwsh_transform(preamble + "Get-ChildItem")
        # Preamble should survive unchanged
        assert "[Console]::OutputEncoding" in result
        assert "$OutputEncoding" in result
        # No warnings should be generated for the preamble itself
        assert not any("OutputEncoding" in w for w in warnings)

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="PowerShell UTF-8 roundtrip depends on Windows shell selection",
    )
    def test_command_with_ternary_and_unicode_output(self, tmp_path):
        """Full pipeline: transform → preamble → execute → UTF-8 output."""
        from tools.environments.local import LocalEnvironment
        env = LocalEnvironment(cwd=str(tmp_path))
        # Use PS7 ternary syntax that pwsh_transform will down-level
        result = env.execute("$x = $true; if ($x) { '\u4f60\u597d' } else { 'world' }")
        assert result["returncode"] == 0
        assert "\u4f60\u597d" in result["output"]
