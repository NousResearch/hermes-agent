"""Tests for tools.command_filter — command allowlist/denylist enforcement."""

import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.command_filter import (
    _COMPILED_DENY,
    _DEFAULT_DENY_PATTERNS,
    _compile_patterns,
    check_command_filter,
    check_command_filter_fast,
    get_allowlist_patterns,
    get_denylist_patterns,
)


class TestDefaultDenylist:
    """Verify the default denylist blocks known-dangerous commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -rf /tmp",
            "rm -fr /home",
            "sudo rm -rf /",
            "mkfs.ext4 /dev/sda1",
            "mkfs -t ext4 /dev/sdb",
            "dd if=/dev/zero of=/dev/sda bs=1M",
            "dd of=/dev/sda1 bs=4M",
            ">/dev/sda",
            ":> /dev/sdb1",
            "shutdown now",
            "reboot",
            "halt -p",
            "poweroff",
            "kill -9 -1",
            "killall -9 bash",
            "chmod -R 777 /",
            ">/etc/passwd",
            ":> /etc/shadow",
        ],
    )
    def test_dangerous_commands_blocked(self, command: str) -> None:
        """Each dangerous command should be blocked by default denylist."""
        ok, reason = check_command_filter_fast(command)
        assert not ok, f"Expected {command!r} to be blocked, but it passed: {reason}"
        assert reason is not None


class TestSafeCommandsAllowed:
    """Verify that safe commands pass through when no allowlist is configured."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat /etc/hostname",
            "echo hello",
            "git status",
            "python3 -c 'print(1+1)'",
            "mkdir -p /tmp/test",
            "curl https://example.com",
            "docker ps",
        ],
    )
    def test_safe_commands_allowed(self, command: str) -> None:
        ok, reason = check_command_filter_fast(command)
        assert ok, f"Expected {command!r} to be allowed, but it was blocked: {reason}"


class TestAllowlist:
    """Test allowlist behavior when configured."""

    def test_allowlist_blocks_non_matching(self, tmp_path: Path, monkeypatch) -> None:
        """When allowlist is set, commands not matching any pattern are blocked."""
        # Mock config to return a restrictive allowlist
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "terminal:\n"
            "  command_allowlist:\n"
            '    - "^ls\\b"\n'
            '    - "^cat\\b"\n',
            encoding="utf-8",
        )

        with patch("tools.command_filter._cache_initialized", False):
            with patch("tools.command_filter._compiled_deny_cache", None):
                with patch("tools.command_filter._compiled_allow_cache", None):
                    with patch("tools.command_filter._load_config_patterns") as mock_load:
                        # Only allow ls and cat
                        mock_load.return_value = ([], [r"^ls\b", r"^cat\b"])
                        with patch("tools.command_filter._compile_patterns") as mock_compile:
                            mock_compile.return_value = (
                                [],
                                [re.compile(r"^ls\b", re.IGNORECASE), re.compile(r"^cat\b", re.IGNORECASE)],
                            )
                            ok, reason = check_command_filter("rm -rf /tmp")
                            assert not ok
                            assert "allowlist" in reason.lower()

    def test_allowlist_permits_matching(self) -> None:
        """When allowlist is set, matching commands are allowed."""
        # The default allowlist is empty, so all safe commands pass
        ok, reason = check_command_filter_fast("ls -la")
        assert ok


class TestDenylistOverridesAllowlist:
    """Denylist should block even if allowlist would permit."""

    def test_deny_wins_over_allow(self, monkeypatch, tmp_path: Path) -> None:
        """A deny pattern should block even when an allow pattern would match."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("tools.command_filter._cache_initialized", False):
            with patch("tools.command_filter._compiled_deny_cache", None):
                with patch("tools.command_filter._compiled_allow_cache", None):
                    with patch("tools.command_filter._load_config_patterns") as mock_load:
                        mock_load.return_value = (
                            [(r"^rm\b", "rm is not allowed")],
                            [r"^rm\b"],  # allowlist includes rm
                        )
                        with patch("tools.command_filter._compile_patterns") as mock_compile:
                            deny_pat = re.compile(r"^rm\b", re.IGNORECASE)
                            allow_pat = re.compile(r"^rm\b", re.IGNORECASE)
                            mock_compile.return_value = (
                                [(deny_pat, "rm is not allowed")],
                                [allow_pat],
                            )
                            ok, reason = check_command_filter("rm -rf /tmp")
                            assert not ok
                            assert "deny" in reason.lower() or "rm is not allowed" in reason


class TestDiagnostics:
    """Test diagnostic/helper functions."""

    def test_get_denylist_patterns(self) -> None:
        """get_denylist_patterns should return compiled patterns."""
        patterns = get_denylist_patterns()
        assert len(patterns) >= len(_DEFAULT_DENY_PATTERNS)
        for pat, desc in patterns:
            assert isinstance(pat, str)
            assert isinstance(desc, str)

    def test_get_allowlist_patterns_empty_by_default(self) -> None:
        """Default allowlist should be empty."""
        patterns = get_allowlist_patterns()
        assert patterns == []


class TestPatternCompilation:
    """Test pattern compilation handles edge cases."""

    def test_invalid_pattern_logged_and_skipped(self, caplog) -> None:
        """Invalid regex patterns should be logged and skipped."""
        compiled_deny, compiled_allow = _compile_patterns(
            [("(**invalid**", "bad pattern")],
            ["[also-invalid"],
        )
        assert len(compiled_deny) == 0
        assert len(compiled_allow) == 0
        assert "Invalid deny pattern" in caplog.text
        assert "Invalid allow pattern" in caplog.text

    def test_valid_patterns_compiled(self) -> None:
        """Valid patterns should compile successfully."""
        compiled_deny, compiled_allow = _compile_patterns(
            [(r"^rm\b", "test rm")],
            [r"^ls\b"],
        )
        assert len(compiled_deny) == 1
        assert len(compiled_allow) == 1
        assert compiled_deny[0][0].match("rm -rf /")
        assert compiled_allow[0].match("ls -la")


class TestTerminalToolIntegration:
    """Integration test: verify terminal_tool respects the filter."""

    def test_terminal_tool_blocks_denylisted(self, monkeypatch, tmp_path: Path) -> None:
        """Terminal tool should return blocked status for denylisted commands."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("terminal:\n  env_type: local\n", encoding="utf-8")

        # Reset the command filter cache so it picks up the test HERMES_HOME
        import tools.command_filter as cf
        cf._cache_initialized = False
        cf._compiled_deny_cache = None
        cf._compiled_allow_cache = None

        from tools.terminal_tool import terminal_tool

        result = json.loads(terminal_tool("kill -9 -1"))
        assert result.get("status") == "blocked"
        assert "kill" in result.get("error", "").lower()
