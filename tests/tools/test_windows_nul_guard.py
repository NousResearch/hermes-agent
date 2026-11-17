"""Tests for Windows-style NUL device redirection guard under Git Bash (Issue #103244)."""

import json
import sys
import unittest
from unittest.mock import patch

from tools.terminal_tool_guards import (
    detect_windows_nul_redirection,
    windows_nul_redirection_block,
)


class TestWindowsNulRedirectionDetection(unittest.TestCase):
    """Test suite for detecting unquoted Windows NUL device redirections."""

    def test_blocks_common_agent_redirections(self):
        payloads = [
            'rg foo "C:/definitely-does-not-exist" 2>NUL || true',
            'gh issue comment 123 --body "done" && del /q "temp.txt" 2>nul || true',
            'gh issue view 456 2>NUL || echo "unavailable"',
            'gh issue view 789 2>NUL | tail -220',
            'gh pr diff 101 --stat 2>NUL || true',
            "printf 'x' 2>NUL",
            "cmd.exe /c dir > NUL",
            "process_data >> nul",
            "build_artifact &> NUL",
            "compile_code 1>nul 2>&1",
            "run_check >& NUL",
        ]
        for cmd in payloads:
            hit, msg = detect_windows_nul_redirection(cmd)
            self.assertTrue(hit, f"Expected command to be detected: {cmd}")
            self.assertIsNotNone(msg)
            self.assertIn("/dev/null", msg)
            self.assertIn("Git Bash", msg)

    def test_allows_posix_null_and_quoted_occurrences(self):
        safe_commands = [
            "rg foo test 2>/dev/null || true",
            "cmd > /dev/null 2>&1",
            'echo "Notice: 2>NUL is invalid on bash"',
            "git commit -m 'fix: resolve 2>NUL issue in script'",
            "python -c 'x = None'",
            "cat <<'EOF'\n2>NUL is inside heredoc\nEOF",
            "grep --null -r pattern .",
            "echo $NUL",
            "npm run build",
        ]
        for cmd in safe_commands:
            hit, msg = detect_windows_nul_redirection(cmd)
            self.assertFalse(hit, f"Command should NOT be blocked: {cmd}")
            self.assertIsNone(msg)

    def test_windows_nul_redirection_block_platform_behavior(self):
        test_cmd = "rg foo 2>NUL"

        # When platform is simulated as win32
        with patch.object(sys, "platform", "win32"):
            res = windows_nul_redirection_block(test_cmd)
            self.assertIsNotNone(res)
            data = json.loads(res)
            self.assertEqual(data.get("exit_code"), 1)
            self.assertEqual(data.get("status"), "blocked")
            self.assertIn("Use POSIX '/dev/null'", data.get("error", ""))

        # When platform is simulated as linux/darwin
        with patch.object(sys, "platform", "linux"):
            res = windows_nul_redirection_block(test_cmd)
            self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
