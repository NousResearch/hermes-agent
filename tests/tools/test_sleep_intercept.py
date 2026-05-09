"""Tests for long-sleep intercept in terminal_tool.

Foreground `sleep N` with N > 30 must be rejected immediately so the agent
cannot be blocked for minutes at a time.

See: NousResearch/hermes-agent#8612
"""

import json

from tools.terminal_tool import terminal_tool


class TestSleepIntercept:
    """Long foreground sleep commands must be rejected with a helpful message."""

    def test_sleep_31_foreground_rejected(self):
        result = json.loads(terminal_tool("sleep 31"))
        assert result["exit_code"] == -1
        assert "rejected" in result["error"].lower()
        assert "31" in result["error"]

    def test_sleep_300_foreground_rejected(self):
        result = json.loads(terminal_tool("sleep 300"))
        assert result["exit_code"] == -1
        assert "background" in result["error"].lower()

    def test_sleep_30_foreground_allowed(self):
        """Threshold is >30, so exactly 30 must NOT be rejected by this guard."""
        result = json.loads(terminal_tool("sleep 30"))
        # Guard did not fire — exit_code is 0 (sleep ran) or not a guard rejection
        assert result.get("exit_code") != -1 or "rejected" not in result.get("error") or ""

    def test_sleep_1_foreground_allowed(self):
        result = json.loads(terminal_tool("sleep 1"))
        assert result.get("exit_code") != -1 or "rejected" not in result.get("error") or ""

    def test_sleep_31_background_allowed(self):
        """background=True bypasses the guard — long sleeps are fine in background."""
        result = json.loads(terminal_tool("sleep 31", background=True))
        error = result.get("error") or ""
        assert "rejected" not in error.lower() or "sleep" not in error.lower()

    def test_sleep_with_leading_whitespace_rejected(self):
        result = json.loads(terminal_tool("  sleep 60"))
        assert result["exit_code"] == -1
        assert "rejected" in result["error"].lower()

    def test_non_sleep_command_unaffected(self):
        """Regular commands must not be caught by the sleep guard."""
        result = json.loads(terminal_tool("echo hello"))
        assert result.get("exit_code") == 0
        assert "rejected" not in (result.get("error") or "").lower()

    def test_error_message_suggests_background(self):
        result = json.loads(terminal_tool("sleep 120"))
        assert "background" in result["error"].lower()

    def test_error_message_mentions_poll(self):
        result = json.loads(terminal_tool("sleep 120"))
        assert "poll" in result["error"].lower()
