"""Tests for sleep command interception in terminal_tool."""

import pytest
from tools.terminal_tool import _check_long_sleep_command, _MAX_FOREGROUND_SLEEP_SECONDS


class TestSleepCommandInterception:
    """Test the _check_long_sleep_command helper function."""

    def test_short_sleep_allowed(self):
        """Sleep commands under the threshold should be allowed."""
        result = _check_long_sleep_command("sleep 30", background=False)
        assert not result["blocked"]

    def test_long_sleep_blocked(self):
        """Sleep commands over the threshold should be blocked in foreground."""
        result = _check_long_sleep_command("sleep 120", background=False)
        assert result["blocked"]
        assert "background=true" in result["message"]
        assert "process(action=\"poll\")" in result["message"]

    def test_long_sleep_allowed_in_background(self):
        """Long sleep commands should be allowed when background=True."""
        result = _check_long_sleep_command("sleep 3600", background=True)
        assert not result["blocked"]

    def test_sleep_with_minutes_unit(self):
        """Sleep with 'm' unit (minutes) should be parsed correctly."""
        # 2m = 120s, over threshold
        result = _check_long_sleep_command("sleep 2m", background=False)
        assert result["blocked"]
        
        # 30s (no unit) under threshold
        result = _check_long_sleep_command("sleep 30", background=False)
        assert not result["blocked"]

    def test_sleep_with_hours_unit(self):
        """Sleep with 'h' unit (hours) should be parsed correctly."""
        result = _check_long_sleep_command("sleep 1h", background=False)
        assert result["blocked"]

    def test_sleep_in_pipeline(self):
        """Sleep command in a pipeline should still be detected."""
        result = _check_long_sleep_command("sleep 300 && tail -15 /var/log/app.log", background=False)
        assert result["blocked"]

    def test_sleep_after_semicolon(self):
        """Sleep command after semicolon should be detected."""
        result = _check_long_sleep_command("echo start; sleep 600; echo done", background=False)
        assert result["blocked"]

    def test_no_sleep_command(self):
        """Commands without sleep should pass through."""
        result = _check_long_sleep_command("ls -la /tmp", background=False)
        assert not result["blocked"]
        
        result = _check_long_sleep_command("tail -f /var/log/syslog", background=False)
        assert not result["blocked"]

    def test_sleep_exact_threshold(self):
        """Sleep exactly at threshold should be allowed."""
        result = _check_long_sleep_command(f"sleep {_MAX_FOREGROUND_SLEEP_SECONDS}", background=False)
        assert not result["blocked"]
        
        # One second over should be blocked
        result = _check_long_sleep_command(f"sleep {_MAX_FOREGROUND_SLEEP_SECONDS + 1}", background=False)
        assert result["blocked"]

    def test_sleep_case_insensitive(self):
        """Sleep detection should be case-insensitive."""
        result = _check_long_sleep_command("SLEEP 300", background=False)
        assert result["blocked"]
        
        result = _check_long_sleep_command("Sleep 300", background=False)
        assert result["blocked"]

    def test_sleepy_not_matched(self):
        """Commands containing 'sleep' as part of another word should not match."""
        # 'sleepy' is not the same as 'sleep'
        result = _check_long_sleep_command("echo sleepy300", background=False)
        assert not result["blocked"]

    def test_error_message_includes_duration(self):
        """Error message should include the detected duration."""
        result = _check_long_sleep_command("sleep 360", background=False)
        assert result["blocked"]
        assert "360s" in result["message"]
