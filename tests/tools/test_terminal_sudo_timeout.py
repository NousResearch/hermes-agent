"""Tests for configurable sudo password prompt timeout.

Covers the HERMES_SUDO_TIMEOUT env var introduced to fix the hardcoded
45s wait that blocked non-interactive automation (cron, gateway) on
every accidental sudo invocation. See security audit finding #3.
"""

import pytest

from tools.terminal_tool import (
    _DEFAULT_SUDO_TIMEOUT,
    _get_sudo_timeout,
    _prompt_for_sudo_password,
)


class TestGetSudoTimeout:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_SUDO_TIMEOUT", raising=False)
        assert _get_sudo_timeout() == _DEFAULT_SUDO_TIMEOUT

    def test_default_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "")
        assert _get_sudo_timeout() == _DEFAULT_SUDO_TIMEOUT

    def test_default_when_env_whitespace(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "   ")
        assert _get_sudo_timeout() == _DEFAULT_SUDO_TIMEOUT

    def test_positive_integer_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "10")
        assert _get_sudo_timeout() == 10

    def test_zero_allowed_for_fast_skip(self, monkeypatch):
        # 0 = "skip immediately without prompting" — fast path for cron/gateway
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "0")
        assert _get_sudo_timeout() == 0

    def test_negative_clamps_to_zero(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "-5")
        assert _get_sudo_timeout() == 0

    def test_invalid_value_falls_back_to_default(self, monkeypatch, caplog):
        # A typo in config should not silently disable the prompt.
        # Fall back to the default and log a warning.
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "not-a-number")
        import logging
        with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
            assert _get_sudo_timeout() == _DEFAULT_SUDO_TIMEOUT
        assert any("HERMES_SUDO_TIMEOUT" in r.message for r in caplog.records)

    def test_large_value_allowed(self, monkeypatch):
        # No artificial upper bound -- user may want minutes for slow typists
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "600")
        assert _get_sudo_timeout() == 600


class TestPromptSkipsImmediatelyWhenTimeoutZero:
    """With HERMES_SUDO_TIMEOUT=0, the prompt must not block at all.

    Without this fix, the cron scheduler could be stalled for 45s per
    accidental sudo invocation in a scheduled job.
    """

    def test_env_zero_skips_without_blocking(self, monkeypatch):
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "0")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        import time
        start = time.monotonic()
        result = _prompt_for_sudo_password()
        elapsed = time.monotonic() - start

        assert result == ""
        # Must return essentially instantly -- anything above a small
        # threshold means the prompt actually blocked.
        assert elapsed < 0.5, f"Expected instant skip, took {elapsed:.2f}s"

    def test_explicit_zero_arg_skips_immediately(self, monkeypatch):
        # Even if env is unset, an explicit 0 arg should still skip.
        monkeypatch.delenv("HERMES_SUDO_TIMEOUT", raising=False)
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        import time
        start = time.monotonic()
        result = _prompt_for_sudo_password(timeout_seconds=0)
        elapsed = time.monotonic() - start

        assert result == ""
        assert elapsed < 0.5

    def test_negative_arg_also_skips(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        assert _prompt_for_sudo_password(timeout_seconds=-1) == ""

    def test_callback_not_invoked_when_timeout_zero(self, monkeypatch):
        """When timeout is 0, the registered UI callback must not run either.

        Otherwise a cron job using a callback-based CLI could still get a
        prompt via the callback path, defeating the point of the fast-skip.
        """
        monkeypatch.setenv("HERMES_SUDO_TIMEOUT", "0")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        called = {"value": False}

        def fake_callback():
            called["value"] = True
            return "should-not-be-returned"

        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "_sudo_password_callback", fake_callback)

        result = _prompt_for_sudo_password()

        assert result == ""
        assert called["value"] is False
