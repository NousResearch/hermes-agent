"""Tests for TUI stale confirmation expiry on session resume (#60209)."""

import time
import pytest

from agent.replay_cleanup import (
    strip_stale_dangerous_confirmations,
    is_dangerous_confirmation,
    _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS,
)


class TestTuiStaleConfirmation:
    def test_stale_confirmation_stripped_on_resume(self):
        """Confirmation older than expiry window should be stripped."""
        now = time.time()
        history = [
            {"role": "user", "content": "can you restart?", "timestamp": now - 100},
            {"role": "assistant", "content": "Type confirm...", "timestamp": now - 99},
            {"role": "user", "content": "confirm forced restart", "timestamp": now - 61},
            {"role": "assistant", "content": "OK restarting", "timestamp": now - 60},
        ]
        result = strip_stale_dangerous_confirmations(history, now=now)
        # The confirmation message should be expired
        confirmations = [
            m for m in result
            if m["role"] == "user" and "confirm forced restart" in (m.get("content") or "")
        ]
        assert len(confirmations) == 0

    def test_fresh_confirmation_preserved(self):
        """Confirmation within expiry window should be kept."""
        now = time.time()
        history = [
            {"role": "user", "content": "can you restart?", "timestamp": now - 30},
            {"role": "assistant", "content": "Type confirm...", "timestamp": now - 29},
            {"role": "user", "content": "confirm forced restart", "timestamp": now - 5},
            {"role": "assistant", "content": "OK restarting", "timestamp": now - 4},
        ]
        result = strip_stale_dangerous_confirmations(history, now=now)
        confirmations = [
            m for m in result
            if m["role"] == "user" and "confirm forced restart" in (m.get("content") or "")
        ]
        assert len(confirmations) == 1

    def test_expiry_window_is_reasonable(self):
        """Expiry window should be 60 seconds (short for safety)."""
        assert _DANGEROUS_CONFIRMATION_EXPIRY_SECONDS == 60.0

    def test_dangerous_patterns_recognized(self):
        """Known dangerous confirmation patterns should be detected."""
        assert is_dangerous_confirmation("confirm forced restart")
        assert is_dangerous_confirmation("confirm forced reboot")
        assert not is_dangerous_confirmation("hello world")
