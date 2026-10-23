"""Regression tests for the browser command-timeout cache (#14331).

Before the fix the cache could be marked resolved while still holding ``None``, so a caller evaluating
``max(timeout, floor)`` crashed with ``TypeError``. ``browser_session_cloud.command_timeout`` (the floor
of the Chromium launch timeout) must answer an int even when the config read fails.
"""

from unittest.mock import patch

import pytest

from tools import browser_session_cloud


@pytest.fixture(autouse=True)
def _fresh_cache():
    browser_session_cloud.reset_caches()
    yield
    browser_session_cloud.reset_caches()


class TestGetCommandTimeoutRace:
    def test_returns_default_when_config_read_raises(self):
        """If config reading blows up, we still return an int (not None), and the memoised value is that int."""
        with patch("hermes_cli.config_effective.load_user_config_effective", side_effect=OSError("boom")):
            result = browser_session_cloud.command_timeout()

        assert isinstance(result, int)
        assert result == browser_session_cloud.DEFAULT_COMMAND_TIMEOUT
        # The cached slot must hold the int, never None.
        assert browser_session_cloud.command_timeout() == browser_session_cloud.DEFAULT_COMMAND_TIMEOUT
