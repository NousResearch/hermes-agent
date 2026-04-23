"""Regression tests for _get_command_timeout race after cleanup_all_browsers.

Issue #14331: When cleanup_all_browsers() resets the timeout cache and a
concurrent thread calls _get_command_timeout() during the window where
_command_timeout_resolved is True but _cached_command_timeout is still None,
the function must return DEFAULT_COMMAND_TIMEOUT instead of None.
"""


class TestGetCommandTimeoutRace:
    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_cached = browser_tool._cached_command_timeout
        self.orig_resolved = browser_tool._command_timeout_resolved

    def teardown_method(self):
        self.browser_tool._cached_command_timeout = self.orig_cached
        self.browser_tool._command_timeout_resolved = self.orig_resolved

    def test_returns_default_when_cache_is_none_but_resolved(self):
        """Simulate the race: resolved=True but cached=None after cleanup."""
        bt = self.browser_tool
        bt._command_timeout_resolved = True
        bt._cached_command_timeout = None

        result = bt._get_command_timeout()
        assert result == bt.DEFAULT_COMMAND_TIMEOUT
        assert isinstance(result, int)

    def test_max_does_not_raise_type_error(self):
        """The call-site uses max(_get_command_timeout(), 60); must not TypeError."""
        bt = self.browser_tool
        bt._command_timeout_resolved = True
        bt._cached_command_timeout = None

        value = max(bt._get_command_timeout(), 60)
        assert value >= 60

    def test_returns_cached_value_when_set(self):
        """Normal path: cached value is an int, return it directly."""
        bt = self.browser_tool
        bt._command_timeout_resolved = True
        bt._cached_command_timeout = 45

        assert bt._get_command_timeout() == 45
