"""Regression tests for the _get_command_timeout cache race (#14331).

Before the fix:
  ``_command_timeout_resolved`` was set to ``True`` *before*
  ``_cached_command_timeout`` was assigned. If the body raised between those
  two statements (e.g. inside ``read_raw_config``), or a re-entrant/concurrent
  reader hit the cache between them, the function returned ``None``. Callers
  then evaluated ``max(None, 60)`` and crashed with::

      TypeError: '>' not supported between instances of 'int' and 'NoneType'

The fix:
  1. assign cache before flipping the resolved flag,
  2. flip the resolved flag *off* before nulling the cache in
     ``cleanup_all_browsers()``,
  3. expose ``_safe_command_timeout()`` as defense in depth.
"""

from unittest.mock import patch


class TestGetCommandTimeoutRace:
    def setup_method(self):
        from tools import browser_tool

        self.bt = browser_tool
        self._orig_cache = browser_tool._cached_command_timeout
        self._orig_resolved = browser_tool._command_timeout_resolved
        browser_tool._cached_command_timeout = None
        browser_tool._command_timeout_resolved = False

    def teardown_method(self):
        self.bt._cached_command_timeout = self._orig_cache
        self.bt._command_timeout_resolved = self._orig_resolved

    def test_returns_default_when_config_read_raises(self):
        """If config reading blows up, we still return an int (not None)."""
        with patch(
            "hermes_cli.config.read_raw_config", side_effect=RuntimeError("boom")
        ):
            result = self.bt._get_command_timeout()

        assert isinstance(result, int)
        assert result == self.bt.DEFAULT_COMMAND_TIMEOUT
        # Cache must be populated (not left as None) once resolved is True.
        assert self.bt._cached_command_timeout is not None
        assert self.bt._command_timeout_resolved is True


    def test_max_call_site_pattern_never_raises(self):
        """The exact expression from browser_navigate must not raise TypeError."""
        # Force the corrupted state the bug used to produce.
        self.bt._command_timeout_resolved = True
        self.bt._cached_command_timeout = None

        # This is the literal line from browser_navigate() after the fix.
        timeout = max(self.bt._safe_command_timeout(), 60)
        assert timeout == 60
