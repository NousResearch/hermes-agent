"""Tests for browser_camofox._get_command_timeout — config-driven timeout."""
from unittest.mock import MagicMock, patch

import pytest


class TestCamofoxCommandTimeout:
    """Verify that the Camofox HTTP backend reads browser.command_timeout."""

    def test_default_is_30(self):
        """When config has no browser.command_timeout, default to 30s."""
        from tools.browser_camofox import _get_command_timeout

        # Clear cache
        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        with patch("tools.browser_camofox.read_raw_config", return_value={}):
            assert _get_command_timeout() == 30

    def test_reads_from_config(self):
        """Read browser.command_timeout from config.yaml."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        cfg = {"browser": {"command_timeout": 90}}
        with patch("tools.browser_camofox.read_raw_config", return_value=cfg):
            assert _get_command_timeout() == 90

    def test_floor_at_5s(self):
        """Config values below 5 are clamped to 5."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        cfg = {"browser": {"command_timeout": 1}}
        with patch("tools.browser_camofox.read_raw_config", return_value=cfg):
            assert _get_command_timeout() == 5

    def test_cached_after_first_call(self):
        """Config is read only once; subsequent calls use cached value."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        mock_read = MagicMock(return_value={"browser": {"command_timeout": 45}})
        with patch("tools.browser_camofox.read_raw_config", mock_read):
            _get_command_timeout()
            _get_command_timeout()
        mock_read.assert_called_once()

    def test_config_read_error_falls_back(self):
        """If config read raises, fall back to 30s."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        mod._cmd_timeout_resolved = False
        mod._cached_cmd_timeout = None

        with patch("tools.browser_camofox.read_raw_config", side_effect=Exception("no config")):
            assert _get_command_timeout() == 30

    def test_never_returns_none_under_concurrent_race_state(self):
        """A concurrent reader must never observe resolved=True with a None cache.

        The tail of ``_get_command_timeout`` used to flip ``_cmd_timeout_resolved``
        to True *before* assigning ``_cached_cmd_timeout``, with a GIL-releasing
        ``read_raw_config()`` file read in between. A second thread hitting the
        early-return guard in that window saw ``resolved=True`` and returned the
        still-``None`` cache, which flows straight into
        ``requests(..., timeout=None)`` — an indefinite hang. Reproduce the
        corrupted intermediate state directly: the guard must re-derive a real
        value instead of returning ``None``.
        """
        import tools.browser_camofox as mod

        mod._cmd_timeout_resolved = True
        mod._cached_cmd_timeout = None

        with patch("tools.browser_camofox.read_raw_config", return_value={}):
            result = mod._get_command_timeout()

        assert result == 30
        assert isinstance(result, int)
        assert mod._get_command_timeout() is not None
