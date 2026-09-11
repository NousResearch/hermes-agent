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
        with mod._cache_lock:
            mod._cmd_timeout_cache.clear()

        with patch("tools.browser_camofox.read_raw_config", return_value={}):
            assert _get_command_timeout() == 30

    def test_config_read_error_falls_back(self):
        """If config read raises, fall back to 30s."""
        from tools.browser_camofox import _get_command_timeout

        import tools.browser_camofox as mod
        with mod._cache_lock:
            mod._cmd_timeout_cache.clear()

        with patch("tools.browser_camofox.read_raw_config", side_effect=Exception("no config")):
            assert _get_command_timeout() == 30

    def test_multiplex_profiles_keep_independent_timeouts(self, tmp_path):
        from tools.browser_camofox import _get_command_timeout
        from agent import secret_scope as ss
        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home
        import tools.browser_camofox as mod

        profile_a = tmp_path / "profile-a"
        profile_a.mkdir()
        profile_b = tmp_path / "profile-b"
        profile_b.mkdir()
        with mod._cache_lock:
            mod._cmd_timeout_cache.clear()

        def _profile_config():
            home = get_hermes_home()
            value = 11 if home.resolve() == profile_a.resolve() else 22
            return {"browser": {"command_timeout": value}}

        ss.set_multiplex_active(True)
        try:
            with patch("tools.browser_camofox.read_raw_config", side_effect=_profile_config):
                with _profile_runtime_scope(profile_a):
                    timeout_a = _get_command_timeout()
                with _profile_runtime_scope(profile_b):
                    timeout_b = _get_command_timeout()
        finally:
            ss.set_multiplex_active(False)
            with mod._cache_lock:
                mod._cmd_timeout_cache.clear()

        assert timeout_a == 11
        assert timeout_b == 22
