"""Tests for real-profile local browser resolution + routing."""
import os
import ntpath
from unittest.mock import Mock, patch

import pytest


class TestRealProfileResolvers:
    def test_data_dir_windows(self):
        import hermes_cli.browser_connect as bc
        with patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\T\AppData\Local"}, clear=False):
            got = bc.real_profile_data_dir("chrome", "Windows")
        # Use ntpath basename checks so this passes on Linux CI too.
        assert got.endswith(ntpath.join("Google", "Chrome", "User Data")) or got.endswith(
            "Google\\Chrome\\User Data"
        )

    def test_data_dir_linux_edge(self):
        import hermes_cli.browser_connect as bc
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/home/t/.config"}, clear=False):
            got = bc.real_profile_data_dir("edge", "Linux")
        assert got == "/home/t/.config/microsoft-edge"

    def test_data_dir_unknown_browser_is_none(self):
        import hermes_cli.browser_connect as bc
        assert bc.real_profile_data_dir("firefox", "Windows") is None

    def test_detect_default_windows_progid_maps(self):
        import hermes_cli.browser_connect as bc
        # Non-Windows host: _detect_default_windows short-circuits via winreg
        # ImportError → None. Assert the ProgId map itself is correct instead.
        m = dict(bc._WINDOWS_PROGID_MAP)
        assert m["chromehtml"] == "chrome"
        assert m["msedgehtm"] == "edge"
        assert m["bravehtml"] == "brave"

    def test_detect_default_non_chromium_is_none(self):
        import hermes_cli.browser_connect as bc
        with patch.object(bc, "_detect_default_linux", return_value=None):
            assert bc.detect_default_chromium("Linux") is None


class TestRealProfileLaunchArgs:
    def _reset(self):
        import tools.browser_tool as bt
        bt._use_real_profile_resolved = False
        bt._cached_use_real_profile = False

    def test_consent_off_is_noop(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=False):
            args, err = bt._real_profile_launch_args()
        assert args == [] and err is None

    def test_non_chromium_default_fails_closed(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None):
            args, err = bt._real_profile_launch_args()
        assert args == []
        assert err and "not a supported Chromium" in err

    def test_chromium_default_injects_profile(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        data_dir = tmp_path / "chrome-user-data"
        data_dir.mkdir()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(data_dir)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value="/usr/bin/google-chrome"):
            args, err = bt._real_profile_launch_args()
        assert err is None
        assert "--profile" in args and str(data_dir) in args
        assert "--executable-path" in args and "/usr/bin/google-chrome" in args

    def test_missing_profile_dir_fails_closed(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(tmp_path / "nope")):
            args, err = bt._real_profile_launch_args()
        assert args == []
        assert err and "profile directory was not found" in err


class TestLocalBrowserRouting:
    def _reset(self):
        import tools.browser_tool as bt
        bt._use_real_profile_resolved = False
        bt._cached_use_real_profile = False

    def test_local_browser_forces_sidecar_with_consent(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt, "_url_is_private", return_value=False), \
             patch.object(bt, "_use_real_profile", return_value=True):
            key = bt._navigation_session_key("t1", "https://example.com", local_browser=True)
        assert key == "t1::local"

    def test_local_browser_without_cloud_provider_keeps_bare_session(self):
        """Pure local backend: the bare session already is the (real-profile)
        local Chromium; a second ``::local`` session would fight it for the
        same user-data-dir."""
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=None), \
             patch.object(bt, "_use_real_profile", return_value=True):
            key = bt._navigation_session_key("t1", "https://example.com", local_browser=True)
        assert key == "t1"

    def test_local_browser_respects_private_url_opt_out(self):
        """auto_local_for_private_urls: false is the user's LAN routing
        decision; a model-supplied flag must not override it."""
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt, "_url_is_private", return_value=True), \
             patch.object(bt, "_auto_local_for_private_urls", return_value=False), \
             patch.object(bt, "_use_real_profile", return_value=True):
            key = bt._navigation_session_key("t1", "http://192.168.1.1/admin", local_browser=True)
        assert key == "t1"

    def test_local_browser_private_url_follows_auto_local_when_allowed(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=Mock()), \
             patch.object(bt, "_url_is_private", return_value=True), \
             patch.object(bt, "_auto_local_for_private_urls", return_value=True), \
             patch.object(bt, "_use_real_profile", return_value=True):
            key = bt._navigation_session_key("t1", "http://192.168.1.1/admin", local_browser=True)
        assert key == "t1::local"

    def test_local_browser_ignored_without_consent(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value=""), \
             patch.object(bt, "_is_camofox_mode", return_value=False), \
             patch.object(bt, "_use_real_profile", return_value=False), \
             patch.object(bt, "_get_cloud_provider", return_value=None):
            key = bt._navigation_session_key("t1", "https://example.com", local_browser=True)
        assert key == "t1"

    def test_cdp_override_still_wins_over_local_browser(self):
        import tools.browser_tool as bt
        self._reset()
        with patch.object(bt, "_get_cdp_override_raw", return_value="ws://x"), \
             patch.object(bt, "_use_real_profile", return_value=True):
            key = bt._navigation_session_key("t1", "https://example.com", local_browser=True)
        assert key == "t1"
