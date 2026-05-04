"""Tests for Lightpanda engine support in browser_tool.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_engine_cache():
    """Reset the module-level engine cache so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_browser_engine = None
    bt._browser_engine_resolved = False


@pytest.fixture(autouse=True)
def _clean_engine_cache():
    """Reset engine cache before and after each test."""
    _reset_engine_cache()
    yield
    _reset_engine_cache()


# ---------------------------------------------------------------------------
# _get_browser_engine
# ---------------------------------------------------------------------------

class TestGetBrowserEngine:
    """Test engine resolution from config and env vars."""

    def test_default_is_auto(self):
        """With no config or env var, engine defaults to 'auto'."""
        from tools.browser_tool import _get_browser_engine
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_ENGINE", None)
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _get_browser_engine() == "auto"

    def test_config_lightpanda(self):
        """Config browser.engine = 'lightpanda' is respected."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "lightpanda"

    def test_config_chrome(self):
        """Config browser.engine = 'chrome' is respected."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "chrome"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "chrome"

    def test_env_var_fallback(self):
        """AGENT_BROWSER_ENGINE env var is used when config has no engine key."""
        from tools.browser_tool import _get_browser_engine
        with patch.dict(os.environ, {"AGENT_BROWSER_ENGINE": "lightpanda"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _get_browser_engine() == "lightpanda"

    def test_config_takes_priority_over_env(self):
        """Config value wins over env var."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "chrome"}}
        with patch.dict(os.environ, {"AGENT_BROWSER_ENGINE": "lightpanda"}):
            with patch("hermes_cli.config.read_raw_config", return_value=cfg):
                assert _get_browser_engine() == "chrome"

    def test_value_is_lowercased(self):
        """Engine value is normalized to lowercase."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "Lightpanda"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "lightpanda"

    def test_invalid_engine_falls_back_to_auto(self):
        """Unknown engine values are rejected and fall back to 'auto'."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "firefox"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_browser_engine() == "auto"

    def test_caching(self):
        """Result is cached — second call doesn't re-read config."""
        from tools.browser_tool import _get_browser_engine
        mock_read = MagicMock(return_value={"browser": {"engine": "lightpanda"}})
        with patch("hermes_cli.config.read_raw_config", mock_read):
            assert _get_browser_engine() == "lightpanda"
            assert _get_browser_engine() == "lightpanda"
            mock_read.assert_called_once()


# ---------------------------------------------------------------------------
# _should_inject_engine
# ---------------------------------------------------------------------------

class TestShouldInjectEngine:
    """Test whether --engine flag is injected based on mode."""

    def test_auto_never_injects(self):
        from tools.browser_tool import _should_inject_engine
        assert _should_inject_engine("auto") is False

    def test_lightpanda_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None):
            assert _should_inject_engine("lightpanda") is True

    def test_chrome_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=None):
            assert _should_inject_engine("chrome") is True

    def test_no_inject_in_camofox_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=True):
            assert _should_inject_engine("lightpanda") is False

    def test_no_inject_with_cdp_override(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value="ws://localhost:9222"):
            assert _should_inject_engine("lightpanda") is False

    def test_no_inject_with_cloud_provider(self):
        from tools.browser_tool import _should_inject_engine
        mock_provider = MagicMock()
        with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch("tools.browser_tool._get_cdp_override", return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", return_value=mock_provider):
            assert _should_inject_engine("lightpanda") is False


# ---------------------------------------------------------------------------
# _needs_lightpanda_fallback
# ---------------------------------------------------------------------------

class TestNeedsLightpandaFallback:
    """Test fallback detection for Lightpanda results."""

    def test_non_lightpanda_never_falls_back(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "timeout"}
        assert _needs_lightpanda_fallback("chrome", "open", result) is False
        assert _needs_lightpanda_fallback("auto", "open", result) is False

    def test_failed_command_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "page.goto: Timeout"}
        assert _needs_lightpanda_fallback("lightpanda", "open", result) is True

    def test_empty_snapshot_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": ""}}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True

    def test_short_snapshot_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": "- none"}}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True

    def test_normal_snapshot_does_not_trigger(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {
            "snapshot": '- heading "Example Domain" [ref=e1]\n- link "Learn more" [ref=e2]'
        }}
        assert _needs_lightpanda_fallback("lightpanda", "snapshot", result) is False

    def test_small_screenshot_triggers_fallback(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Create a tiny file simulating the Lightpanda placeholder PNG
        placeholder = tmp_path / "placeholder.png"
        placeholder.write_bytes(b"\x89PNG" + b"\x00" * 2000)  # ~2KB
        result = {"success": True, "data": {"path": str(placeholder)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is True

    def test_actual_placeholder_size_triggers_fallback(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Lightpanda PR #1766 resized the placeholder to 1920x1080 (~17 KB)
        placeholder = tmp_path / "placeholder_1920.png"
        placeholder.write_bytes(b"\x89PNG" + b"\x00" * 16693)  # actual measured: 16697 bytes
        result = {"success": True, "data": {"path": str(placeholder)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is True

    def test_normal_screenshot_does_not_trigger(self, tmp_path):
        from tools.browser_tool import _needs_lightpanda_fallback
        # Create a larger file simulating a real Chrome screenshot
        real_screenshot = tmp_path / "real.png"
        real_screenshot.write_bytes(b"\x89PNG" + b"\x00" * 50_000)  # ~50KB
        result = {"success": True, "data": {"path": str(real_screenshot)}}
        assert _needs_lightpanda_fallback("lightpanda", "screenshot", result) is False

    def test_successful_open_does_not_trigger(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"title": "Example", "url": "https://example.com"}}
        assert _needs_lightpanda_fallback("lightpanda", "open", result) is False

    def test_close_command_never_triggers_fallback(self):
        """Session-management commands like 'close' are not fallback-eligible."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "session closed"}
        assert _needs_lightpanda_fallback("lightpanda", "close", result) is False

    def test_record_command_never_triggers_fallback(self):
        """The 'record' command is tied to the engine daemon — not retryable."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "recording failed"}
        assert _needs_lightpanda_fallback("lightpanda", "record", result) is False

    def test_unknown_command_does_not_trigger_fallback(self):
        """Commands not in the whitelist should not trigger fallback."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "nope"}
        assert _needs_lightpanda_fallback("lightpanda", "some_future_cmd", result) is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify engine config is in DEFAULT_CONFIG."""

    def test_engine_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "engine" in DEFAULT_CONFIG["browser"]
        assert DEFAULT_CONFIG["browser"]["engine"] == "auto"

    def test_env_var_registered(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "AGENT_BROWSER_ENGINE" in OPTIONAL_ENV_VARS
        entry = OPTIONAL_ENV_VARS["AGENT_BROWSER_ENGINE"]
        assert entry["category"] == "tool"
        assert entry["advanced"] is True


# ---------------------------------------------------------------------------
# cleanup_all_browsers resets engine cache
# ---------------------------------------------------------------------------

class TestCleanupResetsEngineCache:
    """Verify cleanup_all_browsers resets engine-related globals."""

    def test_engine_cache_reset(self):
        import tools.browser_tool as bt
        # Seed the cache
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True
        # cleanup should reset them
        bt.cleanup_all_browsers()
        assert bt._cached_browser_engine is None
        assert bt._browser_engine_resolved is False


# ---------------------------------------------------------------------------
# _engine_override parameter
# ---------------------------------------------------------------------------

class TestEngineOverride:
    """Verify _engine_override bypasses the cached engine."""

    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_override_prevents_engine_injection(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """When _engine_override='auto', --engine flag is NOT injected."""
        import tools.browser_tool as bt

        # Set the global cache to lightpanda
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        # Track the cmd_parts that Popen receives
        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        # We need to mock the file operations too
        with patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value='{"success": true, "data": {}}'))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task1", "snapshot", [], _engine_override="auto")

        # Should NOT contain "--engine" since override is "auto"
        assert len(captured_cmds) == 1
        assert "--engine" not in captured_cmds[0]

    @patch("tools.browser_tool._get_session_info")
    @patch("tools.browser_tool._find_agent_browser", return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", return_value=True)
    @patch("tools.browser_tool._chromium_installed", return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", return_value=None)
    @patch("tools.browser_tool._get_cdp_override", return_value="")
    @patch("tools.browser_tool._is_camofox_mode", return_value=False)
    def test_no_override_uses_cached_engine(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """Without _engine_override, the cached engine is used."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            return mock_proc

        # Return a substantive snapshot so the LP fallback does NOT trigger.
        mock_stdout = '{"success": true, "data": {"snapshot": "- heading \\"Hello\\" [ref=e1]", "refs": {"e1": {}}}}'
        with patch("subprocess.Popen", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid"):
            bt._run_browser_command("task1", "snapshot", [])

        # SHOULD contain "--engine lightpanda"
        assert len(captured_cmds) == 1
        assert "--engine" in captured_cmds[0]
        engine_idx = captured_cmds[0].index("--engine")
        assert captured_cmds[0][engine_idx + 1] == "lightpanda"
